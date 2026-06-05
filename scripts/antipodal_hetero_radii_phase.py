"""Does the right-of-way convention survive a fleet of different-SIZE drones?

The convention's robustness is established across speed heterogeneity (a 4x spread
is fine) and was bounded for controller/direction mismatch. The remaining
heterogeneity axis is SIZE: a roundabout sized for uniform drones must also hold
the bigger ones, which need more clearance. Does a mixed big/small fleet still
round the antipodal hub under the convention, or does size heterogeneity cap it?

Antipodal swap, MPC + pairwise convention, alternating big/small radii around the
ring (mean held at 0.4 m; each drone's safety_margin scales with its own radius so
it keeps its own clearance). Paired by seed, McNemar exact:
  homo_pw     uniform r=0.4, pairwise convention (reference rescue)
  het_pw      alternating big/small radii, pairwise convention
  het_stock   alternating radii, NO convention (deadlock baseline)

NOTE: relies on the peer-collision fix (each drone checked with its OWN radius);
without it big drones would phase through undetected.

  python scripts/antipodal_hetero_radii_phase.py --n 6 --spread-list 0 0.3 0.6 0.9 --episodes 40
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import math
import tempfile
from multiprocessing import Pool
from pathlib import Path

from uav_nav_lab.config import ExperimentConfig
from uav_nav_lab.runner.multi.experiment import run_experiment_multi
from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p

SPEED = 5.0
CX, CY = 25.0, 25.0
RADIUS = 20.0
PW = 10.0


def _radii(n, spread):
    # alternating big / small around the ring, mean ~0.4
    big, small = 0.4 + spread / 2.0, max(0.2, 0.4 - spread / 2.0)
    return [big if k % 2 == 0 else small for k in range(n)]


def _drones(n, spread):
    rs = _radii(n, spread)
    out = []
    for k in range(n):
        a = 2.0 * math.pi * k / n
        out.append({"name": f"d{k}",
                    "start": [round(CX + RADIUS * math.cos(a), 3), round(CY + RADIUS * math.sin(a), 3)],
                    "goal": [round(CX - RADIUS * math.cos(a), 3), round(CY - RADIUS * math.sin(a), 3)],
                    "radius": rs[k], "start_jitter": 0.8})
    return out


def _per_drone(n, spread, conv):
    # each drone's safety_margin scales with its own radius (it does not otherwise
    # know its own size); pairwise convention on when conv.
    rs = _radii(n, spread)
    out = []
    for k in range(n):
        d = {"safety_margin": round(0.1 + rs[k], 3)}
        if conv:
            d["pairwise_bias"] = PW
            d["pairwise_radius"] = 8.0
        out.append(d)
    return out


def _cfg(n, spread, conv, seed, n_eps, max_steps):
    base = {"type": "mpc", "max_speed": SPEED, "replan_period": 0.2, "horizon": 40,
            "dt_plan": 0.05, "n_samples": 32, "resolution": 1.0, "inflate": 1,
            "goal_radius": 1.5, "use_prediction": True, "w_goal": 1.0,
            "w_obs": 100.0, "w_smooth": 0.05, "predictor": {"type": "constant_velocity"},
            "per_drone": _per_drone(n, spread, conv)}
    return {
        "name": f"radii_n{n}_s{spread}_{'pw' if conv else 'stock'}", "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50], "resolution": 1.0,
                     "obstacles": {"type": "none"}, "drones": _drones(n, spread)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": base, "sensor": {"type": "perfect"},
        "output": {"dir": "results/radii_tmp"},
    }


def _run_cell(job):
    arm, n, spread, conv, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, spread, conv, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (arm, spread, by_seed)


def _succ(bs, seeds):
    return sum(bs[s] == "success" for s in seeds)


def _brk(bs, seeds):
    return (sum(bs[s] == "collision" for s in seeds),
            sum(bs[s] == "timeout" for s in seeds))


def _mc(a, b, seeds):
    bb = sum(a[s] == "success" and b[s] != "success" for s in seeds)
    cc = sum(a[s] != "success" and b[s] == "success" for s in seeds)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--spread-list", type=float, nargs="+", default=[0.0, 0.3, 0.6, 0.9])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--out", default="results/antipodal_hetero_radii_phase.json")
    args = ap.parse_args()

    # homo_pw uses spread 0 (it is the spread=0 het_pw cell); we report het_pw and
    # het_stock across spreads, with the spread-0 pw cell as the homogeneous ref.
    jobs = []
    for s in args.spread_list:
        jobs.append(("het_pw", args.n, s, True, args.seed, args.episodes, args.max_steps))
        jobs.append(("het_stock", args.n, s, False, args.seed, args.episodes, args.max_steps))
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    cells = {}
    for arm, s, bs in res:
        cells.setdefault(s, {})[arm] = bs

    report = {"n": args.n, "pairwise_bias": PW, "episodes": args.episodes, "cells": []}
    print(f"\nHeterogeneous-radii antipodal (N={args.n}, MPC, mean r=0.4): does the convention "
          f"survive mixed sizes? (n={args.episodes}, paired; succ [coll/timeout])")
    print(f"{'spread':>6} | {'radii':>11} | {'het_stock':>14} | {'het_pw':>14} | {'pw vs stock':>16}")
    print("-" * 74)
    for s in sorted(cells):
        c = cells[s]
        st, pw = c["het_stock"], c["het_pw"]
        seeds = sorted(set(st) & set(pw))
        m = len(seeds)
        big, small = 0.4 + s / 2.0, max(0.2, 0.4 - s / 2.0)
        def cell(x):
            cc, to = _brk(x, seeds)
            return f"{_succ(x,seeds):>2}/{m:<2}[{cc:>2}c/{to:>2}t]"
        b, cc, p = _mc(st, pw, seeds)
        print(f"{s:>6.1f} | {big:.2f}/{small:.2f} | {cell(st):>14} | {cell(pw):>14} | "
              f"b={b:>2} c={cc:>2} p={p:>6.4f}")
        report["cells"].append({"spread": s, "big": big, "small": small, "m": m,
            "het_stock": _succ(st, seeds), "het_pw": _succ(pw, seeds),
            "het_stock_ct": _brk(st, seeds), "het_pw_ct": _brk(pw, seeds),
            "pw_vs_stock": {"b": b, "c": cc, "p": p}})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
