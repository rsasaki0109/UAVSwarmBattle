"""Is the convention a CONSENSUS device? A split convention (half veer right, half
left) collides — proving the shared DIRECTION, not the tilt, breaks the symmetry.

The right-of-way convention rescues the antipodal hub by tilting every drone the
same way (all RIGHT) into a clockwise roundabout. Is the rescue from the tilt
itself, or from every drone AGREEING on the direction? This splits the fleet: half
the drones obey lateral_bias=+B (veer right) and half lateral_bias=-B (veer left).
If a split convention fails where the unanimous one succeeds, the convention's
power is CONSENSUS on the direction, not the act of tilting.

Antipodal swap, MPC, alternating around the ring, paired by seed:
  stock      lateral_bias 0 (deadlock)
  consensus  every drone lateral_bias +B (unanimous right)
  split      alternating lateral_bias +B / -B (half right, half left)

  python scripts/antipodal_split_convention_phase.py --n-list 4 6 8 --bias 2 --episodes 40
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


def _base():
    return {"type": "mpc", "max_speed": SPEED, "replan_period": 0.2, "horizon": 40,
            "dt_plan": 0.05, "n_samples": 48, "resolution": 1.0, "inflate": 1,
            "goal_radius": 1.5, "safety_margin": 0.5, "use_prediction": True,
            "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
            "predictor": {"type": "constant_velocity"}}


def _planner(arm, n, bias):
    p = _base()
    if arm == "consensus":
        p["lateral_bias"] = bias
    elif arm == "split":
        # alternating right (+) / left (-) around the ring
        p["per_drone"] = [{"lateral_bias": bias if k % 2 == 0 else -bias} for k in range(n)]
    return p


def _drones(n):
    out = []
    for k in range(n):
        a = 2.0 * math.pi * k / n
        out.append({"name": f"d{k}",
                    "start": [round(CX + RADIUS * math.cos(a), 3), round(CY + RADIUS * math.sin(a), 3)],
                    "goal": [round(CX - RADIUS * math.cos(a), 3), round(CY - RADIUS * math.sin(a), 3)],
                    "radius": 0.4, "start_jitter": 0.8})
    return out


def _cfg(n, arm, bias, seed, n_eps, max_steps):
    return {
        "name": f"split_n{n}_{arm}", "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50],
                     "resolution": 1.0, "obstacles": {"type": "none"}, "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": _planner(arm, n, bias),
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/split_tmp"},
    }


def _run_cell(job):
    arm, n, bias, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, arm, bias, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (arm, n, by_seed)


def _succ(bs, seeds):
    return sum(bs[s] == "success" for s in seeds)


def _brk(bs, seeds):
    return (sum(bs[s] == "collision" for s in seeds),
            sum(bs[s] == "timeout" for s in seeds))


def _mc(a, b, seeds):
    bb = sum(a[s] == "success" and b[s] != "success" for s in seeds)
    cc = sum(a[s] != "success" and b[s] == "success" for s in seeds)
    return bb, cc, mcnemar_exact_p(bb, cc)


ARMS = ["stock", "consensus", "split"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--bias", type=float, default=2.0)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--out", default="results/antipodal_split_convention_phase.json")
    args = ap.parse_args()

    jobs = [(arm, n, args.bias, args.seed, args.episodes, args.max_steps)
            for n in args.n_list for arm in ARMS]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    cells = {}
    for arm, n, bs in res:
        cells.setdefault(n, {})[arm] = bs

    report = {"bias": args.bias, "episodes": args.episodes, "cells": []}
    print(f"\nIs the convention a consensus device? antipodal, MPC, bias={args.bias}, "
          f"n={args.episodes} (succ [coll/timeout])")
    print(f"{'N':>2} | {'stock':>14} | {'consensus':>14} | {'split':>14} | "
          f"{'cons vs stock':>16} | {'cons vs split':>16}")
    print("-" * 92)
    for n in sorted(cells):
        c = cells[n]
        st, co, sp = c["stock"], c["consensus"], c["split"]
        seeds = sorted(set(st) & set(co) & set(sp))
        m = len(seeds)
        def cell(x):
            cc, to = _brk(x, seeds)
            return f"{_succ(x,seeds):>2}/{m:<2}[{cc:>2}c/{to:>2}t]"
        b1, c1, p1 = _mc(st, co, seeds)
        b2, c2, p2 = _mc(sp, co, seeds)
        print(f"{n:>2} | {cell(st):>14} | {cell(co):>14} | {cell(sp):>14} | "
              f"b={b1:>2} c={c1:>2} p={p1:>6.4f} | b={b2:>2} c={c2:>2} p={p2:>6.4f}")
        row = {"n": n, "m": m, "consensus_vs_stock": {"b": b1, "c": c1, "p": p1},
               "consensus_vs_split": {"b": b2, "c": c2, "p": p2}}
        for a in ARMS:
            cc, to = _brk(c[a], seeds)
            row[a] = _succ(c[a], seeds)
            row[a + "_ct"] = [cc, to]
        report["cells"].append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
