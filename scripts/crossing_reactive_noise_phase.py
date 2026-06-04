"""Which reactive collision avoider degrades most gracefully under noisy peer sensing?

Every reactive-baseline result so far used PERFECT peer observations. Real fleets
track peers with error. This sweeps Gaussian position noise on the peer tracker
(`sensor: noisy_tracker`) and measures how ORCA, CBF and APF degrade on a
perpendicular crossing where all three succeed under perfect sensing.

The three use peer state differently, so they should not degrade alike:
  ORCA  velocity-obstacle from peer position + velocity (geometric, tight margins)
  CBF   barrier from peer position + velocity (a safety-rate buffer)
  APF   repulsion from peer position only (a soft distance field)

Arms = {orca, cbf, apf} x position_noise_std, homogeneous fleets, paired by seed,
McNemar exact between methods at each noise level.

  python scripts/crossing_reactive_noise_phase.py --n 4 --noise-list 0 0.25 0.5 1.0 --episodes 40
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import tempfile
from multiprocessing import Pool
from pathlib import Path

from uav_nav_lab.config import ExperimentConfig
from uav_nav_lab.runner.multi.experiment import run_experiment_multi
from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p

SPEED = 5.0
GAP = 2.2
LO, HI = 5.0, 45.0


def _planner(kind):
    if kind == "orca":
        return {"type": "orca", "max_speed": SPEED, "replan_period": 0.05, "radius": 0.4,
                "safety_margin": 0.1, "time_horizon": 2.0, "time_step": 0.05,
                "neighbor_dist": 15.0, "goal_radius": 1.5}
    if kind == "cbf":
        return {"type": "cbf", "max_speed": SPEED, "replan_period": 0.05, "radius": 0.4,
                "safety_margin": 0.1, "alpha": 2.0, "time_step": 0.05,
                "neighbor_dist": 15.0, "goal_radius": 1.5}
    return {"type": "apf", "max_speed": SPEED, "replan_period": 0.05, "radius": 0.4,
            "k_att": 1.0, "k_rep": 6.0, "influence_dist": 4.0, "time_step": 0.05,
            "goal_radius": 1.5}


def _drones(n):
    out = []
    span = (n - 1) * GAP
    base = 25.0 - span / 2.0
    for i in range(n):
        y = round(base + i * GAP, 3)
        out.append({"name": f"ax{i}", "start": [LO, y], "goal": [HI, y], "radius": 0.4, "start_jitter": 0.6})
    for i in range(n):
        x = round(base + i * GAP, 3)
        out.append({"name": f"by{i}", "start": [x, LO], "goal": [x, HI], "radius": 0.4, "start_jitter": 0.6})
    return out


def _cfg(n, kind, noise, seed, n_eps, max_steps):
    return {
        "name": f"xnoise_n{n}_{kind}_{noise}", "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50],
                     "resolution": 1.0, "obstacles": {"type": "none"}, "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": _planner(kind),
        "sensor": {"type": "noisy_tracker", "position_noise_std": noise,
                   "velocity_noise_std": 0.0, "delay": 0.0},
        "output": {"dir": "results/xnoise_tmp"},
    }


def _run_cell(job):
    kind, noise, n, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, kind, noise, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (kind, noise, by_seed)


def _succ(bs, seeds):
    return sum(bs[s] == "success" for s in seeds)


def _brk(bs, seeds):
    return (sum(bs[s] == "collision" for s in seeds),
            sum(bs[s] == "timeout" for s in seeds))


def _mc(a, b, seeds):
    bb = sum(a[s] == "success" and b[s] != "success" for s in seeds)
    cc = sum(a[s] != "success" and b[s] == "success" for s in seeds)
    return bb, cc, mcnemar_exact_p(bb, cc)


KINDS = ["orca", "cbf", "apf"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--noise-list", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--out", default="results/crossing_reactive_noise_phase.json")
    args = ap.parse_args()

    jobs = [(k, nz, args.n, args.seed, args.episodes, args.max_steps)
            for nz in args.noise_list for k in KINDS]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    cells = {}
    for kind, noise, bs in res:
        cells.setdefault(noise, {})[kind] = bs

    report = {"n": args.n, "speed": SPEED, "gap": GAP, "episodes": args.episodes, "cells": []}
    print(f"\nReactive avoidance under noisy peer sensing (perpendicular crossing, 2N={2*args.n} "
          f"drones, n={args.episodes}, paired; succ [coll/timeout])")
    print(f"{'noise':>6} | {'orca':>14} | {'cbf':>14} | {'apf':>14} | "
          f"{'apf vs orca':>16} | {'apf vs cbf':>16}")
    print("-" * 96)
    for nz in sorted(cells):
        c = cells[nz]
        o, b, a = c["orca"], c["cbf"], c["apf"]
        seeds = sorted(set(o) & set(b) & set(a))
        m = len(seeds)
        def cell(x):
            co, to = _brk(x, seeds)
            return f"{_succ(x,seeds):>2}/{m:<2}[{co:>2}c/{to:>2}t]"
        b1, c1, p1 = _mc(o, a, seeds)   # apf vs orca
        b2, c2, p2 = _mc(b, a, seeds)   # apf vs cbf
        print(f"{nz:>6} | {cell(o):>14} | {cell(b):>14} | {cell(a):>14} | "
              f"b={b1:>2} c={c1:>2} p={p1:>6.4f} | b={b2:>2} c={c2:>2} p={p2:>6.4f}")
        row = {"noise": nz, "m": m,
               "apf_vs_orca": {"b": b1, "c": c1, "p": p1},
               "apf_vs_cbf": {"b": b2, "c": c2, "p": p2}}
        for k in KINDS:
            co, to = _brk(c[k], seeds)
            row[k] = _succ(c[k], seeds)
            row[k + "_ct"] = [co, to]
        report["cells"].append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
