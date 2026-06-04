"""Does decentralized priority deconfliction break the antipodal hub? (No — it makes
it collide.)

The convention (veer right) and the roundabout (shared ring) break the symmetric
antipodal deadlock by SYMMETRIC participation — every drone moves the same way.
Priority deconfliction is the opposite school: a total order in which lower-priority
drones yield and higher-priority ones proceed, ignoring the lower. This tests it as
a CBF flag `priority_yield` (decentralized order from each peer's observable, fixed
GOAL position, lexicographic; the ego avoids only higher-priority peers, assuming the
rest yield).

Antipodal swap, CBF, n=40, paired by seed, collision-vs-timeout breakdown:
  stock     plain CBF (deadlocks: collision-free TIMEOUT at the hub)
  priority  CBF + priority_yield (the social-hierarchy symmetry-breaker)
  pairwise  CBF + pairwise_bias (the convention, for reference)

Hypothesis: unlike the convention, priority does NOT solve the hub — at a
simultaneous radial convergence the ignored lower-priority peers have no room to
yield, so deadlock turns into COLLISION (a safety regression), while the symmetric
convention rescues.

  python scripts/antipodal_priority_phase.py --n-list 6 8 12 --episodes 40
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


def _planner(arm):
    p = {"type": "cbf", "max_speed": SPEED, "replan_period": 0.05, "radius": 0.4,
         "safety_margin": 0.1, "alpha": 2.0, "neighbor_dist": 15.0,
         "time_step": 0.05, "goal_radius": 1.5}
    if arm == "priority":
        p["priority_yield"] = True
    elif arm == "pairwise":
        p["pairwise_bias"] = 0.5
        p["pairwise_radius"] = 8.0
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


def _cfg(n, arm, seed, n_eps, max_steps):
    return {
        "name": f"prio_n{n}_{arm}", "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50],
                     "resolution": 1.0, "obstacles": {"type": "none"}, "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": _planner(arm),
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/prio_tmp"},
    }


def _run_cell(job):
    arm, n, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, arm, seed, n_eps, max_steps))
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


ARMS = ["stock", "priority", "pairwise"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[6, 8, 12])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--out", default="results/antipodal_priority_phase.json")
    args = ap.parse_args()

    jobs = [(arm, n, args.seed, args.episodes, args.max_steps)
            for n in args.n_list for arm in ARMS]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    cells = {}
    for arm, n, bs in res:
        cells.setdefault(n, {})[arm] = bs

    report = {"episodes": args.episodes, "cells": []}
    print(f"\nAntipodal: priority deconfliction vs the convention (CBF, n={args.episodes}, "
          f"paired; succ [coll/timeout])")
    print(f"{'N':>2} | {'stock':>14} | {'priority':>14} | {'pairwise':>14} | "
          f"{'prio vs stock':>16} | {'pw vs prio':>16}")
    print("-" * 96)
    for n in sorted(cells):
        c = cells[n]
        st, pr, pw = c["stock"], c["priority"], c["pairwise"]
        seeds = sorted(set(st) & set(pr) & set(pw))
        m = len(seeds)
        def cell(x):
            co, to = _brk(x, seeds)
            return f"{_succ(x,seeds):>2}/{m:<2}[{co:>2}c/{to:>2}t]"
        b1, c1, p1 = _mc(st, pr, seeds)
        b2, c2, p2 = _mc(pr, pw, seeds)
        print(f"{n:>2} | {cell(st):>14} | {cell(pr):>14} | {cell(pw):>14} | "
              f"b={b1:>2} c={c1:>2} p={p1:>6.4f} | b={b2:>2} c={c2:>2} p={p2:>6.4f}")
        row = {"n": n, "m": m, "prio_vs_stock": {"b": b1, "c": c1, "p": p1},
               "pw_vs_prio": {"b": b2, "c": c2, "p": p2}}
        for a in ARMS:
            co, to = _brk(c[a], seeds)
            row[a] = _succ(c[a], seeds)
            row[a + "_ct"] = [co, to]
        report["cells"].append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
