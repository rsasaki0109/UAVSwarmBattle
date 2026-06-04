"""Priority deconfliction at the DOORWAY: the sequential conflict where it should
work (the positive counterpart to its failure at the simultaneous hub).

Priority deconfliction [fails the antipodal hub](docs/findings.md#priority-deconfliction-fails-the-symmetric-hub--it-trades-deadlock-for-collision):
at a simultaneous radial convergence the ignored lower-priority peers have nowhere
to yield. The claim was that priority is the tool for SEQUENTIAL conflicts, where
one party can wait. The doorway is exactly that — two opposing streams funnel
through a narrow gap and must take turns. This tests whether priority, useless at
the hub, WORKS at the doorway (completing the dichotomy priority<->sequential,
convention<->simultaneous).

MPC (static-aware; the wall is explicit cells), 2N drones cross both ways through
the gap. Arms, paired by seed:
  stock      no rule (head-on jam in the gap)
  priority   priority_yield (lower-goal stream waits, higher goes first)
  pairwise   pairwise_bias convention (lane discipline)

  python scripts/doorway_priority_phase.py --n 3 --gap-list 6 --episodes 30
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
SIZE = 50
WALL_X = (24, 25)


def _wall_cells(gap):
    half = gap // 2
    lo, hi = 25 - half, 25 + half
    return [[x, y] for x in WALL_X for y in range(SIZE) if not (lo <= y < hi)]


def _planner(arm):
    p = {"type": "mpc", "max_speed": SPEED, "replan_period": 0.2, "horizon": 40,
         "dt_plan": 0.05, "n_samples": 32, "resolution": 1.0, "inflate": 1,
         "goal_radius": 1.5, "safety_margin": 0.5, "use_prediction": True,
         "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
         "predictor": {"type": "constant_velocity"}}
    if arm == "priority":
        p["priority_yield"] = True
    elif arm == "pairwise":
        p["pairwise_bias"] = 10.0
        p["pairwise_radius"] = 8.0
    return p


def _drones(n):
    out = []
    ys = [round(20.0 + i * (10.0 / max(1, n - 1)), 3) for i in range(n)]
    for i, y in enumerate(ys):
        out.append({"name": f"lr{i}", "start": [6.0, y], "goal": [44.0, y], "radius": 0.4, "start_jitter": 0.5})
    for i, y in enumerate(ys):
        out.append({"name": f"rl{i}", "start": [44.0, y], "goal": [6.0, y], "radius": 0.4, "start_jitter": 0.5})
    return out


def _cfg(n, gap, arm, seed, n_eps, max_steps):
    return {
        "name": f"pdoor_n{n}_g{gap}_{arm}", "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [SIZE, SIZE], "resolution": 1.0,
                     "obstacles": {"type": "none", "cells": _wall_cells(gap)}, "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": _planner(arm),
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/pdoor_tmp"},
    }


def _run_cell(job):
    arm, n, gap, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, gap, arm, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (arm, gap, by_seed)


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
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--gap-list", type=int, nargs="+", default=[6])
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--out", default="results/doorway_priority_phase.json")
    args = ap.parse_args()

    jobs = [(arm, args.n, gap, args.seed, args.episodes, args.max_steps)
            for gap in args.gap_list for arm in ARMS]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    cells = {}
    for arm, gap, bs in res:
        cells.setdefault(gap, {})[arm] = bs

    report = {"n": args.n, "episodes": args.episodes, "cells": []}
    print(f"\nDoorway (2N={2*args.n}, MPC): does PRIORITY work at the sequential gap? "
          f"(n={args.episodes}, paired; succ [coll/timeout])")
    print(f"{'gap':>4} | {'stock':>14} | {'priority':>14} | {'pairwise':>14} | "
          f"{'prio vs stock':>16} | {'pw vs stock':>16}")
    print("-" * 92)
    for gap in sorted(cells):
        c = cells[gap]
        st, pr, pw = c["stock"], c["priority"], c["pairwise"]
        seeds = sorted(set(st) & set(pr) & set(pw))
        m = len(seeds)
        def cell(x):
            co, to = _brk(x, seeds)
            return f"{_succ(x,seeds):>2}/{m:<2}[{co:>2}c/{to:>2}t]"
        b1, c1, p1 = _mc(st, pr, seeds)
        b2, c2, p2 = _mc(st, pw, seeds)
        print(f"{gap:>4} | {cell(st):>14} | {cell(pr):>14} | {cell(pw):>14} | "
              f"b={b1:>2} c={c1:>2} p={p1:>6.4f} | b={b2:>2} c={c2:>2} p={p2:>6.4f}")
        row = {"gap": gap, "m": m, "prio_vs_stock": {"b": b1, "c": c1, "p": p1},
               "pw_vs_stock": {"b": b2, "c": c2, "p": p2}}
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
