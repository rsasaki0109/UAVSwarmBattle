"""The doorway bottleneck (the social-mini-games canonical scenario): does the
right-of-way convention create lane discipline through a bidirectional gap?

The whole convention arc lives on the antipodal *hub* — a radial symmetric
convergence. The other canonical hard multi-robot scenario (LivePoint, the
discrete-time-CBF social mini-games) is the DOORWAY: a wall with a narrow gap that
two opposing streams must funnel through. The conflict is a head-on jam *inside*
the gap, a different geometry from the hub. Does the same in-plane right-of-way
that breaks the hub also break the doorway, by splitting the opposing streams onto
consistent sides of the gap (a lane)?

The reactive baselines ignore static occupancy, so this uses the static-aware
sampling MPC. A vertical wall (explicit `cells`) at x=24,25 with a gap at the
centre; N drones cross left->right and N right->left, their goals on the far side
so every one must thread the gap. Arms, paired by seed, McNemar exact:
  stock      MPC, no convention
  global     MPC + lateral_bias
  pairwise   MPC + pairwise_bias

  python scripts/doorway_convention_phase.py --n 4 --gap 6 --episodes 40
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
WALL_X = (24, 25)        # 2-cell-thick wall columns
GY_LO, GY_HI = 5.0, 45.0


def _wall_cells(gap):
    half = gap // 2
    gap_lo, gap_hi = 25 - half, 25 + half  # open rows [gap_lo, gap_hi)
    cells = []
    for x in WALL_X:
        for y in range(SIZE):
            if not (gap_lo <= y < gap_hi):
                cells.append([x, y])
    return cells


def _planner(arm, gb, pb):
    p = {"type": "mpc", "max_speed": SPEED, "replan_period": 0.2, "horizon": 40,
         "dt_plan": 0.05, "n_samples": 32, "resolution": 1.0, "inflate": 1,
         "goal_radius": 1.5, "safety_margin": 0.5, "use_prediction": True,
         "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
         "predictor": {"type": "constant_velocity"}}
    if arm == "global":
        p["lateral_bias"] = gb
    elif arm == "pairwise":
        p["pairwise_bias"] = pb
        p["pairwise_radius"] = 8.0
    return p


def _drones(n):
    # n cross left->right, n cross right->left; y spread over [20,30] so off-centre
    # drones must converge on the gap. Goal on the far side at the same y.
    out = []
    ys = [round(20.0 + i * (10.0 / max(1, n - 1)), 3) for i in range(n)]
    for i, y in enumerate(ys):
        out.append({"name": f"lr{i}", "start": [6.0, y], "goal": [44.0, y],
                    "radius": 0.4, "start_jitter": 0.5})
    for i, y in enumerate(ys):
        out.append({"name": f"rl{i}", "start": [44.0, y], "goal": [6.0, y],
                    "radius": 0.4, "start_jitter": 0.5})
    return out


def _cfg(n, gap, arm, gb, pb, seed, n_eps, max_steps):
    return {
        "name": f"door_n{n}_g{gap}_{arm}", "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [SIZE, SIZE],
                     "resolution": 1.0,
                     "obstacles": {"type": "none", "cells": _wall_cells(gap)},
                     "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": _planner(arm, gb, pb),
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/door_tmp"},
    }


def _run_cell(job):
    arm, n, gap, gb, pb, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, gap, arm, gb, pb, seed, n_eps, max_steps))
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


ARMS = ["stock", "global", "pairwise"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--gap-list", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--global-bias", type=float, default=2.0)
    ap.add_argument("--pairwise-bias", type=float, default=10.0)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--out", default="results/doorway_convention_phase.json")
    args = ap.parse_args()

    jobs = [(arm, args.n, gap, args.global_bias, args.pairwise_bias, args.seed, args.episodes, args.max_steps)
            for gap in args.gap_list for arm in ARMS]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    cells = {}
    for arm, gap, bs in res:
        cells.setdefault(gap, {})[arm] = bs

    report = {"n": args.n, "global_bias": args.global_bias, "pairwise_bias": args.pairwise_bias,
              "episodes": args.episodes, "cells": []}
    print(f"\nDoorway bottleneck (2N={2*args.n} drones, MPC): does the convention make a lane? "
          f"(n={args.episodes}, paired; succ [coll/timeout])")
    print(f"{'gap':>4} | {'stock':>13} | {'global':>13} | {'pairwise':>13} | "
          f"{'global vs stock':>17} | {'pairwise vs stock':>17}")
    print("-" * 96)
    for gap in sorted(cells):
        c = cells[gap]
        st, gl, pw = c["stock"], c["global"], c["pairwise"]
        seeds = sorted(set(st) & set(gl) & set(pw))
        m = len(seeds)
        def cell(x):
            co, to = _brk(x, seeds)
            return f"{_succ(x,seeds):>2}/{m:<2}[{co:>2}c/{to:>2}t]"
        b1, c1, p1 = _mc(st, gl, seeds)
        b2, c2, p2 = _mc(st, pw, seeds)
        print(f"{gap:>4} | {cell(st):>13} | {cell(gl):>13} | {cell(pw):>13} | "
              f"b={b1:>2} c={c1:>2} p={p1:>6.4f} | b={b2:>2} c={c2:>2} p={p2:>6.4f}")
        row = {"gap": gap, "m": m,
               "global_vs_stock": {"b": b1, "c": c1, "p": p1},
               "pairwise_vs_stock": {"b": b2, "c": c2, "p": p2}}
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
