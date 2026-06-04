"""Two reciprocal schools, opposite failure signatures — and one convention that
rescues both.

ORCA (velocity-space, planner.type:orca, PR #85) fails the symmetric antipodal
swap by COLLISION: the reciprocal velocity split funnels everyone onto the hub.
BVC (position-space, planner.type:bvc) cannot collide by construction (buffered
Voronoi cells are disjoint), so it fails the SAME swap by DEADLOCK / TIMEOUT:
the goal-ward face of each agent's cell is cut off at the hub and it stops.

This script puts both stock baselines and the BVC right-of-way ports on the same
N-drone 2D antipodal benchmark, paired by seed, with a collision-vs-timeout
breakdown (the whole point):

  orca_stock    ORCA, no convention        -> expect COLLISION failures
  bvc_stock     BVC, no convention         -> expect TIMEOUT failures (0 coll)
  bvc_global    BVC + global lateral_bias  -> does the convention rescue BVC?
  bvc_pairwise  BVC + pairwise_bias        -> neighbour-conditional version

Hypotheses: (1) the two stock baselines fail the swap with OPPOSITE signatures
(orca collides, bvc times out); (2) the right-of-way convention — proven
planner-agnostic for MPC (#68/#84) and ORCA (#85) — also rescues BVC, the third
family; (3) BVC stays collision-free even when it fails (timeouts only).

  python scripts/antipodal_bvc_phase.py --n-list 4 6 8 12 --episodes 40
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

CX, CY = 25.0, 25.0
RADIUS = 20.0
SPEED = 5.0


def _drones(n):
    out = []
    for k in range(n):
        ang = 2.0 * math.pi * k / n
        sx, sy = CX + RADIUS * math.cos(ang), CY + RADIUS * math.sin(ang)
        gx, gy = CX - RADIUS * math.cos(ang), CY - RADIUS * math.sin(ang)
        out.append({"name": f"d{k}",
                    "start": [round(sx, 3), round(sy, 3)],
                    "goal": [round(gx, 3), round(gy, 3)],
                    "radius": 0.4, "start_jitter": 0.8})
    return out


# Operating point matched to the ORCA baseline (#85): default replan cadence
# (0.5 s) so the reactive controllers reproduce the antipodal deadlock; each
# Reactive controllers run every control step (replan = sim dt); time_step is
# matched. BVC's hard position constraint needs a BRAKE-aware buffer under the
# accel-limited sim: stopping distance v^2/(2 a) = 5^2/12 ~= 2.1 m, so a small
# buffer overshoots and collides; BRAKE_MARGIN covers it (the dynamics-awareness
# the textbook single-integrator BVC omits). CBF's alpha gain brakes gradually,
# so it needs no such margin.
REPLAN = 0.05  # overridden by --replan; internal time_step is matched to it
# BVC buffer sweet spot (under the accel-limited sim): 0.1 overshoots & collides,
# ~2.5 fully collision-free but halts so early the convention can't route, ~1.0
# is the middle band where BVC mostly deadlocks AND the convention can rescue it.
BRAKE_MARGIN = 1.0


def _planner(arm, gbias, pbias):
    if arm.startswith("cbf"):
        p = {"type": "cbf", "max_speed": SPEED, "replan_period": REPLAN,
             "radius": 0.4, "safety_margin": 0.1, "alpha": 2.0,
             "neighbor_dist": 15.0, "time_step": REPLAN, "goal_radius": 1.5}
    else:
        # bvc_nobrake = textbook buffer (overshoots under accel limits);
        # all other bvc arms use the brake-aware buffer.
        sm = 0.1 if arm == "bvc_nobrake" else BRAKE_MARGIN
        p = {"type": "bvc", "max_speed": SPEED, "replan_period": REPLAN,
             "radius": 0.4, "safety_margin": sm, "neighbor_dist": 15.0,
             "time_step": REPLAN, "goal_radius": 1.5, "proj_iters": 20}
    if arm.endswith("_global"):
        p["lateral_bias"] = gbias
    elif arm.endswith("_pairwise"):
        p["pairwise_bias"] = pbias
        p["pairwise_radius"] = 8.0
    return p


def _cfg(n, arm, gbias, pbias, seed, n_eps, max_steps):
    return {
        "name": f"bvc_n{n}_{arm}", "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50],
                     "resolution": 1.0, "obstacles": {"type": "none"},
                     "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": _planner(arm, gbias, pbias),
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/bvc_tmp"},
    }


def _run_cell(job):
    arm, n, gbias, pbias, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, arm, gbias, pbias, seed, n_eps, max_steps))
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


ARMS = ["bvc_nobrake", "bvc_stock", "bvc_pairwise", "cbf_stock", "cbf_pairwise"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[4, 6, 8, 12])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--global-bias", type=float, default=0.2)
    ap.add_argument("--pairwise-bias", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--replan", type=float, default=0.05)
    ap.add_argument("--out", default="results/antipodal_bvc_phase.json")
    args = ap.parse_args()
    global REPLAN
    REPLAN = args.replan

    jobs = [(arm, n, args.global_bias, args.pairwise_bias, args.seed,
             args.episodes, args.max_steps)
            for n in args.n_list for arm in ARMS]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    cells = {}
    for arm, n, bs in res:
        cells.setdefault(n, {})[arm] = bs

    report = {"radius": RADIUS, "speed": SPEED, "episodes": args.episodes,
              "global_bias": args.global_bias, "pairwise_bias": args.pairwise_bias,
              "cells": []}
    print(f"\n2D antipodal: reactive-family failure signatures + convention "
          f"(n={args.episodes}, paired; succ [coll/timeout])")
    print(f"{'N':>2} | " + " | ".join(f"{a:>13}" for a in ARMS) +
          f" | {'bvc pw/stk':>12} | {'cbf pw/stk':>12}")
    print("-" * 130)
    for n in sorted(cells):
        c = cells[n]
        seeds = sorted(set.intersection(*[set(c[a]) for a in ARMS]))
        m = len(seeds)
        def cell(a):
            co, to = _brk(c[a], seeds)
            return f"{_succ(c[a],seeds):>2}/{m:<2}[{co:>2}c/{to:>2}t]"
        bb, cb, pb = _mc(c["bvc_stock"], c["bvc_pairwise"], seeds)
        bc, cc2, pc = _mc(c["cbf_stock"], c["cbf_pairwise"], seeds)
        print(f"{n:>2} | " + " | ".join(f"{cell(a):>13}" for a in ARMS) +
              f" | c={cb:>2} p={pb:>5.3f} | c={cc2:>2} p={pc:>5.3f}")
        row = {"n": n, "m": m,
               "bvc_pw_vs_stock": {"b": bb, "c": cb, "p": pb},
               "cbf_pw_vs_stock": {"b": bc, "c": cc2, "p": pc}}
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
