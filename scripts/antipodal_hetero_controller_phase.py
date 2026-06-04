"""On the symmetric antipodal hub, mixing reciprocal controllers HELPS — the mirror
of its harm on the crossing.

A [perpendicular crossing](docs/findings.md#two-reciprocal-collision-avoiders-are-less-safe-mixed-than-either-is-alone)
showed mixing ORCA and CBF is *less* safe than either alone — protocol heterogeneity
breaks the shared reciprocity. But that geometry has no symmetry to break. The
antipodal swap is the opposite: a *symmetric* convergence where a homogeneous
reactive fleet deadlocks (ORCA collides at the hub, CBF times out). There,
heterogeneity should HELP — alternating two avoidance protocols around the ring
desyncs the mirror-symmetric manoeuvre, the same way mixing predictors does
(docs/findings.md "heterogeneous predictor swarms break the antipodal deadlock by
desync").

N drones, alternating ORCA / CBF around the ring, default replan cadence (so both
homogeneous fleets reproduce the deadlock), paired by seed, McNemar exact:
  all_orca   every drone ORCA   (deadlocks: collisions at the hub)
  all_cbf    every drone CBF    (deadlocks: timeouts at the hub)
  mixed      alternating ORCA/CBF

Hypothesis: `mixed` beats BOTH homogeneous fleets at moderate density (the protocol
mismatch that *crashes* drones on a crossing instead *desyncs* the symmetric hub
convergence here), with the effect fading as density rises.

  python scripts/antipodal_hetero_controller_phase.py --n-list 4 6 8 --episodes 40
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
REPLAN = 0.5  # default cadence — both homogeneous fleets deadlock here


def _orca():
    return {"type": "orca", "max_speed": SPEED, "radius": 0.4, "safety_margin": 0.1,
            "time_horizon": 2.0, "time_step": 0.25, "neighbor_dist": 15.0,
            "goal_radius": 1.5}


def _cbf():
    return {"type": "cbf", "max_speed": SPEED, "radius": 0.4, "safety_margin": 0.1,
            "alpha": 2.0, "time_step": REPLAN, "neighbor_dist": 15.0, "goal_radius": 1.5}


def _drones(n):
    out = []
    for k in range(n):
        a = 2.0 * math.pi * k / n
        out.append({"name": f"d{k}",
                    "start": [round(CX + RADIUS * math.cos(a), 3), round(CY + RADIUS * math.sin(a), 3)],
                    "goal": [round(CX - RADIUS * math.cos(a), 3), round(CY - RADIUS * math.sin(a), 3)],
                    "radius": 0.4, "start_jitter": 0.8})
    return out


def _per_drone(n, arm):
    if arm == "all_orca":
        return [_orca() for _ in range(n)]
    if arm == "all_cbf":
        return [_cbf() for _ in range(n)]
    return [_orca() if k % 2 == 0 else _cbf() for k in range(n)]  # alternating


def _cfg(n, arm, seed, n_eps, max_steps):
    return {
        "name": f"antihetero_n{n}_{arm}", "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50],
                     "resolution": 1.0, "obstacles": {"type": "none"},
                     "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": {"type": "orca", "max_speed": SPEED, "replan_period": REPLAN,
                    "per_drone": _per_drone(n, arm)},
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/antihetero_tmp"},
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


ARMS = ["all_orca", "all_cbf", "mixed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--out", default="results/antipodal_hetero_controller_phase.json")
    args = ap.parse_args()

    jobs = [(arm, n, args.seed, args.episodes, args.max_steps)
            for n in args.n_list for arm in ARMS]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    cells = {}
    for arm, n, bs in res:
        cells.setdefault(n, {})[arm] = bs

    report = {"speed": SPEED, "radius": RADIUS, "replan": REPLAN,
              "episodes": args.episodes, "cells": []}
    print(f"\nAntipodal: does mixing reciprocal controllers HELP (desync the symmetric "
          f"deadlock)? (n={args.episodes}, paired; succ [coll/timeout])")
    print(f"{'N':>2} | {'all_orca':>14} | {'all_cbf':>14} | {'mixed':>14} | "
          f"{'mixed vs orca':>18} | {'mixed vs cbf':>18}")
    print("-" * 96)
    for n in sorted(cells):
        c = cells[n]
        o, b, mx = c["all_orca"], c["all_cbf"], c["mixed"]
        seeds = sorted(set(o) & set(b) & set(mx))
        m = len(seeds)
        co, to = _brk(o, seeds); cb, tb = _brk(b, seeds); cm, tm = _brk(mx, seeds)
        b1, c1, p1 = _mc(o, mx, seeds)
        b2, c2, p2 = _mc(b, mx, seeds)
        print(f"{n:>2} | {_succ(o,seeds):>2}/{m:<2}[{co:>2}c/{to:>2}t] | "
              f"{_succ(b,seeds):>2}/{m:<2}[{cb:>2}c/{tb:>2}t] | "
              f"{_succ(mx,seeds):>2}/{m:<2}[{cm:>2}c/{tm:>2}t] | "
              f"b={b1:>2} c={c1:>2} p={p1:>7.4f} | b={b2:>2} c={c2:>2} p={p2:>7.4f}")
        report["cells"].append({"n": n, "m": m,
            "all_orca": _succ(o, seeds), "all_orca_ct": [co, to],
            "all_cbf": _succ(b, seeds), "all_cbf_ct": [cb, tb],
            "mixed": _succ(mx, seeds), "mixed_ct": [cm, tm],
            "mixed_vs_orca": {"b": b1, "c": c1, "p": p1},
            "mixed_vs_cbf": {"b": b2, "c": c2, "p": p2}})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
