"""Explicit roundabout (Merry-Go-Round) vs the implicit cost-bias convention:
scalability vs speed on the antipodal swap.

The lab's `lateral_bias` / `pairwise_bias` conventions break the antipodal deadlock
*implicitly* (a small cost nudge) and cheaply (~8 % makespan overhead), but they
have a density cliff: a fixed bias decays as the hub fills (a stronger bias is
needed as N grows). Merry-Go-Round (`planner.type: roundabout`) breaks it
*explicitly* — all drones ride one shared CCW ring — which is collision-free by
construction at any density, but rides a half-circumference arc instead of the
diameter.

This puts them head to head across N, reporting success AND makespan (joint
`final_t`; free-flight ideal = 2R/speed = 8.0 s):
  roundabout    explicit shared ring (Merry-Go-Round)
  mpc_global    MPC + lateral_bias (implicit, fixed strength)
  mpc_pairwise  MPC + pairwise_bias (implicit, adaptive)

Expectation: the implicit conventions are faster at low N but cliff as N grows;
the explicit roundabout holds ~100 % with near-constant makespan but is slower.

  python scripts/antipodal_roundabout_phase.py --n-list 6 12 16 --episodes 20
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

SPEED = 5.0
CX, CY = 25.0, 25.0
RADIUS = 20.0
IDEAL = 2 * RADIUS / SPEED


def _planner(arm, gb, pb):
    if arm == "roundabout":
        return {"type": "roundabout", "max_speed": SPEED, "replan_period": 0.05,
                "center": [CX, CY], "ring_radius": RADIUS, "exit_angle": 0.35,
                "time_step": 0.05, "goal_radius": 1.5}
    p = {"type": "mpc", "max_speed": SPEED, "replan_period": 0.2, "horizon": 40,
         "dt_plan": 0.05, "n_samples": 32, "resolution": 1.0, "inflate": 1,
         "goal_radius": 1.5, "safety_margin": 0.5, "use_prediction": True,
         "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
         "predictor": {"type": "constant_velocity"}}
    if arm == "mpc_global":
        p["lateral_bias"] = gb
    elif arm == "mpc_pairwise":
        p["pairwise_bias"] = pb
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


def _cfg(n, arm, gb, pb, seed, n_eps, max_steps):
    return {
        "name": f"rab_n{n}_{arm}", "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50],
                     "resolution": 1.0, "obstacles": {"type": "none"}, "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": _planner(arm, gb, pb),
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/rab_tmp"},
    }


def _run_cell(job):
    arm, n, gb, pb, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, arm, gb, pb, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = (d["outcome"], float(d.get("final_t", 0.0)))
    return (arm, n, by_seed)


def _succ(bs, seeds):
    return sum(bs[s][0] == "success" for s in seeds)


def _mksp(bs, seeds):
    v = [bs[s][1] for s in seeds if bs[s][0] == "success"]
    return sum(v) / len(v) if v else float("nan")


ARMS = ["roundabout", "mpc_global", "mpc_pairwise"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[6, 12, 16])
    ap.add_argument("--global-bias", type=float, default=2.0)
    ap.add_argument("--pairwise-bias", type=float, default=10.0)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--out", default="results/antipodal_roundabout_phase.json")
    args = ap.parse_args()

    jobs = [(arm, n, args.global_bias, args.pairwise_bias, args.seed, args.episodes, args.max_steps)
            for n in args.n_list for arm in ARMS]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    cells = {}
    for arm, n, bs in res:
        cells.setdefault(n, {})[arm] = bs

    report = {"ideal_makespan": IDEAL, "episodes": args.episodes, "cells": []}
    print(f"\nExplicit roundabout (Merry-Go-Round) vs implicit convention — antipodal, "
          f"n={args.episodes} (succ ; makespan vs ideal {IDEAL:.1f}s)")
    print(f"{'N':>3} | {'roundabout':>20} | {'mpc_global':>20} | {'mpc_pairwise':>20}")
    print("-" * 74)
    for n in sorted(cells):
        c = cells[n]
        seeds = sorted(set(c["roundabout"]) & set(c["mpc_global"]) & set(c["mpc_pairwise"]))
        m = len(seeds)
        def cell(a):
            return f"{_succ(c[a],seeds):>2}/{m:<2} {_mksp(c[a],seeds):>6.2f}s"
        print(f"{n:>3} | {cell('roundabout'):>20} | {cell('mpc_global'):>20} | {cell('mpc_pairwise'):>20}")
        row = {"n": n, "m": m}
        for a in ARMS:
            row[a + "_succ"] = _succ(c[a], seeds)
            row[a + "_makespan"] = _mksp(c[a], seeds)
        report["cells"].append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
