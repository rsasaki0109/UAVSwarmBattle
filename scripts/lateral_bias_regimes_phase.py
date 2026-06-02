"""Is the right-of-way lateral_bias a general, safe coordination primitive, or
an antipodal-specific trick that must stay default-off?

scripts/antipodal_rightofway_phase.py proved lateral_bias takes goal-aware
prediction to 100% on the antipodal swap. This asks the two questions that
decide whether the knob is recommendable beyond that one scenario:

  GENERALITY — does it also lift a DIFFERENT symmetric congestion (a dense
               N-vs-N perpendicular crossing, not a ring)?
  SAFETY     — does it HARM the regimes where there is no symmetric deadlock to
               break: the proven 2-drone crossing (asymmetric encounter), and a
               single drone threading a static+dynamic obstacle field?

Each regime runs bias 0.0 vs 2.0, paired by seed, McNemar exact. A good knob:
helps dense_cross, no-harm on crossing2 and single_dyn.
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
from uav_nav_lab.runner.experiment import run_experiment
from uav_nav_lab.runner.multi.experiment import run_experiment_multi
from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p

MPC = dict(
    type="mpc", max_speed=5.0, replan_period=0.2, horizon=40, dt_plan=0.05,
    n_samples=32, resolution=1.0, inflate=1, goal_radius=1.5, safety_margin=0.5,
    use_prediction=True, w_goal=1.0, w_obs=100.0, w_smooth=0.05,
)


def _mpc(bias, predictor="game_theoretic"):
    p = dict(MPC)
    p["lateral_bias"] = bias
    p["predictor"] = {"type": predictor}
    return p


def _cfg_single_dyn(bias, seed, n_eps):
    # Single drone threading a static + dynamic obstacle field (the core
    # avoidance scenario; cf. examples/exp_compare_mpc.yaml). No peers, so this
    # isolates whether a constant rightward bias HARMS goal-reach in clutter.
    return {
        "name": f"single_dyn_b{bias}", "seed": seed, "num_episodes": n_eps,
        "scenario": {
            "type": "grid_world", "size": [50, 50],
            "start": [2.0, 2.0], "goal": [45.0, 45.0], "resolution": 1.0,
            "obstacles": {"type": "random", "count": 25, "seed": 7},
            "dynamic_obstacles": [
                {"start": [25.0, 10.0], "velocity": [0.0, 6.0], "reflect": True, "radius": 0.8},
                {"start": [10.0, 30.0], "velocity": [5.0, 0.0], "reflect": True, "radius": 0.8},
                {"start": [35.0, 25.0], "velocity": [-4.0, 3.0], "reflect": True, "radius": 0.8},
            ],
        },
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": 1500,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": _mpc(bias, predictor="constant_velocity"),
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/lbr_tmp"},
    }


def _crossing_drones(n_per_dir):
    # n_per_dir drones west->east + n_per_dir south->north, staggered into lanes
    # that all funnel through the centre. Symmetric multi-conflict, distinct from
    # the antipodal ring.
    drones = []
    span = 12.0
    for k in range(n_per_dir):
        off = (k - (n_per_dir - 1) / 2.0) * (span / max(1, n_per_dir))
        drones.append({"name": f"e{k}", "start": [4.0, 25.0 + off],
                       "goal": [46.0, 25.0 + off], "radius": 0.4, "start_jitter": 0.8})
        drones.append({"name": f"n{k}", "start": [25.0 + off, 4.0],
                       "goal": [25.0 + off, 46.0], "radius": 0.4, "start_jitter": 0.8})
    return drones


def _cfg_multi(name, drones, bias, seed, n_eps):
    return {
        "name": f"{name}_b{bias}", "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50],
                     "resolution": 1.0, "obstacles": {"type": "none"}, "drones": drones},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": 1000,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": _mpc(bias, predictor="game_theoretic"),
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/lbr_tmp"},
    }


def _cfg_crossing2(bias, seed, n_eps):
    drones = [
        {"name": "east", "start": [4.0, 25.0], "goal": [46.0, 25.0], "radius": 0.4, "start_jitter": 0.8},
        {"name": "north", "start": [25.0, 4.0], "goal": [25.0, 46.0], "radius": 0.4, "start_jitter": 0.8},
    ]
    return _cfg_multi("crossing2", drones, bias, seed, n_eps)


def _cfg_dense_cross(bias, seed, n_eps):
    return _cfg_multi("dense_cross", _crossing_drones(3), bias, seed, n_eps)


def _headon_drones(n_lanes):
    # n_lanes pairs flying HEAD-ON in the same lane (west->east meets east->west).
    # Unlike a perpendicular crossing this forces opposing convergence — the same
    # mechanism as the antipodal swap, but a corridor topology rather than a ring.
    drones = []
    span = 14.0
    for k in range(n_lanes):
        off = (k - (n_lanes - 1) / 2.0) * (span / max(1, n_lanes))
        drones.append({"name": f"e{k}", "start": [4.0, 25.0 + off],
                       "goal": [46.0, 25.0 + off], "radius": 0.4, "start_jitter": 0.8})
        drones.append({"name": f"w{k}", "start": [46.0, 25.0 + off],
                       "goal": [4.0, 25.0 + off], "radius": 0.4, "start_jitter": 0.8})
    return drones


def _cfg_headon(bias, seed, n_eps):
    return _cfg_multi("headon", _headon_drones(3), bias, seed, n_eps)


REGIMES = {
    "single_dyn":  ("single", _cfg_single_dyn,  "SAFETY: single drone in static+dynamic clutter"),
    "crossing2":   ("multi",  _cfg_crossing2,   "SAFETY: proven 2-drone asymmetric crossing"),
    "dense_cross": ("multi",  _cfg_dense_cross, "GENERALITY: dense 3v3 symmetric crossing"),
    "headon":      ("multi",  _cfg_headon,     "GENERALITY: 3-lane head-on corridor swap"),
}


def _run_cell(job):
    regime, bias, seed, n_eps = job
    kind, builder, _ = REGIMES[regime]
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(builder(bias, seed, n_eps))
        if kind == "single":
            out = run_experiment(cfg, Path(td))
            pattern = "episode_[0-9]*.json"
            files = [f for f in sorted(Path(out).glob(pattern)) if "joint" not in f.name]
        else:
            out = run_experiment_multi(cfg, Path(td))
            files = sorted(Path(out).glob("episode_*_joint.json"))
        by_seed = {}
        for jf in files:
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (regime, bias, by_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--bias", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES))
    ap.add_argument("--out", default="results/lateral_bias_regimes_phase.json")
    args = ap.parse_args()

    jobs = []
    for r in args.regimes:
        jobs.append((r, 0.0, args.seed, args.episodes))
        jobs.append((r, args.bias, args.seed, args.episodes))
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    cells = {}
    for r, bias, bs in res:
        cells.setdefault(r, {})[bias] = bs

    report = {"episodes": args.episodes, "bias": args.bias, "cells": []}
    print(f"\nlateral_bias 0.0 vs {args.bias}, n={args.episodes}, paired by seed")
    print(f"{'regime':>12} | {'no-bias':>8} | {'bias':>8} | b(b0>bB) c(bB>b0) | McNemar p | verdict")
    print("-" * 92)
    for r in args.regimes:
        b0, bB = cells[r][0.0], cells[r][args.bias]
        seeds = sorted(set(b0) & set(bB))
        m = len(seeds)
        s0 = sum(b0[s] == "success" for s in seeds)
        sB = sum(bB[s] == "success" for s in seeds)
        b = sum(b0[s] == "success" and bB[s] != "success" for s in seeds)  # bias HURT
        c = sum(b0[s] != "success" and bB[s] == "success" for s in seeds)  # bias HELPED
        p = mcnemar_exact_p(b, c)
        if p < 0.05 and c > b:
            v = "HELPS"
        elif p < 0.05 and b > c:
            v = "HARMS"
        else:
            v = "no-harm (tie)"
        print(f"{r:>12} | {s0:>3}/{m:<4} | {sB:>3}/{m:<4} | b={b:<3} c={c:<3}      | "
              f"p={p:.4f}  | {v}")
        report["cells"].append({"regime": r, "m": m, "succ_nobias": s0,
                                "succ_bias": sB, "b": b, "c": c, "p": p, "verdict": v})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
