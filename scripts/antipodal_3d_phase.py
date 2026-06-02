"""Is the antipodal-swap inversion a 2D planar-confinement artifact? Does the
vertical escape dimension dissolve it?

scripts/antipodal_predictor_phase.py proved (in 2D) that goal-aware prediction
INVERTS to a liability on the antipodal swap: every drone's correct, shared,
symmetric forecast makes the fleet mirror-swerve into a re-collision at the hub.
But that was on a plane, where the only way past a head-on peer is sideways --
into the same congested hub. In 3D each drone can also climb or dive, an escape
axis the symmetric in-plane forecast never contests.

This embeds the SAME antipodal ring (radius, dynamics, jitter, predictor) in a
3D voxel world with vertical room (z in [0, Z]) and asks:
  2D gt vs 3D gt : does the vertical dimension dissolve the inversion?
  3D gt vs 3D cv : does the predictor still matter (or invert) once 3D is free?
Same MPC (N-D: it samples a Fibonacci sphere in 3D), paired by seed, McNemar.
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
Z_SIZE = 16
Z_MID = 8.0


def _drones(n, dim):
    out = []
    for k in range(n):
        ang = 2.0 * math.pi * k / n
        sx, sy = CX + RADIUS * math.cos(ang), CY + RADIUS * math.sin(ang)
        gx, gy = CX - RADIUS * math.cos(ang), CY - RADIUS * math.sin(ang)
        if dim == 2:
            start, goal = [round(sx, 3), round(sy, 3)], [round(gx, 3), round(gy, 3)]
        else:
            start = [round(sx, 3), round(sy, 3), Z_MID]
            goal = [round(gx, 3), round(gy, 3), Z_MID]
        out.append({"name": f"d{k}", "start": start, "goal": goal,
                    "radius": 0.4, "start_jitter": 0.8})
    return out


def _cfg(n, dim, predictor, bias, seed, n_eps):
    if dim == 2:
        scenario = {"type": "multi_drone_grid", "size": [50, 50], "resolution": 1.0,
                    "obstacles": {"type": "none"}, "drones": _drones(n, 2)}
        sim_type = "dummy_2d"
    else:
        scenario = {"type": "multi_drone_voxel", "size": [50, 50, Z_SIZE],
                    "resolution": 1.0, "obstacles": {"type": "none"},
                    "drones": _drones(n, 3)}
        sim_type = "dummy_3d"
    return {
        "name": f"antipodal{dim}d_n{n}_{predictor}_b{bias}", "seed": seed,
        "num_episodes": n_eps,
        "scenario": scenario,
        "simulator": {"type": sim_type, "dt": 0.05, "max_steps": 1000,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": {"type": "mpc", "max_speed": 5.0, "replan_period": 0.2,
                    "horizon": 40, "dt_plan": 0.05, "n_samples": 48,
                    "resolution": 1.0, "inflate": 1, "goal_radius": 1.5,
                    "safety_margin": 0.5, "use_prediction": True,
                    "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
                    "lateral_bias": bias, "predictor": {"type": predictor}},
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/antipodal3d_tmp"},
    }


def _run_cell(job):
    label, n, dim, predictor, bias, seed, n_eps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, dim, predictor, bias, seed, n_eps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (label, n, by_seed)


def _succ(bs, seeds):
    return sum(bs[s] == "success" for s in seeds)


def _mc(a, b, seeds):
    bb = sum(a[s] == "success" and b[s] != "success" for s in seeds)
    cc = sum(a[s] != "success" and b[s] == "success" for s in seeds)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="results/antipodal_3d_phase.json")
    args = ap.parse_args()

    # arms: 2D gt, 3D gt, 3D cv  (all bias 0 -- testing geometry, not the fix)
    jobs = []
    for n in args.n_list:
        jobs.append(("gt2d", n, 2, "game_theoretic", 0.0, args.seed, args.episodes))
        jobs.append(("gt3d", n, 3, "game_theoretic", 0.0, args.seed, args.episodes))
        jobs.append(("cv3d", n, 3, "constant_velocity", 0.0, args.seed, args.episodes))
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    cells = {}
    for label, n, bs in res:
        cells.setdefault(n, {})[label] = bs

    report = {"radius": RADIUS, "z_size": Z_SIZE, "episodes": args.episodes, "cells": []}
    print(f"\n2D vs 3D antipodal swap (z in [0,{Z_SIZE}]), n={args.episodes}, paired")
    print(f"{'N':>2} | {'gt2d':>6} | {'gt3d':>6} | {'cv3d':>6} | "
          f"{'3d vs 2d gt (c-b,p)':>22} | {'3d gt vs cv (c-b,p)':>22}")
    print("-" * 92)
    for n in sorted(cells):
        g2, g3, c3 = cells[n]["gt2d"], cells[n]["gt3d"], cells[n]["cv3d"]
        seeds = sorted(set(g2) & set(g3) & set(c3))
        m = len(seeds)
        b1, c1, p1 = _mc(g2, g3, seeds)   # c1-b1>0 => 3D better than 2D
        b2, c2, p2 = _mc(c3, g3, seeds)   # c2-b2>0 => gt better than cv in 3D
        print(f"{n:>2} | {_succ(g2,seeds):>3}/{m:<2} | {_succ(g3,seeds):>3}/{m:<2} | "
              f"{_succ(c3,seeds):>3}/{m:<2} | b={b1} c={c1} p={p1:>7.4f}     | "
              f"b={b2} c={c2} p={p2:>7.4f}")
        report["cells"].append({
            "n": n, "m": m, "gt2d": _succ(g2, seeds), "gt3d": _succ(g3, seeds),
            "cv3d": _succ(c3, seeds),
            "gt3d_vs_gt2d": {"b": b1, "c": c1, "p": p1},
            "gt3d_vs_cv3d": {"b": b2, "c": c2, "p": p2},
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
