"""Does the antipodal deadlock dissolve in 3-D for a REACTIVE controller too?

The goal-aware-predictor inversion dissolves when the antipodal ring is lifted
into a voxel world: the vertical axis gives every drone a symmetry escape, so the
2-D deadlock vanishes (docs/findings.md "the antipodal inversion dissolves in
3-D"). That was a *forecast* result. This asks the same of a *reactive* baseline:
lift the CBF safety filter (planner.type: cbf, now 2-D/3-D) onto the 3-D
antipodal swap and see whether its hub deadlock (collision-free timeout, no
right-of-way) dissolves the same way.

Two arms per N, paired by seed, McNemar exact, collision-vs-timeout breakdown:
  cbf_2d  multi_drone_grid  (the planar deadlock: stock CBF times out at the hub)
  cbf_3d  multi_drone_voxel (same ring at z=mid; does the vertical axis free it?)

If cbf_3d succeeds where cbf_2d deadlocks, the reactive deadlock is a planar
artifact too — symmetry-breaking by an added dimension, not only by a convention.
Also watches whether the even-N resonance recurs in 3-D for the reactive method.

  python scripts/antipodal_cbf_3d_phase.py --n-list 4 6 8 --episodes 40
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
SPEED = 5.0


def _drones(n, dim):
    out = []
    for k in range(n):
        ang = 2.0 * math.pi * k / n
        sx, sy = CX + RADIUS * math.cos(ang), CY + RADIUS * math.sin(ang)
        gx, gy = CX - RADIUS * math.cos(ang), CY - RADIUS * math.sin(ang)
        d = {"name": f"d{k}", "radius": 0.4, "start_jitter": 0.8}
        if dim == 3:
            d["start"] = [round(sx, 3), round(sy, 3), Z_MID]
            d["goal"] = [round(gx, 3), round(gy, 3), Z_MID]
        else:
            d["start"] = [round(sx, 3), round(sy, 3)]
            d["goal"] = [round(gx, 3), round(gy, 3)]
        out.append(d)
    return out


def _planner(kind):
    if kind == "mpc":
        # The sampling planner that DOES explore the free vertical axis (the
        # contrast: it dissolves the 3-D antipodal deadlock where the reactive
        # CBF cannot). cv predictor, no convention.
        return {"type": "mpc", "max_speed": SPEED, "replan_period": 0.2,
                "horizon": 40, "dt_plan": 0.05, "n_samples": 48,
                "resolution": 1.0, "inflate": 1, "goal_radius": 1.5,
                "safety_margin": 0.5, "use_prediction": True,
                "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
                "predictor": {"type": "constant_velocity"}}
    return {"type": "cbf", "max_speed": SPEED, "replan_period": 0.05,
            "radius": 0.4, "safety_margin": 0.1, "alpha": 2.0,
            "neighbor_dist": 15.0, "time_step": 0.05, "goal_radius": 1.5}


def _cfg(n, dim, kind, seed, n_eps, max_steps):
    if dim == 3:
        scenario = {"type": "multi_drone_voxel", "size": [50, 50, Z_SIZE],
                    "resolution": 1.0, "obstacles": {"type": "none"},
                    "drones": _drones(n, 3)}
        simulator = {"type": "dummy_3d", "dt": 0.05, "max_steps": max_steps,
                     "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4}
    else:
        scenario = {"type": "multi_drone_grid", "size": [50, 50],
                    "resolution": 1.0, "obstacles": {"type": "none"},
                    "drones": _drones(n, 2)}
        simulator = {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                     "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4}
    return {"name": f"cbf{dim}d_n{n}", "seed": seed, "num_episodes": n_eps,
            "scenario": scenario, "simulator": simulator, "planner": _planner(kind),
            "sensor": {"type": "perfect"}, "output": {"dir": "results/cbf3d_tmp"}}


def _run_cell(job):
    label, n, dim, kind, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, dim, kind, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (label, n, by_seed)


def _succ(bs, seeds):
    return sum(bs[s] == "success" for s in seeds)


def _brk(bs, seeds):
    return (sum(bs[s] == "collision" for s in seeds),
            sum(bs[s] == "timeout" for s in seeds))


def _mc(a, b, seeds):
    bb = sum(a[s] == "success" and b[s] != "success" for s in seeds)
    cc = sum(a[s] != "success" and b[s] == "success" for s in seeds)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--out", default="results/antipodal_cbf_3d_phase.json")
    args = ap.parse_args()

    jobs = []
    for n in args.n_list:
        jobs.append(("cbf_2d", n, 2, "cbf", args.seed, args.episodes, args.max_steps))
        jobs.append(("cbf_3d", n, 3, "cbf", args.seed, args.episodes, args.max_steps))
        jobs.append(("mpc_3d", n, 3, "mpc", args.seed, args.episodes, args.max_steps))
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    cells = {}
    for label, n, bs in res:
        cells.setdefault(n, {})[label] = bs

    report = {"radius": RADIUS, "speed": SPEED, "episodes": args.episodes, "cells": []}
    print(f"\nCBF antipodal: does the reactive deadlock dissolve in 3-D? "
          f"(n={args.episodes}, paired; succ [coll/timeout])")
    print(f"{'N':>2} | {'cbf_2d':>14} | {'cbf_3d':>14} | {'mpc_3d':>14} | "
          f"{'mpc_3d vs cbf_3d':>22}")
    print("-" * 80)
    for n in sorted(cells):
        c = cells[n]
        c2, c3, m3 = c["cbf_2d"], c["cbf_3d"], c["mpc_3d"]
        seeds = sorted(set(c2) & set(c3) & set(m3))
        m = len(seeds)
        co2, to2 = _brk(c2, seeds)
        co3, to3 = _brk(c3, seeds)
        cm3, tm3 = _brk(m3, seeds)
        b, cc, p = _mc(c3, m3, seeds)  # does MPC dissolve where CBF cannot?
        print(f"{n:>2} | {_succ(c2,seeds):>2}/{m:<2}[{co2:>2}c/{to2:>2}t] | "
              f"{_succ(c3,seeds):>2}/{m:<2}[{co3:>2}c/{to3:>2}t] | "
              f"{_succ(m3,seeds):>2}/{m:<2}[{cm3:>2}c/{tm3:>2}t] | "
              f"b={b:>2} c={cc:>2} p={p:>8.5f}")
        report["cells"].append({"n": n, "m": m,
            "cbf_2d": _succ(c2, seeds), "cbf_2d_ct": [co2, to2],
            "cbf_3d": _succ(c3, seeds), "cbf_3d_ct": [co3, to3],
            "mpc_3d": _succ(m3, seeds), "mpc_3d_ct": [cm3, tm3],
            "mpc3d_vs_cbf3d": {"b": b, "c": cc, "p": p}})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
