"""Under noisy peer sensing, does the ego-only GLOBAL convention overtake the
peer-dependent PAIRWISE one — re-inverting their clean-sensing order?

The pairwise right-of-way strictly dominates the global veer-right under perfect
sensing ([no over-rotation cliff](docs/findings.md#on-orca-too-a-pairwise-right-of-way-removes-the-global-rules-over-rotation-timeout-cliff)).
But the two read different state to decide the tilt:
  global  (lateral_bias) : tilts right of the EGO's own goal heading — reads NO
                           peer state at all, so it is sensing-INDEPENDENT.
  pairwise(pairwise_bias): tilts toward the bearing to each nearby PEER — reads
                           peer positions, so noise corrupts the tilt direction.

On the antipodal hub both rescue the deadlock under perfect sensing. This sweeps
Gaussian position noise on the peer tracker and asks whether the global rule's
sensing-independence makes it the more robust symmetry-breaker as sensing degrades
(the underlying ORCA avoidance reads peers in both arms, so any divergence is the
CONVENTION's sensing dependence).

ORCA at the #85/#87 operating point, paired by seed, McNemar exact:
  stock     no convention (deadlocks)
  global    lateral_bias = 0.2
  pairwise  pairwise_bias = 0.5

  python scripts/antipodal_convention_noise_phase.py --n 6 --noise-list 0 0.25 0.5 1.0 --episodes 40
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
REPLAN = 0.5


def _planner(arm, gb, pb):
    p = {"type": "orca", "max_speed": SPEED, "replan_period": REPLAN, "radius": 0.4,
         "safety_margin": 0.1, "time_horizon": 2.0, "time_step": 0.25,
         "neighbor_dist": 15.0, "goal_radius": 1.5}
    if arm == "global":
        p["lateral_bias"] = gb
    elif arm == "pairwise":
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


def _cfg(n, arm, gb, pb, noise, seed, n_eps, max_steps):
    return {
        "name": f"convnoise_n{n}_{arm}_{noise}", "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50],
                     "resolution": 1.0, "obstacles": {"type": "none"}, "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": _planner(arm, gb, pb),
        "sensor": {"type": "noisy_tracker", "position_noise_std": noise,
                   "velocity_noise_std": 0.0, "delay": 0.0},
        "output": {"dir": "results/convnoise_tmp"},
    }


def _run_cell(job):
    arm, noise, n, gb, pb, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, arm, gb, pb, noise, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (arm, noise, by_seed)


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
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--noise-list", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--global-bias", type=float, default=0.2)
    ap.add_argument("--pairwise-bias", type=float, default=0.5)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--out", default="results/antipodal_convention_noise_phase.json")
    args = ap.parse_args()

    jobs = [(arm, nz, args.n, args.global_bias, args.pairwise_bias, args.seed, args.episodes, args.max_steps)
            for nz in args.noise_list for arm in ARMS]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    cells = {}
    for arm, nz, bs in res:
        cells.setdefault(nz, {})[arm] = bs

    report = {"n": args.n, "global_bias": args.global_bias, "pairwise_bias": args.pairwise_bias,
              "episodes": args.episodes, "cells": []}
    print(f"\nAntipodal N={args.n}: ego-only GLOBAL vs peer-dependent PAIRWISE convention under "
          f"position noise (n={args.episodes}, paired; succ [coll/timeout])")
    print(f"{'noise':>6} | {'stock':>13} | {'global':>13} | {'pairwise':>13} | "
          f"{'global vs pairwise':>20}")
    print("-" * 82)
    for nz in sorted(cells):
        c = cells[nz]
        st, gl, pw = c["stock"], c["global"], c["pairwise"]
        seeds = sorted(set(st) & set(gl) & set(pw))
        m = len(seeds)
        def cell(x):
            co, to = _brk(x, seeds)
            return f"{_succ(x,seeds):>2}/{m:<2}[{co:>2}c/{to:>2}t]"
        b, cc, p = _mc(pw, gl, seeds)  # c-b>0 => global better than pairwise
        print(f"{nz:>6} | {cell(st):>13} | {cell(gl):>13} | {cell(pw):>13} | "
              f"b={b:>2} c={cc:>2} p={p:>7.4f}")
        row = {"noise": nz, "m": m, "global_vs_pairwise": {"b": b, "c": cc, "p": p}}
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
