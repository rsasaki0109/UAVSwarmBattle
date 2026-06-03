"""Does a PAIRWISE (winding-number-style) right-of-way beat the GLOBAL veer-right?

The `lateral_bias` convention [[right-of-way fix]] breaks the antipodal deadlock by
having every drone veer RIGHT of its OWN goal heading -- a single global rule, applied
unconditionally. We have since proved that exact unconditionality is its weakness:

  * 3D N=4 HARM (antipodal_n8_even_resonance): where there is NO deadlock (cv 30/30),
    turning the global bias on DRIVES cv 30 -> 0/30 -- it manufactures a 4-way pinwheel
    that re-collides. The convention fires even when no one is in conflict.
  * 2D DENSITY CLIFF (user's convention-cliff study): a FIXED global bias decays with the
    hub crowd (N=16 bias=2 -> ~65%); you must keep cranking the single scalar.

The 2025 literature (Winding Number-Aware MPC, arXiv:2511.15239; Merry-Go-Round,
arXiv:2503.05848) argues the durable symmetry-breaker is PAIRWISE/RELATIVE, not a global
heading bias: each agent passes each NEARBY neighbour on a consistent relative side, so a
drone in no conflict is left alone and the rule scales with the actual pair geometry.

This script tests that claim against our own two known failure modes of the global rule.
New planner knob `pairwise_bias` (+ `pairwise_radius`): prefer candidate directions that
pass each neighbour on the right, weighted exp(-d/radius). Three arms, paired by seed,
McNemar exact:

  b0        no convention            (the reference: deadlock in 2D, free vertical in 3D)
  global    lateral_bias = B_global  (the known, unconditional convention)
  pairwise  pairwise_bias = B_pair   (the neighbour-conditional challenger)

Headline cells:
  --dim 3 --n-list 4 6 8   does pairwise AVOID the N=4 harm while still rescuing N>=6?
  --dim 2 --n-list 8 12 16 does pairwise SCALE past the density cliff without a bigger scalar?

  python scripts/antipodal_pairwise_convention_phase.py --dim 3 --n-list 4 6 8 --episodes 30
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
BASE_SPEED = 5.0


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


# arm -> (lateral_bias, pairwise_bias)
def _arm_planner(arm, gb, pb, prad, predictor):
    planner = {"type": "mpc", "max_speed": BASE_SPEED, "replan_period": 0.2,
               "horizon": 40, "dt_plan": 0.05, "n_samples": 48,
               "resolution": 1.0, "inflate": 1, "goal_radius": 1.5,
               "safety_margin": 0.5, "use_prediction": True,
               "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
               "predictor": {"type": predictor}}
    if arm == "global":
        planner["lateral_bias"] = gb
    elif arm == "pairwise":
        planner["pairwise_bias"] = pb
        planner["pairwise_radius"] = prad
    return planner


def _cfg(n, dim, arm, gb, pb, prad, predictor, seed, n_eps, max_steps):
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
    return {
        "name": f"antipodal{dim}d_n{n}_{arm}", "seed": seed, "num_episodes": n_eps,
        "scenario": scenario, "simulator": simulator,
        "planner": _arm_planner(arm, gb, pb, prad, predictor),
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/antipodal_pairwise_tmp"},
    }


def _run_cell(job):
    label, n, dim, arm, gb, pb, prad, predictor, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(
            _cfg(n, dim, arm, gb, pb, prad, predictor, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed, brk = {}, {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (label, n, by_seed)


def _succ(bs, seeds):
    return sum(bs[s] == "success" for s in seeds)


def _brk(bs, seeds):
    coll = sum(bs[s] == "collision" for s in seeds)
    to = sum(bs[s] == "timeout" for s in seeds)
    return coll, to


def _mc(a, b, seeds):
    # c-b>0 => b better than a
    bb = sum(a[s] == "success" and b[s] != "success" for s in seeds)
    cc = sum(a[s] != "success" and b[s] == "success" for s in seeds)
    return bb, cc, mcnemar_exact_p(bb, cc)


ARMS = ["b0", "global", "pairwise"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, choices=[2, 3], default=3)
    ap.add_argument("--n-list", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--global-bias", type=float, default=2.0)
    ap.add_argument("--pairwise-bias", type=float, default=10.0)
    ap.add_argument("--pairwise-radius", type=float, default=8.0)
    ap.add_argument("--predictor", default="constant_velocity")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--out", default="results/antipodal_pairwise_convention_phase.json")
    args = ap.parse_args()

    jobs = []
    for n in args.n_list:
        for arm in ARMS:
            jobs.append((arm, n, args.dim, arm, args.global_bias, args.pairwise_bias,
                         args.pairwise_radius, args.predictor, args.seed,
                         args.episodes, args.max_steps))
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    cells = {}
    for label, n, bs in res:
        cells.setdefault(n, {})[label] = bs

    report = {"dim": args.dim, "radius": RADIUS, "predictor": args.predictor,
              "global_bias": args.global_bias, "pairwise_bias": args.pairwise_bias,
              "pairwise_radius": args.pairwise_radius, "episodes": args.episodes,
              "cells": []}
    print(f"\n{args.dim}D antipodal: PAIRWISE vs GLOBAL right-of-way "
          f"(pred={args.predictor}, gb={args.global_bias}, pb={args.pairwise_bias}, "
          f"prad={args.pairwise_radius}, n={args.episodes}, paired)")
    print(f"{'N':>2} | {'b0':>7} | {'global':>7} | {'pairwise':>8} | "
          f"{'global vs b0':>20} | {'pairwise vs b0':>20} | {'pairwise vs global':>20}")
    print("-" * 120)
    for n in sorted(cells):
        c = cells[n]
        b0, gl, pw = c["b0"], c["global"], c["pairwise"]
        seeds = sorted(set(b0) & set(gl) & set(pw))
        m = len(seeds)
        b1, c1, p1 = _mc(b0, gl, seeds)   # global vs b0
        b2, c2, p2 = _mc(b0, pw, seeds)   # pairwise vs b0
        b3, c3, p3 = _mc(gl, pw, seeds)   # pairwise vs global
        cb0, tb0 = _brk(b0, seeds)
        cgl, tgl = _brk(gl, seeds)
        cpw, tpw = _brk(pw, seeds)
        print(f"{n:>2} | {_succ(b0,seeds):>3}/{m:<3} | {_succ(gl,seeds):>3}/{m:<3} | "
              f"{_succ(pw,seeds):>4}/{m:<3} | "
              f"b={b1:>2} c={c1:>2} p={p1:>7.4f} | b={b2:>2} c={c2:>2} p={p2:>7.4f} | "
              f"b={b3:>2} c={c3:>2} p={p3:>7.4f}")
        report["cells"].append({
            "n": n, "m": m,
            "b0": _succ(b0, seeds), "global": _succ(gl, seeds), "pairwise": _succ(pw, seeds),
            "b0_coll_to": [cb0, tb0], "global_coll_to": [cgl, tgl], "pairwise_coll_to": [cpw, tpw],
            "global_vs_b0": {"b": b1, "c": c1, "p": p1},
            "pairwise_vs_b0": {"b": b2, "c": c2, "p": p2},
            "pairwise_vs_global": {"b": b3, "c": c3, "p": p3},
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
