"""Is the 3D high-density predictor collapse a SYMMETRY failure or a FORECAST
failure?

scripts/antipodal_3d_phase.py proved that lifting the antipodal ring into 3D
dissolves the goal-aware-predictor inversion (game_theoretic 100% at every N),
but that at the top density the dumb constant_velocity predictor COLLAPSES
(N=6: cv 0/40 while gt 40/40). docs/findings.md offered a mechanism: once the
vertical axis is the only slack left, six drones can only stack their
climbs/dives without re-colliding if the forecast knows where peers are GOING --
i.e. the collapse is forecast-dependent.

That mechanism is an assertion. This script tests it the honest way, by asking
whether a cheap, goal-BLIND symmetry-breaker recovers cv:

  if `planner.lateral_bias` (the in-plane right-of-way convention, default off --
  it just biases each candidate heading to veer RIGHT of the goal direction)
  rescues cv at the collapsed cells, the failure was SYMMETRY (a shared
  rotational convention spreads the in-plane hub and frees the vertical axis),
  REFINING the findings.md claim;

  if lateral_bias does NOT rescue cv but gt still wins, the failure is genuinely
  FORECAST-dependent (you must know peer goals to phase the vertical escape),
  CONFIRMING it.

Four arms per N, paired by seed, McNemar exact, all in the 3D voxel world:
  cv_b0   constant_velocity, bias 0  (the collapse)
  cv_bias constant_velocity, bias 2  (does convention rescue the dumb forecast?)
  gt_b0   game_theoretic,   bias 0  (the survivor / reference)
  gt_bias game_theoretic,   bias 2  (is the convention harmless on the survivor?)
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


def _drones(n):
    out = []
    for k in range(n):
        ang = 2.0 * math.pi * k / n
        sx, sy = CX + RADIUS * math.cos(ang), CY + RADIUS * math.sin(ang)
        gx, gy = CX - RADIUS * math.cos(ang), CY - RADIUS * math.sin(ang)
        out.append({"name": f"d{k}",
                    "start": [round(sx, 3), round(sy, 3), Z_MID],
                    "goal": [round(gx, 3), round(gy, 3), Z_MID],
                    "radius": 0.4, "start_jitter": 0.8})
    return out


def _cfg(n, predictor, bias, seed, n_eps, max_steps):
    return {
        "name": f"antipodal3d_n{n}_{predictor}_b{bias}", "seed": seed,
        "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_voxel", "size": [50, 50, Z_SIZE],
                     "resolution": 1.0, "obstacles": {"type": "none"},
                     "drones": _drones(n)},
        "simulator": {"type": "dummy_3d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": {"type": "mpc", "max_speed": 5.0, "replan_period": 0.2,
                    "horizon": 40, "dt_plan": 0.05, "n_samples": 48,
                    "resolution": 1.0, "inflate": 1, "goal_radius": 1.5,
                    "safety_margin": 0.5, "use_prediction": True,
                    "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
                    "lateral_bias": bias, "predictor": {"type": predictor}},
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/antipodal3dsym_tmp"},
    }


def _run_cell(job):
    label, n, predictor, bias, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, predictor, bias, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (label, n, by_seed)


def _succ(bs, seeds):
    return sum(bs[s] == "success" for s in seeds)


def _mc(a, b, seeds):
    # c-b>0 => b is better than a
    bb = sum(a[s] == "success" and b[s] != "success" for s in seeds)
    cc = sum(a[s] != "success" and b[s] == "success" for s in seeds)
    return bb, cc, mcnemar_exact_p(bb, cc)


ARMS = [("cv_b0", "constant_velocity", 0.0),
        ("cv_bias", "constant_velocity", None),  # bias filled from args
        ("gt_b0", "game_theoretic", 0.0),
        ("gt_bias", "game_theoretic", None)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[5, 6, 7])
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--bias", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=6)
    # A crossing succeeds in ~200 steps; an unresolved deadlock never recovers,
    # so 600 steps is ample for every success and halves the deadlock-cell cost.
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--out", default="results/antipodal_3d_symmetry_phase.json")
    args = ap.parse_args()

    jobs = []
    for n in args.n_list:
        for label, pred, bias in ARMS:
            b = args.bias if bias is None else bias
            jobs.append((label, n, pred, b, args.seed, args.episodes, args.max_steps))
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    cells = {}
    for label, n, bs in res:
        cells.setdefault(n, {})[label] = bs

    report = {"radius": RADIUS, "z_size": Z_SIZE, "bias": args.bias,
              "episodes": args.episodes, "cells": []}
    print(f"\n3D antipodal: is the cv collapse SYMMETRY or FORECAST? "
          f"bias={args.bias}, n={args.episodes}, paired")
    print(f"{'N':>2} | {'cv_b0':>6} | {'cv_bias':>7} | {'gt_b0':>6} | {'gt_bias':>7} | "
          f"{'cvbias vs cvb0':>16} | {'gtb0 vs cvb0':>14} | {'cvbias vs gtb0':>16}")
    print("-" * 110)
    for n in sorted(cells):
        c = cells[n]
        cv0, cvb, gt0, gtb = c["cv_b0"], c["cv_bias"], c["gt_b0"], c["gt_bias"]
        seeds = sorted(set(cv0) & set(cvb) & set(gt0) & set(gtb))
        m = len(seeds)
        b1, c1, p1 = _mc(cv0, cvb, seeds)   # convention rescues cv?
        b2, c2, p2 = _mc(cv0, gt0, seeds)   # reproduce #73 collapse
        b3, c3, p3 = _mc(gt0, cvb, seeds)   # convention match forecast?
        print(f"{n:>2} | {_succ(cv0,seeds):>3}/{m:<2} | {_succ(cvb,seeds):>4}/{m:<2} | "
              f"{_succ(gt0,seeds):>3}/{m:<2} | {_succ(gtb,seeds):>4}/{m:<2} | "
              f"b={b1} c={c1} p={p1:>6.4f} | b={b2} c={c2} p={p2:>6.4f} | "
              f"b={b3} c={c3} p={p3:>6.4f}")
        report["cells"].append({
            "n": n, "m": m,
            "cv_b0": _succ(cv0, seeds), "cv_bias": _succ(cvb, seeds),
            "gt_b0": _succ(gt0, seeds), "gt_bias": _succ(gtb, seeds),
            "cvbias_vs_cvb0": {"b": b1, "c": c1, "p": p1},
            "gtb0_vs_cvb0": {"b": b2, "c": c2, "p": p2},
            "cvbias_vs_gtb0": {"b": b3, "c": c3, "p": p3},
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
