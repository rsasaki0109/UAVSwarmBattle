"""Antipodal-swap predictor phase: does goal-aware peer modelling
(game_theoretic) lift the N-drone deadlock, or does it tie const_velocity
the way the symmetric swap defeats reactive avoidance?

Setup: N drones equally spaced on a circle, each heading to its antipode
through the centre (the canonical hard swarm benchmark — every drone's
straight-line path crosses the same congested hub). Obstacles none, so the
ONLY thing the planner must solve is peer coordination. Same MPC / dynamics
as the proven 2-drone crossing study (examples/exp_multi_drone_crossing_*),
just generalised to N on a ring. We swap ONLY the predictor and pair by seed.

  game_theoretic : models each peer as steering toward its goal (it is fed
                   peer goals) → can anticipate the swap before the hub jams.
  constant_velocity : coasts each peer on current velocity → reacts late.

Joint success = all drones reach goal (episode_*_joint.json). Paired McNemar
exact test per N.
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

CENTER = (25.0, 25.0)
RADIUS = 20.0
PREDICTORS = ["constant_velocity", "game_theoretic"]


def _drones(n):
    out = []
    for k in range(n):
        ang = 2.0 * math.pi * k / n
        sx = CENTER[0] + RADIUS * math.cos(ang)
        sy = CENTER[1] + RADIUS * math.sin(ang)
        gx = CENTER[0] - RADIUS * math.cos(ang)
        gy = CENTER[1] - RADIUS * math.sin(ang)
        out.append({
            "name": f"d{k}",
            "start": [round(sx, 3), round(sy, 3)],
            "goal": [round(gx, 3), round(gy, 3)],
            "radius": 0.4,
            "start_jitter": 0.8,
        })
    return out


def _cfg(n, predictor, seed, n_eps):
    return {
        "name": f"antipodal_n{n}_{predictor}",
        "seed": seed,
        "num_episodes": n_eps,
        "scenario": {
            "type": "multi_drone_grid",
            "size": [50, 50],
            "resolution": 1.0,
            "obstacles": {"type": "none"},
            "drones": _drones(n),
        },
        "simulator": {
            "type": "dummy_2d",
            "dt": 0.05,
            "max_steps": 1000,
            "max_accel": 6.0,
            "goal_radius": 1.5,
            "drone_radius": 0.4,
        },
        "planner": {
            "type": "mpc",
            "max_speed": 5.0,
            "replan_period": 0.2,
            "horizon": 40,
            "dt_plan": 0.05,
            "n_samples": 32,
            "resolution": 1.0,
            "inflate": 1,
            "goal_radius": 1.5,
            "safety_margin": 0.5,
            "use_prediction": True,
            "w_goal": 1.0,
            "w_obs": 100.0,
            "w_smooth": 0.05,
            "predictor": {"type": predictor},
        },
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/antipodal_tmp"},
    }


def _run_cell(args):
    n, predictor, seed, n_eps = args
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, predictor, seed, n_eps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (n, predictor, by_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default="results/antipodal_predictor_phase.json")
    ap.add_argument("--plot-from", default=None,
                    help="skip the sweep; load this report json and only (re)draw the PNG")
    args = ap.parse_args()

    if args.plot_from:
        report = json.loads(Path(args.plot_from).read_text())
        _plot(report, Path(args.plot_from).with_suffix(".png"))
        print(f"plotted {Path(args.plot_from).with_suffix('.png')}")
        return

    jobs = [(n, p, args.seed, args.episodes) for n in args.n_list for p in PREDICTORS]
    with Pool(min(args.workers, len(jobs))) as pool:
        results = pool.map(_run_cell, jobs)

    cells = {}
    for n, p, by_seed in results:
        cells.setdefault(n, {})[p] = by_seed

    report = {"radius": RADIUS, "episodes": args.episodes, "cells": []}
    print(f"\n{'N':>2} | {'cv succ':>8} | {'gt succ':>8} | b(cv>gt) c(gt>cv) | McNemar p")
    print("-" * 60)
    for n in sorted(cells):
        cv = cells[n]["constant_velocity"]
        gt = cells[n]["game_theoretic"]
        seeds = sorted(set(cv) & set(gt))
        cv_s = sum(cv[s] == "success" for s in seeds)
        gt_s = sum(gt[s] == "success" for s in seeds)
        cv_col = sum(cv[s] == "collision" for s in seeds)
        gt_col = sum(gt[s] == "collision" for s in seeds)
        cv_to = sum(cv[s] == "timeout" for s in seeds)
        gt_to = sum(gt[s] == "timeout" for s in seeds)
        b = sum(cv[s] == "success" and gt[s] != "success" for s in seeds)
        c = sum(cv[s] != "success" and gt[s] == "success" for s in seeds)
        p = mcnemar_exact_p(b, c)
        m = len(seeds)
        print(f"{n:>2} | {cv_s:>3}/{m:<4} | {gt_s:>3}/{m:<4} | "
              f"b={b:<3} c={c:<3}       | p={p:.4f} | "
              f"cv col/to={cv_col}/{cv_to} gt col/to={gt_col}/{gt_to}")
        report["cells"].append({
            "n": n, "m": m, "cv_success": cv_s, "gt_success": gt_s,
            "cv_collision": cv_col, "gt_collision": gt_col,
            "cv_timeout": cv_to, "gt_timeout": gt_to,
            "b": b, "c": c, "p": p,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    _plot(report, Path(args.out).with_suffix(".png"))
    print(f"plotted {Path(args.out).with_suffix('.png')}")


def _plot(report, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = sorted(report["cells"], key=lambda d: d["n"])
    ns = [d["n"] for d in cells]
    m = [d["m"] for d in cells]
    cv = [100.0 * d["cv_success"] / d["m"] for d in cells]
    gt = [100.0 * d["gt_success"] / d["m"] for d in cells]
    delta = [100.0 * (d["c"] - d["b"]) / d["m"] for d in cells]  # gt − cv, paired
    ps = [d["p"] for d in cells]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax1.plot(ns, cv, "o-", color="#1f77b4", lw=2, ms=8, label="constant_velocity (coast)")
    ax1.plot(ns, gt, "s-", color="#d62728", lw=2, ms=8, label="game_theoretic (goal-aware)")
    ax1.set_xlabel("N drones (antipodal swap on a ring)")
    ax1.set_ylabel("joint success rate (%)")
    ax1.set_title("Goal-aware peer prediction: wins head-on (N=2),\n"
                  "inverts to a liability on the symmetric swap (N≥3)")
    ax1.set_xticks(ns)
    ax1.set_ylim(-3, 103)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")

    colors = ["#2ca02c" if d > 0 else "#d62728" for d in delta]
    ax2.bar(ns, delta, color=colors, alpha=0.85)
    ax2.axhline(0, color="k", lw=0.8)
    for x, d, p in zip(ns, delta, ps):
        ax2.annotate(f"p={p:.3f}" if p >= 1e-4 else "p<1e-4",
                     (x, d), textcoords="offset points",
                     xytext=(0, 6 if d >= 0 else -14), ha="center", fontsize=8)
    ax2.set_xlabel("N drones")
    ax2.set_ylabel("paired Δ joint success: game_theoretic − const_velocity (pp)")
    ax2.set_title("Sign flips at N=2→3 (McNemar paired-by-seed)")
    ax2.set_xticks(ns)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
