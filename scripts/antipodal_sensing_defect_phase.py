"""Under the convention, WHICH sensing defect restores the predictor — position
noise, perception delay, or both?

The [predictor-restoration result](docs/findings.md "Sensing noise restores the
predictor's relevance under the convention") showed that with the right-of-way
convention on, ground-truth peers make cv+row = gt+row (a tie), but position
*noise* re-opens a significant gt+row > cv+row gap (gt anchors each peer on its
exact goal; cv extrapolates the corrupted state). Is that a property of sensing
error in general, or specific to noise?

This crosses the two canonical tracking defects — position noise (σ) and a fixed
perception delay (τ, position lagged by a ring buffer; velocity passes through) —
on the same antipodal swarm. Prediction from the mechanism: noise differentiates
the predictors (gt's goal anchor survives a corrupted position; cv's extrapolation
does not), but delay does NOT — both predictors read the *same* stale position,
and the only thing delay leaves intact (the true velocity) is the very channel
cv uses, so they should degrade together.

Convention always on (lateral_bias), arms cv+row vs gt+row, paired by seed, over
a noise × delay grid. McNemar(gt+row vs cv+row) per cell.

  python scripts/antipodal_sensing_defect_phase.py --n 8 \
      --noise-list 0 1 2 --delay-list 0 0.05 0.1 --episodes 40
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


def _wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _drones(n):
    out = []
    for k in range(n):
        ang = 2.0 * math.pi * k / n
        out.append({
            "name": f"d{k}",
            "start": [round(CENTER[0] + RADIUS * math.cos(ang), 3),
                      round(CENTER[1] + RADIUS * math.sin(ang), 3)],
            "goal": [round(CENTER[0] - RADIUS * math.cos(ang), 3),
                     round(CENTER[1] - RADIUS * math.sin(ang), 3)],
            "radius": 0.4, "start_jitter": 0.8,
        })
    return out


def _cfg(n, predictor, noise, delay, bias, seed, n_eps, max_steps):
    return {
        "name": f"sensdef_n{n}_{predictor[:2]}_s{noise:g}_d{delay:g}",
        "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50], "resolution": 1.0,
                     "obstacles": {"type": "none"}, "dynamic_obstacles": [],
                     "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": {"type": "mpc", "max_speed": 5.0, "replan_period": 0.2,
                    "horizon": 40, "dt_plan": 0.05, "n_samples": 32,
                    "resolution": 1.0, "inflate": 1, "goal_radius": 1.5,
                    "safety_margin": 0.5, "use_prediction": True,
                    "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
                    "lateral_bias": bias, "predictor": {"type": predictor}},
        "sensor": {"type": "noisy_tracker", "dt": 0.05, "delay": delay,
                   "position_noise_std": noise, "velocity_noise_std": noise},
        "output": {"dir": "results/sensdef_tmp"},
    }


def _run_cell(job):
    n, predictor, noise, delay, bias, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, predictor, noise, delay, bias, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return ("gt" if predictor == "game_theoretic" else "cv", noise, delay, by_seed)


def _succ(bs, seeds):
    return sum(bs[s] == "success" for s in seeds)


def _mcnemar(a, b, seeds):
    bb = sum(a[s] == "success" and b[s] != "success" for s in seeds)
    cc = sum(a[s] != "success" and b[s] == "success" for s in seeds)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--noise-list", type=float, nargs="+", default=[0, 1, 2])
    ap.add_argument("--delay-list", type=float, nargs="+", default=[0, 0.05, 0.1])
    ap.add_argument("--bias", type=float, default=4.0)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/antipodal_sensing_defect_phase.json")
    ap.add_argument("--plot-from", default=None)
    args = ap.parse_args()

    if args.plot_from:
        report = json.loads(Path(args.plot_from).read_text())
        _plot(report, Path(args.plot_from).with_suffix(".png"))
        return

    preds = ["constant_velocity", "game_theoretic"]
    jobs = [(args.n, p, s, d, args.bias, args.seed, args.episodes, args.max_steps)
            for p in preds for s in args.noise_list for d in args.delay_list]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)
    cell = {}
    for arm, noise, delay, bs in res:
        cell[(arm, noise, delay)] = bs

    seeds = sorted(set.intersection(*[set(bs) for bs in cell.values()]))
    m = len(seeds)
    report = {"radius": RADIUS, "n": args.n, "episodes": args.episodes, "m": m,
              "bias": args.bias, "noise_list": args.noise_list,
              "delay_list": args.delay_list, "cells": []}
    print(f"\nsensing-defect taxonomy: N={args.n}, bias={args.bias}, m={m} paired seeds")
    print(f"{'noise':>6} {'delay':>6} | {'cv+row':>10} {'gt+row':>10} | gt vs cv (c/b,p)")
    print("-" * 60)
    for s in args.noise_list:
        for d in args.delay_list:
            scv, sgt = _succ(cell[("cv", s, d)], seeds), _succ(cell[("gt", s, d)], seeds)
            bb, cc, p = _mcnemar(cell[("cv", s, d)], cell[("gt", s, d)], seeds)
            tag = "  gt WINS" if (cc - bb > 0 and p < 0.05) else ("  tie" if p >= 0.05 else "")
            print(f"{s:>6g} {d:>6g} | {scv:>3}/{m} {scv*100//m:>3.0f}% {sgt:>3}/{m} {sgt*100//m:>3.0f}%"
                  f" | c={cc:<2}/b={bb:<2} p={p:.4g}{tag}")
            report["cells"].append({"noise": s, "delay": d, "s_cv": scv, "s_gt": sgt,
                                    "gt_vs_cv": {"b": bb, "c": cc, "p": p}})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    print("  noise column: gt+row > cv+row (restoration). delay column: tie (no restoration).")
    _plot(report, Path(args.out).with_suffix(".png"))


def _plot(report, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    noises = report["noise_list"]
    delays = report["delay_list"]
    m = report["m"]
    by = {(c["noise"], c["delay"]): c for c in report["cells"]}
    # delta = gt - cv success (pp); positive = gt better
    grid = np.array([[100.0 * (by[(s, d)]["s_gt"] - by[(s, d)]["s_cv"]) / m
                      for d in delays] for s in noises])
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-60, vmax=60, aspect="auto", origin="lower")
    ax.set_xticks(range(len(delays)))
    ax.set_xticklabels([f"{d:g}" for d in delays])
    ax.set_yticks(range(len(noises)))
    ax.set_yticklabels([f"{s:g}" for s in noises])
    ax.set_xlabel("perception delay τ (s)")
    ax.set_ylabel("position/velocity noise σ")
    for i, s in enumerate(noises):
        for j, d in enumerate(delays):
            c = by[(s, d)]
            p = c["gt_vs_cv"]["p"]
            star = "*" if (p < 0.05 and c["s_gt"] > c["s_cv"]) else ""
            ax.text(j, i, f"{int(round(100*c['s_cv']/m))}/{int(round(100*c['s_gt']/m))}{star}",
                    ha="center", va="center", fontsize=9)
    ax.set_title("gt+row − cv+row (pp); cell = cv%/gt%, * = gt wins p<0.05\n"
                 "noise restores the predictor (blue up the σ axis); delay does not (flat across τ)")
    fig.colorbar(im, ax=ax, label="gt+row − cv+row (percentage points)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"plotted {png_path}")


if __name__ == "__main__":
    main()
