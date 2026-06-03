"""Does a decentralized right-of-way lateral bias lift the antipodal-swap
deadlock that goal-aware prediction otherwise amplifies?

Companion to scripts/antipodal_predictor_phase.py, which proved that the
game_theoretic predictor — the winner on a 2-drone crossing — INVERTS to a
significant liability on the N-drone antipodal swap, because every drone's
correct, shared, symmetric forecast makes the fleet mirror-swerve into a new
symmetric arrangement that re-collides at the hub. The named fix was an
explicit symmetry-breaker, which the CPU MPC stack lacked.

This script tests that fix: a new `planner.lateral_bias` knob (a small global
cost preference for veering to the RIGHT of the goal heading — a decentralized
right-of-way rule that turns the symmetric convergence into a clockwise
roundabout). Arms, paired by seed:

  cv      : constant_velocity, lateral_bias 0   (the surprise winner)
  gt      : game_theoretic,    lateral_bias 0   (the inverted loser)
  gt+row  : game_theoretic,    lateral_bias B   (the fix)

Reports McNemar exact for gt+row vs gt (does the fix help?) and gt+row vs cv
(does the fix beat the myopic winner?). --bias-sweep calibrates B at one N.
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


def _cfg(n, predictor, bias, seed, n_eps):
    return {
        "name": f"antipodal_n{n}_{predictor}_b{bias}",
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
            "type": "dummy_2d", "dt": 0.05, "max_steps": 1000,
            "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4,
        },
        "planner": {
            "type": "mpc", "max_speed": 5.0, "replan_period": 0.2,
            "horizon": 40, "dt_plan": 0.05, "n_samples": 32,
            "resolution": 1.0, "inflate": 1, "goal_radius": 1.5,
            "safety_margin": 0.5, "use_prediction": True,
            "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
            "lateral_bias": bias,
            "predictor": {"type": predictor},
        },
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/antipodal_row_tmp"},
    }


def _run_cell(job):
    label, n, predictor, bias, seed, n_eps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, predictor, bias, seed, n_eps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (label, n, by_seed)


def _succ(by_seed, seeds):
    return sum(by_seed[s] == "success" for s in seeds)


def _mcnemar(a, b, seeds):
    # b-side = a-success/b-fail ; c-side = a-fail/b-success ; c-b>0 => b better
    bb = sum(a[s] == "success" and b[s] != "success" for s in seeds)
    cc = sum(a[s] != "success" and b[s] == "success" for s in seeds)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--bias", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bias-sweep", type=float, nargs="+", default=None,
                    help="calibrate: at a single N (first of --n-list), run gt at each of these biases")
    ap.add_argument("--out", default="results/antipodal_rightofway_phase.json")
    ap.add_argument("--overlay", nargs="+", default=None,
                    help="report jsons (one per bias) to overlay cv + gt+row vs N (no run)")
    args = ap.parse_args()

    if args.overlay is not None:
        _plot_overlay(args.overlay, args.out)
        return

    if args.bias_sweep is not None:
        n = args.n_list[0]
        jobs = [(f"gt_b{b}", n, "game_theoretic", b, args.seed, args.episodes)
                for b in args.bias_sweep]
        jobs.append(("cv_b0", n, "constant_velocity", 0.0, args.seed, args.episodes))
        with Pool(min(args.workers, len(jobs))) as pool:
            res = pool.map(_run_cell, jobs)
        cells = {label: bs for label, _, bs in res}
        cv = cells["cv_b0"]
        print(f"\nbias calibration at N={n} (n={args.episodes}); cv baseline = "
              f"{_succ(cv, sorted(cv))}/{len(cv)}")
        print(f"{'bias':>6} | {'gt succ':>8} | vs cv (b/c, p)")
        print("-" * 48)
        for b in args.bias_sweep:
            gt = cells[f"gt_b{b}"]
            seeds = sorted(set(gt) & set(cv))
            bb, cc, p = _mcnemar(cv, gt, seeds)
            print(f"{b:>6} | {_succ(gt, seeds):>3}/{len(seeds):<4} | b={bb} c={cc} p={p:.4f}")
        return

    jobs = []
    for n in args.n_list:
        jobs.append(("cv", n, "constant_velocity", 0.0, args.seed, args.episodes))
        jobs.append(("gt", n, "game_theoretic", 0.0, args.seed, args.episodes))
        jobs.append(("gtrow", n, "game_theoretic", args.bias, args.seed, args.episodes))
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    cells = {}
    for label, n, bs in res:
        cells.setdefault(n, {})[label] = bs

    report = {"radius": RADIUS, "episodes": args.episodes, "bias": args.bias, "cells": []}
    print(f"\nright-of-way lateral_bias={args.bias}, n={args.episodes}, paired by seed")
    print(f"{'N':>2} | {'cv':>6} | {'gt':>6} | {'gt+row':>6} | "
          f"{'row vs gt (c-b,p)':>20} | {'row vs cv (c-b,p)':>20}")
    print("-" * 86)
    for n in sorted(cells):
        cv, gt, row = cells[n]["cv"], cells[n]["gt"], cells[n]["gtrow"]
        seeds = sorted(set(cv) & set(gt) & set(row))
        m = len(seeds)
        cv_s, gt_s, row_s = _succ(cv, seeds), _succ(gt, seeds), _succ(row, seeds)
        b1, c1, p1 = _mcnemar(gt, row, seeds)   # c1-b1>0 => row better than gt
        b2, c2, p2 = _mcnemar(cv, row, seeds)   # c2-b2>0 => row better than cv
        print(f"{n:>2} | {cv_s:>3}/{m:<2} | {gt_s:>3}/{m:<2} | {row_s:>3}/{m:<2} | "
              f"b={b1} c={c1} p={p1:>6.4f}    | b={b2} c={c2} p={p2:>6.4f}")
        report["cells"].append({
            "n": n, "m": m, "cv": cv_s, "gt": gt_s, "gtrow": row_s,
            "row_vs_gt": {"b": b1, "c": c1, "p": p1},
            "row_vs_cv": {"b": b2, "c": c2, "p": p2},
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
    m = cells[0]["m"]
    cv = [100.0 * d["cv"] / d["m"] for d in cells]
    gt = [100.0 * d["gt"] / d["m"] for d in cells]
    row = [100.0 * d["gtrow"] / d["m"] for d in cells]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(ns, cv, "o-", color="#1f77b4", lw=2, ms=8, label="constant_velocity (no bias)")
    ax.plot(ns, gt, "s-", color="#d62728", lw=2, ms=8, label="game_theoretic (no bias)")
    ax.plot(ns, row, "D-", color="#2ca02c", lw=2.5, ms=9,
            label=f"game_theoretic + right-of-way (bias={report['bias']})")
    ax.set_xlabel("N drones (antipodal swap on a ring)")
    ax.set_ylabel("joint success rate (%)")
    ax.set_title("A decentralized right-of-way lateral bias lifts the\n"
                 "antipodal-swap deadlock that goal-aware prediction amplifies")
    ax.set_xticks(ns)
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


def _plot_overlay(files, out):
    """Overlay cv, gt(no bias), and gt+row at several biases vs N — the convention cliff
    and how a stronger bias pushes it out."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports = [json.loads(Path(f).read_text()) for f in files]
    reports.sort(key=lambda r: r.get("bias", 0))
    base = reports[0]
    cells0 = sorted(base["cells"], key=lambda d: d["n"])
    ns = [d["n"] for d in cells0]
    cv = [100.0 * d["cv"] / d["m"] for d in cells0]
    gt = [100.0 * d["gt"] / d["m"] for d in cells0]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(ns, cv, "o--", color="#1f77b4", lw=1.8, ms=7, label="constant_velocity (no bias)")
    ax.plot(ns, gt, "x--", color="#7f7f7f", lw=1.5, ms=7, label="game_theoretic (no bias)")
    greens = ["#a1d99b", "#41ab5d", "#006d2c", "#00441b"]
    for i, r in enumerate(reports):
        cells = sorted(r["cells"], key=lambda d: d["n"])
        row = [100.0 * d["gtrow"] / d["m"] for d in cells]
        xs = [d["n"] for d in cells]
        ax.plot(xs, row, "D-", color=greens[i % len(greens)], lw=2.4, ms=8,
                label=f"game_theoretic + right-of-way (bias={r['bias']:g})")
    ax.set_xlabel("N drones (antipodal swap on a ring, fixed radius → rising hub density)")
    ax.set_ylabel("joint success rate (%)")
    ax.set_title("The convention cliff: a fixed right-of-way bias decays with N,\n"
                 "but a stronger bias pushes the cliff out")
    ax.set_xticks(ns)
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    out_path = Path(out)
    if out_path.suffix == "":
        out_path = out_path / "convention_cliff.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote overlay {out_path}")


if __name__ == "__main__":
    main()
