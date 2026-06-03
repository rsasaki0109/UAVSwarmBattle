"""Does a HETEROGENEOUS predictor mix break the antipodal deadlock by itself?

The proven antipodal inversion (docs/findings.md "Goal-aware peer prediction wins
head-on and inverts to a liability on the symmetric swap") showed that when EVERY drone
runs the same goal-aware `game_theoretic` predictor, N≥3 antipodal swaps collapse: all
drones share the SAME symmetric forecast, all mirror-swerve into the same new arrangement,
and re-collide at the hub (N=6: gt ~1/40). The fix shipped so far is an EXPLICIT convention
— `planner.lateral_bias` right-of-way — that injects an asymmetry every drone obeys.

This script tests an IMPLICIT alternative: if the deadlock is caused by a SHARED symmetric
forecast, then simply making the forecasts DIFFER should break the symmetry without any
convention. Run half the swarm on `game_theoretic` and half on `constant_velocity`: the two
groups now predict their peers differently, desynchronise, and (hypothesis) thread through
the hub where a uniform fleet jams.

Three arms, paired by seed, same ring geometry / MPC / dynamics as
antipodal_predictor_phase.py (CENTER=(25,25), RADIUS=20, max_accel=6, start_jitter=0.8):
  all_cv  : every drone constant_velocity (the reactive baseline)
  all_gt  : every drone game_theoretic    (the inverted, deadlocking fleet)
  mixed   : first half game_theoretic, second half constant_velocity (per_drone)

Joint success = all drones reach goal, no inter-drone collision. McNemar exact:
  mixed vs all_gt  (does heterogeneity RESCUE the deadlocked fleet?)
  mixed vs all_cv  (does it also beat the dumb-but-symmetry-free baseline?)

Calibrate N first — the inversion is strongest at N=4/6; pick an N where all_gt is off the
floor-vs-ceiling so the mix can discriminate:
    python scripts/antipodal_heterogeneous_phase.py --n-list 4 6 --episodes 20
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
ARMS = ["all_cv", "all_gt", "mixed"]
MIX_PATTERN = "block"  # "block" = first half gt; "alternating" = gt,cv,gt,cv...


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


def _arm_predictors(arm, n):
    """Return (base_predictor, per_drone_list_or_None) for an arm."""
    if arm == "all_cv":
        return "constant_velocity", None
    if arm == "all_gt":
        return "game_theoretic", None
    if arm == "mixed":
        # half goal-aware, half coasting — two desynchronised groups.
        # block: first n//2 are gt (contiguous arc); alternating: gt,cv,gt,cv...
        half = n // 2
        per = []
        for i in range(n):
            if MIX_PATTERN == "alternating":
                is_gt = (i % 2 == 0)
            else:
                is_gt = (i < half)
            pred = "game_theoretic" if is_gt else "constant_velocity"
            per.append({"predictor": {"type": pred}})
        return "game_theoretic", per
    raise ValueError(arm)


def _cfg(n, arm, seed, n_eps):
    base_pred, per_drone = _arm_predictors(arm, n)
    planner = {
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
        "predictor": {"type": base_pred},
    }
    if per_drone is not None:
        planner["per_drone"] = per_drone
    return {
        "name": f"antihet_n{n}_{arm}",
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
        "planner": planner,
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/antihet_tmp"},
    }


def _run_cell(args):
    n, arm, seed, n_eps = args
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, arm, seed, n_eps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (n, arm, by_seed)


def _mcnemar(a, b_arm, seeds):
    """b = a-success & other-fail; c = a-fail & other-success. c-b>0 => other better."""
    b = sum(a[s] == "success" and b_arm[s] != "success" for s in seeds)
    c = sum(a[s] != "success" and b_arm[s] == "success" for s in seeds)
    return b, c, mcnemar_exact_p(b, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/antipodal_heterogeneous_phase.json")
    ap.add_argument("--mix-pattern", choices=["block", "alternating"], default="block",
                    help="how the mixed arm places gt vs cv around the ring")
    ap.add_argument("--plot-from", default=None)
    args = ap.parse_args()

    global MIX_PATTERN
    MIX_PATTERN = args.mix_pattern

    if args.plot_from:
        report = json.loads(Path(args.plot_from).read_text())
        _plot(report, Path(args.plot_from).with_suffix(".png"))
        print(f"plotted {Path(args.plot_from).with_suffix('.png')}")
        return

    jobs = [(n, arm, args.seed, args.episodes) for n in args.n_list for arm in ARMS]
    with Pool(min(args.workers, len(jobs))) as pool:
        results = pool.map(_run_cell, jobs)

    cells = {}
    for n, arm, by_seed in results:
        cells.setdefault(n, {})[arm] = by_seed

    report = {"radius": RADIUS, "episodes": args.episodes, "cells": []}
    print(f"\n{'N':>2} | {'cv':>7} | {'gt':>7} | {'mixed':>7} | "
          f"{'mix vs gt (c/b,p)':>20} | {'mix vs cv (c/b,p)':>20}")
    print("-" * 88)
    for n in sorted(cells):
        cv, gt, mx = cells[n]["all_cv"], cells[n]["all_gt"], cells[n]["mixed"]
        seeds = sorted(set(cv) & set(gt) & set(mx))
        m = len(seeds)
        cv_s = sum(cv[s] == "success" for s in seeds)
        gt_s = sum(gt[s] == "success" for s in seeds)
        mx_s = sum(mx[s] == "success" for s in seeds)
        # mixed vs gt: c = mixed wins where gt fails
        b_g, c_g, p_g = _mcnemar(gt, mx, seeds)
        b_c, c_c, p_c = _mcnemar(cv, mx, seeds)
        print(f"{n:>2} | {cv_s:>3}/{m:<3} | {gt_s:>3}/{m:<3} | {mx_s:>3}/{m:<3} | "
              f"c={c_g:<2}/b={b_g:<2} p={p_g:>6.4f} | "
              f"c={c_c:<2}/b={b_c:<2} p={p_c:>6.4f}")
        report["cells"].append({
            "n": n, "m": m, "cv_success": cv_s, "gt_success": gt_s, "mixed_success": mx_s,
            "mixed_vs_gt": {"b": b_g, "c": c_g, "p": p_g},
            "mixed_vs_cv": {"b": b_c, "c": c_c, "p": p_c},
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    print("  c (mix vs gt) > 0 = heterogeneity RESCUES the deadlocked uniform-gt fleet;")
    print("  c (mix vs cv) > 0 = it also beats the symmetry-free dumb baseline.")
    _plot(report, Path(args.out).with_suffix(".png"))
    print(f"plotted {Path(args.out).with_suffix('.png')}")


def _plot(report, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = sorted(report["cells"], key=lambda d: d["n"])
    ns = [d["n"] for d in cells]
    cv = [100.0 * d["cv_success"] / d["m"] for d in cells]
    gt = [100.0 * d["gt_success"] / d["m"] for d in cells]
    mx = [100.0 * d["mixed_success"] / d["m"] for d in cells]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax1.plot(ns, cv, "o-", color="#1f77b4", lw=2, ms=8, label="all constant_velocity")
    ax1.plot(ns, gt, "s-", color="#d62728", lw=2, ms=8, label="all game_theoretic (deadlocks)")
    ax1.plot(ns, mx, "D-", color="#2ca02c", lw=2, ms=8, label="mixed gt/cv (heterogeneous)")
    ax1.set_xlabel("N drones (antipodal swap on a ring)")
    ax1.set_ylabel("joint success rate (%)")
    ax1.set_title("Does a heterogeneous predictor mix break the\nantipodal deadlock without a convention?")
    ax1.set_xticks(ns)
    ax1.set_ylim(-3, 103)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="best", fontsize=9)

    dg = [100.0 * (d["mixed_vs_gt"]["c"] - d["mixed_vs_gt"]["b"]) / d["m"] for d in cells]
    dc = [100.0 * (d["mixed_vs_cv"]["c"] - d["mixed_vs_cv"]["b"]) / d["m"] for d in cells]
    x = range(len(ns))
    w = 0.36
    ax2.bar([i - w / 2 for i in x], dg, w, color="#d62728", alpha=0.85, label="mixed − all_gt")
    ax2.bar([i + w / 2 for i in x], dc, w, color="#1f77b4", alpha=0.85, label="mixed − all_cv")
    ax2.axhline(0, color="k", lw=0.8)
    for i, d in zip(x, cells):
        pg = d["mixed_vs_gt"]["p"]
        ax2.annotate(f"p={pg:.3f}" if pg >= 1e-4 else "p<1e-4",
                     (i - w / 2, dg[i]), textcoords="offset points",
                     xytext=(0, 4 if dg[i] >= 0 else -12), ha="center", fontsize=7)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(ns)
    ax2.set_xlabel("N drones")
    ax2.set_ylabel("paired Δ joint success (pp)")
    ax2.set_title("Heterogeneity vs uniform fleets (McNemar paired)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
