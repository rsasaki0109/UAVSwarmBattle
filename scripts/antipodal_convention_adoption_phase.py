"""Does the right-of-way convention need a critical mass, or does partial adoption work?

The [[right-of-way fix]] lifts the antipodal deadlock to 100 % when EVERY drone obeys the
same `lateral_bias` veer-right convention. The heterogeneous-PREDICTOR study found the
opposite kind of mix helps: half gt / half cv DESYNCs the shared symmetric forecast and
rescues the uniform-gt deadlock — diversity (desync) breaks symmetry. This script asks the
mirror question for the CONVENTION: if only SOME drones obey the right-of-way rule and the
rest run the deadlocking goal-aware predictor with no bias, does coordination degrade
gracefully, kick in at a critical mass, or collapse unless everyone complies?

Hypothesis (the mirror of the predictor result): a convention is a COORDINATION device, not
a desync device — it only works if shared. Partial adoption should leave the non-adopting
drones to mirror-swerve into the hub and re-collide, so joint success should stay low until
adoption is near-total — desync helps prediction, coordination helps convention.

Setup: N antipodal drones, ALL on game_theoretic (the deadlocking predictor). `k` of them
also carry `lateral_bias`=B (adopters), the rest carry 0 (free-riders). Sweep k = 0..N via
`planner.per_drone`. Joint success = all reach goal, no inter-drone collision. Paired by
seed, McNemar exact vs full adoption (k=N).

  python scripts/antipodal_convention_adoption_phase.py --n 6 --bias 2 --episodes 40
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
ADOPT_PATTERN = "block"   # "block" = first k adopt; "spread" = every-other-ish


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
            "radius": 0.4,
            "start_jitter": 0.8,
        })
    return out


def _adopters(n, k):
    """Indices of the k drones that obey the convention."""
    if ADOPT_PATTERN == "spread" and 0 < k < n:
        # spread adopters as evenly as possible around the ring
        return set(round(i * n / k) % n for i in range(k))
    return set(range(k))  # block: first k


def _cfg(n, k, bias, seed, n_eps):
    adopt = _adopters(n, k)
    per_drone = [{"lateral_bias": bias if i in adopt else 0.0} for i in range(n)]
    return {
        "name": f"antiadopt_n{n}_k{k}",
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
            "type": "dummy_2d", "dt": 0.05, "max_steps": 700,
            "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4,
        },
        "planner": {
            "type": "mpc", "max_speed": 5.0, "replan_period": 0.2,
            "horizon": 40, "dt_plan": 0.05, "n_samples": 32,
            "resolution": 1.0, "inflate": 1, "goal_radius": 1.5,
            "safety_margin": 0.5, "use_prediction": True,
            "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
            "lateral_bias": 0.0,
            "predictor": {"type": "game_theoretic"},
            "per_drone": per_drone,
        },
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/antiadopt_tmp"},
    }


def _run_cell(job):
    n, k, bias, seed, n_eps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, k, bias, seed, n_eps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (k, by_seed)


def _succ(bs, seeds):
    return sum(bs[s] == "success" for s in seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--bias", type=float, default=2.0)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pattern", choices=["block", "spread"], default="block")
    ap.add_argument("--out", default="results/antipodal_convention_adoption_phase.json")
    ap.add_argument("--plot-from", default=None)
    ap.add_argument("--overlay", nargs="+", default=None,
                    help="report jsons (one per N) to overlay on the adoption-fraction axis")
    args = ap.parse_args()

    global ADOPT_PATTERN
    ADOPT_PATTERN = args.pattern

    if args.overlay:
        _plot_overlay(args.overlay, args.out)
        return

    if args.plot_from:
        report = json.loads(Path(args.plot_from).read_text())
        _plot(report, Path(args.plot_from).with_suffix(".png"))
        return

    ks = list(range(args.n + 1))
    jobs = [(args.n, k, args.bias, args.seed, args.episodes) for k in ks]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = dict(pool.map(_run_cell, jobs))

    full = res[args.n]  # k=N reference (full adoption)
    report = {"n": args.n, "bias": args.bias, "episodes": args.episodes,
              "pattern": args.pattern, "cells": []}
    print(f"\nconvention adoption sweep: N={args.n}, bias={args.bias}, n={args.episodes}, "
          f"pattern={args.pattern}")
    print(f"{'k adopt':>7} | {'frac':>5} | {'joint succ':>10} | {'vs full (c/b, p)':>18}")
    print("-" * 54)
    for k in ks:
        bs = res[k]
        seeds = sorted(set(bs) & set(full))
        m = len(seeds)
        s = _succ(bs, seeds)
        # b = this cell succ & full fail; c = this fail & full succ. (full is the reference)
        b = sum(bs[x] == "success" and full[x] != "success" for x in seeds)
        c = sum(bs[x] != "success" and full[x] == "success" for x in seeds)
        p = mcnemar_exact_p(b, c)
        print(f"{k:>7} | {k/args.n:>5.2f} | {s:>3}/{m:<3} {s/m*100:>3.0f}% | "
              f"c={c:<2}/b={b:<2} p={p:>6.4f}")
        report["cells"].append({"k": k, "frac": k / args.n, "m": m, "success": s,
                                "vs_full": {"b": b, "c": c, "p": p}})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    print("  Graceful (linear in frac) | critical-mass (step) | all-or-nothing "
          "(flat-low until k=N) — which one?")
    _plot(report, Path(args.out).with_suffix(".png"))


def _plot(report, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = sorted(report["cells"], key=lambda d: d["k"])
    fracs = [100.0 * d["frac"] for d in cells]
    succ = [100.0 * d["success"] / d["m"] for d in cells]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(fracs, succ, "o-", color="#8e44ad", lw=2.4, ms=9)
    ax.plot([0, 100], [succ[0], succ[-1]], "--", color="#bbbbbb", lw=1.3,
            label="linear (graceful) reference")
    ax.set_xlabel(f"% of fleet obeying right-of-way (N={report['n']}, bias={report['bias']:g})")
    ax.set_ylabel("joint success rate (%)")
    ax.set_title("Does the right-of-way convention need a critical mass?\n"
                 "Partial adoption on the antipodal swap (rest = deadlocking gt)")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"plotted {png_path}")


def _plot_overlay(files, out):
    """Overlay several N on the adoption-fraction axis — the convention's free-rider
    tolerance shrinks with crowd density."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports = [json.loads(Path(f).read_text()) for f in files]
    reports.sort(key=lambda r: r["n"])
    colors = ["#8e44ad", "#c0392b", "#2980b9", "#27ae60"]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for i, r in enumerate(reports):
        cells = sorted(r["cells"], key=lambda d: d["k"])
        fracs = [100.0 * d["frac"] for d in cells]
        succ = [100.0 * d["success"] / d["m"] for d in cells]
        ax.plot(fracs, succ, "o-", color=colors[i % len(colors)], lw=2.3, ms=8,
                label=f"N={r['n']} (free-rider tol: "
                      f"{'1 drone' if r['n'] == 6 else 'none — full only'})")
    ax.plot([0, 100], [0, 100], "--", color="#bbbbbb", lw=1.2, label="linear (graceful) reference")
    ax.set_xlabel("% of fleet obeying the right-of-way convention (rest = deadlocking gt)")
    ax.set_ylabel("joint success rate (%)")
    ax.set_title("The right-of-way convention needs (near-)full adoption\n"
                 "and its free-rider tolerance shrinks with crowd density")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    out_path = Path(out)
    if out_path.suffix == "":
        out_path = out_path / "convention_adoption.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote overlay {out_path}")


if __name__ == "__main__":
    main()
