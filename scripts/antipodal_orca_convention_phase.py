"""Does the right-of-way convention generalise from our sampling MPC to the
canonical ORCA reciprocal planner?

Background. This repo's convention arc proved that a decentralized right-of-way
lateral bias lifts the antipodal-swap deadlock that goal-aware prediction
amplifies under the CPU MPC stack (the symmetric shared forecast makes the
fleet mirror-swerve into a new symmetric arrangement that re-collides at the
hub; a "veer right" rule turns it into a clockwise roundabout). The open
question the audit flagged: every one of those results was measured against
*other MPC arms*, never against the literature's canonical reactive baseline,
ORCA (van den Berg et al. 2011, "Reciprocal n-Body Collision Avoidance").

ORCA is the *reciprocal* school's answer to the same problem: no forecast, no
sampling — each agent splits the avoidance 50/50 and solves a tiny velocity-
space linear program. It is famously prone to symmetric deadlock on exactly
this antipodal benchmark. So ORCA is both the missing baseline AND a clean test
of whether the convention is a property of *our planner* or of *the geometry*:
if the right-of-way bias rescues ORCA too, the convention is planner-agnostic.

Arms = a sweep of ORCA's ``lateral_bias`` at fixed N (so the only thing that
changes is the convention strength), all paired by seed. bias=0 is stock ORCA.
Reports per-bias joint success + Wilson 95% CI, and McNemar exact for the
band-centre bias vs stock (rescue) and vs an over-strong bias (overshoot) — the
inverted-U operating band, the same shape as the MPC convention cliff.

  python scripts/antipodal_orca_convention_phase.py --n-list 6 8 \
      --bias-list 0 0.1 0.2 0.3 0.4 0.5 --episodes 40
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


def _cfg(n, bias, seed, n_eps, max_steps):
    return {
        "name": f"orcaconv_n{n}_b{bias:g}",
        "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50], "resolution": 1.0,
                     "obstacles": {"type": "none"}, "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": {"type": "orca", "max_speed": 5.0, "radius": 0.4,
                    "time_horizon": 2.0, "time_step": 0.25, "neighbor_dist": 15.0,
                    "safety_margin": 0.1, "goal_radius": 1.5, "lateral_bias": bias},
        "sensor": {"type": "perfect"}, "output": {"dir": "results/orcaconv_tmp"},
    }


def _run_cell(job):
    n, bias, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, bias, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (n, bias, by_seed)


def _succ(bs, seeds):
    return sum(bs[s] == "success" for s in seeds)


def _mcnemar(a, b, seeds):
    bb = sum(a[s] == "success" and b[s] != "success" for s in seeds)
    cc = sum(a[s] != "success" and b[s] == "success" for s in seeds)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[6, 8])
    ap.add_argument("--bias-list", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/antipodal_orca_convention_phase.json")
    ap.add_argument("--plot-from", default=None)
    args = ap.parse_args()

    if args.plot_from:
        report = json.loads(Path(args.plot_from).read_text())
        _plot(report, Path(args.plot_from).with_suffix(".png"))
        return

    jobs = [(n, b, args.seed, args.episodes, args.max_steps)
            for n in args.n_list for b in args.bias_list]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)
    cells = {}
    for n, bias, bs in res:
        cells.setdefault(n, {})[bias] = bs

    report = {"radius": RADIUS, "episodes": args.episodes,
              "bias_list": args.bias_list, "cells": []}
    print(f"\nORCA right-of-way convention: RADIUS={RADIUS}, n={args.episodes}")
    for n in sorted(cells):
        c = cells[n]
        biases = sorted(c)
        seeds = sorted(set.intersection(*[set(c[b]) for b in biases]))
        m = len(seeds)
        sv = {b: _succ(c[b], seeds) for b in biases}
        # band centre = the bias with the most successes (ties -> smallest)
        best = max(biases, key=lambda b: (sv[b], -b))
        stock = min(biases)  # bias 0 = stock ORCA
        over = max(biases)   # strongest bias = overshoot side
        b_r, c_r, p_r = _mcnemar(c[stock], c[best], seeds)   # rescue
        b_o, c_o, p_o = _mcnemar(c[over], c[best], seeds)    # overshoot
        print(f"\n N={n} (m={m} paired seeds)")
        print(f"   {'bias':>5} {'succ':>7} {'rate':>6}  95% CI")
        for b in biases:
            lo, hi = _wilson(sv[b], m)
            tag = "  <- band centre" if b == best else ""
            print(f"   {b:>5g} {sv[b]:>3}/{m:<3} {sv[b]*100/m:>5.0f}%  "
                  f"[{lo*100:4.0f},{hi*100:4.0f}]{tag}")
        print(f"   RESCUE    best({best:g}) vs stock(0):    "
              f"c={c_r}/b={b_r} p={p_r:.4g}")
        print(f"   OVERSHOOT best({best:g}) vs over({over:g}): "
              f"c={c_o}/b={b_o} p={p_o:.4g}")
        report["cells"].append({
            "n": n, "m": m,
            "succ": {f"{b:g}": sv[b] for b in biases},
            "best": best, "stock": stock, "over": over,
            "rescue": {"b": b_r, "c": c_r, "p": p_r},
            "overshoot": {"b": b_o, "c": c_o, "p": p_o},
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    print("  inverted-U band => convention generalises to ORCA, but is double-edged.")
    _plot(report, Path(args.out).with_suffix(".png"))


def _plot(report, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    biases = report["bias_list"]
    cells = sorted(report["cells"], key=lambda d: d["n"])
    colors = ["#2ca02c", "#e67e22", "#1f77b4", "#9467bd"]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for i, d in enumerate(cells):
        m = d["m"]
        ys = [100.0 * d["succ"][f"{b:g}"] / m for b in biases]
        ax.plot(biases, ys, "o-", color=colors[i % len(colors)], lw=2.2, ms=8,
                label=f"N={d['n']} (R={report['radius']:g})")
    ax.axvline(0.0, color="#888", ls=":", lw=1)
    ax.set_xlabel("ORCA lateral_bias (right-of-way convention strength)")
    ax.set_ylabel("antipodal joint success rate (%)")
    ax.set_title("Right-of-way convention ported to ORCA (van den Berg 2011)\n"
                 "bias=0 is stock ORCA; an inverted-U band rescues the deadlock")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"plotted {png_path}")


if __name__ == "__main__":
    main()
