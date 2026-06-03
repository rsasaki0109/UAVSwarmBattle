"""Is the right-of-way convention cliff a function of N, or of hub DENSITY (N/R)?

The [[convention cliff study]] showed a fixed `lateral_bias`=2 decays as N grows on a
fixed-radius ring (N=8->16: 97.5 -> 65 %), and a stronger bias pushes the cliff out. But N
was a proxy: on a fixed-radius ring, raising N raises the hub density (more drones cross the
same centre). The mechanistic question is whether the cliff is really about the *count* N or
about the *density* — the number of drones per unit ring circumference, N / (2*pi*R), i.e.
the linear density of the antipodal traffic through the hub.

If density is the single control variable, then two (N, R) points with the SAME N/R should
give the SAME joint success even though one has twice the drones — a density collapse. If
they don't collapse, N and R are separate effects (also a finding).

This sweeps matched-density pairs: (N, R) chosen so N/R is held across a small and a large
fleet. CENTER=(25,25), so R<=24 keeps the ring inside the 50x50 world. Arms gt / gt+row
(bias=2), paired by seed, joint success, McNemar exact. The headline plot is success vs N/R
with N=6 and N=12 overlaid: if they land on one curve, the cliff is density, not count.

  python scripts/antipodal_density_phase.py --episodes 40
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
BIAS = 2.0

# Matched-density (N, R) pairs: each row shares N/R across N=6 and N=12.
# N/R in {0.30, 0.60, 0.90, 1.20}; R<=24 to stay inside the 50x50 world.
PAIRS = [
    (0.30, [(6, 20.0), (12, 40.0)]),   # (12,40) is out of bounds -> dropped at build
    (0.60, [(6, 10.0), (12, 20.0)]),
    (0.90, [(6, 6.667), (12, 13.333)]),
    (1.20, [(6, 5.0), (12, 10.0)]),
]


def _drones(n, radius):
    out = []
    for k in range(n):
        ang = 2.0 * math.pi * k / n
        out.append({
            "name": f"d{k}",
            "start": [round(CENTER[0] + radius * math.cos(ang), 3),
                      round(CENTER[1] + radius * math.sin(ang), 3)],
            "goal": [round(CENTER[0] - radius * math.cos(ang), 3),
                     round(CENTER[1] - radius * math.sin(ang), 3)],
            "radius": 0.4,
            "start_jitter": 0.8,
        })
    return out


def _cfg(n, radius, bias, seed, n_eps):
    return {
        "name": f"antidens_n{n}_r{radius:g}_b{bias:g}",
        "seed": seed,
        "num_episodes": n_eps,
        "scenario": {
            "type": "multi_drone_grid", "size": [50, 50], "resolution": 1.0,
            "obstacles": {"type": "none"}, "drones": _drones(n, radius),
        },
        "simulator": {
            "type": "dummy_2d", "dt": 0.05, "max_steps": 800,
            "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4,
        },
        "planner": {
            "type": "mpc", "max_speed": 5.0, "replan_period": 0.2,
            "horizon": 40, "dt_plan": 0.05, "n_samples": 32,
            "resolution": 1.0, "inflate": 1, "goal_radius": 1.5,
            "safety_margin": 0.5, "use_prediction": True,
            "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
            "lateral_bias": bias, "predictor": {"type": "game_theoretic"},
        },
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/antidens_tmp"},
    }


def _run_cell(job):
    n, radius, bias, seed, n_eps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, radius, bias, seed, n_eps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (n, radius, by_seed)


def _succ(bs, seeds):
    return sum(bs[s] == "success" for s in seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--bias", type=float, default=BIAS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/antipodal_density_phase.json")
    ap.add_argument("--plot-from", default=None)
    args = ap.parse_args()

    if args.plot_from:
        report = json.loads(Path(args.plot_from).read_text())
        _plot(report, Path(args.plot_from).with_suffix(".png"))
        return

    # build in-bounds (N, R) cells
    cells = []
    for dens, pts in PAIRS:
        for (n, r) in pts:
            if CENTER[0] + r <= 49.5 and CENTER[1] + r <= 49.5:
                cells.append((dens, n, r))
    jobs = [(n, r, args.bias, args.seed, args.episodes) for (_, n, r) in cells]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = {(n, r): bs for (n, r, bs) in pool.map(_run_cell, jobs)}

    report = {"bias": args.bias, "episodes": args.episodes, "cells": []}
    print(f"\ndensity collapse: gt+row bias={args.bias}, n={args.episodes}, paired by seed")
    print(f"{'N/R':>6} | {'N':>3} | {'R':>6} | {'joint succ':>10}")
    print("-" * 40)
    for (dens, n, r) in cells:
        bs = res[(n, r)]
        seeds = sorted(bs)
        s = _succ(bs, seeds)
        m = len(seeds)
        print(f"{dens:>6.2f} | {n:>3} | {r:>6.2f} | {s:>3}/{m:<3} {s/m*100:>3.0f}%")
        report["cells"].append({"density": dens, "n": n, "r": r, "m": m, "success": s})

    # within each matched-density pair, McNemar N=6 vs N=12 (paired by seed)
    print(f"\n{'N/R':>6} | N=6 vs N=12 (same density): McNemar")
    print("-" * 46)
    for dens, pts in PAIRS:
        got = {n: res.get((n, r)) for (n, r) in pts if (n, r) in res}
        if len(got) == 2 and 6 in got and 12 in got:
            a, b = got[6], got[12]
            seeds = sorted(set(a) & set(b))
            bb = sum(a[s] == "success" and b[s] != "success" for s in seeds)
            cc = sum(a[s] != "success" and b[s] == "success" for s in seeds)
            p = mcnemar_exact_p(bb, cc)
            same = "COLLAPSE (tie)" if p > 0.05 else "SPLIT (differ)"
            print(f"{dens:>6.2f} | b={bb:<2} c={cc:<2} p={p:>6.4f}  {same}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    print("  If N=6 and N=12 land on one success-vs-(N/R) curve, the cliff is DENSITY, not count.")
    _plot(report, Path(args.out).with_suffix(".png"))


def _plot(report, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = report["cells"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for n, color, mk in [(6, "#8e44ad", "o"), (12, "#c0392b", "s")]:
        pts = sorted([c for c in cells if c["n"] == n], key=lambda c: c["density"])
        if pts:
            ax.plot([c["density"] for c in pts],
                    [100.0 * c["success"] / c["m"] for c in pts],
                    mk + "-", color=color, lw=2.2, ms=9, label=f"N={n}")
    ax.set_xlabel("hub linear density  N / R  (drones per unit ring radius)")
    ax.set_ylabel("joint success rate (%)")
    ax.set_title(f"Is the convention cliff density or count?\n"
                 f"gt + right-of-way (bias={report['bias']:g}); overlap = density collapse")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"plotted {png_path}")


if __name__ == "__main__":
    main()
