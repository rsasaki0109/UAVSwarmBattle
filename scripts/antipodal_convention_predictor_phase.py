"""Once the right-of-way convention is on, does the PREDICTOR still matter?

Two results frame this:
  * Goal-aware prediction INVERTS on the antipodal swap (gt deadlocks where dumb cv
    threads through) -- [[goal-aware predictor inversion]].
  * A lateral_bias right-of-way convention lifts gt back to 100% by breaking the symmetry
    -- [[right-of-way fix]] -- and at 2-drone head-on, cv+row even BEATS gt (the convention
    SUBSTITUTES for the predictor) -- [[right-of-way substitution]].

So at N=2 the convention dominates the predictor. Does that hold for the N-drone antipodal
swarm? If breaking the symmetry is what matters, then WITH the convention on, the forecast
should be irrelevant: cv+row ~ gt+row. If the predictor still does independent work (e.g.
tracking peers' curve-back inside the roundabout), gt+row should beat cv+row.

Four arms, paired by seed, fixed RADIUS=20 so N sets hub density (the regime where gt+row
has a cliff): cv (no bias), gt (no bias), cv+row, gt+row, all at lateral_bias B for the
+row arms. McNemar exact for cv+row vs gt+row (the dominance test) and each +row vs its
no-row self.

  python scripts/antipodal_convention_predictor_phase.py --n-list 8 12 16 --bias 2 --episodes 40
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
# (label, predictor, uses_bias)
ARMS = [("cv", "constant_velocity", False),
        ("gt", "game_theoretic", False),
        ("cvrow", "constant_velocity", True),
        ("gtrow", "game_theoretic", True)]


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


def _cfg(n, predictor, bias, seed, n_eps):
    return {
        "name": f"anticonvpred_n{n}_{predictor}_b{bias:g}",
        "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50], "resolution": 1.0,
                     "obstacles": {"type": "none"}, "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": 800,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": {"type": "mpc", "max_speed": 5.0, "replan_period": 0.2,
                    "horizon": 40, "dt_plan": 0.05, "n_samples": 32,
                    "resolution": 1.0, "inflate": 1, "goal_radius": 1.5,
                    "safety_margin": 0.5, "use_prediction": True,
                    "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
                    "lateral_bias": bias, "predictor": {"type": predictor}},
        "sensor": {"type": "perfect"}, "output": {"dir": "results/anticonvpred_tmp"},
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


def _succ(bs, seeds):
    return sum(bs[s] == "success" for s in seeds)


def _mcnemar(a, b, seeds):
    bb = sum(a[s] == "success" and b[s] != "success" for s in seeds)
    cc = sum(a[s] != "success" and b[s] == "success" for s in seeds)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--bias", type=float, default=2.0)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/antipodal_convention_predictor_phase.json")
    ap.add_argument("--plot-from", default=None)
    args = ap.parse_args()

    if args.plot_from:
        report = json.loads(Path(args.plot_from).read_text())
        _plot(report, Path(args.plot_from).with_suffix(".png"))
        return

    jobs = []
    for n in args.n_list:
        for label, pred, uses_bias in ARMS:
            jobs.append((label, n, pred, args.bias if uses_bias else 0.0,
                         args.seed, args.episodes))
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)
    cells = {}
    for label, n, bs in res:
        cells.setdefault(n, {})[label] = bs

    report = {"radius": RADIUS, "bias": args.bias, "episodes": args.episodes, "cells": []}
    print(f"\nconvention x predictor: bias={args.bias}, RADIUS={RADIUS}, n={args.episodes}")
    print(f"{'N':>3} | {'cv':>5} {'gt':>5} {'cv+row':>6} {'gt+row':>6} | "
          f"{'gtrow vs cvrow (c/b,p)':>22}")
    print("-" * 70)
    for n in sorted(cells):
        c = cells[n]
        seeds = sorted(set(c["cv"]) & set(c["gt"]) & set(c["cvrow"]) & set(c["gtrow"]))
        m = len(seeds)
        sv = {k: _succ(c[k], seeds) for k in ("cv", "gt", "cvrow", "gtrow")}
        # c-b>0 => gtrow better than cvrow (predictor still does work under convention)
        b, cc, p = _mcnemar(c["cvrow"], c["gtrow"], seeds)
        print(f"{n:>3} | {sv['cv']*100//m:>4.0f}% {sv['gt']*100//m:>4.0f}% "
              f"{sv['cvrow']*100//m:>5.0f}% {sv['gtrow']*100//m:>5.0f}% | "
              f"c={cc:<2}/b={b:<2} p={p:>6.4f}")
        report["cells"].append({"n": n, "m": m, **{f"{k}_s": sv[k] for k in sv},
                                "gtrow_vs_cvrow": {"b": b, "c": cc, "p": p}})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    print("  cv+row ~ gt+row (tie) => convention DOMINATES predictor (symmetry, not forecast).")
    print("  gt+row > cv+row       => predictor still does work inside the roundabout.")
    _plot(report, Path(args.out).with_suffix(".png"))


def _plot(report, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = sorted(report["cells"], key=lambda d: d["n"])
    ns = [d["n"] for d in cells]
    series = [("cv_s", "cv (no bias)", "#1f77b4", "o", "--"),
              ("gt_s", "gt (no bias)", "#7f7f7f", "x", "--"),
              ("cvrow_s", "cv + right-of-way", "#e67e22", "D", "-"),
              ("gtrow_s", "gt + right-of-way", "#2ca02c", "s", "-")]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for key, lab, col, mk, ls in series:
        ax.plot(ns, [100.0 * d[key] / d["m"] for d in cells], mk + ls, color=col,
                lw=2.2, ms=8, label=lab)
    ax.set_xlabel("N drones (antipodal swap, fixed R=20 -> rising hub density)")
    ax.set_ylabel("joint success rate (%)")
    ax.set_title("Once right-of-way is on, does the predictor matter?\n"
                 "cv+row vs gt+row overlap = convention dominates the forecast")
    ax.set_xticks(ns)
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"plotted {png_path}")


if __name__ == "__main__":
    main()
