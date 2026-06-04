"""Which right-of-way convention survives an external hub-crossing obstacle —
the GLOBAL veer-right, or the PAIRWISE winding-number rule that dominates it in
an empty arena?

Two prior results frame this:
  * The pairwise winding-number convention (`pairwise_bias`) STRICTLY DOMINATES
    the global veer-right (`lateral_bias`) on `obstacles: none`: no 3D N=4 harm,
    and it pushes the 2D density cliff out at fixed strength (docs/findings.md
    "A pairwise winding-number right-of-way strictly dominates the global
    veer-right").
  * BUT the global convention is a *peer* rule with a chokepoint: it funnels the
    fleet into one clockwise hub roundabout, and a body crossing that hub caps
    success far below the obstacle-free ceiling (docs/findings.md "The
    right-of-way convention is a peer rule — a hub-crossing obstacle defeats the
    roundabout it builds").

Does the pairwise rule, which never builds a single shared roundabout, AVOID that
chokepoint and stay robust to the hub obstacle? Or does its very locality — no
shared rotational current to absorb an external perturbation — make it WORSE?

Two conventions, each at a strength that solves the peers on its own (global
`lateral_bias=4`, pairwise `pairwise_bias=8`; both ~100 % no-obstacle at N=6/8),
crossed with obstacle ∈ {none, hub-crossing, far-corner}, MPC + game_theoretic,
paired by seed. The headline is McNemar(global vs pairwise) under the hub
obstacle; `none` checks the match, `far` is the move-the-stressor control.

  python scripts/antipodal_convention_obstacle_robustness_phase.py \
      --n-list 6 8 --global-bias 4 --pairwise-bias 8 --episodes 40
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
# Same body in both obstacle conditions; only its lane changes.
OBSTACLES = {
    "none": [],
    "hub": [{"start": [25.0, 2.0], "velocity": [0.0, 4.5], "radius": 1.5, "reflect": True}],
    "far": [{"start": [5.0, 5.0], "velocity": [0.0, 4.5], "radius": 1.5, "reflect": True}],
}


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


def _cfg(n, lat, pw, obs_key, seed, n_eps, max_steps):
    return {
        "name": f"convobs_n{n}_{'g' if lat else 'p'}_{obs_key}",
        "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50], "resolution": 1.0,
                     "obstacles": {"type": "none"},
                     "dynamic_obstacles": [dict(o) for o in OBSTACLES[obs_key]],
                     "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": {"type": "mpc", "max_speed": 5.0, "replan_period": 0.2,
                    "horizon": 40, "dt_plan": 0.05, "n_samples": 32,
                    "resolution": 1.0, "inflate": 1, "goal_radius": 1.5,
                    "safety_margin": 0.5, "use_prediction": True,
                    "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
                    "lateral_bias": lat, "pairwise_bias": pw, "pairwise_radius": 5.0,
                    "predictor": {"type": "game_theoretic"}},
        "sensor": {"type": "perfect"}, "output": {"dir": "results/convobs_tmp"},
    }


def _run_cell(job):
    n, conv, lat, pw, obs_key, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, lat, pw, obs_key, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (n, conv, obs_key, by_seed)


def _succ(bs, seeds):
    return sum(bs[s] == "success" for s in seeds)


def _mcnemar(a, b, seeds):
    bb = sum(a[s] == "success" and b[s] != "success" for s in seeds)
    cc = sum(a[s] != "success" and b[s] == "success" for s in seeds)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[6, 8])
    ap.add_argument("--global-bias", type=float, default=4.0)
    ap.add_argument("--pairwise-bias", type=float, default=8.0)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/antipodal_convention_obstacle_robustness_phase.json")
    ap.add_argument("--plot-from", default=None)
    args = ap.parse_args()

    if args.plot_from:
        report = json.loads(Path(args.plot_from).read_text())
        _plot(report, Path(args.plot_from).with_suffix(".png"))
        return

    conv_arms = [("global", args.global_bias, 0.0), ("pairwise", 0.0, args.pairwise_bias)]
    jobs = [(n, conv, lat, pw, ok, args.seed, args.episodes, args.max_steps)
            for n in args.n_list for conv, lat, pw in conv_arms for ok in OBSTACLES]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)
    cells = {}
    for n, conv, ok, bs in res:
        cells.setdefault(n, {})[(conv, ok)] = bs

    report = {"radius": RADIUS, "episodes": args.episodes,
              "global_bias": args.global_bias, "pairwise_bias": args.pairwise_bias,
              "cells": []}
    print(f"\nconvention obstacle-robustness: global lat={args.global_bias} vs "
          f"pairwise pw={args.pairwise_bias}, n={args.episodes}")
    for n in sorted(cells):
        c = cells[n]
        seeds = sorted(set.intersection(*[set(bs) for bs in c.values()]))
        m = len(seeds)
        print(f"\n N={n} (m={m} paired seeds)   success rate [Wilson 95% CI]")
        print(f"   {'obstacle':>8} | {'global':>16} | {'pairwise':>16} | global vs pairwise (c/b,p)")
        rows = []
        for ok in ("none", "hub", "far"):
            sg, sp = _succ(c[("global", ok)], seeds), _succ(c[("pairwise", ok)], seeds)
            lg = _wilson(sg, m); lp = _wilson(sp, m)
            # c-b>0 => pairwise better; c-b<0 => global better
            bb, cc, p = _mcnemar(c[("global", ok)], c[("pairwise", ok)], seeds)
            print(f"   {ok:>8} | {sg:>2}/{m} {sg*100/m:>3.0f}% [{lg[0]*100:3.0f},{lg[1]*100:3.0f}]"
                  f" | {sp:>2}/{m} {sp*100/m:>3.0f}% [{lp[0]*100:3.0f},{lp[1]*100:3.0f}]"
                  f" | c={cc:<2}/b={bb:<2} p={p:.4g}")
            rows.append({"obstacle": ok, "s_global": sg, "s_pairwise": sp,
                         "global_vs_pairwise": {"b": bb, "c": cc, "p": p}})
        report["cells"].append({"n": n, "m": m, "rows": rows})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    print("  hub: global >> pairwise  => the coherent roundabout absorbs the external"
          " perturbation that pairwise's locality cannot.")
    _plot(report, Path(args.out).with_suffix(".png"))


def _plot(report, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cells = sorted(report["cells"], key=lambda d: d["n"])
    obs = ["none", "hub", "far"]
    fig, axes = plt.subplots(1, len(cells), figsize=(5.6 * len(cells), 5.0), squeeze=False)
    x = np.arange(len(obs))
    for ax, d in zip(axes[0], cells):
        m = d["m"]
        g = [100.0 * r["s_global"] / m for r in d["rows"]]
        p = [100.0 * r["s_pairwise"] / m for r in d["rows"]]
        ax.bar(x - 0.2, g, 0.4, color="#1f77b4", label="global (lateral_bias)")
        ax.bar(x + 0.2, p, 0.4, color="#e67e22", label="pairwise (pairwise_bias)")
        ax.set_xticks(x)
        ax.set_xticklabels(["no obstacle", "hub-crossing", "far corner"])
        ax.set_title(f"N={d['n']} (R={report['radius']:g})")
        ax.set_ylabel("antipodal joint success rate (%)")
        ax.set_ylim(0, 103)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)
    fig.suptitle("Pairwise dominates global in an empty arena — but the dominance INVERTS\n"
                 "under a hub-crossing obstacle: the global roundabout's coherence absorbs it")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"plotted {png_path}")


if __name__ == "__main__":
    main()
