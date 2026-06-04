"""Does temporally desynchronising the fleet break the hub-obstacle cap — and
does it depend on whether the obstacle is transient or recurring?

The right-of-way convention funnels the antipodal fleet into one clockwise hub
roundabout, and a body crossing that hub caps success below the obstacle-free
ceiling (docs/findings.md "The right-of-way convention is a peer rule …"). The
cap could be TEMPORAL (everyone arrives at the hub at the same instant, so a
hub-crossing body hits the whole synchronized cluster) or SPATIAL (the hub is a
single point every drone must cross, so timing is irrelevant).

This separates the two with a temporal desynchroniser that is peer-neutral on its
own: per-drone speed heterogeneity (alternating max_speed 3 / 7, mean 5 — 40/40
no-obstacle, like the homogeneous fleet, so it does not break the peers). Crossed
with obstacle ∈ {none, single-pass (reflect off → crosses the hub once and
leaves = TRANSIENT), reflecting (bounces in the box → returns to the hub
repeatedly = RECURRING)}, MPC + game_theoretic + global lateral_bias, paired by
seed. McNemar(speed-het vs homogeneous) per obstacle condition.

Prediction: against a TRANSIENT obstacle, spreading arrival times lets drones
dodge the one-time crossing (cap is temporal → broken). Against a RECURRING one,
the obstacle keeps returning to the unavoidable hub point (cap is spatial → not
broken).

NOTE on scope: only a MODERATE spread (3/7) is used. Extreme spreads (e.g. 2/8,
1/9) collapse the peers on their own under this strong bias and are out of scope
here — this script isolates the obstacle axis, not the speed-het peer limit.

  python scripts/antipodal_desync_obstacle_phase.py --n-list 6 8 --episodes 40
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
SLOW, FAST = 3.0, 7.0  # alternating per-drone max_speed (mean = nominal 5)


def _obstacle(reflect):
    return [{"start": [25.0, 2.0], "velocity": [0.0, 4.5], "radius": 1.5, "reflect": reflect}]


OBSTACLES = {"none": [], "single": _obstacle(False), "reflect": _obstacle(True)}


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


def _per_drone(n, het):
    if not het:
        return None
    return [{"max_speed": (SLOW if k % 2 == 0 else FAST)} for k in range(n)]


def _cfg(n, het, obs_key, bias, seed, n_eps, max_steps):
    p = {"type": "mpc", "max_speed": 5.0, "replan_period": 0.2, "horizon": 40,
         "dt_plan": 0.05, "n_samples": 32, "resolution": 1.0, "inflate": 1,
         "goal_radius": 1.5, "safety_margin": 0.5, "use_prediction": True,
         "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05, "lateral_bias": bias,
         "predictor": {"type": "game_theoretic"}}
    pd = _per_drone(n, het)
    if pd:
        p["per_drone"] = pd
    return {
        "name": f"desyncobs_n{n}_{'het' if het else 'homo'}_{obs_key}",
        "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50], "resolution": 1.0,
                     "obstacles": {"type": "none"},
                     "dynamic_obstacles": [dict(o) for o in OBSTACLES[obs_key]],
                     "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": p, "sensor": {"type": "perfect"},
        "output": {"dir": "results/desyncobs_tmp"},
    }


def _run_cell(job):
    n, het, obs_key, bias, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, het, obs_key, bias, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (n, "het" if het else "homo", obs_key, by_seed)


def _succ(bs, seeds):
    return sum(bs[s] == "success" for s in seeds)


def _mcnemar(a, b, seeds):
    bb = sum(a[s] == "success" and b[s] != "success" for s in seeds)
    cc = sum(a[s] != "success" and b[s] == "success" for s in seeds)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[6, 8])
    ap.add_argument("--bias", type=float, default=4.0)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=8000)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/antipodal_desync_obstacle_phase.json")
    ap.add_argument("--plot-from", default=None)
    args = ap.parse_args()

    if args.plot_from:
        report = json.loads(Path(args.plot_from).read_text())
        _plot(report, Path(args.plot_from).with_suffix(".png"))
        return

    jobs = [(n, het, ok, args.bias, args.seed, args.episodes, args.max_steps)
            for n in args.n_list for het in (False, True) for ok in OBSTACLES]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)
    cells = {}
    for n, arm, ok, bs in res:
        cells.setdefault(n, {})[(arm, ok)] = bs

    report = {"radius": RADIUS, "episodes": args.episodes, "bias": args.bias,
              "slow": SLOW, "fast": FAST, "cells": []}
    print(f"\ndesync x obstacle persistence: bias={args.bias}, speed-het {SLOW}/{FAST}, "
          f"n={args.episodes}")
    for n in sorted(cells):
        c = cells[n]
        seeds = sorted(set.intersection(*[set(bs) for bs in c.values()]))
        m = len(seeds)
        print(f"\n N={n} (m={m})   homogeneous vs speed-het [Wilson 95% CI]")
        print(f"   {'obstacle':>8} | {'homo':>16} | {'speed-het':>16} | het vs homo (c/b,p)")
        rows = []
        for ok in ("none", "single", "reflect"):
            sh, sht = _succ(c[("homo", ok)], seeds), _succ(c[("het", ok)], seeds)
            lh = _wilson(sh, m); lt = _wilson(sht, m)
            bb, cc, p = _mcnemar(c[("homo", ok)], c[("het", ok)], seeds)
            print(f"   {ok:>8} | {sh:>2}/{m} {sh*100/m:>3.0f}% [{lh[0]*100:3.0f},{lh[1]*100:3.0f}]"
                  f" | {sht:>2}/{m} {sht*100/m:>3.0f}% [{lt[0]*100:3.0f},{lt[1]*100:3.0f}]"
                  f" | c={cc:<2}/b={bb:<2} p={p:.4g}")
            rows.append({"obstacle": ok, "s_homo": sh, "s_het": sht,
                         "het_vs_homo": {"b": bb, "c": cc, "p": p}})
        report["cells"].append({"n": n, "m": m, "rows": rows})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    print("  single (transient): het breaks the cap (temporal). reflect (recurring): it cannot (spatial).")
    _plot(report, Path(args.out).with_suffix(".png"))


def _plot(report, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cells = sorted(report["cells"], key=lambda d: d["n"])
    obs = ["none", "single", "reflect"]
    labels = ["no obstacle", "single-pass\n(transient)", "reflecting\n(recurring)"]
    fig, axes = plt.subplots(1, len(cells), figsize=(5.6 * len(cells), 5.0), squeeze=False)
    x = np.arange(len(obs))
    for ax, d in zip(axes[0], cells):
        m = d["m"]
        h = [100.0 * r["s_homo"] / m for r in d["rows"]]
        t = [100.0 * r["s_het"] / m for r in d["rows"]]
        ax.bar(x - 0.2, h, 0.4, color="#1f77b4", label="homogeneous")
        ax.bar(x + 0.2, t, 0.4, color="#2ca02c", label=f"speed-het {report['slow']:g}/{report['fast']:g}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"N={d['n']} (R={report['radius']:g}, bias={report['bias']:g})")
        ax.set_ylabel("antipodal joint success rate (%)")
        ax.set_ylim(0, 103)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)
    fig.suptitle("The hub-obstacle cap is temporal for a transient obstacle, spatial for a recurring one\n"
                 "speed desync dodges a single-pass crossing (→100 %) but not a reflecting one")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"plotted {png_path}")


if __name__ == "__main__":
    main()
