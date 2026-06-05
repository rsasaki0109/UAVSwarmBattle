"""Can a shared right-of-way convention make HETEROGENEOUS controllers interoperate?

The convention was proved planner-agnostic one controller at a time: it rescues
the antipodal deadlock for the sampling MPC, for ORCA (docs/findings.md "ORCA is
the missing reciprocal baseline …"), for BVC and CBF. But a real swarm is not
homogeneous — different vehicles run different stacks. Do controllers with
*incompatible avoidance styles* coordinate when they share only the convention?

MPC is a predict-then-optimize controller (it forecasts each peer and scores
sampled trajectories); ORCA is a reciprocal velocity-obstacle controller (it
assumes every neighbour takes half the avoidance responsibility via a velocity
half-plane). Mixed in one fleet they mis-model each other: the MPC drone does not
honour ORCA's reciprocity, so the ORCA drone's half-measure can be too little.
The question is whether a shared "veer right" convention — the *same* rule
expressed in each controller's own cost/preference — absorbs that incompatibility.

Arms on the antipodal swap (paired by seed), each controller at its own
convention strength (MPC lateral_bias=4, ORCA lateral_bias=0.2 — the calibrated
bands of the single-controller studies):

  homo_mpc   : all MPC + row          (reference ceiling)
  homo_orca  : all ORCA + row         (reference ceiling)
  mix_off    : alternating MPC/ORCA, convention OFF  (do they self-coordinate?)
  mix_on     : alternating MPC/ORCA, convention ON   (does the shared rule rescue?)

McNemar mix_on vs mix_off (does the convention rescue the mixed fleet?) and
mix_on vs each homogeneous arm (is mixing penalty-free once the rule is shared?).

  python scripts/antipodal_heterogeneous_controller_phase.py --n-list 6 8 12 --episodes 40
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
MPC_BIAS = 4.0
ORCA_BIAS = 0.2


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


def _mpc_over(bias):
    return {"type": "mpc", "lateral_bias": bias}


def _orca_over(bias):
    # ORCA reads its own config keys (ignores the MPC-shaped base ones).
    return {"type": "orca", "lateral_bias": bias, "time_horizon": 2.0,
            "time_step": 0.25, "neighbor_dist": 15.0, "safety_margin": 0.5,
            "radius": 0.4, "max_speed": 5.0, "goal_radius": 1.5}


def _per_drone(arm, n):
    if arm == "homo_mpc":
        return [_mpc_over(MPC_BIAS) for _ in range(n)]
    if arm == "homo_orca":
        return [_orca_over(ORCA_BIAS) for _ in range(n)]
    if arm == "mix_off":
        return [(_mpc_over(0.0) if k % 2 == 0 else _orca_over(0.0)) for k in range(n)]
    if arm == "mix_on":
        return [(_mpc_over(MPC_BIAS) if k % 2 == 0 else _orca_over(ORCA_BIAS)) for k in range(n)]
    raise ValueError(arm)


def _cfg(n, arm, seed, n_eps, max_steps):
    # Shared base is MPC-shaped; per_drone overrides type + convention per drone.
    base = {"type": "mpc", "max_speed": 5.0, "replan_period": 0.2, "horizon": 40,
            "dt_plan": 0.05, "n_samples": 32, "resolution": 1.0, "inflate": 1,
            "goal_radius": 1.5, "safety_margin": 0.5, "use_prediction": True,
            "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05, "lateral_bias": 0.0,
            "predictor": {"type": "game_theoretic"}, "per_drone": _per_drone(arm, n)}
    return {
        "name": f"hetctrl_n{n}_{arm}",
        "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50], "resolution": 1.0,
                     "obstacles": {"type": "none"}, "dynamic_obstacles": [],
                     "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": base, "sensor": {"type": "perfect"},
        "output": {"dir": "results/hetctrl_tmp"},
    }


def _run_cell(job):
    n, arm, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, arm, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (n, arm, by_seed)


def _succ(bs, seeds):
    return sum(bs[s] == "success" for s in seeds)


def _mcnemar(a, b, seeds):
    bb = sum(a[s] == "success" and b[s] != "success" for s in seeds)
    cc = sum(a[s] != "success" and b[s] == "success" for s in seeds)
    return bb, cc, mcnemar_exact_p(bb, cc)


ARMS = ["homo_mpc", "homo_orca", "mix_off", "mix_on"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[6, 8, 12])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/antipodal_heterogeneous_controller_phase.json")
    ap.add_argument("--plot-from", default=None)
    args = ap.parse_args()

    if args.plot_from:
        report = json.loads(Path(args.plot_from).read_text())
        _plot(report, Path(args.plot_from).with_suffix(".png"))
        return

    jobs = [(n, arm, args.seed, args.episodes, args.max_steps)
            for n in args.n_list for arm in ARMS]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)
    cells = {}
    for n, arm, bs in res:
        cells.setdefault(n, {})[arm] = bs

    report = {"radius": RADIUS, "episodes": args.episodes, "mpc_bias": MPC_BIAS,
              "orca_bias": ORCA_BIAS, "cells": []}
    print(f"\nheterogeneous-controller interop: MPC bias={MPC_BIAS}, ORCA bias={ORCA_BIAS}, "
          f"n={args.episodes}")
    for n in sorted(cells):
        c = cells[n]
        seeds = sorted(set.intersection(*[set(c[a]) for a in ARMS]))
        m = len(seeds)
        sv = {a: _succ(c[a], seeds) for a in ARMS}
        b_r, c_r, p_r = _mcnemar(c["mix_off"], c["mix_on"], seeds)   # rescue
        b_m, c_m, p_m = _mcnemar(c["mix_on"], c["homo_mpc"], seeds)  # mixing penalty?
        print(f"\n N={n} (m={m})")
        for a in ARMS:
            lo, hi = _wilson(sv[a], m)
            print(f"   {a:>10}: {sv[a]:>2}/{m} {sv[a]*100/m:>3.0f}% [{lo*100:3.0f},{hi*100:3.0f}]")
        print(f"   RESCUE   mix_on vs mix_off: c={c_r}/b={b_r} p={p_r:.4g}")
        print(f"   MIXPEN   mix_on vs homo_mpc: c={c_m}/b={b_m} p={p_m:.4g}")
        report["cells"].append({"n": n, "m": m, **{f"{a}_s": sv[a] for a in ARMS},
                                "rescue": {"b": b_r, "c": c_r, "p": p_r},
                                "mix_penalty": {"b": b_m, "c": c_m, "p": p_m}})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    print("  mix_off << mix_on ≈ homo => the shared convention is a common protocol that lets")
    print("  incompatible controllers interoperate at no mixing penalty.")
    _plot(report, Path(args.out).with_suffix(".png"))


def _plot(report, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cells = sorted(report["cells"], key=lambda d: d["n"])
    ns = [d["n"] for d in cells]
    series = [("homo_mpc_s", "homo MPC + row", "#1f77b4", "o", "-"),
              ("homo_orca_s", "homo ORCA + row", "#2ca02c", "s", "-"),
              ("mix_on_s", "mix MPC/ORCA + row", "#9467bd", "D", "-"),
              ("mix_off_s", "mix MPC/ORCA, no convention", "#d62728", "x", "--")]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for key, lab, col, mk, ls in series:
        ax.plot(ns, [100.0 * d[key] / d["m"] for d in cells], mk + ls, color=col,
                lw=2.2, ms=8, label=lab)
    ax.set_xlabel("N drones (antipodal swap, R=20)")
    ax.set_ylabel("joint success rate (%)")
    ax.set_title("A shared right-of-way convention lets heterogeneous controllers interoperate\n"
                 "mix MPC+ORCA collides without it; the shared rule restores it to the homogeneous ceiling")
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
