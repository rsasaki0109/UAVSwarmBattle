"""Does the right-of-way convention survive an external moving obstacle crossing
the hub — and can a stronger convention pay for it?

The entire convention arc lives on `obstacles: none`: every antipodal-swap result
measures *peer* coordination in an empty arena. This script adds the one thing
that arena never had — a scene **dynamic obstacle** that crosses the central hub
while the fleet converges — and asks three things, all paired by seed:

  1. Is the moving obstacle itself a symmetry-breaker? If *any* asymmetry lifts the
     deadlock, a hub-crossing obstacle should rescue the no-convention fleet.
     (Hypothesis from the arc: NO — the peer convergence is symmetric regardless of
     a third body; the obstacle is just another threat, not a tie-breaker.)
  2. Does the right-of-way convention still help with the obstacle present? (It is a
     *peer* rule; the obstacle is not a peer.)
  3. Does the obstacle degrade the convention, and can a STRONGER bias buy it back?
     (Hypothesis: the bias rescues peer-deadlock but cannot rescue obstacle
     collisions — it addresses the wrong threat, so success caps below the
     obstacle-free ceiling no matter how strong the convention.)

Cells = N x lateral_bias x obstacle{absent,present}, MPC + game_theoretic, paired by
seed. Reports the 2x2 McNemars (rescue with/without obstacle; obstacle effect on each
convention arm) plus, when several biases are swept, the convention's ceiling under
the obstacle.

  python scripts/antipodal_obstacle_convention_phase.py --n-list 6 8 \
      --bias-list 0 1 2 4 --episodes 40
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
# A single body entering from the bottom edge, crossing the hub vertically, and
# reflecting inside the 50x50 box so it remains a sustained threat near the centre
# while the fleet converges (deterministic — variance comes from drone jitter).
OBSTACLE = {"start": [25.0, 2.0], "velocity": [0.0, 4.5], "radius": 1.5, "reflect": True}


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


def _cfg(n, bias, obs, seed, n_eps, max_steps):
    return {
        "name": f"antiobs_n{n}_b{bias:g}_o{int(obs)}",
        "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50], "resolution": 1.0,
                     "obstacles": {"type": "none"},
                     "dynamic_obstacles": [dict(OBSTACLE)] if obs else [],
                     "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": {"type": "mpc", "max_speed": 5.0, "replan_period": 0.2,
                    "horizon": 40, "dt_plan": 0.05, "n_samples": 32,
                    "resolution": 1.0, "inflate": 1, "goal_radius": 1.5,
                    "safety_margin": 0.5, "use_prediction": True,
                    "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
                    "lateral_bias": bias, "predictor": {"type": "game_theoretic"}},
        "sensor": {"type": "perfect"}, "output": {"dir": "results/antiobs_tmp"},
    }


def _run_cell(job):
    n, bias, obs, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, bias, obs, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (n, bias, int(obs), by_seed)


def _succ(bs, seeds):
    return sum(bs[s] == "success" for s in seeds)


def _mcnemar(a, b, seeds):
    bb = sum(a[s] == "success" and b[s] != "success" for s in seeds)
    cc = sum(a[s] != "success" and b[s] == "success" for s in seeds)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[6, 8])
    ap.add_argument("--bias-list", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=6000)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/antipodal_obstacle_convention_phase.json")
    ap.add_argument("--plot-from", default=None)
    args = ap.parse_args()

    if args.plot_from:
        report = json.loads(Path(args.plot_from).read_text())
        _plot(report, Path(args.plot_from).with_suffix(".png"))
        return

    jobs = [(n, b, o, args.seed, args.episodes, args.max_steps)
            for n in args.n_list for b in args.bias_list for o in (False, True)]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)
    cells = {}
    for n, bias, obs, bs in res:
        cells.setdefault(n, {})[(bias, obs)] = bs

    report = {"radius": RADIUS, "episodes": args.episodes, "obstacle": OBSTACLE,
              "bias_list": args.bias_list, "cells": []}
    print(f"\nconvention x hub-crossing obstacle: RADIUS={RADIUS}, n={args.episodes}")
    bias0 = min(args.bias_list)
    for n in sorted(cells):
        c = cells[n]
        seeds = sorted(set.intersection(*[set(bs) for bs in c.values()]))
        m = len(seeds)
        sv = {k: _succ(c[k], seeds) for k in c}
        print(f"\n N={n} (m={m} paired seeds)   success rate [Wilson 95% CI]")
        print(f"   {'bias':>5} | {'no-obstacle':>18} | {'+obstacle':>18} | obstacle effect (c/b,p)")
        rows = []
        for b in args.bias_list:
            s0, s1 = sv[(b, 0)], sv[(b, 1)]
            lo0, hi0 = _wilson(s0, m); lo1, hi1 = _wilson(s1, m)
            # obstacle effect on this bias arm: present vs absent
            bb, cc, p = _mcnemar(c[(b, 0)], c[(b, 1)], seeds)
            print(f"   {b:>5g} | {s0:>2}/{m} {s0*100/m:>3.0f}% [{lo0*100:3.0f},{hi0*100:3.0f}]"
                  f" | {s1:>2}/{m} {s1*100/m:>3.0f}% [{lo1*100:3.0f},{hi1*100:3.0f}]"
                  f" | c={cc:<2}/b={bb:<2} p={p:.4g}")
            rows.append({"bias": b, "s_noobs": s0, "s_obs": s1,
                         "obstacle_effect": {"b": bb, "c": cc, "p": p}})
        # the convention rescue at the strongest bias, with vs without obstacle
        bmax = max(args.bias_list)
        r_no = _mcnemar(c[(bias0, 0)], c[(bmax, 0)], seeds)   # rescue, no obstacle
        r_ob = _mcnemar(c[(bias0, 1)], c[(bmax, 1)], seeds)   # rescue, with obstacle
        print(f"   RESCUE no-obs  stock({bias0:g}) vs row({bmax:g}): c={r_no[1]}/b={r_no[0]} p={r_no[2]:.4g}")
        print(f"   RESCUE +obs    stock({bias0:g}) vs row({bmax:g}): c={r_ob[1]}/b={r_ob[0]} p={r_ob[2]:.4g}")
        report["cells"].append({"n": n, "m": m, "rows": rows,
                                "rescue_noobs": {"b": r_no[0], "c": r_no[1], "p": r_no[2]},
                                "rescue_obs": {"b": r_ob[0], "c": r_ob[1], "p": r_ob[2]}})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    _plot(report, Path(args.out).with_suffix(".png"))


def _plot(report, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    biases = report["bias_list"]
    cells = sorted(report["cells"], key=lambda d: d["n"])
    fig, axes = plt.subplots(1, len(cells), figsize=(5.6 * len(cells), 5.0), squeeze=False)
    for ax, d in zip(axes[0], cells):
        m = d["m"]
        no = [100.0 * r["s_noobs"] / m for r in d["rows"]]
        ob = [100.0 * r["s_obs"] / m for r in d["rows"]]
        ax.plot(biases, no, "o-", color="#2ca02c", lw=2.2, ms=8, label="no obstacle")
        ax.plot(biases, ob, "s--", color="#d62728", lw=2.2, ms=8, label="+ hub-crossing obstacle")
        ax.set_title(f"N={d['n']} (R={report['radius']:g})")
        ax.set_xlabel("right-of-way lateral_bias")
        ax.set_ylabel("antipodal joint success rate (%)")
        ax.set_ylim(-3, 103)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Right-of-way is a PEER rule: a stronger convention rescues the deadlock\n"
                 "but cannot pay for an external moving obstacle (the gap is the wrong-threat cap)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"plotted {png_path}")


if __name__ == "__main__":
    main()
