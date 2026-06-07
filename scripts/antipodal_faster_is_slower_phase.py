"""Faster-is-slower at the antipodal hub: the doorway optimum-speed effect is
general, and at the hub the high-speed branch is the centripetal-budget limit.

The doorway study (docs/findings.md "Doorway success is an inverted-U in desired
speed") found an optimal cruise speed at a bottleneck: too fast and drones cannot
brake at the gap. Is that specific to a static gap, or general to swarm
convergence? The antipodal hub is the other canonical conflict — a radial
convergence the right-of-way convention turns into a roundabout. Here greed costs
in a *different* currency: the roundabout is a curved lane, so going faster raises
the centripetal demand v^2/r, and once it exceeds the acceleration budget the
drone cannot hold the lane and is flung into the hub. That is exactly the
boundary the heterogeneous-acceleration study hit from the other side (high speed
broke the sluggish drones); here we hit it with a HOMOGENEOUS fleet and read off
the critical speed.

N antipodal swap, MPC + lateral_bias (roundabout on), `max_accel` held fixed at 6
(the budget), sweep `max_speed`. Per speed: joint success, the collision/timeout
split, and makespan of the successes. McNemar pairs the peak-success speed against
each higher speed. Prediction: success collapses above a critical speed
v_crit ~ sqrt(max_accel * r), where r is the roundabout radius, and the failures
are collisions (centripetal limit), not timeouts.

  python scripts/antipodal_faster_is_slower_phase.py --n 6 \
      --speeds 3 5 7 9 11 13 15 --episodes 40
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

CX, CY = 25.0, 25.0
RADIUS = 20.0


def _drones(n):
    out = []
    for k in range(n):
        a = 2.0 * math.pi * k / n
        out.append({"name": f"d{k}",
                    "start": [round(CX + RADIUS * math.cos(a), 3),
                              round(CY + RADIUS * math.sin(a), 3)],
                    "goal": [round(CX - RADIUS * math.cos(a), 3),
                             round(CY - RADIUS * math.sin(a), 3)],
                    "radius": 0.4, "start_jitter": 0.8})
    return out


def _cfg(n, speed, bias, max_accel, seed, n_eps, max_steps):
    return {
        "name": f"hubfis_n{n}_v{speed}", "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50],
                     "resolution": 1.0, "obstacles": {"type": "none"},
                     "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": max_accel, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": {"type": "mpc", "max_speed": speed, "replan_period": 0.2,
                    "horizon": 40, "dt_plan": 0.05, "n_samples": 32,
                    "resolution": 1.0, "inflate": 1, "goal_radius": 1.5,
                    "safety_margin": 0.5, "use_prediction": True,
                    "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
                    "lateral_bias": bias, "predictor": {"type": "constant_velocity"}},
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/hubfis_tmp"},
    }


def _run_cell(job):
    n, speed, bias, max_accel, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(
            _cfg(n, speed, bias, max_accel, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = (d["outcome"], float(d.get("final_t", 0.0)))
    return (speed, by_seed)


def _succ(bs, seeds):
    return sum(bs[s][0] == "success" for s in seeds)


def _brk(bs, seeds):
    return (sum(bs[s][0] == "collision" for s in seeds),
            sum(bs[s][0] == "timeout" for s in seeds))


def _makespan(bs, seeds):
    v = [bs[s][1] for s in seeds if bs[s][0] == "success"]
    return sum(v) / len(v) if v else float("nan")


def _mc(a, b, seeds):
    bb = sum(a[s][0] == "success" and b[s][0] != "success" for s in seeds)
    cc = sum(a[s][0] != "success" and b[s][0] == "success" for s in seeds)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--speeds", type=float, nargs="+", default=[3, 5, 7, 9, 11, 13, 15])
    ap.add_argument("--bias", type=float, default=2.0)
    ap.add_argument("--max-accel", type=float, default=6.0)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--out", default="results/antipodal_faster_is_slower_phase.json")
    args = ap.parse_args()

    jobs = [(args.n, v, args.bias, args.max_accel, args.seed, args.episodes,
             args.max_steps) for v in args.speeds]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)
    cells = dict(res)

    seeds = sorted(set.intersection(*[set(bs) for bs in cells.values()]))
    succ = {v: _succ(cells[v], seeds) for v in args.speeds}
    peak = max(args.speeds, key=lambda v: succ[v])

    report = {"n": args.n, "bias": args.bias, "max_accel": args.max_accel,
              "episodes": args.episodes, "m": len(seeds), "peak_speed": peak,
              "cells": {}, "tests": {}}
    print(f"\nFaster-is-slower @ antipodal hub N={args.n}, bias={args.bias}, "
          f"max_accel={args.max_accel}, paired m={len(seeds)}")
    print(f"{'speed':>6} | {'success':>8} | {'coll/to':>8} | {'makespan':>9} | "
          f"vs peak v={peak:g}")
    print("-" * 70)
    for v in args.speeds:
        bs = cells[v]
        s = succ[v]; co, to = _brk(bs, seeds); mk = _makespan(bs, seeds)
        report["cells"][f"v{v}"] = {"success": s, "collision": co, "timeout": to,
                                    "makespan": mk}
        if v == peak:
            print(f"{v:>6g} | {s:>3}/{len(seeds):<3} | {co:>3}/{to:<3} | "
                  f"{mk:>9.2f} | (peak)")
        else:
            b, c, p = _mc(cells[peak], bs, seeds)
            report["tests"][f"v{v}_vs_peak"] = {"b": b, "c": c, "p": p}
            print(f"{v:>6g} | {s:>3}/{len(seeds):<3} | {co:>3}/{to:<3} | "
                  f"{mk:>9.2f} | b={b} c={c} p={p:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
