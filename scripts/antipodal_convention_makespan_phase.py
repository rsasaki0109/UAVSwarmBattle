"""The price of the convention: how much makespan does the roundabout cost, and
is the adaptive pairwise rule cheaper than the rigid global one?

The whole convention arc measured only binary success. But the right-of-way works
by turning a head-on convergence into a roundabout, which is a detour — so the
fleets it rescues pay in completion TIME. This measures the makespan (joint
`final_t`, the moment the last drone reaches goal) of the convention-rescued
antipodal fleet against the free-flight ideal (a drone crossing the diameter
2R at cruise speed, ~8.0 s here), and compares the two conventions on the cells
where both succeed.

MPC on the antipodal swap, N sweep, arms global (lateral_bias) and pairwise
(pairwise_bias). For each: success rate, mean makespan over successful episodes,
overhead vs the ideal; and a per-seed paired makespan comparison (sign test via
McNemar) on seeds where BOTH succeed.

  python scripts/antipodal_convention_makespan_phase.py --n-list 2 3 4 5 6 --episodes 40
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

SPEED = 5.0
CX, CY = 25.0, 25.0
RADIUS = 20.0
IDEAL = 2 * RADIUS / SPEED  # free-flight makespan crossing the diameter (~8.0 s)


def _planner(arm, gb, pb):
    p = {"type": "mpc", "max_speed": SPEED, "replan_period": 0.2, "horizon": 40,
         "dt_plan": 0.05, "n_samples": 32, "resolution": 1.0, "inflate": 1,
         "goal_radius": 1.5, "safety_margin": 0.5, "use_prediction": True,
         "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
         "predictor": {"type": "constant_velocity"}}
    if arm == "global":
        p["lateral_bias"] = gb
    elif arm == "pairwise":
        p["pairwise_bias"] = pb
        p["pairwise_radius"] = 8.0
    return p


def _drones(n):
    out = []
    for k in range(n):
        a = 2.0 * math.pi * k / n
        out.append({"name": f"d{k}",
                    "start": [round(CX + RADIUS * math.cos(a), 3), round(CY + RADIUS * math.sin(a), 3)],
                    "goal": [round(CX - RADIUS * math.cos(a), 3), round(CY - RADIUS * math.sin(a), 3)],
                    "radius": 0.4, "start_jitter": 0.8})
    return out


def _cfg(n, arm, gb, pb, seed, n_eps, max_steps):
    return {
        "name": f"mksp_n{n}_{arm}", "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50],
                     "resolution": 1.0, "obstacles": {"type": "none"}, "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": _planner(arm, gb, pb),
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/mksp_tmp"},
    }


def _run_cell(job):
    arm, n, gb, pb, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, arm, gb, pb, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = (d["outcome"], float(d.get("final_t", 0.0)))
    return (arm, n, by_seed)


def _succ_seeds(bs):
    return {s for s, (o, _) in bs.items() if o == "success"}


def _mean_makespan(bs, seeds):
    vals = [bs[s][1] for s in seeds]
    return sum(vals) / len(vals) if vals else float("nan")


ARMS = ["global", "pairwise"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--global-bias", type=float, default=2.0)
    ap.add_argument("--pairwise-bias", type=float, default=10.0)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--out", default="results/antipodal_convention_makespan_phase.json")
    args = ap.parse_args()

    jobs = [(arm, n, args.global_bias, args.pairwise_bias, args.seed, args.episodes, args.max_steps)
            for n in args.n_list for arm in ARMS]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    cells = {}
    for arm, n, bs in res:
        cells.setdefault(n, {})[arm] = bs

    report = {"ideal_makespan": IDEAL, "global_bias": args.global_bias,
              "pairwise_bias": args.pairwise_bias, "episodes": args.episodes, "cells": []}
    print(f"\nPrice of the convention — antipodal makespan vs free-flight ideal "
          f"({IDEAL:.1f}s), MPC, n={args.episodes}")
    print(f"{'N':>2} | {'global succ/mksp/ovh':>26} | {'pairwise succ/mksp/ovh':>26} | "
          f"{'pw faster? (g/p, p)':>20}")
    print("-" * 86)
    for n in sorted(cells):
        c = cells[n]
        g, p = c["global"], c["pairwise"]
        gs, ps = _succ_seeds(g), _succ_seeds(p)
        both = sorted(gs & ps)
        gmk = _mean_makespan(g, gs)
        pmk = _mean_makespan(p, ps)
        # paired sign test on `both`: who is faster per seed
        g_faster = sum(1 for s in both if g[s][1] < p[s][1] - 1e-6)
        p_faster = sum(1 for s in both if p[s][1] < g[s][1] - 1e-6)
        pv = mcnemar_exact_p(g_faster, p_faster)
        print(f"{n:>2} | {len(gs):>2}/40 {gmk:>6.2f}s +{gmk-IDEAL:>5.2f} | "
              f"{len(ps):>2}/40 {pmk:>6.2f}s +{pmk-IDEAL:>5.2f} | "
              f"g={g_faster:>2} p={p_faster:>2} pv={pv:>6.4f}")
        report["cells"].append({"n": n,
            "global_succ": len(gs), "global_mean_makespan": gmk,
            "pairwise_succ": len(ps), "pairwise_mean_makespan": pmk,
            "n_both": len(both), "global_faster": g_faster, "pairwise_faster": p_faster,
            "sign_p": pv})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
