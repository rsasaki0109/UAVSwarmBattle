"""Does the non-reciprocal APF fail the antipodal swap, and does the right-of-way
convention rescue it?

APF (artificial potential field) is the one reactive baseline here that is NOT
reciprocal: each agent independently descends an attract-to-goal + repel-from-peers
field, with no model of the peer. ORCA / CBF / BVC all assume a cooperating peer
that takes its share; APF assumes nothing, so it is the test of whether the swarm
convention generalises beyond the reciprocal family to a pure gradient controller.

The symmetric antipodal hub is a stationary point of the field (attraction toward
the antipode vs repulsion from the converging peers). With constant-speed steering
(cruise along the gradient, matching the other baselines) the fleet plows into that
point and COLLIDES; the classic variable-speed APF would instead stall there (the
textbook local minimum). Either way the symmetric hub defeats stock APF.

Arms per N, paired by seed, McNemar exact, collision-vs-timeout split:
  apf_stock     no convention
  apf_pairwise  + pairwise right-of-way (in-plane tilt)

  python scripts/antipodal_apf_phase.py --n-list 4 6 8 --episodes 40
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


def _drones(n):
    out = []
    for k in range(n):
        a = 2.0 * math.pi * k / n
        out.append({"name": f"d{k}",
                    "start": [round(CX + RADIUS * math.cos(a), 3), round(CY + RADIUS * math.sin(a), 3)],
                    "goal": [round(CX - RADIUS * math.cos(a), 3), round(CY - RADIUS * math.sin(a), 3)],
                    "radius": 0.4, "start_jitter": 0.8})
    return out


def _cfg(n, pw, seed, n_eps, max_steps):
    planner = {"type": "apf", "max_speed": SPEED, "replan_period": 0.05,
               "radius": 0.4, "k_att": 1.0, "k_rep": 6.0, "influence_dist": 4.0,
               "time_step": 0.05, "goal_radius": 1.5}
    if pw > 0.0:
        planner["pairwise_bias"] = pw
        planner["pairwise_radius"] = 8.0
    return {"name": f"apf_n{n}_pw{pw}", "seed": seed, "num_episodes": n_eps,
            "scenario": {"type": "multi_drone_grid", "size": [50, 50],
                         "resolution": 1.0, "obstacles": {"type": "none"},
                         "drones": _drones(n)},
            "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                          "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
            "planner": planner, "sensor": {"type": "perfect"},
            "output": {"dir": "results/apf_tmp"}}


def _run_cell(job):
    label, n, pw, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, pw, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (label, n, by_seed)


def _succ(bs, seeds):
    return sum(bs[s] == "success" for s in seeds)


def _brk(bs, seeds):
    return (sum(bs[s] == "collision" for s in seeds),
            sum(bs[s] == "timeout" for s in seeds))


def _mc(a, b, seeds):
    bb = sum(a[s] == "success" and b[s] != "success" for s in seeds)
    cc = sum(a[s] != "success" and b[s] == "success" for s in seeds)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--pairwise-bias", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--out", default="results/antipodal_apf_phase.json")
    args = ap.parse_args()

    jobs = []
    for n in args.n_list:
        jobs.append(("apf_stock", n, 0.0, args.seed, args.episodes, args.max_steps))
        jobs.append(("apf_pairwise", n, args.pairwise_bias, args.seed, args.episodes, args.max_steps))
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    cells = {}
    for label, n, bs in res:
        cells.setdefault(n, {})[label] = bs

    report = {"speed": SPEED, "radius": RADIUS, "pairwise_bias": args.pairwise_bias,
              "episodes": args.episodes, "cells": []}
    print(f"\nAPF antipodal: does the non-reciprocal hub failure yield to the "
          f"convention? (pw={args.pairwise_bias}, n={args.episodes}, paired)")
    print(f"{'N':>2} | {'apf_stock':>14} | {'apf_pairwise':>14} | {'pw vs stock (b/c,p)':>22}")
    print("-" * 64)
    for n in sorted(cells):
        c = cells[n]
        st, pw = c["apf_stock"], c["apf_pairwise"]
        seeds = sorted(set(st) & set(pw))
        m = len(seeds)
        cs, ts = _brk(st, seeds)
        cp, tp = _brk(pw, seeds)
        b, cc, p = _mc(st, pw, seeds)
        print(f"{n:>2} | {_succ(st,seeds):>2}/{m:<2}[{cs:>2}c/{ts:>2}t] | "
              f"{_succ(pw,seeds):>2}/{m:<2}[{cp:>2}c/{tp:>2}t] | "
              f"b={b:>2} c={cc:>2} p={p:>8.5f}")
        report["cells"].append({"n": n, "m": m,
            "apf_stock": _succ(st, seeds), "apf_stock_ct": [cs, ts],
            "apf_pairwise": _succ(pw, seeds), "apf_pairwise_ct": [cp, tp],
            "pw_vs_stock": {"b": b, "c": cc, "p": p}})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
