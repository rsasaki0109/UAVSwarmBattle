"""Do a local side-commitment (HRVO) and a global right-of-way convention COMPOSE,
or does the convention dominate once it is on?

HRVO's side-commitment is a LOCAL, pairwise symmetry-breaker that partially breaks
the antipodal deadlock but decays with density (it tops out as the crowd grows).
The right-of-way convention is a GLOBAL rotation rule that scales further. This
wires the convention into HRVO (`pairwise_bias`) and asks, on the antipodal hub
across density:

  - does the convention rescue HRVO the way it rescues ORCA?  (hrvo+row vs hrvo)
  - does HRVO's local commitment ADD anything once the global rule is on, i.e.
    push the density cliff further out than the convention alone? (hrvo+row vs orca+row)

Self-contained single-integrator antipodal sim at the commitment operating point
(replan_period=0.5, where the hub deadlock appears), per-seed ring jitter, paired
by seed; arms hrvo / hrvo+row / orca / orca+row.

  python scripts/antipodal_hrvo_convention_phase.py --n-list 8 12 16 --episodes 40 --workers 6
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from uav_nav_lab.planner import PLANNER_REGISTRY
from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p

SPEED = 5.0
DT = 0.05
RING = 15.0
CX, CY = 25.0, 25.0
COLL = 0.8
BIAS = 0.5
PR = 6.0
ARMS = ("hrvo", "hrvo+row", "orca", "orca+row")


def _planner(arm):
    c = {"max_speed": SPEED, "radius": 0.4, "safety_margin": 0.1, "time_horizon": 2.0,
         "neighbor_dist": 15.0, "goal_radius": 1.5, "time_step": DT}
    base = arm
    if arm.endswith("+row"):
        base = arm[:-4]
        c["pairwise_bias"] = BIAS
        c["pairwise_radius"] = PR
    p = PLANNER_REGISTRY.get(base).from_config(c)
    p.reset()
    return p


def _layout(n, rng):
    starts, goals = [], []
    for i in range(n):
        a = 2.0 * math.pi * i / n + rng.uniform(-0.05, 0.05)
        r = RING + rng.uniform(-0.4, 0.4)
        starts.append(np.array([CX + r * math.cos(a), CY + r * math.sin(a)]))
        goals.append(np.array([CX - r * math.cos(a), CY - r * math.sin(a)]))
    return starts, goals


def _episode(arm, n, seed, max_steps=600, replan_period=0.5):
    rng = random.Random(seed)
    starts, goals = _layout(n, rng)
    pos = [s.copy() for s in starts]
    vel = [np.zeros(2) for _ in range(n)]
    plan = [_planner(arm) for _ in range(n)]
    arrived = [False] * n
    collided = False
    rp = max(1, round(replan_period / DT))
    step = 0
    for step in range(max_steps):
        if step % rp == 0:
            peers = [{"position": pos[j].copy(), "velocity": vel[j].copy(), "radius": 0.4}
                     for j in range(n)]
            for i in range(n):
                if arrived[i]:
                    vel[i] = np.zeros(2); continue
                plan[i].set_current_state(pos[i], vel[i])
                others = [peers[j] for j in range(n) if j != i]
                vel[i] = plan[i].plan(pos[i], goals[i], None, dynamic_obstacles=others).target_velocity
        for i in range(n):
            if not arrived[i]:
                pos[i] = pos[i] + vel[i] * DT
                if float(np.linalg.norm(pos[i] - goals[i])) < 1.5:
                    arrived[i] = True
        for i in range(n):
            for j in range(i + 1, n):
                if float(np.linalg.norm(pos[i] - pos[j])) < COLL:
                    collided = True
        if collided or all(arrived):
            break
    if collided:
        return "collision"
    return "success" if all(arrived) else "timeout"


def _cell(task):
    arm, n, seed = task
    return arm, n, seed, _episode(arm, n, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="results/antipodal_hrvo_convention_phase.json")
    args = ap.parse_args()

    tasks = [(arm, n, args.seed + e) for n in args.n_list for arm in ARMS
             for e in range(args.episodes)]
    with Pool(args.workers) as pool:
        results = pool.map(_cell, tasks)
    rows = {n: {a: {} for a in ARMS} for n in args.n_list}
    for arm, n, seed, out in results:
        rows[n][arm][seed] = out

    seeds = [args.seed + e for e in range(args.episodes)]
    report = {"episodes": args.episodes, "n_list": args.n_list, "bias": BIAS, "cells": {}, "comparisons": {}}
    print(f"\nHRVO local commitment x global convention on the antipodal hub (bias={BIAS}, n={args.episodes})")
    for n in args.n_list:
        print(f"\n--- N={n} ---")
        print(f"{'arm':>9} | {'success':>9} | {'coll':>5} | {'t/o':>5}")
        print("-" * 36)
        for a in ARMS:
            out = list(rows[n][a].values())
            s = out.count("success"); c = out.count("collision"); t = out.count("timeout")
            report["cells"][f"N{n}_{a}"] = {"success": s, "collision": c, "timeout": t}
            print(f"{a:>9} | {s:>3}/{args.episodes:<3} | {c:>5} | {t:>5}")

        def mc(a, b):
            ao = sum(1 for sd in seeds if rows[n][a][sd] == "success" and rows[n][b][sd] != "success")
            bo = sum(1 for sd in seeds if rows[n][b][sd] == "success" and rows[n][a][sd] != "success")
            return ao, bo, mcnemar_exact_p(bo, ao)

        for a, b in (("hrvo+row", "hrvo"), ("orca+row", "orca"), ("hrvo+row", "orca+row")):
            ao, bo, p = mc(a, b)
            report["comparisons"][f"N{n}_{a}_vs_{b}"] = {"a_only": ao, "b_only": bo, "mcnemar_p": p}
            print(f"  {a} vs {b}: {a}+{ao} / {b}+{bo}; McNemar p={p:.2e}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
