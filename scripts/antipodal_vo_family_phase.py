"""Does HRVO's side-commitment break the antipodal-hub deadlock that stock
RVO/ORCA suffer? Bridging the oscillation arc to the symmetry/convention arc.

The antipodal hub (N drones on a ring, each heading to the diametrically opposite
point) is the canonical symmetry trap: every path crosses the centre at once, and
purely reciprocal avoiders deadlock there — stock ORCA collides at the hub for
N>=6, and the whole convention line of work exists to break that symmetry with an
explicit right-of-way rule. HRVO (just added) cures RVO's oscillation by
committing each agent to ONE SIDE of every obstacle. A side-commitment is itself a
local symmetry-breaker — so the sharp question is whether it ALSO dissolves the
antipodal deadlock for free, with no explicit convention.

Self-contained single-integrator antipodal sim (same dynamics as the oscillation
studies), arms rvo / hrvo / orca, swept over N. Small per-seed jitter on the ring
so the symmetric cell is not a single deterministic outcome (avoids the
zero-variance blowout trap). Outcome paired by seed; success = all reach, no
collision; failures split into collision vs timeout.

  python scripts/antipodal_vo_family_phase.py --n-list 4 6 8 --episodes 40 --workers 6
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
ARMS = ("rvo", "hrvo", "orca")


def _planner(kind):
    c = {"max_speed": SPEED, "radius": 0.4, "safety_margin": 0.1, "time_horizon": 2.0,
         "neighbor_dist": 15.0, "goal_radius": 1.5, "time_step": DT}
    p = PLANNER_REGISTRY.get(kind).from_config(c)
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


def _h_ok(v):
    return float(np.hypot(v[0], v[1])) > 0.1


def _episode(kind, n, seed, max_steps=600, replan_period=0.5):
    rng = random.Random(seed)
    starts, goals = _layout(n, rng)
    pos = [s.copy() for s in starts]
    vel = [np.zeros(2) for _ in range(n)]
    plan = [_planner(kind) for _ in range(n)]
    arrived = [False] * n
    collided = False
    # Commitment: agents hold their planned velocity for replan_period seconds.
    # The antipodal hub deadlock is operating-point sensitive -- it appears with
    # commitment (replan_period ~0.5) and is hidden by full per-step reactivity.
    rp_steps = max(1, round(replan_period / DT))
    for step in range(max_steps):
        if step % rp_steps != 0:
            for i in range(n):
                pos[i] = pos[i] + vel[i] * DT
                if not arrived[i] and float(np.linalg.norm(pos[i] - goals[i])) < 1.5:
                    arrived[i] = True
            for i in range(n):
                for j in range(i + 1, n):
                    if float(np.linalg.norm(pos[i] - pos[j])) < COLL:
                        collided = True
            if collided or all(arrived):
                break
            continue
        peers = [{"position": pos[j].copy(), "velocity": vel[j].copy(), "radius": 0.4}
                 for j in range(n)]
        for i in range(n):
            if arrived[i]:
                vel[i] = np.zeros(2); continue
            plan[i].set_current_state(pos[i], vel[i])
            others = [peers[j] for j in range(n) if j != i]
            vel[i] = plan[i].plan(pos[i], goals[i], None, dynamic_obstacles=others).target_velocity
        for i in range(n):
            pos[i] = pos[i] + vel[i] * DT
            if not arrived[i] and float(np.linalg.norm(pos[i] - goals[i])) < 1.5:
                arrived[i] = True
        for i in range(n):
            for j in range(i + 1, n):
                if float(np.linalg.norm(pos[i] - pos[j])) < COLL:
                    collided = True
        if collided or all(arrived):
            break
    if collided:
        return "collision"
    if all(arrived):
        return "success"
    return "timeout"


def _cell(task):
    kind, n, seed = task
    return kind, n, seed, _episode(kind, n, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="results/antipodal_vo_family_phase.json")
    args = ap.parse_args()

    tasks = [(kind, n, args.seed + e) for n in args.n_list for kind in ARMS
             for e in range(args.episodes)]
    with Pool(args.workers) as pool:
        results = pool.map(_cell, tasks)

    # rows[n][kind][seed] = outcome
    rows = {n: {k: {} for k in ARMS} for n in args.n_list}
    for kind, n, seed, out in results:
        rows[n][kind][seed] = out

    seeds = [args.seed + e for e in range(args.episodes)]
    report = {"episodes": args.episodes, "n_list": args.n_list, "cells": {}, "comparisons": {}}
    print(f"\nAntipodal hub: does HRVO's side-commitment break the deadlock? (n={args.episodes})")
    for n in args.n_list:
        print(f"\n--- N={n} (ring={RING}) ---")
        print(f"{'arm':>6} | {'success':>9} | {'collision':>9} | {'timeout':>8}")
        print("-" * 42)
        for k in ARMS:
            out = list(rows[n][k].values())
            s = out.count("success"); c = out.count("collision"); t = out.count("timeout")
            report["cells"][f"N{n}_{k}"] = {"success": s, "collision": c, "timeout": t}
            print(f"{k:>6} | {s:>3}/{args.episodes:<3} | {c:>9} | {t:>8}")

        def succ_mc(a, b):  # a succeeds where b fails
            ao = sum(1 for sd in seeds if rows[n][a][sd] == "success" and rows[n][b][sd] != "success")
            bo = sum(1 for sd in seeds if rows[n][b][sd] == "success" and rows[n][a][sd] != "success")
            return ao, bo, mcnemar_exact_p(bo, ao)

        for a, b in (("hrvo", "rvo"), ("hrvo", "orca")):
            ao, bo, p = succ_mc(a, b)
            report["comparisons"][f"N{n}_{a}_vs_{b}"] = {"a_only": ao, "b_only": bo, "mcnemar_p": p}
            print(f"  {a} vs {b}: {a}-only {ao}, {b}-only {bo}; McNemar p={p:.2e}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
