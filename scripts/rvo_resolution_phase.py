"""Is RVO's oscillation a sampling artifact? Sweep RVO's velocity-sampling
resolution and watch whether the oscillation converges to ORCA's continuous floor.

The lineage study (vo_rvo_orca_ladder_phase.py) found that RVO's oscillation
collapses only at the RVO->ORCA rung and attributed it to a MECHANISM: VO and RVO
share a *sampled* velocity selection, so their jitter is dominated by the
discretization, and ORCA removes it by being *continuous* (a half-plane LP). That
claim makes a sharp, falsifiable prediction this script tests directly:

  if oscillation is a discretization artifact, then refining RVO's sampling grid
  (more candidate angles) should drive its oscillation rate DOWN toward ORCA's
  continuous floor.

If instead RVO stays jittery at high resolution, the mechanism claim is wrong --
the cure would be the half-plane STRUCTURE (side commitment), not mere continuity.

Same self-contained single-integrator crossing as the ladder study. Arms are RVO
at a range of n_angles, plus ORCA as the continuous reference. Oscillation as a
rate (rad/s). Paired by seed. Parallel across (arm, seed) cells.

  python scripts/rvo_resolution_phase.py --angles 6 12 24 48 96 --episodes 30
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
GAP = 2.6
LO, HI = 6.0, 44.0
COLL = 0.8


def _planner(kind, n_angles=24):
    c = {"max_speed": SPEED, "radius": 0.4, "safety_margin": 0.1, "time_horizon": 2.0,
         "neighbor_dist": 15.0, "goal_radius": 1.5, "time_step": DT,
         "n_speeds": 4, "n_angles": n_angles}
    p = PLANNER_REGISTRY.get(kind).from_config(c)
    p.reset()
    return p


def _layout(n, rng):
    span = (n - 1) * GAP
    base = 25.0 - span / 2.0
    starts, goals = [], []
    for i in range(n):
        y = base + i * GAP + rng.uniform(-0.4, 0.4)
        starts.append(np.array([LO, y])); goals.append(np.array([HI, y]))
    for i in range(n):
        x = base + i * GAP + rng.uniform(-0.4, 0.4)
        starts.append(np.array([x, LO])); goals.append(np.array([x, HI]))
    return starts, goals


def _h_ok(v):
    return float(np.hypot(v[0], v[1])) > 0.1


def _episode(kind, n_angles, n, seed, max_steps=500):
    rng = random.Random(seed)
    starts, goals = _layout(n, rng)
    m = 2 * n
    pos = [s.copy() for s in starts]
    vel = [np.zeros(2) for _ in range(m)]
    plan = [_planner(kind, n_angles) for _ in range(m)]
    osc = [0.0] * m
    prev_h = [None] * m
    arrived = [False] * m
    collided = False
    step = 0
    for step in range(max_steps):
        peers = [{"position": pos[j].copy(), "velocity": vel[j].copy(), "radius": 0.4}
                 for j in range(m)]
        for i in range(m):
            if arrived[i]:
                vel[i] = np.zeros(2); continue
            plan[i].set_current_state(pos[i], vel[i])
            others = [peers[j] for j in range(m) if j != i]
            vel[i] = plan[i].plan(pos[i], goals[i], None, dynamic_obstacles=others).target_velocity
            if _h_ok(vel[i]):
                h = math.atan2(vel[i][1], vel[i][0])
                if prev_h[i] is not None:
                    osc[i] += abs((h - prev_h[i] + math.pi) % (2 * math.pi) - math.pi)
                prev_h[i] = h
        for i in range(m):
            pos[i] = pos[i] + vel[i] * DT
            if not arrived[i] and float(np.linalg.norm(pos[i] - goals[i])) < 1.5:
                arrived[i] = True
        for i in range(m):
            for j in range(i + 1, m):
                if float(np.linalg.norm(pos[i] - pos[j])) < COLL:
                    collided = True
        if collided or all(arrived):
            break
    success = (not collided) and all(arrived)
    elapsed = (step + 1) * DT
    osc_rate = (sum(osc) / m) / max(elapsed, 1e-6)
    return success, osc_rate


def _cell(task):
    key, kind, na, n, seed = task
    s, r = _episode(kind, na, n, seed)
    return key, seed, s, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--angles", type=int, nargs="+", default=[6, 12, 24, 48, 96])
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="results/rvo_resolution_phase.json")
    args = ap.parse_args()

    arms = [(f"rvo@{a}", "rvo", a) for a in args.angles] + [("orca", "orca", 24)]
    tasks = [(key, kind, na, args.n, args.seed + e)
             for (key, kind, na) in arms for e in range(args.episodes)]

    with Pool(args.workers) as pool:
        results = pool.map(_cell, tasks)

    rows = {key: {} for key, _, _ in arms}
    for key, seed, s, r in results:
        rows[key][seed] = (s, r)

    print(f"\nRVO sampling-resolution sweep vs ORCA floor (2N={2*args.n}, n={args.episodes})")
    print(f"{'arm':>10} | {'success':>9} | {'osc rate (rad/s)':>17}")
    print("-" * 44)
    summ = {}
    for key, _, na in arms:
        vals = list(rows[key].values())
        succ = sum(1 for s, _ in vals if s)
        rate = sum(r for _, r in vals) / len(vals)
        summ[key] = {"success": succ, "mean_osc_rate": rate, "n_angles": na if key != "orca" else None}
        print(f"{key:>10} | {succ:>3}/{args.episodes:<3} | {rate:>17.3f}")

    seeds = [args.seed + e for e in range(args.episodes)]
    coarse = f"rvo@{args.angles[0]}"
    fine = f"rvo@{args.angles[-1]}"

    def sign(a, b):  # a higher osc rate than b?
        am = sum(1 for s in seeds if rows[a][s][1] > rows[b][s][1] + 1e-6)
        bm = sum(1 for s in seeds if rows[b][s][1] > rows[a][s][1] + 1e-6)
        return am, bm, mcnemar_exact_p(bm, am)

    cf_a, cf_b, cf_p = sign(coarse, fine)
    fo_a, fo_b, fo_p = sign(fine, "orca")

    # paired SUCCESS: does refining the grid cost safety? (coarse succeeds where fine fails?)
    succ_cf_b = sum(1 for s in seeds if rows[coarse][s][0] and not rows[fine][s][0])  # coarse wins
    succ_cf_c = sum(1 for s in seeds if rows[fine][s][0] and not rows[coarse][s][0])  # fine wins
    succ_p = mcnemar_exact_p(succ_cf_b, succ_cf_c)

    print()
    print(f"refine helps osc? {coarse} > {fine} on {cf_a}/{args.episodes} seeds (vs {cf_b}); "
          f"p={cf_p:.2e}; ratio {summ[coarse]['mean_osc_rate']/max(summ[fine]['mean_osc_rate'],1e-6):.1f}x")
    print(f"reaches floor?    {fine} > orca on {fo_a}/{args.episodes} seeds (vs {fo_b}); "
          f"p={fo_p:.2e}; ratio {summ[fine]['mean_osc_rate']/max(summ['orca']['mean_osc_rate'],1e-6):.1f}x")
    print(f"refine costs safety? {coarse} succeeds where {fine} fails on {succ_cf_b} seeds "
          f"(reverse {succ_cf_c}); McNemar p={succ_p:.2e}")

    report = {"n": args.n, "episodes": args.episodes, "arms": summ,
              "coarse_vs_fine_osc": {"a": coarse, "b": fine, "a_higher": cf_a, "b_higher": cf_b, "sign_p": cf_p},
              "fine_vs_orca_osc": {"a": fine, "a_higher": fo_a, "b_higher": fo_b, "sign_p": fo_p},
              "coarse_vs_fine_success": {"coarse_only": succ_cf_b, "fine_only": succ_cf_c, "mcnemar_p": succ_p}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
