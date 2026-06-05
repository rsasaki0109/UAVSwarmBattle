"""Does HRVO remove RVO's oscillation while staying in the sampled velocity-obstacle
framework? The constructive test of the lineage's corrected mechanism.

The resolution sweep (rvo_resolution_phase.py) refuted "ORCA cures oscillation by
being continuous": refining RVO's sampling does not help. The corrected claim is
that the cure is a *structural* commitment to one side of the obstacle. HRVO
(Snape et al. 2011) is exactly that structure added to a sampled VO/RVO — it does
NOT use ORCA's half-plane LP and is not "more continuous" than RVO. So if the
corrected claim is right, HRVO should collapse the oscillation toward ORCA's floor
while remaining a sampled selector.

Same self-contained single-integrator crossing as the other lineage studies. Arms
vo / rvo / hrvo / orca; oscillation as a rate (rad/s); paired by seed. Parallel.

  python scripts/hrvo_oscillation_phase.py --n 4 --episodes 40 --workers 6
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
ARMS = ("vo", "rvo", "hrvo", "orca")


def _planner(kind):
    c = {"max_speed": SPEED, "radius": 0.4, "safety_margin": 0.1, "time_horizon": 2.0,
         "neighbor_dist": 15.0, "goal_radius": 1.5, "time_step": DT}
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


def _episode(kind, n, seed, max_steps=500):
    rng = random.Random(seed)
    starts, goals = _layout(n, rng)
    m = 2 * n
    pos = [s.copy() for s in starts]
    vel = [np.zeros(2) for _ in range(m)]
    plan = [_planner(kind) for _ in range(m)]
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
    kind, n, seed = task
    s, r = _episode(kind, n, seed)
    return kind, seed, s, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="results/hrvo_oscillation_phase.json")
    args = ap.parse_args()

    tasks = [(kind, args.n, args.seed + e) for kind in ARMS for e in range(args.episodes)]
    with Pool(args.workers) as pool:
        results = pool.map(_cell, tasks)
    rows = {k: {} for k in ARMS}
    for kind, seed, s, r in results:
        rows[kind][seed] = (s, r)

    seeds = [args.seed + e for e in range(args.episodes)]
    print(f"\nHRVO vs the lineage (2N={2*args.n}, self-contained sim, n={args.episodes})")
    print(f"{'arm':>6} | {'success':>9} | {'osc rate (rad/s)':>17}")
    print("-" * 40)
    summ = {}
    for k in ARMS:
        succ = sum(1 for s, _ in rows[k].values() if s)
        rate = sum(r for _, r in rows[k].values()) / args.episodes
        summ[k] = {"success": succ, "mean_osc_rate": rate}
        print(f"{k:>6} | {succ:>3}/{args.episodes:<3} | {rate:>17.3f}")

    def sign(a, b):  # a higher osc rate than b?
        am = sum(1 for s in seeds if rows[a][s][1] > rows[b][s][1] + 1e-6)
        bm = sum(1 for s in seeds if rows[b][s][1] > rows[a][s][1] + 1e-6)
        return am, bm, mcnemar_exact_p(bm, am)

    def succ_mc(a, b):  # a succeeds where b fails
        a_only = sum(1 for s in seeds if rows[a][s][0] and not rows[b][s][0])
        b_only = sum(1 for s in seeds if rows[b][s][0] and not rows[a][s][0])
        return a_only, b_only, mcnemar_exact_p(b_only, a_only)

    print("\noscillation rate (paired):")
    comps = {}
    for a, b in (("rvo", "hrvo"), ("hrvo", "orca"), ("rvo", "orca")):
        am, bm, p = sign(a, b)
        ratio = summ[a]["mean_osc_rate"] / max(summ[b]["mean_osc_rate"], 1e-6)
        comps[f"osc_{a}_vs_{b}"] = {"a_higher": am, "b_higher": bm, "sign_p": p, "ratio": ratio}
        print(f"  {a} > {b}: {am}/{args.episodes} (vs {bm}); p={p:.2e}; ratio {ratio:.1f}x")
    print("\nsuccess (paired McNemar):")
    for a, b in (("hrvo", "rvo"), ("hrvo", "orca")):
        ao, bo, p = succ_mc(a, b)
        comps[f"succ_{a}_vs_{b}"] = {"a_only": ao, "b_only": bo, "mcnemar_p": p}
        print(f"  {a} vs {b}: {a}-only {ao}, {b}-only {bo}; p={p:.2e}")

    report = {"n": args.n, "episodes": args.episodes, "arms": summ, "comparisons": comps}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
