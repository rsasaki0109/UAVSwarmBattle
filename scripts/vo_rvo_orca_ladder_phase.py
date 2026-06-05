"""The reciprocal-avoidance lineage as an oscillation ladder:
VO (1998, non-reciprocal) -> RVO (2008, reciprocal) -> ORCA (2011, half-plane LP).

Companion to scripts/rvo_orca_oscillation_phase.py, which proved ORCA removes
RVO's oscillation. This adds the FOUNDATIONAL ancestor, VO (Fiorini & Shiller
1998), to test the lineage hypothesis: VO is NON-reciprocal — each agent avoids
as if it alone is responsible — so two VO agents over-react to the same encounter
and should oscillate *more* than RVO, which splits the avoidance in half. The
expected ordering is osc(VO) > osc(RVO) >> osc(ORCA).

Same self-contained single-integrator sim (the model all three assume) so the
full velocity trajectory is observable; oscillation = total absolute heading
variation of the velocity (rad). Perpendicular crossing of 2N drones, jittered
per seed, paired by seed; arms vo / rvo / orca.

  python scripts/vo_rvo_orca_ladder_phase.py --n 4 --episodes 40
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

from uav_nav_lab.planner import PLANNER_REGISTRY
from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p

SPEED = 5.0
DT = 0.05
GAP = 2.6
LO, HI = 6.0, 44.0
COLL = 0.8
ARMS = ("vo", "rvo", "orca")


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
    arrive_t = [None] * m
    collided = False
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
                arrived[i] = True; arrive_t[i] = (step + 1) * DT
        for i in range(m):
            for j in range(i + 1, m):
                if float(np.linalg.norm(pos[i] - pos[j])) < COLL:
                    collided = True
        if collided or all(arrived):
            break
    success = (not collided) and all(arrived)
    mean_osc = sum(osc) / m
    elapsed = (step + 1) * DT
    osc_rate = mean_osc / max(elapsed, 1e-6)  # rad/s: removes the early-collision-termination confound
    makespan = max([t for t in arrive_t if t is not None], default=float("nan")) if all(arrived) else float("nan")
    return success, mean_osc, makespan, osc_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--out", default="results/vo_rvo_orca_ladder_phase.json")
    args = ap.parse_args()

    rows = {k: [] for k in ARMS}
    for kind in ARMS:
        for e in range(args.episodes):
            rows[kind].append(_episode(kind, args.n, args.seed + e))

    def summ(k):
        succ = sum(1 for s, _, _, _ in rows[k] if s)
        oscs = [o for _, o, _, _ in rows[k]]
        rates = [r for _, _, _, r in rows[k]]
        mks = [t for _, _, t, _ in rows[k] if not math.isnan(t)]
        return (succ, sum(oscs) / len(oscs), sum(rates) / len(rates),
                (sum(mks) / len(mks) if mks else float("nan")))

    print(f"\nVO -> RVO -> ORCA oscillation ladder (2N={2*args.n}, self-contained sim, n={args.episodes})")
    print(f"{'arm':>5} | {'success':>9} | {'osc (rad)':>11} | {'osc rate (rad/s)':>17} | {'makespan':>10}")
    print("-" * 64)
    stats = {}
    for k in ARMS:
        s, o, r, mk = summ(k)
        stats[k] = {"success": s, "mean_osc": o, "mean_osc_rate": r, "mean_makespan": mk}
        print(f"{k:>5} | {s:>3}/{args.episodes:<3} | {o:>11.2f} | {r:>17.3f} | {mk:>8.2f}s")

    # paired comparisons along the ladder, on osc RATE (rad/s) to remove the
    # early-collision-termination confound (collided runs end sooner -> less
    # accumulated total oscillation regardless of how jittery they were).
    def sign_p(a, b, idx):  # a higher than b on field idx?
        a_more = sum(1 for i in range(args.episodes) if rows[a][i][idx] > rows[b][i][idx] + 1e-6)
        b_more = sum(1 for i in range(args.episodes) if rows[b][i][idx] > rows[a][i][idx] + 1e-6)
        return a_more, b_more, mcnemar_exact_p(b_more, a_more)

    print("\nper-seed paired oscillation RATE (rad/s):")
    comps = {}
    for a, b in (("vo", "rvo"), ("rvo", "orca"), ("vo", "orca")):
        am, bm, p = sign_p(a, b, 3)
        ratio = stats[a]["mean_osc_rate"] / max(stats[b]["mean_osc_rate"], 1e-6)
        comps[f"{a}_vs_{b}"] = {"a_higher": am, "b_higher": bm, "sign_p": p, "rate_ratio": ratio}
        print(f"  rate {a} > {b}: {am}/{args.episodes} seeds (vs {bm}); sign p={p:.2e}; ratio {ratio:.1f}x")

    report = {"n": args.n, "episodes": args.episodes, "arms": stats,
              "comparisons_osc_rate": comps}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
