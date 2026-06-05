"""Reproduce and quantify the classic RVO -> ORCA improvement: ORCA removes RVO's
oscillation (the "reciprocal dance").

RVO (van den Berg 2008) and ORCA (2011) are both reciprocal velocity-space
avoiders; ORCA's contribution was the half-plane LP that eliminates the
oscillation RVO suffers from its discrete penalty-ranked velocity selection. This
runs both on the same crossing in a self-contained single-integrator sim (the
model both methods assume) so the full velocity trajectory is observable, and
measures, per drone:
  - OSCILLATION = total absolute heading variation of the velocity (rad) — the
    direct signature of the reciprocal dance;
  - success (reached goal, never within the collision radius);
  - arrival time.

Perpendicular crossing of 2N drones, jittered per seed; arms rvo / orca; paired
by seed (mean per-drone oscillation, success, time).

  python scripts/rvo_orca_oscillation_phase.py --n 4 --episodes 40
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
COLL = 0.8  # 2 * drone radius


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
    for i in range(n):  # +x stream
        y = base + i * GAP + rng.uniform(-0.4, 0.4)
        starts.append(np.array([LO, y])); goals.append(np.array([HI, y]))
    for i in range(n):  # +y stream
        x = base + i * GAP + rng.uniform(-0.4, 0.4)
        starts.append(np.array([x, LO])); goals.append(np.array([x, HI]))
    return starts, goals


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
        # collision check
        for i in range(m):
            for j in range(i + 1, m):
                if float(np.linalg.norm(pos[i] - pos[j])) < COLL:
                    collided = True
        if collided or all(arrived):
            break
    success = (not collided) and all(arrived)
    mean_osc = sum(osc) / m
    makespan = max([t for t in arrive_t if t is not None], default=float("nan")) if all(arrived) else float("nan")
    return success, mean_osc, makespan


def _h_ok(v):
    return float(np.hypot(v[0], v[1])) > 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--out", default="results/rvo_orca_oscillation_phase.json")
    args = ap.parse_args()

    rows = {"rvo": [], "orca": []}
    for kind in ("rvo", "orca"):
        for e in range(args.episodes):
            rows[kind].append(_episode(kind, args.n, args.seed + e))

    def summ(k):
        succ = sum(1 for s, _, _ in rows[k] if s)
        oscs = [o for _, o, _ in rows[k]]
        mks = [t for _, _, t in rows[k] if not math.isnan(t)]
        return succ, sum(oscs) / len(oscs), (sum(mks) / len(mks) if mks else float("nan"))

    rs, ro_, rmk = summ("rvo")
    os_, oo, omk = summ("orca")
    # paired: per-seed which has higher oscillation (expect rvo every time)
    rvo_more = sum(1 for i in range(args.episodes) if rows["rvo"][i][1] > rows["orca"][i][1] + 1e-6)
    orca_more = sum(1 for i in range(args.episodes) if rows["orca"][i][1] > rows["rvo"][i][1] + 1e-6)
    p_osc = mcnemar_exact_p(orca_more, rvo_more)

    print(f"\nRVO vs ORCA on the crossing (2N={2*args.n}, self-contained sim, n={args.episodes})")
    print(f"{'arm':>5} | {'success':>9} | {'mean osc (rad)':>15} | {'mean makespan':>14}")
    print("-" * 52)
    print(f"{'rvo':>5} | {rs:>3}/{args.episodes:<3} | {ro_:>15.2f} | {rmk:>12.2f}s")
    print(f"{'orca':>5} | {os_:>3}/{args.episodes:<3} | {oo:>15.2f} | {omk:>12.2f}s")
    print(f"\noscillation: RVO higher on {rvo_more}/{args.episodes} seeds, ORCA higher on "
          f"{orca_more} (sign p={p_osc:.2e}); RVO/ORCA osc ratio = {ro_/max(oo,1e-6):.1f}x")

    report = {"n": args.n, "episodes": args.episodes,
              "rvo": {"success": rs, "mean_osc": ro_, "mean_makespan": rmk},
              "orca": {"success": os_, "mean_osc": oo, "mean_makespan": omk},
              "osc_rvo_higher": rvo_more, "osc_orca_higher": orca_more, "osc_sign_p": p_osc,
              "osc_ratio": ro_ / max(oo, 1e-6)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
