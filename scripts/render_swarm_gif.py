"""Render eye-catching side-by-side GIFs of the 2-D reciprocal-avoidance planners.

World-coordinate single-integrator sim (the model VO/RVO/HRVO/ORCA assume), so it
needs no occupancy grid: each pane drives one planner and records trajectories,
then matplotlib animates the panes together with fading trails, goal markers, and
a collision flash. Used for the README gallery.

  # antipodal hub: ORCA collides where HRVO rounds it
  python scripts/render_swarm_gif.py --scenario antipodal --arms orca hrvo \
      --n 6 --seed 4017 --out docs/images/swarm_antipodal_orca_vs_hrvo.gif

  # crossing: RVO's reciprocal dance vs ORCA's smooth glide
  python scripts/render_swarm_gif.py --scenario crossing --arms rvo orca \
      --n 3 --seed 4000 --out docs/images/swarm_crossing_rvo_vs_orca.gif
"""
import argparse
import math
import random

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from uav_nav_lab.planner import PLANNER_REGISTRY

SPEED = 5.0
DT = 0.05
COLL = 0.8
CX, CY = 25.0, 25.0

LABELS = {"vo": "VO (1998)", "rvo": "RVO (2008)", "hrvo": "HRVO (2009)", "orca": "ORCA (2011)"}


def _planner(kind):
    c = {"max_speed": SPEED, "radius": 0.4, "safety_margin": 0.1, "time_horizon": 2.0,
         "neighbor_dist": 15.0, "goal_radius": 1.5, "time_step": DT}
    p = PLANNER_REGISTRY.get(kind).from_config(c)
    p.reset()
    return p


def _layout(scenario, n, rng):
    starts, goals = [], []
    if scenario == "antipodal":
        ring = 15.0
        for i in range(n):
            a = 2.0 * math.pi * i / n + rng.uniform(-0.05, 0.05)
            r = ring + rng.uniform(-0.4, 0.4)
            starts.append(np.array([CX + r * math.cos(a), CY + r * math.sin(a)]))
            goals.append(np.array([CX - r * math.cos(a), CY - r * math.sin(a)]))
    else:  # crossing: two perpendicular streams
        gap, lo, hi = 2.6, 10.0, 40.0
        span = (n - 1) * gap
        base = 25.0 - span / 2.0
        for i in range(n):
            y = base + i * gap + rng.uniform(-0.3, 0.3)
            starts.append(np.array([lo, y])); goals.append(np.array([hi, y]))
        for i in range(n):
            x = base + i * gap + rng.uniform(-0.3, 0.3)
            starts.append(np.array([x, lo])); goals.append(np.array([x, hi]))
    return starts, goals


def _simulate(kind, scenario, n, seed, max_steps=400, replan_period=0.5):
    rng = random.Random(seed)
    starts, goals = _layout(scenario, n, rng)
    m = len(starts)
    pos = [s.copy() for s in starts]
    vel = [np.zeros(2) for _ in range(m)]
    plan = [_planner(kind) for _ in range(m)]
    arrived = [False] * m
    traj = [[p.copy() for p in pos]]
    collide_step = [None] * m
    rp_steps = max(1, round(replan_period / DT))
    for step in range(max_steps):
        if step % rp_steps == 0:
            peers = [{"position": pos[j].copy(), "velocity": vel[j].copy(), "radius": 0.4}
                     for j in range(m)]
            for i in range(m):
                if arrived[i]:
                    vel[i] = np.zeros(2); continue
                plan[i].set_current_state(pos[i], vel[i])
                others = [peers[j] for j in range(m) if j != i]
                vel[i] = plan[i].plan(pos[i], goals[i], None, dynamic_obstacles=others).target_velocity
        for i in range(m):
            if not arrived[i]:
                pos[i] = pos[i] + vel[i] * DT
                if float(np.linalg.norm(pos[i] - goals[i])) < 1.5:
                    arrived[i] = True
        for i in range(m):
            for j in range(i + 1, m):
                if float(np.linalg.norm(pos[i] - pos[j])) < COLL:
                    if collide_step[i] is None:
                        collide_step[i] = step
                    if collide_step[j] is None:
                        collide_step[j] = step
        traj.append([p.copy() for p in pos])
        if all(arrived):
            break
    return np.array(traj), np.array(goals), collide_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["antipodal", "crossing"], default="antipodal")
    ap.add_argument("--arms", nargs="+", default=["orca", "hrvo"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=4017)
    ap.add_argument("--replan-period", type=float, default=0.5)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--trail", type=int, default=40)
    ap.add_argument("--out", default="docs/images/swarm.gif")
    args = ap.parse_args()

    sims = [_simulate(k, args.scenario, args.n, args.seed, replan_period=args.replan_period)
            for k in args.arms]
    T = max(s[0].shape[0] for s in sims)
    m = sims[0][0].shape[1]
    cmap = plt.get_cmap("turbo")
    colors = [cmap(i / max(m - 1, 1)) for i in range(m)]

    fig, axes = plt.subplots(1, len(args.arms), figsize=(5.0 * len(args.arms), 5.0))
    if len(args.arms) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#0d1117")

    pad = 6.0
    allpts = np.vstack([s[0].reshape(-1, 2) for s in sims])
    xlo, ylo = allpts.min(0) - pad
    xhi, yhi = allpts.max(0) + pad

    arts = []
    for ax, kind, (traj, goals, _) in zip(axes, args.arms, sims):
        ax.set_facecolor("#0d1117")
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        ax.set_title(LABELS.get(kind, kind), color="#e6edf3", fontsize=15, pad=10, fontweight="bold")
        ax.scatter(goals[:, 0], goals[:, 1], marker="x", s=70, c=colors, linewidths=2.0, alpha=0.55)
        trails = [ax.plot([], [], "-", lw=2.0, color=colors[i], alpha=0.6)[0] for i in range(m)]
        dots = ax.scatter([s[0] for s in traj[0]], [s[1] for s in traj[0]],
                          s=130, c=colors, edgecolors="white", linewidths=1.0, zorder=5)
        flash = ax.scatter([], [], s=420, facecolors="none", edgecolors="#ff3b30", linewidths=3.0, zorder=6)
        arts.append((traj, trails, dots, flash))

    def update(f):
        out = []
        for (traj, trails, dots, flash) in arts:
            tt = min(f, traj.shape[0] - 1)
            cur = traj[tt]
            dots.set_offsets(cur)
            lo = max(0, tt - args.trail)
            for i in range(m):
                seg = traj[lo:tt + 1, i, :]
                trails[i].set_data(seg[:, 0], seg[:, 1])
            # flash any pair currently within the collision distance
            hot = []
            for i in range(cur.shape[0]):
                for j in range(i + 1, cur.shape[0]):
                    if float(np.linalg.norm(cur[i] - cur[j])) < COLL:
                        hot.extend([cur[i], cur[j]])
            flash.set_offsets(np.array(hot) if hot else np.empty((0, 2)))
            out += trails + [dots, flash]
        return out

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / args.fps, blit=True)
    fig.tight_layout()
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=90,
              savefig_kwargs={"facecolor": "#0d1117"})
    print(f"wrote {args.out}  ({T} frames, {len(args.arms)} panes, {m} drones)")


if __name__ == "__main__":
    main()
