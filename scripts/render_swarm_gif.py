"""Render eye-catching side-by-side (or single-pane) GIFs of the 2-D planners.

World-coordinate single-integrator sim (the model VO/RVO/HRVO/ORCA assume), so it
needs no occupancy grid: each pane drives one planner and records trajectories,
then matplotlib animates the panes with fading trails, goal markers, a collision
flash, and — optionally — fast moving obstacles and a gusty wind field the drones
have to fly through. Used for the README gallery.

  # antipodal hub: ORCA collides where HRVO rounds it
  python scripts/render_swarm_gif.py --scenario antipodal --arms orca hrvo \
      --n 6 --seed 4002 --out docs/images/swarm_antipodal_orca_vs_hrvo.gif

  # crossing: RVO's reciprocal dance vs ORCA's smooth glide
  python scripts/render_swarm_gif.py --scenario crossing --arms rvo orca \
      --n 4 --seed 4006 --trail 10000 --replan-period 0.05 \
      --out docs/images/swarm_crossing_rvo_vs_orca.gif

  # aggressive: weave drones through fast sweeping obstacles
  python scripts/render_swarm_gif.py --scenario crossing --arms hrvo --n 4 \
      --obstacles 5 --replan-period 0.1 --out docs/images/swarm_obstacle_gauntlet.gif

  # a big rotating roundabout
  python scripts/render_swarm_gif.py --scenario antipodal --arms roundabout \
      --n 18 --out docs/images/swarm_big_roundabout.gif

  # fly a crossing through a gusty crosswind
  python scripts/render_swarm_gif.py --scenario crossing --arms hrvo --n 4 \
      --wind 2.5 0.0 --gust 2.0 --replan-period 0.1 --out docs/images/swarm_wind.gif
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

LABELS = {"vo": "VO (1998)", "rvo": "RVO (2008)", "hrvo": "HRVO (2009)",
          "orca": "ORCA (2011)", "roundabout": "Merry-Go-Round", "mpc": "MPC"}


def _planner(kind):
    c = {"max_speed": SPEED, "radius": 0.4, "safety_margin": 0.1, "time_horizon": 2.0,
         "neighbor_dist": 15.0, "goal_radius": 1.5, "time_step": DT,
         "center": (CX, CY), "ring_radius": 16.0}
    p = PLANNER_REGISTRY.get(kind).from_config(c)
    p.reset()
    return p


def _layout(scenario, n, rng):
    starts, goals = [], []
    if scenario == "antipodal":
        ring = 16.0
        for i in range(n):
            a = 2.0 * math.pi * i / n + rng.uniform(-0.05, 0.05)
            r = ring + rng.uniform(-0.4, 0.4)
            starts.append(np.array([CX + r * math.cos(a), CY + r * math.sin(a)]))
            goals.append(np.array([CX - r * math.cos(a), CY - r * math.sin(a)]))
    else:  # crossing: two perpendicular streams
        gap, lo, hi = 2.6, 8.0, 42.0
        span = (n - 1) * gap
        base = 25.0 - span / 2.0
        for i in range(n):
            y = base + i * gap + rng.uniform(-0.3, 0.3)
            starts.append(np.array([lo, y])); goals.append(np.array([hi, y]))
        for i in range(n):
            x = base + i * gap + rng.uniform(-0.3, 0.3)
            starts.append(np.array([x, lo])); goals.append(np.array([x, hi]))
    return starts, goals


def _spawn_obstacles(k, rng):
    """Fast bodies that sweep across the central hub region (bounced in a box)."""
    obs = []
    for _ in range(k):
        side = rng.randint(0, 3)
        sp = rng.uniform(4.0, 6.5)
        if side == 0:    # from left, moving +x
            p = [6.0, rng.uniform(14.0, 36.0)]; v = [sp, rng.uniform(-1.5, 1.5)]
        elif side == 1:  # from right
            p = [44.0, rng.uniform(14.0, 36.0)]; v = [-sp, rng.uniform(-1.5, 1.5)]
        elif side == 2:  # from bottom
            p = [rng.uniform(14.0, 36.0), 6.0]; v = [rng.uniform(-1.5, 1.5), sp]
        else:            # from top
            p = [rng.uniform(14.0, 36.0), 44.0]; v = [rng.uniform(-1.5, 1.5), -sp]
        obs.append({"position": np.array(p, dtype=float),
                    "velocity": np.array(v, dtype=float), "radius": 1.3})
    return obs


def _wind(t, base, gust, period=2.5):
    if base is None:
        return np.zeros(2)
    g = 1.0 + gust * math.sin(2.0 * math.pi * t / period)
    return np.array(base, dtype=float) * g


def _h_ok(v):
    return float(np.hypot(v[0], v[1])) > 0.1


def _simulate(kind, scenario, n, seed, max_steps=400, replan_period=0.5,
              n_obstacles=0, wind_base=None, wind_gust=0.0):
    rng = random.Random(seed)
    starts, goals = _layout(scenario, n, rng)
    m = len(starts)
    pos = [s.copy() for s in starts]
    vel = [np.zeros(2) for _ in range(m)]
    plan = [_planner(kind) for _ in range(m)]
    arrived = [False] * m
    traj = [[p.copy() for p in pos]]
    obstacles = _spawn_obstacles(n_obstacles, rng) if n_obstacles else []
    obs_traj = [[o["position"].copy() for o in obstacles]]
    rp_steps = max(1, round(replan_period / DT))
    for step in range(max_steps):
        t = step * DT
        for o in obstacles:  # move obstacles (bounce inside the box)
            o["position"] = o["position"] + o["velocity"] * DT
            for d in range(2):
                if o["position"][d] < 4.0 and o["velocity"][d] < 0:
                    o["velocity"][d] *= -1
                if o["position"][d] > 46.0 and o["velocity"][d] > 0:
                    o["velocity"][d] *= -1
        if step % rp_steps == 0:
            peers = [{"position": pos[j].copy(), "velocity": vel[j].copy(), "radius": 0.4}
                     for j in range(m)]
            obs_dicts = [{"position": o["position"].copy(), "velocity": o["velocity"].copy(),
                          "radius": o["radius"]} for o in obstacles]
            for i in range(m):
                if arrived[i]:
                    vel[i] = np.zeros(2); continue
                plan[i].set_current_state(pos[i], vel[i])
                others = [peers[j] for j in range(m) if j != i] + obs_dicts
                vel[i] = plan[i].plan(pos[i], goals[i], None, dynamic_obstacles=others).target_velocity
        w = _wind(t, wind_base, wind_gust)
        for i in range(m):
            if not arrived[i]:
                pos[i] = pos[i] + (vel[i] + w) * DT
                if float(np.linalg.norm(pos[i] - goals[i])) < 1.5:
                    arrived[i] = True
        traj.append([p.copy() for p in pos])
        obs_traj.append([o["position"].copy() for o in obstacles])
        if all(arrived):
            break
    return np.array(traj), np.array(goals), np.array(obs_traj) if obstacles else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["antipodal", "crossing"], default="antipodal")
    ap.add_argument("--arms", nargs="+", default=["orca", "hrvo"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=4017)
    ap.add_argument("--replan-period", type=float, default=0.5)
    ap.add_argument("--obstacles", type=int, default=0)
    ap.add_argument("--wind", type=float, nargs=2, default=None, metavar=("WX", "WY"))
    ap.add_argument("--gust", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--trail", type=int, default=40)
    ap.add_argument("--out", default="docs/images/swarm.gif")
    args = ap.parse_args()

    sims = [_simulate(k, args.scenario, args.n, args.seed, replan_period=args.replan_period,
                      n_obstacles=args.obstacles, wind_base=args.wind, wind_gust=args.gust)
            for k in args.arms]
    T = max(s[0].shape[0] for s in sims)
    m = sims[0][0].shape[1]
    cmap = plt.get_cmap("turbo")
    colors = [cmap(i / max(m - 1, 1)) for i in range(m)]

    fig, axes = plt.subplots(1, len(args.arms), figsize=(5.2 * len(args.arms), 5.2))
    if len(args.arms) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#0d1117")

    pad = 3.0
    allpts = np.vstack([s[0].reshape(-1, 2) for s in sims])
    xlo, ylo = allpts.min(0) - pad
    xhi, yhi = allpts.max(0) + pad

    arts = []
    for ax, kind, (traj, goals, obs_traj) in zip(axes, args.arms, sims):
        ax.set_facecolor("#0d1117")
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        title = LABELS.get(kind, kind)
        if args.wind is not None and (args.wind[0] or args.wind[1]):
            title += "  +wind"
        if args.obstacles:
            title += f"  +{args.obstacles} obstacles"
        ax.set_title(title, color="#e6edf3", fontsize=15, pad=10, fontweight="bold")
        ax.scatter(goals[:, 0], goals[:, 1], marker="x", s=70, c=colors, linewidths=2.0, alpha=0.5)
        if args.wind is not None and (args.wind[0] or args.wind[1]):
            wn = np.array(args.wind, float); wn = wn / (np.linalg.norm(wn) + 1e-9)
            ax.annotate("", xy=(xlo + 5 + wn[0] * 3, ylo + 4 + wn[1] * 3),
                        xytext=(xlo + 5, ylo + 4),
                        arrowprops=dict(arrowstyle="-|>", color="#58a6ff", lw=2.5, alpha=0.85))
        trails = [ax.plot([], [], "-", lw=2.0, color=colors[i], alpha=0.6)[0] for i in range(m)]
        dots = ax.scatter([s[0] for s in traj[0]], [s[1] for s in traj[0]],
                          s=130, c=colors, edgecolors="white", linewidths=1.0, zorder=5)
        n_obs = obs_traj.shape[1] if obs_traj is not None else 0
        obs_dots = ax.scatter([], [], s=900, facecolors="#da3633", edgecolors="#ff7b72",
                              linewidths=1.5, alpha=0.85, zorder=4) if n_obs else None
        flash = ax.scatter([], [], s=420, facecolors="none", edgecolors="#ffd33d", linewidths=3.0, zorder=6)
        arts.append((traj, obs_traj, trails, dots, obs_dots, flash))

    def update(f):
        out = []
        for (traj, obs_traj, trails, dots, obs_dots, flash) in arts:
            tt = min(f, traj.shape[0] - 1)
            cur = traj[tt]
            dots.set_offsets(cur)
            lo = max(0, tt - args.trail)
            for i in range(m):
                seg = traj[lo:tt + 1, i, :]
                trails[i].set_data(seg[:, 0], seg[:, 1])
            ocur = None
            if obs_dots is not None:
                ocur = obs_traj[min(tt, obs_traj.shape[0] - 1)]
                obs_dots.set_offsets(ocur)
            hot = []
            for i in range(cur.shape[0]):
                for j in range(i + 1, cur.shape[0]):
                    if float(np.linalg.norm(cur[i] - cur[j])) < COLL:
                        hot.extend([cur[i], cur[j]])
                if ocur is not None:
                    for o in ocur:
                        if float(np.linalg.norm(cur[i] - o)) < 0.4 + 1.3:
                            hot.append(cur[i])
            flash.set_offsets(np.array(hot) if hot else np.empty((0, 2)))
            out += trails + [dots, flash]
            if obs_dots is not None:
                out += [obs_dots]
        return out

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / args.fps, blit=True)
    fig.tight_layout()
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=90,
              savefig_kwargs={"facecolor": "#0d1117"})
    print(f"wrote {args.out}  ({T} frames, {len(args.arms)} panes, {m} drones, "
          f"{args.obstacles} obstacles, wind={args.wind})")


if __name__ == "__main__":
    main()
