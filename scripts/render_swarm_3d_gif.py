"""Render a 3-D swarm GIF: drones swap across a sphere through a shared centre,
avoiding in full 3-D with a slowly orbiting camera.

A world-coordinate 3-D single-integrator sim (no occupancy grid) driven by the
3-D-capable APF planner: N drones sit on a sphere and head for the antipodal
point, so every path crosses the centre. In 3-D the avoidance has a vertical
escape the 2-D hub lacks, so the fleet threads the centre as a swirling 3-D
flow. Rendered with matplotlib's 3-D axes (depth-shaded markers, fading 3-D
trails, a wireframe box, dark theme) and a camera azimuth that rotates over the
clip for a cinematic, simulator-like look.

  python scripts/render_swarm_3d_gif.py --n 12 --seed 1 --out docs/images/swarm_3d_sphere.gif
"""
import argparse
import math
import random

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from matplotlib.animation import FuncAnimation, PillowWriter

from uav_nav_lab.planner import PLANNER_REGISTRY

SPEED = 5.0
DT = 0.05
COLL = 0.8
C = np.array([25.0, 25.0, 25.0])
R = 16.0


def _planner():
    p = PLANNER_REGISTRY.get("apf").from_config(
        {"max_speed": SPEED, "radius": 0.4, "safety_margin": 0.2, "time_step": DT,
         "goal_radius": 1.5})
    p.reset()
    return p


def _fib_sphere(n, rng):
    """N near-uniform points on a sphere (Fibonacci), lightly jittered."""
    pts = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - 2.0 * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1.0 - y * y))
        th = phi * i + rng.uniform(-0.05, 0.05)
        pts.append(np.array([r * math.cos(th), y, r * math.sin(th)]))
    return pts


def _field_layout(n, rng):
    """Cross a volume: drones on the -x face, goals on the +x face (an asteroid
    field they fly through)."""
    starts, goals = [], []
    for _ in range(n):
        y = rng.uniform(C[1] - R, C[1] + R)
        z = rng.uniform(C[2] - R, C[2] + R)
        starts.append(np.array([C[0] - R, y, z]))
        goals.append(np.array([C[0] + R, y + rng.uniform(-3, 3), z + rng.uniform(-3, 3)]))
    return starts, goals


def _spawn_obstacles3d(k, rng):
    obs = []
    for _ in range(k):
        p = C + np.array([rng.uniform(-R * 0.5, R * 0.5) for _ in range(3)])
        v = np.array([rng.uniform(-2.0, 2.0) for _ in range(3)])
        obs.append({"position": p, "velocity": v, "radius": rng.uniform(1.6, 2.6)})
    return obs


def _simulate(n, seed, max_steps=320, replan_period=0.2, scenario="antipodal",
              n_obstacles=0, wind=None):
    rng = random.Random(seed)
    if scenario == "field":
        starts, goals = _field_layout(n, rng)
    else:
        dirs = _fib_sphere(n, rng)
        starts = [C + R * d for d in dirs]
        goals = [C - R * d for d in dirs]   # antipodal
    pos = [s.copy() for s in starts]
    vel = [np.zeros(3) for _ in range(n)]
    plan = [_planner() for _ in range(n)]
    arrived = [False] * n
    traj = [[p.copy() for p in pos]]
    obstacles = _spawn_obstacles3d(n_obstacles, rng) if n_obstacles else []
    obs_r = [o["radius"] for o in obstacles]
    obs_traj = [[o["position"].copy() for o in obstacles]]
    w = np.array(wind, dtype=float) if wind is not None else np.zeros(3)
    rp = max(1, round(replan_period / DT))
    for step in range(max_steps):
        for o in obstacles:  # drift, bounce in the box
            o["position"] = o["position"] + o["velocity"] * DT
            for d in range(3):
                if abs(o["position"][d] - C[d]) > R and \
                        (o["position"][d] - C[d]) * o["velocity"][d] > 0:
                    o["velocity"][d] *= -1
        if step % rp == 0:
            peers = [{"position": pos[j].copy(), "velocity": vel[j].copy(), "radius": 0.4}
                     for j in range(n)]
            obs_d = [{"position": o["position"].copy(), "velocity": o["velocity"].copy(),
                      "radius": o["radius"]} for o in obstacles]
            for i in range(n):
                if arrived[i]:
                    vel[i] = np.zeros(3); continue
                if hasattr(plan[i], "set_current_state"):
                    plan[i].set_current_state(pos[i], vel[i])
                others = [peers[j] for j in range(n) if j != i] + obs_d
                vel[i] = plan[i].plan(pos[i], goals[i], None, dynamic_obstacles=others).target_velocity
        for i in range(n):
            if not arrived[i]:
                pos[i] = pos[i] + (vel[i] + w) * DT
                if float(np.linalg.norm(pos[i] - goals[i])) < 1.5:
                    arrived[i] = True
        traj.append([p.copy() for p in pos])
        obs_traj.append([o["position"].copy() for o in obstacles])
        if all(arrived):
            break
    return np.array(traj), np.array(goals), (np.array(obs_traj) if obstacles else None), obs_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--trail", type=int, default=45)
    ap.add_argument("--replan-period", type=float, default=0.2)
    ap.add_argument("--scenario", choices=["antipodal", "field"], default="antipodal")
    ap.add_argument("--obstacles", type=int, default=0)
    ap.add_argument("--wind", type=float, nargs=3, default=None, metavar=("WX", "WY", "WZ"))
    ap.add_argument("--out", default="docs/images/swarm_3d_sphere.gif")
    args = ap.parse_args()

    traj, goals, obs_traj, obs_r = _simulate(
        args.n, args.seed, replan_period=args.replan_period, scenario=args.scenario,
        n_obstacles=args.obstacles, wind=args.wind)
    T, m = traj.shape[0], traj.shape[1]
    colors = [plt.get_cmap("turbo")(i / max(m - 1, 1)) for i in range(m)]

    fig = plt.figure(figsize=(6.4, 6.4))
    fig.patch.set_facecolor("#0d1117")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0d1117")
    lo, hi = C - (R + 1.0), C + (R + 1.0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#0d1117")
        axis.pane.set_edgecolor("#30363d")
        axis.pane.set_alpha(1.0)
    ax.grid(False)

    ax.scatter(goals[:, 0], goals[:, 1], goals[:, 2], marker="x", s=40, c=colors, alpha=0.4)
    trails = [ax.plot([], [], [], "-", lw=1.8, color=colors[i], alpha=0.55)[0] for i in range(m)]
    dots = ax.scatter(traj[0, :, 0], traj[0, :, 1], traj[0, :, 2],
                      s=90, c=colors, edgecolors="white", linewidths=0.8, depthshade=True)
    n_obs = obs_traj.shape[1] if obs_traj is not None else 0
    obs_dots = None
    if n_obs:
        osz = [float(np.clip(40.0 * r * r, 120, 900)) for r in obs_r]
        obs_dots = ax.scatter(obs_traj[0, :, 0], obs_traj[0, :, 1], obs_traj[0, :, 2],
                              s=osz, c="#da3633", edgecolors="#ff7b72", linewidths=1.0,
                              alpha=0.8, depthshade=True)
    label = ("3-D obstacle field — APF threads a drifting asteroid field"
             if args.scenario == "field" else "3-D antipodal swarm — APF, full 3-D avoidance")
    title = ax.set_title(label, color="#e6edf3", fontsize=14, fontweight="bold", pad=6)

    def update(f):
        tt = min(f, T - 1)
        cur = traj[tt]
        dots._offsets3d = (cur[:, 0], cur[:, 1], cur[:, 2])
        lo_i = max(0, tt - args.trail)
        for i in range(m):
            seg = traj[lo_i:tt + 1, i, :]
            trails[i].set_data(seg[:, 0], seg[:, 1])
            trails[i].set_3d_properties(seg[:, 2])
        if obs_dots is not None:
            oc = obs_traj[min(tt, obs_traj.shape[0] - 1)]
            obs_dots._offsets3d = (oc[:, 0], oc[:, 1], oc[:, 2])
        ax.view_init(elev=22.0, azim=(30.0 + 0.55 * f) % 360.0)  # slow orbit
        return trails + [dots, title] + ([obs_dots] if obs_dots is not None else [])

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=90,
              savefig_kwargs={"facecolor": "#0d1117"})
    print(f"wrote {args.out}  ({T} frames, {m} drones, 3-D)")


if __name__ == "__main__":
    main()
