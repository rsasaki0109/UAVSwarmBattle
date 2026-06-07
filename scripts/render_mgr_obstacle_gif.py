"""Side-by-side GIF: under a hub-crossing obstacle the always-on peer convention
TRANSITS the hub (some drones get through) while the triggered Merry-Go-Round
ORBITS the contested centre and gets mowed down.

Left:  cbf_pairwise — CBF + the always-on pairwise winding convention. The fleet
       drives a clockwise CURRENT through the hub; each drone transits quickly and
       continues to its goal, so the crossing body only catches a few.
Right: mgr — the SAME CBF base, but each drone detects the deadlock and ORBITS the
       cluster centroid. The roundabout HOLDS the fleet circulating in the swept
       region, so the reflecting obstacle sweeps through the ring and collides.

Both run the registered planners through the same accel-limited single-integrator
dynamics the lab's dummy_2d sim uses, with the SAME reflecting hub obstacle fed to
each planner via dynamic_obstacles. Loads nothing — it rolls out live.

  python scripts/render_mgr_obstacle_gif.py --n 6
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from uav_nav_lab.planner import PLANNER_REGISTRY  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BG = "#0d1117"
PANEL = "#161b22"
GRID = "#21262d"
CRASH = "#f85149"
GOAL = "#3fb950"
OBS = "#f0883e"
SPEED = 5.0
R = 20.0
RADIUS = 0.4
OBS_R = 1.5
OBS_SPEED = 4.5
OBS_BOUND = 23.0  # reflect the obstacle within +/- this in y (crosses the hub)


def _planner(kind):
    cfg = {"max_speed": SPEED, "radius": RADIUS, "alpha": 2.0, "time_step": 0.1,
           "neighbor_dist": 15.0, "safety_margin": 0.1, "goal_radius": 1.5}
    if kind == "cbf_pairwise":
        cfg["pairwise_bias"] = 1.0
        cfg["pairwise_radius"] = 8.0
        return PLANNER_REGISTRY.get("cbf").from_config(cfg)
    return PLANNER_REGISTRY.get(kind).from_config(cfg)


def _antipodal(n, rng):
    ang = 2.0 * np.pi * np.arange(n) / n
    start = R * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    start = start + rng.normal(0.0, 0.8, start.shape)
    return start, -start.copy()


def _rollout(kind, start, goal, *, dt=0.05, replan=0.1, max_steps=900,
             max_accel=6.0):
    n = len(start)
    planners = [_planner(kind) for _ in range(n)]
    for p in planners:
        p.reset()
    pos = start.copy().astype(float)
    vel = np.zeros((n, 2))
    cmd = np.zeros((n, 2))
    done = np.zeros(n, dtype=bool)
    hit = np.zeros(n, dtype=bool)
    last = np.full(n, -1e9)
    obs_p = np.array([0.0, -OBS_BOUND])
    obs_v = np.array([0.0, OBS_SPEED])
    traj = [pos.copy()]
    obs_traj = [obs_p.copy()]
    for step in range(max_steps):
        t = step * dt
        obs_dyn = {"position": obs_p.copy(), "velocity": obs_v.copy(), "radius": OBS_R}
        for i in range(n):
            if done[i]:
                cmd[i] = 0.0
                continue
            if t - last[i] >= replan - 1e-9:
                peers = [{"position": pos[j], "velocity": vel[j], "radius": RADIUS}
                         for j in range(n) if j != i and not done[j]]
                peers.append(dict(obs_dyn))
                planners[i].set_current_state(pos[i], vel[i])
                pl = planners[i].plan(pos[i], goal[i], None, dynamic_obstacles=peers)
                cmd[i] = pl.target_velocity if pl.target_velocity is not None else 0.0
                last[i] = t
        for i in range(n):
            if done[i]:
                continue
            dv = cmd[i] - vel[i]
            nm = float(np.linalg.norm(dv))
            if nm > max_accel * dt:
                dv *= max_accel * dt / nm
            vel[i] = vel[i] + dv
            pos[i] = pos[i] + vel[i] * dt
            if float(np.linalg.norm(pos[i] - goal[i])) <= 1.5:
                done[i] = True
        # advance + reflect the obstacle
        obs_p = obs_p + obs_v * dt
        if abs(obs_p[1]) >= OBS_BOUND:
            obs_v[1] = -obs_v[1]
            obs_p[1] = float(np.clip(obs_p[1], -OBS_BOUND, OBS_BOUND))
        for i in range(n):
            if not done[i] and float(np.linalg.norm(pos[i] - obs_p)) < RADIUS + OBS_R:
                hit[i] = True
        for i in range(n):
            for j in range(i + 1, n):
                if not done[i] and not done[j] and \
                        float(np.linalg.norm(pos[i] - pos[j])) < 2 * RADIUS:
                    hit[i] = hit[j] = True
        traj.append(pos.copy())
        obs_traj.append(obs_p.copy())
        if done.all():
            break
    return traj, obs_traj, hit


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=4001)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=260)
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "images" / "swarm_mgr_obstacle.gif"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    start, goal = _antipodal(args.n, rng)
    runs = {
        "Peer convention — a current THROUGH the hub": _rollout("cbf_pairwise", start, goal),
        "Merry-Go-Round — ORBITS the hub, mowed down": _rollout("mgr", start, goal),
    }
    nf = min(max(len(r[0]) for r in runs.values()), args.max_frames)

    def pad(seq):
        s = list(seq)
        while len(s) < nf:
            s.append(s[-1])
        return s[:nf]

    data = {k: (pad(v[0]), pad(v[1]), v[2]) for k, v in runs.items()}
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, args.n))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 5.0))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.03, wspace=0.06)
    art = {}
    lim = R + 4.0
    for ax, (title, (traj, otraj, hit)) in zip(axes, runs.items()):
        ax.set_facecolor(PANEL)
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        survived = int((~hit).sum())
        ax.set_title(title, color=(GOAL if survived == args.n else CRASH),
                     fontsize=11.5, fontweight="bold", pad=8)
        ax.scatter(goal[:, 0], goal[:, 1], s=90, marker="*", c=colors, alpha=0.45, zorder=2)
        trails = [ax.plot([], [], "-", color=colors[i], lw=1.3, alpha=0.55, zorder=3)[0]
                  for i in range(args.n)]
        scat = ax.scatter(start[:, 0], start[:, 1], s=90, c=colors,
                          edgecolors="white", linewidths=0.6, zorder=5)
        obs_dot = ax.scatter([0], [-OBS_BOUND], s=OBS_R ** 2 * 120, c=OBS,
                             edgecolors="white", linewidths=0.8, alpha=0.9, zorder=6)
        art[title] = dict(trails=trails, scat=scat, obs=obs_dot, hit=hit)

    fig.suptitle(f"A body crossing the hub, N={args.n}: transit survives, orbit gets caught",
                 color="#c9d1d9", fontsize=13, y=0.975)

    def update(f):
        out = []
        for title in runs:
            a = art[title]
            T, OT, hit = data[title]
            pos = T[f]
            a["scat"].set_offsets(pos)
            # colour collided drones red once the obstacle has reached them
            fc = [(CRASH if hit[i] else None) for i in range(args.n)]
            ec = np.array([[0.97, 0.32, 0.29, 1.0] if hit[i] else [1, 1, 1, 1]
                           for i in range(args.n)])
            a["scat"].set_edgecolors(ec)
            a["obs"].set_offsets([OT[f]])
            for i, tr in enumerate(a["trails"]):
                arr = np.array([T[k][i] for k in range(f + 1)])
                tr.set_data(arr[:, 0], arr[:, 1])
            out += a["trails"] + [a["scat"], a["obs"]]
        return out

    anim = FuncAnimation(fig, update, frames=nf, interval=1000 / args.fps, blit=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"[gif] {out}  ({out.stat().st_size // 1024} KB, {nf} frames)")
    for title, (_, _, hit) in runs.items():
        print(f"   {title!r}: {int((~hit).sum())}/{args.n} survived")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
