"""Side-by-side GIF of the antipodal swap: stock CBF deadlocks at the hub, the
decentralized triggered Merry-Go-Round turns it into a roundabout.

Left:  cbf  — a plain control-barrier-function avoider; the symmetric hub brakes
       every agent to a safe stop and the swarm DEADLOCKS (timeout), frozen.
Right: mgr  — the SAME CBF base, but each agent detects the local deadlock, agrees
       on a common ring centre from sensing alone (no handed symmetry), and the
       fleet spirals counter-clockwise through the hub and peels off to goal.

Both run the registered planners through the same single-integrator dynamics the
lab's dummy_2d sim uses (accel-limited velocity tracking). Loads nothing — it
rolls out live.

  python scripts/render_mgr_gif.py --n 8
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
SPEED = 5.0
R = 20.0
RADIUS = 0.4


def _planner(kind):
    cfg = {"max_speed": SPEED, "radius": RADIUS, "alpha": 2.0, "time_step": 0.1,
           "neighbor_dist": 15.0, "safety_margin": 0.1, "goal_radius": 1.5}
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
    last = np.full(n, -1e9)
    traj = [pos.copy()]
    collided_at = None
    for step in range(max_steps):
        t = step * dt
        for i in range(n):
            if done[i]:
                cmd[i] = 0.0
                continue
            if t - last[i] >= replan - 1e-9:
                peers = [{"position": pos[j], "velocity": vel[j], "radius": RADIUS}
                         for j in range(n) if j != i and not done[j]]
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
        for i in range(n):
            for j in range(i + 1, n):
                if not done[i] and not done[j] and \
                        float(np.linalg.norm(pos[i] - pos[j])) < 2 * RADIUS:
                    collided_at = collided_at or len(traj)
        traj.append(pos.copy())
        if done.all():
            break
    return traj, bool(done.all()), collided_at


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=240)
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "images" / "swarm_mgr_roundabout.gif"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    start, goal = _antipodal(args.n, rng)
    runs = {"plain CBF — deadlock": _rollout("cbf", start, goal),
            "Merry-Go-Round — decentralized roundabout": _rollout("mgr", start, goal)}
    nf = min(max(len(r[0]) for r in runs.values()), args.max_frames)

    def pad(traj):
        t = list(traj)
        while len(t) < nf:
            t.append(t[-1])
        return t[:nf]

    data = {k: pad(v[0]) for k, v in runs.items()}
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, args.n))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 5.0))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.03, wspace=0.06)
    art = {}
    lim = R + 4.0
    for ax, (title, (traj, ok, coll)) in zip(axes, runs.items()):
        ax.set_facecolor(PANEL)
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_title(title, color=(GOAL if ok else CRASH), fontsize=12,
                     fontweight="bold", pad=8)
        ax.scatter(goal[:, 0], goal[:, 1], s=90, marker="*", c=colors, alpha=0.45, zorder=2)
        trails = [ax.plot([], [], "-", color=colors[i], lw=1.3, alpha=0.55, zorder=3)[0]
                  for i in range(args.n)]
        scat = ax.scatter(start[:, 0], start[:, 1], s=90, c=colors,
                          edgecolors="white", linewidths=0.6, zorder=5)
        art[title] = dict(trails=trails, scat=scat, ok=ok)

    fig.suptitle(f"The antipodal swap, N={args.n}: a triggered roundabout breaks the CBF deadlock",
                 color="#c9d1d9", fontsize=13, y=0.975)

    def update(f):
        out = []
        for title in runs:
            a = art[title]
            T = data[title]
            pos = T[f]
            a["scat"].set_offsets(pos)
            for i, tr in enumerate(a["trails"]):
                arr = np.array([T[k][i] for k in range(f + 1)])
                tr.set_data(arr[:, 0], arr[:, 1])
            out += a["trails"] + [a["scat"]]
        return out

    anim = FuncAnimation(fig, update, frames=nf, interval=1000 / args.fps, blit=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"[gif] {out}  ({out.stat().st_size // 1024} KB, {nf} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
