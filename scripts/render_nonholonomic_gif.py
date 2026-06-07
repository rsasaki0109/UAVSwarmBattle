"""Side-by-side GIF: the right-of-way convention on NON-HOLONOMIC drones.

Both panels are the antipodal swap flown by drones that cannot strafe — each is a
unicycle (forward drive + rate-limited turn, drawn as an oriented triangle), the
same kinematics as the lab's `dummy_unicycle` sim. The MPC controller is identical;
only the convention differs:

Left:  no convention (lateral_bias = 0) — the symmetric hub jams; drones that
       cannot side-step turn into each other and stall / collide.
Right: the right-of-way convention (lateral_bias = 2) — every drone veers right,
       turning the head-on convergence into a roundabout the non-holonomic fleet
       can actually fly.

  python scripts/render_nonholonomic_gif.py --n 6 --turn-rate 1.0
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
CRASH = "#f85149"
GOAL = "#3fb950"
SPEED = 5.0
R = 20.0
CX, CY = 25.0, 25.0   # MPC uses a grid → keep all coordinates positive (matches the runner)
RADIUS = 0.4


def _planner(bias):
    cfg = {"max_speed": SPEED, "replan_period": 0.2, "horizon": 40, "dt_plan": 0.05,
           "n_samples": 32, "resolution": 1.0, "inflate": 1, "goal_radius": 1.5,
           "safety_margin": 0.5, "use_prediction": True, "w_goal": 1.0, "w_obs": 100.0,
           "w_smooth": 0.05, "predictor": {"type": "constant_velocity"}}
    if bias:
        cfg["lateral_bias"] = bias
    return PLANNER_REGISTRY.get("mpc").from_config(cfg)


def _antipodal(n, rng):
    ang = 2.0 * np.pi * np.arange(n) / n
    ring = R * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    centre = np.array([CX, CY])
    start = centre + ring + rng.normal(0.0, 0.8, ring.shape)
    goal = centre - ring
    return start, goal


def _rollout(bias, start, goal, turn_rate, *, dt=0.05, replan=0.2, max_steps=700,
             max_accel=6.0):
    n = len(start)
    occ = np.zeros((50, 50), dtype=bool)   # empty grid (obstacles: none); MPC needs a real map
    planners = [_planner(bias) for _ in range(n)]
    for p in planners:
        p.reset()
    pos = start.copy().astype(float)
    vel = np.zeros((n, 2))
    head = np.arctan2(goal[:, 1] - start[:, 1], goal[:, 0] - start[:, 0])
    cmd = np.zeros((n, 2))
    done = np.zeros(n, dtype=bool)
    last = np.full(n, -1e9)
    traj, heads = [pos.copy()], [head.copy()]
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
                pl = planners[i].plan(pos[i], goal[i], occ, dynamic_obstacles=peers)
                cmd[i] = pl.target_velocity if pl.target_velocity is not None else 0.0
                last[i] = t
        for i in range(n):
            if done[i]:
                continue
            desired_speed = float(np.linalg.norm(cmd[i]))
            if desired_speed > 1e-6:
                ddir = float(np.arctan2(cmd[i, 1], cmd[i, 0]))
                err = (ddir - head[i] + np.pi) % (2 * np.pi) - np.pi
                head[i] += float(np.clip(err, -turn_rate * dt, turn_rate * dt))
            cur = float(np.linalg.norm(vel[i]))
            new = max(0.0, cur + float(np.clip(desired_speed - cur, -max_accel * dt, max_accel * dt)))
            vel[i] = new * np.array([np.cos(head[i]), np.sin(head[i])])
            pos[i] = pos[i] + vel[i] * dt
            if float(np.linalg.norm(pos[i] - goal[i])) <= 1.5:
                done[i] = True
        for i in range(n):
            for j in range(i + 1, n):
                if not done[i] and not done[j] and \
                        float(np.linalg.norm(pos[i] - pos[j])) < 2 * RADIUS:
                    collided_at = collided_at or len(traj)
        traj.append(pos.copy()); heads.append(head.copy())
        if done.all():
            break
    return traj, heads, bool(done.all()), collided_at


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--turn-rate", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=260)
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "images" / "swarm_nonholonomic.gif"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    start, goal = _antipodal(args.n, rng)
    runs = {f"no convention — non-holonomic jam": _rollout(0.0, start, goal, args.turn_rate),
            f"right-of-way convention — roundabout": _rollout(2.0, start, goal, args.turn_rate)}
    nf = min(max(len(r[0]) for r in runs.values()), args.max_frames)

    def pad(seq):
        s = list(seq)
        while len(s) < nf:
            s.append(s[-1])
        return s[:nf]

    colors = plt.cm.turbo(np.linspace(0.05, 0.95, args.n))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 5.0))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.03, wspace=0.06)
    lim = R + 4.0
    panels = []
    for ax, (title, (traj, heads, ok, coll)) in zip(axes, runs.items()):
        ax.set_facecolor(PANEL)
        ax.set_xlim(CX - lim, CX + lim); ax.set_ylim(CY - lim, CY + lim)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, color="#e6edf3", fontsize=11)
        ax.scatter(goal[:, 0], goal[:, 1], s=70, marker="*", color=colors, edgecolors="none", alpha=0.35)
        P0, H0 = traj[0], heads[0]
        # heading arrows (quiver) + position dots — the arrow shows the drone cannot strafe
        quiv = ax.quiver(P0[:, 0], P0[:, 1], np.cos(H0), np.sin(H0), color=colors,
                         scale=22, width=0.012, zorder=5)
        dots = ax.scatter(P0[:, 0], P0[:, 1], c=colors, s=28, edgecolors="k",
                          linewidths=0.4, zorder=6)
        panels.append((ax, pad(traj), pad(heads), quiv, dots, coll))

    def update(f):
        arts = []
        for ax, traj, heads, quiv, dots, coll in panels:
            P, H = traj[f], heads[f]
            quiv.set_offsets(P)
            quiv.set_UVC(np.cos(H), np.sin(H))
            dots.set_offsets(P)
            if coll is not None and f >= coll:
                dots.set_edgecolors(CRASH)
            arts.extend([quiv, dots])
        return arts

    anim = FuncAnimation(fig, update, frames=nf, interval=1000 // args.fps, blit=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=args.fps), dpi=85)
    oks = {k: v[2] for k, v in runs.items()}
    print(f"wrote {out}  reached-all: {oks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
