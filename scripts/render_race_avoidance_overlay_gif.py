#!/usr/bin/env python3
"""Render the README race encounter as an overlay, not side-by-side.

The side-by-side GIF is honest but weak: the reader has to mentally
compare two panes. This renderer overlays the failed vanilla path and
the low-temperature path in the same camera frame, against the same
moving sweeper, so avoidance reads directly as path separation around
the red safety halo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import numpy as np
import yaml

from render_race_hero_gif import draw_track, obstacle_trajectory, set_limits
from uav_nav_lab.viz.episode_gif import load_drones, trajectory_arrays


INK = "#172033"
OBSTACLE_COLOR = "#d6261f"
FAIL_COLOR = "#ef4444"
AVOID_COLOR = "#16a34a"


def load_cfg(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected mapping")
    return data


def parse_run(raw: str) -> tuple[Path, str]:
    if ":" in raw:
        path_s, label = raw.rsplit(":", 1)
    else:
        path_s, label = raw, Path(raw).name
    return Path(path_s), label


def first_collision_step(drone_log: dict, t_max: int) -> int:
    for idx, step in enumerate(drone_log.get("steps", [])):
        if bool(step.get("collision", False)):
            return idx
    if drone_log.get("outcome") == "collision":
        return max(0, len(drone_log.get("steps", [])) - 1)
    return t_max


def clearance(
    pos: np.ndarray,
    obs_pos: np.ndarray,
    *,
    obstacle_radius: float,
    drone_radius: float,
) -> float:
    return float(np.linalg.norm(pos - obs_pos) - obstacle_radius - drone_radius)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--failed-run", required=True, help="`path:label` for contact arm")
    ap.add_argument("--avoid-run", required=True, help="`path:label` for avoiding arm")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--focus-drone", type=int, default=3)
    ap.add_argument("--focus-obstacle", type=int, default=0)
    ap.add_argument("--start-step", type=int, default=520)
    ap.add_argument("--end-step", type=int, default=632)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--trail", type=int, default=76)
    ap.add_argument("--future", type=int, default=52)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--xlim", type=float, nargs=2, default=[14.0, 26.0])
    ap.add_argument("--ylim", type=float, nargs=2, default=[25.0, 36.0])
    ap.add_argument(
        "--title",
        default="Same moving sweeper: vanilla contacts, low-temp detours",
    )
    args = ap.parse_args()

    failed_dir, failed_label = parse_run(args.failed_run)
    avoid_dir, avoid_label = parse_run(args.avoid_run)
    cfg = load_cfg(Path(args.config))
    scenario = cfg.get("scenario", {})
    world = np.asarray(scenario.get("size", [40.0, 40.0, 14.0]), dtype=float)
    obstacles = scenario.get("dynamic_obstacles", []) or []
    obstacle = obstacles[args.focus_obstacle]
    obstacle_radius = float(obstacle.get("radius", 0.5))
    drone_radius = float(scenario.get("drone_radius", 0.4))

    failed_drones = load_drones(failed_dir, args.ep, n_drones=4)
    avoid_drones = load_drones(avoid_dir, args.ep, n_drones=4)
    t_pad = max(
        max(len(d["steps"]) for d in failed_drones),
        max(len(d["steps"]) for d in avoid_drones),
    )
    failed_true, failed_ref, failed_coll = trajectory_arrays(failed_drones, T_pad=t_pad)
    avoid_true, avoid_ref, avoid_coll = trajectory_arrays(avoid_drones, T_pad=t_pad)
    t_max = min(failed_true.shape[1], avoid_true.shape[1])

    obs_traj = obstacle_trajectory(
        np.asarray(obstacle["start"], dtype=float),
        np.asarray(obstacle["velocity"], dtype=float),
        args.dt,
        t_max,
        world,
    )
    ref0 = avoid_ref[0]
    center = np.array(
        [float(ref0[:, 0].mean()), float(ref0[:, 1].mean()), float(ref0[:, 2].mean())]
    )
    rx = float((ref0[:, 0].max() - ref0[:, 0].min()) / 2.0)
    ry = float((ref0[:, 1].max() - ref0[:, 1].min()) / 2.0)

    frames = list(
        range(
            max(0, args.start_step),
            min(t_max, args.end_step),
            max(1, args.stride),
        )
    )
    if not frames:
        raise SystemExit("empty frame range")

    fig, ax = plt.subplots(figsize=(7.4, 5.45))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.83)
    draw_track(ax, center, rx, ry, world)
    set_limits(ax, args.xlim, args.ylim)
    ax.plot(
        obs_traj[:, 0],
        obs_traj[:, 1],
        color=OBSTACLE_COLOR,
        linewidth=4.0,
        alpha=0.12,
        zorder=5,
    )

    fail_line, = ax.plot([], [], color=FAIL_COLOR, linewidth=3.0, alpha=0.92, zorder=10)
    avoid_line, = ax.plot([], [], color=AVOID_COLOR, linewidth=3.4, alpha=0.95, zorder=11)
    ref_line, = ax.plot(
        [],
        [],
        color="#334155",
        linewidth=1.2,
        alpha=0.45,
        linestyle=(0, (3, 3)),
        zorder=8,
    )
    obs_past, = ax.plot([], [], color=OBSTACLE_COLOR, linewidth=2.0, alpha=0.75, zorder=9)
    obs_future, = ax.plot(
        [],
        [],
        color=OBSTACLE_COLOR,
        linewidth=1.4,
        alpha=0.35,
        linestyle=(0, (3, 3)),
        zorder=9,
    )
    radius_sum = obstacle_radius + drone_radius
    obs_halo = Circle(
        (0.0, 0.0),
        radius_sum,
        facecolor=OBSTACLE_COLOR,
        edgecolor="none",
        alpha=0.14,
        zorder=7,
    )
    obs_pt = Circle(
        (0.0, 0.0),
        obstacle_radius,
        facecolor=OBSTACLE_COLOR,
        edgecolor="#111827",
        linewidth=1.1,
        alpha=0.95,
        zorder=12,
    )
    fail_pt = Circle(
        (0.0, 0.0),
        drone_radius,
        facecolor=FAIL_COLOR,
        edgecolor="#ffffff",
        linewidth=1.0,
        alpha=0.95,
        zorder=13,
    )
    avoid_pt = Circle(
        (0.0, 0.0),
        drone_radius,
        facecolor=AVOID_COLOR,
        edgecolor="#ffffff",
        linewidth=1.0,
        alpha=0.98,
        zorder=14,
    )
    for patch in [obs_halo, obs_pt, fail_pt, avoid_pt]:
        ax.add_patch(patch)

    fail_clear, = ax.plot([], [], color=FAIL_COLOR, linewidth=2.0, alpha=0.8, zorder=15)
    avoid_clear, = ax.plot([], [], color=AVOID_COLOR, linewidth=2.4, alpha=0.95, zorder=15)
    title_text = fig.suptitle("", fontsize=13.5, color=INK, fontweight="semibold")
    status_text = ax.text(
        0.03,
        0.96,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "#ffffff",
            "edgecolor": "#cbd5e1",
            "alpha": 0.9,
        },
        zorder=20,
    )
    legend_text = ax.text(
        0.97,
        0.96,
        f"red: {failed_label}\ngreen: {avoid_label}\ndash: race line",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "#ffffff",
            "edgecolor": "#cbd5e1",
            "alpha": 0.86,
        },
        zorder=20,
    )

    fd = args.focus_drone
    failed_hit_step = int(failed_coll[fd])
    avoid_hit_step = int(avoid_coll[fd])

    def update(k: int):
        k0 = max(0, k - args.trail)
        k1 = min(t_max - 1, k + args.future)
        failed_k = min(k, failed_true.shape[1] - 1)
        avoid_k = min(k, avoid_true.shape[1] - 1)
        obs = obs_traj[k]

        fail_line.set_data(
            failed_true[fd, k0 : failed_k + 1, 0],
            failed_true[fd, k0 : failed_k + 1, 1],
        )
        avoid_line.set_data(
            avoid_true[fd, k0 : avoid_k + 1, 0],
            avoid_true[fd, k0 : avoid_k + 1, 1],
        )
        ref_line.set_data(
            avoid_ref[fd, k0 : avoid_k + 1, 0],
            avoid_ref[fd, k0 : avoid_k + 1, 1],
        )
        obs_past.set_data(obs_traj[k0 : k + 1, 0], obs_traj[k0 : k + 1, 1])
        obs_future.set_data(obs_traj[k : k1 + 1, 0], obs_traj[k : k1 + 1, 1])

        obs_xy = (float(obs[0]), float(obs[1]))
        obs_halo.center = obs_xy
        obs_pt.center = obs_xy
        fail_pos = failed_true[fd, failed_k, :]
        avoid_pos = avoid_true[fd, avoid_k, :]
        fail_pt.center = (float(fail_pos[0]), float(fail_pos[1]))
        avoid_pt.center = (float(avoid_pos[0]), float(avoid_pos[1]))
        fail_clear.set_data([fail_pos[0], obs[0]], [fail_pos[1], obs[1]])
        avoid_clear.set_data([avoid_pos[0], obs[0]], [avoid_pos[1], obs[1]])

        fail_c = clearance(
            fail_pos,
            obs,
            obstacle_radius=obstacle_radius,
            drone_radius=drone_radius,
        )
        avoid_c = clearance(
            avoid_pos,
            obs,
            obstacle_radius=obstacle_radius,
            drone_radius=drone_radius,
        )
        failed_collided = failed_hit_step <= k
        avoid_collided = avoid_hit_step <= k
        if failed_collided:
            fail_pt.set_facecolor("#ffffff")
            fail_pt.set_edgecolor(FAIL_COLOR)
            fail_label_now = f"red contact @ {failed_hit_step * args.dt:.2f}s"
        else:
            fail_pt.set_facecolor(FAIL_COLOR)
            fail_pt.set_edgecolor("#ffffff")
            fail_label_now = f"red clearance {fail_c:+.2f} m"
        if avoid_collided:
            avoid_label_now = f"green contact @ {avoid_hit_step * args.dt:.2f}s"
            avoid_pt.set_facecolor("#ffffff")
            avoid_pt.set_edgecolor(AVOID_COLOR)
        else:
            avoid_label_now = f"green clearance {avoid_c:+.2f} m"
            avoid_pt.set_facecolor(AVOID_COLOR)
            avoid_pt.set_edgecolor("#ffffff")

        status_text.set_text(f"{fail_label_now}\n{avoid_label_now}")
        title_text.set_text(f"{args.title}   t = {k * args.dt:.2f} s")
        return [
            fail_line,
            avoid_line,
            ref_line,
            obs_past,
            obs_future,
            obs_halo,
            obs_pt,
            fail_pt,
            avoid_pt,
            fail_clear,
            avoid_clear,
            title_text,
            status_text,
            legend_text,
        ]

    anim = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 // args.fps,
        blit=False,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"wrote {out}  ({len(frames)} frames @ {args.fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
