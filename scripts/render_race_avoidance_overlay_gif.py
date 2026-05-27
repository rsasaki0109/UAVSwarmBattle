#!/usr/bin/env python3
"""Render the README race encounter as an overlay, not side-by-side.

The side-by-side GIF is honest but weak: the reader has to mentally
compare two panes. This renderer overlays the failed vanilla path, the
low-temperature path, and optionally a no-sweeper control in the same
camera frame, against the same moving sweeper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle
import numpy as np
import yaml

from render_race_hero_gif import draw_track, obstacle_trajectory, set_limits
from uav_nav_lab.viz.episode_gif import load_drones, trajectory_arrays


INK = "#172033"
OBSTACLE_COLOR = "#d6261f"
FAIL_COLOR = "#ef4444"
AVOID_COLOR = "#16a34a"
GHOST_COLOR = "#475569"
STATIC_COLOR = "#111827"


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


def draw_static_boxes(ax, scenario: dict) -> None:
    boxes = ((scenario.get("obstacles") or {}).get("boxes") or [])
    for box in boxes:
        center = np.asarray(box.get("center", []), dtype=float)
        size = np.asarray(box.get("size", []), dtype=float)
        if center.size < 2 or size.size < 2:
            continue
        rect = Rectangle(
            (float(center[0] - 0.5 * size[0]), float(center[1] - 0.5 * size[1])),
            float(size[0]),
            float(size[1]),
            facecolor=STATIC_COLOR,
            edgecolor="#000000",
            linewidth=1.2,
            alpha=0.30,
            zorder=6,
        )
        ax.add_patch(rect)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--failed-run", required=True, help="`path:label` for contact arm")
    ap.add_argument("--avoid-run", required=True, help="`path:label` for avoiding arm")
    ap.add_argument(
        "--ghost-run",
        help="optional `path:label` for the no-sweeper control trajectory",
    )
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
        default="Same seed: obstacle bends low-temp away from no-sweeper ghost",
    )
    args = ap.parse_args()

    failed_dir, failed_label = parse_run(args.failed_run)
    avoid_dir, avoid_label = parse_run(args.avoid_run)
    ghost_dir: Path | None = None
    ghost_label: str | None = None
    if args.ghost_run:
        ghost_dir, ghost_label = parse_run(args.ghost_run)
    cfg = load_cfg(Path(args.config))
    scenario = cfg.get("scenario", {})
    world = np.asarray(scenario.get("size", [40.0, 40.0, 14.0]), dtype=float)
    obstacles = scenario.get("dynamic_obstacles", []) or []
    if args.focus_obstacle < 0:
        selected_obstacles = list(enumerate(obstacles))
    else:
        selected_obstacles = [(args.focus_obstacle, obstacles[args.focus_obstacle])]
    if not selected_obstacles:
        raise SystemExit("no dynamic obstacles to render")
    drone_radius = float(scenario.get("drone_radius", 0.4))

    failed_drones = load_drones(failed_dir, args.ep, n_drones=4)
    avoid_drones = load_drones(avoid_dir, args.ep, n_drones=4)
    drone_groups = [failed_drones, avoid_drones]
    if ghost_dir is not None:
        ghost_drones = load_drones(ghost_dir, args.ep, n_drones=4)
        drone_groups.append(ghost_drones)
    else:
        ghost_drones = None
    t_pad = max(
        max(len(d["steps"]) for d in failed_drones),
        max(len(d["steps"]) for d in avoid_drones),
        *(max(len(d["steps"]) for d in group) for group in drone_groups[2:]),
    )
    failed_true, failed_ref, failed_coll = trajectory_arrays(failed_drones, T_pad=t_pad)
    avoid_true, avoid_ref, avoid_coll = trajectory_arrays(avoid_drones, T_pad=t_pad)
    if ghost_drones is not None:
        ghost_true, ghost_ref, ghost_coll = trajectory_arrays(ghost_drones, T_pad=t_pad)
        t_max = min(failed_true.shape[1], avoid_true.shape[1], ghost_true.shape[1])
    else:
        ghost_true = ghost_ref = ghost_coll = None
        t_max = min(failed_true.shape[1], avoid_true.shape[1])

    obs_trajs = [
        (
            idx,
            float(obstacle.get("radius", 0.5)),
            obstacle_trajectory(
                np.asarray(obstacle["start"], dtype=float),
                np.asarray(obstacle["velocity"], dtype=float),
                args.dt,
                t_max,
                world,
            ),
        )
        for idx, obstacle in selected_obstacles
    ]
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
    draw_static_boxes(ax, scenario)
    set_limits(ax, args.xlim, args.ylim)
    for _, _, obs_traj in obs_trajs:
        ax.plot(
            obs_traj[:, 0],
            obs_traj[:, 1],
            color=OBSTACLE_COLOR,
            linewidth=4.0,
            alpha=0.12,
            zorder=5,
        )

    fail_line, = ax.plot([], [], color=FAIL_COLOR, linewidth=3.0, alpha=0.92, zorder=10)
    ghost_line = None
    if ghost_true is not None:
        ghost_line, = ax.plot(
            [],
            [],
            color=GHOST_COLOR,
            linewidth=2.4,
            alpha=0.82,
            linestyle=(0, (5, 3)),
            zorder=10,
        )
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
    obs_artists = []
    for _, obstacle_radius, _ in obs_trajs:
        obs_past, = ax.plot(
            [],
            [],
            color=OBSTACLE_COLOR,
            linewidth=2.0,
            alpha=0.75,
            zorder=9,
        )
        obs_future, = ax.plot(
            [],
            [],
            color=OBSTACLE_COLOR,
            linewidth=1.4,
            alpha=0.35,
            linestyle=(0, (3, 3)),
            zorder=9,
        )
        obs_halo = Circle(
            (0.0, 0.0),
            obstacle_radius + drone_radius,
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
        obs_artists.append(
            {
                "past": obs_past,
                "future": obs_future,
                "halo": obs_halo,
                "point": obs_pt,
                "radius": obstacle_radius,
            }
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
    ghost_pt = None
    if ghost_true is not None:
        ghost_pt = Circle(
            (0.0, 0.0),
            drone_radius,
            facecolor=GHOST_COLOR,
            edgecolor="#ffffff",
            linewidth=1.0,
            alpha=0.96,
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
    patches = [
        patch
        for row in obs_artists
        for patch in (row["halo"], row["point"])
    ] + [fail_pt, avoid_pt]
    if ghost_pt is not None:
        patches.insert(3, ghost_pt)
    for patch in patches:
        ax.add_patch(patch)

    fail_clear, = ax.plot([], [], color=FAIL_COLOR, linewidth=2.0, alpha=0.8, zorder=15)
    ghost_clear = None
    if ghost_true is not None:
        ghost_clear, = ax.plot(
            [],
            [],
            color=GHOST_COLOR,
            linewidth=1.7,
            alpha=0.75,
            linestyle=(0, (2, 2)),
            zorder=15,
        )
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
        (
            f"red: {failed_label}\ngreen: {avoid_label}\n"
            + (f"gray: {ghost_label}\n" if ghost_label else "")
            + "dash: race line"
        ),
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
        ghost_k = min(k, ghost_true.shape[1] - 1) if ghost_true is not None else None
        def closest_obstacle(pos: np.ndarray):
            best = None
            for (idx, radius, obs_traj), _artist in zip(obs_trajs, obs_artists):
                obs = obs_traj[k]
                c = clearance(
                    pos,
                    obs,
                    obstacle_radius=radius,
                    drone_radius=drone_radius,
                )
                if best is None or c < best[0]:
                    best = (c, idx, radius, obs)
            assert best is not None
            return best

        fail_line.set_data(
            failed_true[fd, k0 : failed_k + 1, 0],
            failed_true[fd, k0 : failed_k + 1, 1],
        )
        avoid_line.set_data(
            avoid_true[fd, k0 : avoid_k + 1, 0],
            avoid_true[fd, k0 : avoid_k + 1, 1],
        )
        if ghost_true is not None and ghost_line is not None and ghost_k is not None:
            ghost_line.set_data(
                ghost_true[fd, k0 : ghost_k + 1, 0],
                ghost_true[fd, k0 : ghost_k + 1, 1],
            )
        ref_line.set_data(
            avoid_ref[fd, k0 : avoid_k + 1, 0],
            avoid_ref[fd, k0 : avoid_k + 1, 1],
        )
        for (_idx, _radius, obs_traj), artist in zip(obs_trajs, obs_artists):
            artist["past"].set_data(obs_traj[k0 : k + 1, 0], obs_traj[k0 : k + 1, 1])
            artist["future"].set_data(obs_traj[k : k1 + 1, 0], obs_traj[k : k1 + 1, 1])
            obs = obs_traj[k]
            obs_xy = (float(obs[0]), float(obs[1]))
            artist["halo"].center = obs_xy
            artist["point"].center = obs_xy
        fail_pos = failed_true[fd, failed_k, :]
        avoid_pos = avoid_true[fd, avoid_k, :]
        fail_c, fail_obs_idx, _fail_radius, fail_obs = closest_obstacle(fail_pos)
        avoid_c, avoid_obs_idx, _avoid_radius, avoid_obs = closest_obstacle(avoid_pos)
        fail_pt.center = (float(fail_pos[0]), float(fail_pos[1]))
        avoid_pt.center = (float(avoid_pos[0]), float(avoid_pos[1]))
        fail_clear.set_data([fail_pos[0], fail_obs[0]], [fail_pos[1], fail_obs[1]])
        avoid_clear.set_data([avoid_pos[0], avoid_obs[0]], [avoid_pos[1], avoid_obs[1]])
        ghost_c = None
        ghost_obs_idx = None
        if ghost_true is not None and ghost_pt is not None and ghost_clear is not None and ghost_k is not None:
            ghost_pos = ghost_true[fd, ghost_k, :]
            ghost_c, ghost_obs_idx, _ghost_radius, ghost_obs = closest_obstacle(ghost_pos)
            ghost_pt.center = (float(ghost_pos[0]), float(ghost_pos[1]))
            ghost_clear.set_data([ghost_pos[0], ghost_obs[0]], [ghost_pos[1], ghost_obs[1]])
        failed_collided = failed_hit_step <= k
        avoid_collided = avoid_hit_step <= k
        if failed_collided:
            fail_pt.set_facecolor("#ffffff")
            fail_pt.set_edgecolor(FAIL_COLOR)
            fail_label_now = f"red contact @ {failed_hit_step * args.dt:.2f}s"
        else:
            fail_pt.set_facecolor(FAIL_COLOR)
            fail_pt.set_edgecolor("#ffffff")
            fail_label_now = f"red min clearance {fail_c:+.2f} m (obs {fail_obs_idx})"
        if avoid_collided:
            avoid_label_now = f"green contact @ {avoid_hit_step * args.dt:.2f}s"
            avoid_pt.set_facecolor("#ffffff")
            avoid_pt.set_edgecolor(AVOID_COLOR)
        else:
            avoid_label_now = f"green min clearance {avoid_c:+.2f} m (obs {avoid_obs_idx})"
            avoid_pt.set_facecolor(AVOID_COLOR)
            avoid_pt.set_edgecolor("#ffffff")

        ghost_label_now = (
            f"gray virtual min clearance {ghost_c:+.2f} m (obs {ghost_obs_idx})"
            if ghost_c is not None
            else None
        )
        status_text.set_text(
            "\n".join(
                row for row in [fail_label_now, avoid_label_now, ghost_label_now] if row
            )
        )
        title_text.set_text(f"{args.title}   t = {k * args.dt:.2f} s")
        artists = [
            fail_line,
            avoid_line,
            ref_line,
            fail_pt,
            avoid_pt,
            fail_clear,
            avoid_clear,
            title_text,
            status_text,
            legend_text,
        ]
        for row in obs_artists:
            artists.extend([row["past"], row["future"], row["halo"], row["point"]])
        for artist in [ghost_line, ghost_pt, ghost_clear]:
            if artist is not None:
                artists.append(artist)
        return artists

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
