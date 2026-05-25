"""Render a README-focused top-down drone-race GIF.

This renderer is deliberately presentation-oriented: it consumes real
multi-drone episode logs, but draws them as a clean 2D race scene with an
oval track, gate markers, moving sweeper obstacles, reference ghosts, and
actual drone trails. It is meant for the first README visual where the
important question is whether the reader can immediately see "drone race",
"dynamic obstacle", and "avoidance".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Ellipse, Rectangle
import numpy as np
import yaml

from uav_nav_lab.viz.episode_gif import (
    DRONE_COLORS,
    load_drones,
    trajectory_arrays,
)


OBSTACLE_COLOR = "#d6261f"
TRACK_FACE = "#e8ecef"
BG_COLOR = "#f8fafc"
INK = "#172033"


def parse_run(raw: str) -> tuple[Path, str]:
    if ":" in raw:
        path_s, label = raw.rsplit(":", 1)
    else:
        path_s, label = raw, Path(raw).name
    return Path(path_s), label


def obstacle_trajectory(
    start: np.ndarray,
    velocity: np.ndarray,
    dt: float,
    n_steps: int,
    world_size: np.ndarray,
) -> np.ndarray:
    traj = np.zeros((n_steps, 3), dtype=float)
    pos = start.astype(float).copy()
    vel = velocity.astype(float).copy()
    for k in range(n_steps):
        traj[k] = pos
        pos = pos + vel * dt
        for i in range(3):
            upper = float(world_size[i])
            if pos[i] < 0:
                pos[i] = -pos[i]
                vel[i] = -vel[i]
            elif pos[i] > upper:
                pos[i] = 2 * upper - pos[i]
                vel[i] = -vel[i]
    return traj


def oval_xy(
    center: np.ndarray, radius_x: float, radius_y: float, n_pts: int = 360
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts)
    return center[0] + radius_x * np.cos(theta), center[1] + radius_y * np.sin(theta)


def draw_checkered_start(ax, x: float, y: float, height: float = 5.0) -> None:
    n = 10
    cell_h = height / n
    width = 0.62
    for i in range(n):
        color = "#111827" if i % 2 == 0 else "#ffffff"
        rect = Rectangle(
            (x - width / 2.0, y - height / 2.0 + i * cell_h),
            width,
            cell_h,
            facecolor=color,
            edgecolor="#111827",
            linewidth=0.25,
            zorder=6,
        )
        ax.add_patch(rect)


def draw_track(ax, center: np.ndarray, rx: float, ry: float, world: np.ndarray) -> None:
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, float(world[0]))
    ax.set_ylim(0, float(world[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#cbd5e1", linewidth=0.55, alpha=0.42)
    ax.tick_params(labelsize=8, colors="#64748b", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    lane_half_width = 2.2
    outer = Ellipse(
        center[:2],
        width=2.0 * (rx + lane_half_width),
        height=2.0 * (ry + lane_half_width),
        facecolor=TRACK_FACE,
        edgecolor=INK,
        linewidth=1.2,
        zorder=1,
    )
    inner = Ellipse(
        center[:2],
        width=max(1.0, 2.0 * (rx - lane_half_width)),
        height=max(1.0, 2.0 * (ry - lane_half_width)),
        facecolor=BG_COLOR,
        edgecolor=INK,
        linewidth=1.0,
        zorder=2,
    )
    ax.add_patch(outer)
    ax.add_patch(inner)

    x, y = oval_xy(center, rx, ry)
    ax.plot(
        x,
        y,
        color="#475569",
        linewidth=1.0,
        linestyle=(0, (4, 4)),
        zorder=3,
    )

    gate_w = 4.8
    gates = [
        ((center[0], center[1] + ry), "h"),
        ((center[0] - rx, center[1]), "v"),
        ((center[0], center[1] - ry), "h"),
    ]
    for (gx, gy), orient in gates:
        if orient == "h":
            ax.plot(
                [gx - gate_w / 2.0, gx + gate_w / 2.0],
                [gy, gy],
                color=INK,
                linewidth=2.4,
                solid_capstyle="round",
                zorder=6,
            )
        else:
            ax.plot(
                [gx, gx],
                [gy - gate_w / 2.0, gy + gate_w / 2.0],
                color=INK,
                linewidth=2.4,
                solid_capstyle="round",
                zorder=6,
            )
    draw_checkered_start(ax, float(center[0] + rx), float(center[1]), height=gate_w)


def pane_outcomes(drones: list[dict]) -> tuple[int, int]:
    outcomes = [d.get("outcome", "?") for d in drones]
    return sum(o == "collision" for o in outcomes), len(outcomes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="`path:label` entries")
    ap.add_argument(
        "--config",
        required=True,
        help="Scenario YAML used for obstacle paths",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--trail", type=int, default=120)
    ap.add_argument("--future", type=int, default=90)
    ap.add_argument("--start-step", type=int, default=0)
    ap.add_argument("--end-step", type=int, default=None)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--n-drones", type=int, default=4)
    ap.add_argument(
        "--title",
        default="Post-fix drone race: moving sweepers force real avoidance",
    )
    args = ap.parse_args()

    runs = [parse_run(raw) for raw in args.runs]
    if len(runs) < 1 or len(runs) > 4:
        raise SystemExit("--runs must list 1..4 entries")

    cfg = yaml.safe_load(Path(args.config).read_text())
    scenario = cfg.get("scenario", {})
    world = np.asarray(scenario.get("size", [40.0, 40.0, 14.0]), dtype=float)
    dynamic_specs = scenario.get("dynamic_obstacles", []) or []
    obstacles = [
        {
            "start": np.asarray(spec["start"], dtype=float),
            "vel": np.asarray(spec["velocity"], dtype=float),
            "radius": float(spec.get("radius", 0.5)),
        }
        for spec in dynamic_specs
    ]
    drone_radius = float(scenario.get("drone_radius", 0.4))

    all_drones: list[list[dict]] = []
    for run_dir, _ in runs:
        all_drones.append(load_drones(run_dir, args.ep, n_drones=args.n_drones))

    t_pad = max(max(len(d["steps"]) for d in pane) for pane in all_drones)
    true_arr: list[np.ndarray] = []
    ref_arr: list[np.ndarray] = []
    coll_step_arr: list[np.ndarray] = []
    for pane in all_drones:
        true_p, ref_p, coll_step = trajectory_arrays(pane, T_pad=t_pad)
        true_arr.append(true_p)
        ref_arr.append(ref_p)
        coll_step_arr.append(coll_step)

    t_max = min(arr.shape[1] for arr in true_arr)
    end_step = min(t_max, args.end_step if args.end_step is not None else t_max)
    start_step = int(np.clip(args.start_step, 0, max(0, end_step - 1)))
    frames = list(range(start_step, end_step, max(1, args.stride)))
    if not frames:
        raise SystemExit("empty frame range")

    obs_trajs = [
        obstacle_trajectory(o["start"], o["vel"], args.dt, t_max, world)
        for o in obstacles
    ]

    ref0 = ref_arr[0][0]
    center = np.array(
        [float(ref0[:, 0].mean()), float(ref0[:, 1].mean()), float(ref0[:, 2].mean())]
    )
    rx = float((ref0[:, 0].max() - ref0[:, 0].min()) / 2.0)
    ry = float((ref0[:, 1].max() - ref0[:, 1].min()) / 2.0)

    n_panes = len(runs)
    fig, axes_raw = plt.subplots(
        1,
        n_panes,
        figsize=(5.8 * n_panes, 5.35),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes = list(axes_raw[0])
    fig.subplots_adjust(
        left=0.035,
        right=0.985,
        bottom=0.07,
        top=0.82,
        wspace=0.08,
    )

    trail_lines: list[list] = []
    ref_lines: list[list] = []
    drone_pts: list[list[Circle]] = []
    ref_pts: list[list[Circle]] = []
    obs_past_lines: list[list] = []
    obs_future_lines: list[list] = []
    obs_pts: list[list[Circle]] = []
    obs_halos: list[list[Circle]] = []
    pane_titles: list = []

    for pane, ax in enumerate(axes):
        draw_track(ax, center, rx, ry, world)
        for otraj in obs_trajs:
            ax.plot(
                otraj[:, 0],
                otraj[:, 1],
                color=OBSTACLE_COLOR,
                linewidth=3.0,
                alpha=0.11,
                zorder=4,
            )

        pane_trails = []
        pane_ref_lines = []
        pane_drone_pts = []
        pane_ref_pts = []
        for i in range(args.n_drones):
            c = DRONE_COLORS[i % len(DRONE_COLORS)]
            ln, = ax.plot([], [], color=c, linewidth=2.6, alpha=0.92, zorder=9)
            ref_ln, = ax.plot([], [], color=c, linewidth=1.2, alpha=0.45, zorder=8)
            ref_pt = Circle(
                (0.0, 0.0),
                0.34,
                facecolor="none",
                edgecolor=c,
                linewidth=1.3,
                alpha=0.58,
                zorder=10,
            )
            pt = Circle(
                (0.0, 0.0),
                drone_radius,
                facecolor=c,
                edgecolor="#ffffff",
                linewidth=1.0,
                alpha=0.98,
                zorder=12,
            )
            ax.add_patch(ref_pt)
            ax.add_patch(pt)
            pane_trails.append(ln)
            pane_ref_lines.append(ref_ln)
            pane_ref_pts.append(ref_pt)
            pane_drone_pts.append(pt)
        trail_lines.append(pane_trails)
        ref_lines.append(pane_ref_lines)
        ref_pts.append(pane_ref_pts)
        drone_pts.append(pane_drone_pts)

        pane_obs_past = []
        pane_obs_future = []
        pane_obs_pts = []
        pane_obs_halos = []
        for o in obstacles:
            past_ln, = ax.plot(
                [],
                [],
                color=OBSTACLE_COLOR,
                linewidth=2.0,
                alpha=0.7,
                zorder=7,
            )
            fut_ln, = ax.plot(
                [],
                [],
                color=OBSTACLE_COLOR,
                linewidth=1.3,
                alpha=0.36,
                linestyle=(0, (3, 3)),
                zorder=7,
            )
            halo = Circle(
                (0.0, 0.0),
                float(o["radius"]) + drone_radius,
                facecolor=OBSTACLE_COLOR,
                edgecolor="none",
                alpha=0.12,
                zorder=5,
            )
            pt = Circle(
                (0.0, 0.0),
                float(o["radius"]),
                facecolor=OBSTACLE_COLOR,
                edgecolor="#111827",
                linewidth=1.0,
                alpha=0.94,
                zorder=11,
            )
            ax.add_patch(halo)
            ax.add_patch(pt)
            pane_obs_past.append(past_ln)
            pane_obs_future.append(fut_ln)
            pane_obs_halos.append(halo)
            pane_obs_pts.append(pt)
        obs_past_lines.append(pane_obs_past)
        obs_future_lines.append(pane_obs_future)
        obs_halos.append(pane_obs_halos)
        obs_pts.append(pane_obs_pts)

        title = ax.set_title("", fontsize=11, color=INK, pad=8)
        pane_titles.append(title)

    title_text = fig.suptitle("", fontsize=13.5, color=INK, fontweight="semibold")
    flash_window = max(2, int(round(0.45 / args.dt)))

    def update(k: int):
        k0 = max(0, k - args.trail)
        k1 = min(t_max - 1, k + args.future)
        t_s = k * args.dt
        artists: list = [title_text]
        for pane, (_, label) in enumerate(runs):
            tp = true_arr[pane]
            rp = ref_arr[pane]
            coll_steps = coll_step_arr[pane]
            live_coll = int((coll_steps < k).sum())
            final_coll, total = pane_outcomes(all_drones[pane])
            mean_err = float(np.linalg.norm(tp[:, k, :] - rp[:, k, :], axis=1).mean())
            pane_titles[pane].set_text(
                f"{label}\n"
                f"err {mean_err:.2f} m  live coll {live_coll}/{total}  "
                f"final {final_coll}/{total}"
            )
            artists.append(pane_titles[pane])

            for i in range(args.n_drones):
                c = DRONE_COLORS[i % len(DRONE_COLORS)]
                trail_lines[pane][i].set_data(
                    tp[i, k0 : k + 1, 0],
                    tp[i, k0 : k + 1, 1],
                )
                ref_lines[pane][i].set_data(
                    [rp[i, k, 0], tp[i, k, 0]],
                    [rp[i, k, 1], tp[i, k, 1]],
                )
                ref_pts[pane][i].center = (float(rp[i, k, 0]), float(rp[i, k, 1]))
                drone_pts[pane][i].center = (float(tp[i, k, 0]), float(tp[i, k, 1]))
                dk = k - int(coll_steps[i])
                if dk < 0:
                    drone_pts[pane][i].set_facecolor(c)
                    drone_pts[pane][i].set_edgecolor("#ffffff")
                    drone_pts[pane][i].set_alpha(0.98)
                elif dk < flash_window:
                    drone_pts[pane][i].set_facecolor("#ffffff")
                    drone_pts[pane][i].set_edgecolor("#111827")
                    drone_pts[pane][i].set_alpha(1.0)
                else:
                    drone_pts[pane][i].set_facecolor("#64748b")
                    drone_pts[pane][i].set_edgecolor("#ffffff")
                    drone_pts[pane][i].set_alpha(0.62)
                artists.extend(
                    [
                        trail_lines[pane][i],
                        ref_lines[pane][i],
                        ref_pts[pane][i],
                        drone_pts[pane][i],
                    ]
                )

            for o_idx, otraj in enumerate(obs_trajs):
                obs_past_lines[pane][o_idx].set_data(
                    otraj[k0 : k + 1, 0],
                    otraj[k0 : k + 1, 1],
                )
                obs_future_lines[pane][o_idx].set_data(
                    otraj[k : k1 + 1, 0],
                    otraj[k : k1 + 1, 1],
                )
                center_xy = (float(otraj[k, 0]), float(otraj[k, 1]))
                obs_pts[pane][o_idx].center = center_xy
                obs_halos[pane][o_idx].center = center_xy
                artists.extend(
                    [
                        obs_past_lines[pane][o_idx],
                        obs_future_lines[pane][o_idx],
                        obs_halos[pane][o_idx],
                        obs_pts[pane][o_idx],
                    ]
                )
        title_text.set_text(f"{args.title}   t = {t_s:.2f} s")
        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 // args.fps,
        blit=False,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"wrote {out_path}  ({len(frames)} frames @ {args.fps} fps, {n_panes} panes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
