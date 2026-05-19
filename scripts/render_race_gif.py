"""Render a side-by-side top-down GIF of the drone-race oval circuit
with a bouncing dynamic obstacle, comparing 2 or 3 planners.

Loads `episode_NNN_drone_*.json` from each run directory, computes the
oval reference polyline, animates the 4 drones around it, and overlays
the dynamic obstacle (analytically recomputed from CLI params since it
is not logged per-step).

Usage (2-pane):
    python3 scripts/render_race_gif.py \\
        --runs results/race_oval4_mpc:MPC \\
               results/race_oval4_gpu_mppi:GPU\\ MPPI \\
        --out docs/images/compare_race_oval4.gif

Usage (3-pane):
    python3 scripts/render_race_gif.py \\
        --runs results/race_oval4_mpc:MPC \\
               results/race_oval4_gpu_mppi:GPU\\ MPPI \\
               results/race_oval4_gpu_mppi_smart_v4:Smart\\ MPPI\\ v4 \\
        --out docs/images/compare_race_oval4.gif
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers '3d' projection)
import numpy as np


DRONE_COLORS = ["#e8443b", "#3aa54a", "#3865bf", "#d49b1c"]
OBSTACLE_COLOR = "#cc1f1f"


def load_drones(run_dir: Path, ep: int, n_drones: int = 4) -> list[dict]:
    drones = []
    for i in range(n_drones):
        p = run_dir / f"episode_{ep:03d}_drone_{i:02d}.json"
        drones.append(json.loads(p.read_text()))
    return drones


def trajectory_arrays(drones: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (true_pos[D,T,3], ref_pos[D,T,3]) arrays."""
    D = len(drones)
    T = min(len(d["steps"]) for d in drones)
    true_pos = np.zeros((D, T, 3))
    ref_pos = np.zeros((D, T, 3))
    for i, d in enumerate(drones):
        for k in range(T):
            s = d["steps"][k]
            true_pos[i, k] = s["true_pos"]
            ref_pos[i, k] = s.get("reference_pos", s["true_pos"])
    return true_pos, ref_pos


def obstacle_trajectory(
    start: np.ndarray,
    velocity: np.ndarray,
    dt: float,
    n_steps: int,
    world_size: np.ndarray,
) -> np.ndarray:
    """Recompute the dynamic obstacle trajectory analytically — the
    runner does not log it per-step. Mirrors `_DynamicObstacle3D.step`."""
    traj = np.zeros((n_steps, 3))
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


def oval_polyline(
    center: np.ndarray, radius_x: float, radius_y: float, n_pts: int = 200
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0, 2 * np.pi, n_pts)
    x = center[0] + radius_x * np.cos(theta)
    y = center[1] + radius_y * np.sin(theta)
    z = np.full_like(theta, center[2])
    return x, y, z


def setup_axis(ax, world: np.ndarray, center: np.ndarray):
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # Hide z axis labels — pure top-down view
    ax.set_zticklabels([])
    ax.set_zlabel("")
    ax.set_xlim(0, world[0])
    ax.set_ylim(0, world[1])
    ax.set_zlim(0, world[2])
    # Near-pure top-down view: very high elevation, neutral azimuth.
    # elev=90 is degenerate in matplotlib 3D so we use 89.
    ax.view_init(elev=89, azim=-90)


def parse_run(s: str) -> tuple[Path, str]:
    """Parse `path:label` (label optional)."""
    if ":" in s:
        path_s, label = s.rsplit(":", 1)
    else:
        path_s, label = s, Path(s).name
    return Path(path_s), label


def episode_outcomes(drones: list[dict]) -> list[str]:
    return [d.get("outcome", "?") for d in drones]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="One or more `path:label` (label optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--trail", type=int, default=40)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--obstacle-start", type=float, nargs=3, default=[20.0, 8.0, 7.0])
    ap.add_argument("--obstacle-vel", type=float, nargs=3, default=[0.0, 6.0, 0.0])
    ap.add_argument("--obstacle-radius", type=float, default=1.2)
    ap.add_argument("--world", type=float, nargs=3, default=[40.0, 40.0, 14.0])
    args = ap.parse_args()

    runs = [parse_run(r) for r in args.runs]
    n_panes = len(runs)
    if n_panes < 2 or n_panes > 4:
        raise SystemExit("--runs must list 2..4 entries")

    all_drones: list[list[dict]] = []
    true_arr: list[np.ndarray] = []
    ref_arr: list[np.ndarray] = []
    for run_dir, _ in runs:
        drones = load_drones(run_dir, args.ep)
        all_drones.append(drones)
        true_p, ref_p = trajectory_arrays(drones)
        true_arr.append(true_p)
        ref_arr.append(ref_p)

    # Recover oval geometry from the first run's reference trajectory
    ref0 = ref_arr[0][0]
    center = np.array(
        [float(ref0[:, 0].mean()), float(ref0[:, 1].mean()), float(ref0[:, 2].mean())]
    )
    rx = float((ref0[:, 0].max() - ref0[:, 0].min()) / 2.0)
    ry = float((ref0[:, 1].max() - ref0[:, 1].min()) / 2.0)
    world = np.asarray(args.world, dtype=float)
    ox, oy, oz = oval_polyline(center, rx, ry)

    # Dynamic obstacle trajectory (analytical)
    T_max = min(t.shape[1] for t in true_arr)
    obs_traj = obstacle_trajectory(
        np.asarray(args.obstacle_start, dtype=float),
        np.asarray(args.obstacle_vel, dtype=float),
        args.dt,
        T_max,
        world,
    )

    fig = plt.figure(figsize=(6 * n_panes + 1, 6))
    axes: list = []
    trail_lines: list[list] = []
    drone_pts: list[list] = []
    obs_trail_artists: list = []
    obs_pt_artists: list = []
    for pane in range(n_panes):
        ax = fig.add_subplot(1, n_panes, pane + 1, projection="3d")
        setup_axis(ax, world, center)
        ax.plot(ox, oy, oz, color="#666666", linewidth=1.0, alpha=0.7,
                linestyle="--")
        axes.append(ax)
        pane_trails: list = []
        pane_pts: list = []
        for i in range(4):
            c = DRONE_COLORS[i]
            ln, = ax.plot([], [], [], color=c, linewidth=1.6, alpha=0.85)
            pt, = ax.plot([], [], [], "o", color=c, markersize=7)
            pane_trails.append(ln)
            pane_pts.append(pt)
        trail_lines.append(pane_trails)
        drone_pts.append(pane_pts)
        ot, = ax.plot([], [], [], color=OBSTACLE_COLOR, linewidth=0.8, alpha=0.4)
        op, = ax.plot([], [], [], "o", color=OBSTACLE_COLOR,
                      markersize=14, markeredgecolor="black", markeredgewidth=0.7)
        obs_trail_artists.append(ot)
        obs_pt_artists.append(op)

    title_text = fig.suptitle("", fontsize=13)
    frames = list(range(0, T_max, args.stride))

    def update(k: int):
        k0 = max(0, k - args.trail)
        artists: list = []
        for pane, (_, label) in enumerate(runs):
            tp = true_arr[pane]
            for i in range(4):
                trail_lines[pane][i].set_data(tp[i, k0:k+1, 0], tp[i, k0:k+1, 1])
                trail_lines[pane][i].set_3d_properties(tp[i, k0:k+1, 2])
                drone_pts[pane][i].set_data(tp[i, k:k+1, 0], tp[i, k:k+1, 1])
                drone_pts[pane][i].set_3d_properties(tp[i, k:k+1, 2])
            obs_trail_artists[pane].set_data(obs_traj[k0:k+1, 0], obs_traj[k0:k+1, 1])
            obs_trail_artists[pane].set_3d_properties(obs_traj[k0:k+1, 2])
            obs_pt_artists[pane].set_data(obs_traj[k:k+1, 0], obs_traj[k:k+1, 1])
            obs_pt_artists[pane].set_3d_properties(obs_traj[k:k+1, 2])
            err = np.linalg.norm(tp[:, k, :] - ref_arr[pane][:, k, :], axis=1).mean()
            # Episode-level outcome (final): count collisions among the 4 drones.
            outcomes = episode_outcomes(all_drones[pane])
            n_coll = sum(1 for o in outcomes if o == "collision")
            axes[pane].set_title(
                f"{label}   track err {err:.2f} m   final coll {n_coll}/4",
                fontsize=12,
            )
            artists.extend(trail_lines[pane])
            artists.extend(drone_pts[pane])
            artists.append(obs_trail_artists[pane])
            artists.append(obs_pt_artists[pane])
        t_s = k * args.dt
        title_text.set_text(
            f"Drone race (4 drones, oval circuit) + bouncing intruder   t = {t_s:.2f} s"
        )
        artists.append(title_text)
        return artists

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // args.fps, blit=False)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=args.fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"wrote {out_path}  ({len(frames)} frames @ {args.fps} fps, {n_panes} panes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
