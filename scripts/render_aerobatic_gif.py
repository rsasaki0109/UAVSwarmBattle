"""Render a side-by-side 3D GIF of the aerobatic synchronized-loop
comparison (MPC vs GPU MPPI).

Loads `episode_NNN_drone_*.json` from two run directories, plots the
reference loop trajectory as a thin grey line, and animates the 4
drones around it. Left subplot = MPC, right subplot = GPU MPPI.

Usage:
    python3 scripts/render_aerobatic_gif.py \
        results/aerobatic_loop4_mpc \
        results/aerobatic_loop4_gpu_mppi \
        docs/images/compare_aerobatic_loop4.gif \
        [--ep 0] [--fps 20] [--stride 2]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d' projection)
import numpy as np

from uav_nav_lab.viz.episode_gif import DRONE_COLORS, load_drones, trajectory_arrays


def setup_axis(ax, title: str, center: np.ndarray, radius: float):
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    margin = radius + 2.0
    ax.set_xlim(center[0] - margin, center[0] + margin)
    ax.set_ylim(center[1] - margin, center[1] + margin)
    ax.set_zlim(center[2] - margin, center[2] + margin)
    ax.view_init(elev=12, azim=-60)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mpc_dir")
    ap.add_argument("mppi_dir")
    ap.add_argument("out_path")
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--stride", type=int, default=2,
                    help="step stride (1 = every step, 2 = every 2nd step, …)")
    ap.add_argument("--trail", type=int, default=30,
                    help="number of past steps to show as fading trail")
    args = ap.parse_args()

    mpc_drones = load_drones(Path(args.mpc_dir), args.ep)
    mppi_drones = load_drones(Path(args.mppi_dir), args.ep)
    true_mpc, ref_mpc, _ = trajectory_arrays(mpc_drones, fit="min")
    true_mppi, ref_mppi, _ = trajectory_arrays(mppi_drones, fit="min")

    # Extract loop center & radius from the reference trajectory itself
    # (all drones share the same loop center)
    center = ref_mpc[0].mean(axis=0)
    rel = ref_mpc[0] - center
    radius = float(np.linalg.norm(rel, axis=1).mean())

    # Build a dense reference loop polyline once (same for both subplots)
    n_pts = 200
    theta = np.linspace(0, 2 * np.pi, n_pts)
    # The reference is a vertical loop in the xz plane at y=center_y
    loop_x = center[0] + radius * np.cos(theta)
    loop_y = np.full_like(theta, center[1])
    loop_z = center[2] + radius * np.sin(theta)

    fig = plt.figure(figsize=(12, 6))
    ax_mpc = fig.add_subplot(1, 2, 1, projection="3d")
    ax_mppi = fig.add_subplot(1, 2, 2, projection="3d")
    for ax in (ax_mpc, ax_mppi):
        setup_axis(ax, "", center, radius)
        ax.plot(loop_x, loop_y, loop_z, color="#888888", linewidth=0.7, alpha=0.6)

    # Animated artists
    trail_lines_mpc, drone_pts_mpc = [], []
    trail_lines_mppi, drone_pts_mppi = [], []
    for i in range(4):
        c = DRONE_COLORS[i]
        ln, = ax_mpc.plot([], [], [], color=c, linewidth=1.5, alpha=0.8)
        pt, = ax_mpc.plot([], [], [], "o", color=c, markersize=8)
        trail_lines_mpc.append(ln)
        drone_pts_mpc.append(pt)
        ln, = ax_mppi.plot([], [], [], color=c, linewidth=1.5, alpha=0.8)
        pt, = ax_mppi.plot([], [], [], "o", color=c, markersize=8)
        trail_lines_mppi.append(ln)
        drone_pts_mppi.append(pt)

    title_text = fig.suptitle("", fontsize=13)

    T = min(true_mpc.shape[1], true_mppi.shape[1])
    frames = list(range(0, T, args.stride))

    def update(frame_idx: int):
        k = frame_idx
        for i in range(4):
            k0 = max(0, k - args.trail)
            trail_lines_mpc[i].set_data(
                true_mpc[i, k0:k+1, 0], true_mpc[i, k0:k+1, 1])
            trail_lines_mpc[i].set_3d_properties(true_mpc[i, k0:k+1, 2])
            drone_pts_mpc[i].set_data(
                true_mpc[i, k:k+1, 0], true_mpc[i, k:k+1, 1])
            drone_pts_mpc[i].set_3d_properties(true_mpc[i, k:k+1, 2])
            trail_lines_mppi[i].set_data(
                true_mppi[i, k0:k+1, 0], true_mppi[i, k0:k+1, 1])
            trail_lines_mppi[i].set_3d_properties(true_mppi[i, k0:k+1, 2])
            drone_pts_mppi[i].set_data(
                true_mppi[i, k:k+1, 0], true_mppi[i, k:k+1, 1])
            drone_pts_mppi[i].set_3d_properties(true_mppi[i, k:k+1, 2])
        # Per-frame tracking error labels (instantaneous)
        err_mpc = np.linalg.norm(true_mpc[:, k, :] - ref_mpc[:, k, :], axis=1).mean()
        err_mppi = np.linalg.norm(true_mppi[:, k, :] - ref_mppi[:, k, :], axis=1).mean()
        ax_mpc.set_title(f"MPC  (track err {err_mpc:.2f} m)", fontsize=12)
        ax_mppi.set_title(f"GPU MPPI  (track err {err_mppi:.2f} m)", fontsize=12)
        t_s = (k * 0.05)  # dt assumed 0.05
        title_text.set_text(
            f"Aerobatic synchronized loop (4 drones, phase-offset 90°)   t = {t_s:.2f} s"
        )
        artists = trail_lines_mpc + drone_pts_mpc + trail_lines_mppi + drone_pts_mppi
        return artists + [title_text]

    anim = FuncAnimation(
        fig, update, frames=frames, interval=1000 // args.fps, blit=False
    )
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=args.fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"wrote {out_path}  ({len(frames)} frames @ {args.fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
