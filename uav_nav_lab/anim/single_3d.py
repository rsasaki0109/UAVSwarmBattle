"""Single-drone 3D episode → animated GIF with rotating camera + MPPI rollouts."""

from __future__ import annotations

from typing import Any

from ..config import ExperimentConfig
from ._common import (
    dynamic_obstacle_positions_at,
    frame_indices_for_episode,
    replan_at_or_before,
)


def animate_episode_3d(plt, animation, cfg: ExperimentConfig, ep: dict, scenario, fps: int) -> Any:
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers projection

    res = scenario.resolution
    nx, ny, nz = scenario.occupancy.shape
    steps = ep["steps"]
    if not steps:
        return None

    dt = float(cfg.simulator.get("dt", 0.05))
    frame_indices = frame_indices_for_episode(len(steps), dt, fps)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0, nx * res)
    ax.set_ylim(0, ny * res)
    ax.set_zlim(0, nz * res)

    # Static obstacles: scatter once (cheap vs re-drawing each frame).
    static_occ = getattr(scenario, "_static_occ", scenario.occupancy)
    ix, iy, iz = np.where(static_occ)
    if ix.size > 0:
        ax.scatter(
            (ix + 0.5) * res,
            (iy + 0.5) * res,
            (iz + 0.5) * res,
            c="gray", alpha=0.25, s=18, marker="s",
        )
    ax.scatter(*scenario.start, c="green", s=80, label="start")
    ax.scatter(*scenario.goal, c="red", marker="*", s=160, label="goal")

    (traj_line,) = ax.plot([], [], [], "-", color="tab:blue", lw=1.5, label="true")
    drone_pt = ax.scatter([], [], [], c="tab:blue", s=60, depthshade=True)
    dyn_scatter = ax.scatter([], [], [], s=120, c="tab:red", marker="o", edgecolors="black")

    # MPPI rollout overlay: prepare a fixed pool of translucent lines so we
    # can update geometry per frame without re-creating artists. Pool size
    # follows the largest replan in the log; replans without `rollouts`
    # (non-sampling planners) hide all lines for that frame.
    replans = ep.get("replans", []) or []
    rollouts_max = max(
        (len(r.get("rollouts") or []) for r in replans),
        default=0,
    )
    rollout_lines = [
        ax.plot([], [], [], "-", color="tab:cyan", lw=0.6, alpha=0.30, zorder=2)[0]
        for _ in range(rollouts_max)
    ]
    (best_rollout_line,) = ax.plot(
        [], [], [], "-", color="tab:orange", lw=1.2, alpha=0.85, zorder=3,
        label="MPPI best" if rollouts_max else None,
    )

    title = ax.set_title("")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left", fontsize=8)

    # Re-derive dynamic obstacle trajectories from the replay config so we
    # show their actual positions at each frame (rather than the stale
    # snapshots saved in step records, which only have what the *sensor*
    # reported).
    dyn_specs = list(cfg.scenario.get("dynamic_obstacles", []) or [])
    bounds = (nx * res, ny * res, nz * res)

    def update(frame_i: int) -> tuple[Any, ...]:
        j = frame_indices[frame_i]
        # Trajectory up to j.
        tx = [steps[k]["true_pos"][0] for k in range(j + 1)]
        ty = [steps[k]["true_pos"][1] for k in range(j + 1)]
        tz = [steps[k]["true_pos"][2] for k in range(j + 1)]
        traj_line.set_data(tx, ty)
        traj_line.set_3d_properties(tz)
        drone_pt._offsets3d = ([tx[-1]], [ty[-1]], [tz[-1]])
        dx, dy, dz = dynamic_obstacle_positions_at(j, dyn_specs, dt, bounds)
        dyn_scatter._offsets3d = (dx, dy, dz)

        # Refresh rollout overlay from the most recent replan with rollouts.
        cur_replan = replan_at_or_before(replans, steps[j]["t"]) if rollout_lines else None
        rolls = (cur_replan.get("rollouts") if cur_replan else None) or []
        best_idx = int(cur_replan.get("best_rollout_idx", -1)) if cur_replan else -1
        for k, line in enumerate(rollout_lines):
            if k < len(rolls):
                pts = rolls[k]
                line.set_data([p[0] for p in pts], [p[1] for p in pts])
                line.set_3d_properties([p[2] for p in pts])
            else:
                line.set_data([], [])
                line.set_3d_properties([])
        if 0 <= best_idx < len(rolls):
            pts = rolls[best_idx]
            best_rollout_line.set_data([p[0] for p in pts], [p[1] for p in pts])
            best_rollout_line.set_3d_properties([p[2] for p in pts])
        else:
            best_rollout_line.set_data([], [])
            best_rollout_line.set_3d_properties([])

        # Slow rotating view, +120° over the episode for a sense of depth.
        ax.view_init(elev=22.0, azim=-60.0 + (frame_i / max(1, len(frame_indices) - 1)) * 120.0)
        title.set_text(
            f"ep {ep['meta']['episode']:03d}  outcome={ep.get('outcome','?')}  "
            f"t={steps[j]['t']:.1f}s"
        )
        return (traj_line, drone_pt, dyn_scatter, best_rollout_line, *rollout_lines, title)

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices), interval=1000 / fps, blit=False
    )
    return fig, anim
