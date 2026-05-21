"""Multi-drone 3D episode → animated GIF (rotating camera, per-drone rollouts)."""

from __future__ import annotations

from typing import Any

from ..config import ExperimentConfig
from ._common import (
    PALETTE,
    dynamic_obstacle_positions_at,
    frame_indices_for_episode,
    replan_at_or_before,
)


def animate_episode_multi_3d(
    plt, animation, cfg: ExperimentConfig, drones_eps: list[dict], scenario, fps: int
) -> Any:
    """Render all N drones from a single multi-drone 3D episode in one GIF.

    Mirrors the single-drone 3D animator (rotating view, static voxels
    as a faint scatter) but draws every drone's trajectory + dot in a
    per-drone palette colour and scatters per-drone start / goal
    markers — same convention as :func:`animate_episode_multi_2d`.
    """
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers projection

    res = scenario.resolution
    nx, ny, nz = scenario.occupancy.shape
    drones_eps = sorted(drones_eps, key=lambda e: e["meta"].get("drone_id", 0))
    if not drones_eps or not drones_eps[0]["steps"]:
        return None

    n_steps = max(len(e["steps"]) for e in drones_eps)
    dt = float(cfg.simulator.get("dt", 0.05))
    frame_indices = frame_indices_for_episode(n_steps, dt, fps)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0, nx * res)
    ax.set_ylim(0, ny * res)
    ax.set_zlim(0, nz * res)

    static_occ = getattr(scenario, "_static_occ", scenario.occupancy)
    ix, iy, iz = np.where(static_occ)
    if ix.size > 0:
        ax.scatter(
            (ix + 0.5) * res,
            (iy + 0.5) * res,
            (iz + 0.5) * res,
            c="gray", alpha=0.20, s=14, marker="s",
        )

    traj_lines: list[Any] = []
    drone_pts: list[Any] = []
    rollout_pools: list[list[Any]] = []  # per-drone list of rollout artist lines
    best_rollout_lines: list[Any] = []   # per-drone best-rollout artist
    for ep in drones_eps:
        i = int(ep["meta"].get("drone_id", 0))
        color = PALETTE[i % len(PALETTE)]
        name = ep["meta"].get("drone_name", f"d{i}")
        outcome = ep.get("outcome", "?")
        (line,) = ax.plot([], [], [], "-", color=color, lw=1.3,
                          label=f"{name} ({outcome})")
        pt = ax.scatter([], [], [], c=color, s=60, depthshade=True,
                        edgecolors="black", linewidths=0.5)
        traj_lines.append(line)
        drone_pts.append(pt)
        if i < len(scenario.drones):
            d = scenario.drones[i]
            ax.scatter(*d.start, c=color, s=70, marker="o",
                       edgecolors="black", linewidths=0.5)
            ax.scatter(*d.goal, c=color, s=140, marker="*",
                       edgecolors="black", linewidths=0.5)
        # Pre-allocate rollout-overlay artists per-drone, sized by this
        # drone's largest replan. Best rollout gets a slightly thicker,
        # brighter line in the drone's palette colour.
        replans = ep.get("replans", []) or []
        rollouts_max = max(
            (len(r.get("rollouts") or []) for r in replans),
            default=0,
        )
        pool = [
            ax.plot([], [], [], "-", color=color, lw=0.5, alpha=0.18, zorder=2)[0]
            for _ in range(rollouts_max)
        ]
        rollout_pools.append(pool)
        (best_line,) = ax.plot(
            [], [], [], "-", color=color, lw=1.1, alpha=0.85, zorder=3,
        )
        best_rollout_lines.append(best_line)

    title = ax.set_title("")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left", fontsize=7)

    # Dynamic-obstacle replay (same approach as the single-drone 3D anim).
    dyn_specs = list(cfg.scenario.get("dynamic_obstacles", []) or [])
    bounds = (nx * res, ny * res, nz * res)
    dyn_scatter = ax.scatter([], [], [], c="red", s=80, alpha=0.6, edgecolors="black", linewidths=0.5)

    n_frames = len(frame_indices)

    def update(idx_in_frames: int):
        i = frame_indices[idx_in_frames]
        for ep, line, pt, pool, best_line in zip(
            drones_eps, traj_lines, drone_pts, rollout_pools, best_rollout_lines
        ):
            steps = ep["steps"]
            j = min(i, len(steps) - 1)
            tx = [steps[k]["true_pos"][0] for k in range(j + 1)]
            ty = [steps[k]["true_pos"][1] for k in range(j + 1)]
            tz = [steps[k]["true_pos"][2] for k in range(j + 1)]
            line.set_data(tx, ty)
            line.set_3d_properties(tz)
            pt._offsets3d = ([tx[-1]], [ty[-1]], [tz[-1]])

            # Rollout overlay for this drone (if it logged rollouts).
            replans = ep.get("replans", []) or []
            cur_replan = (
                replan_at_or_before(replans, steps[j]["t"]) if pool else None
            )
            rolls = (cur_replan.get("rollouts") if cur_replan else None) or []
            best_idx = (
                int(cur_replan.get("best_rollout_idx", -1)) if cur_replan else -1
            )
            for k, rl_line in enumerate(pool):
                if k < len(rolls):
                    pts = rolls[k]
                    rl_line.set_data([p[0] for p in pts], [p[1] for p in pts])
                    rl_line.set_3d_properties([p[2] for p in pts])
                else:
                    rl_line.set_data([], [])
                    rl_line.set_3d_properties([])
            if 0 <= best_idx < len(rolls):
                pts = rolls[best_idx]
                best_line.set_data([p[0] for p in pts], [p[1] for p in pts])
                best_line.set_3d_properties([p[2] for p in pts])
            else:
                best_line.set_data([], [])
                best_line.set_3d_properties([])

        dx, dy, dz = dynamic_obstacle_positions_at(i, dyn_specs, dt, bounds)
        dyn_scatter._offsets3d = (dx, dy, dz)

        ax.view_init(elev=22.0, azim=-60.0 + (idx_in_frames / max(1, n_frames - 1)) * 120.0)
        outcomes = [e.get("outcome", "?") for e in drones_eps]
        joint = "all_success" if all(o == "success" for o in outcomes) else "mixed"
        title.set_text(
            f"ep {drones_eps[0]['meta']['episode']:03d}  joint={joint}  "
            f"t={i * dt:.2f}s"
        )
        all_rollouts: list[Any] = []
        for pool in rollout_pools:
            all_rollouts.extend(pool)
        return (*traj_lines, *drone_pts, *all_rollouts,
                *best_rollout_lines, dyn_scatter, title)

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=False
    )
    return fig, anim
