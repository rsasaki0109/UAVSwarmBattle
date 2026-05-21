"""Multi-drone 2D episode → animated GIF (one pane, N coloured drones)."""

from __future__ import annotations

from typing import Any

from ..config import ExperimentConfig
from ._common import PALETTE, frame_indices_for_episode


def animate_episode_multi_2d(
    plt, animation, cfg: ExperimentConfig, drones_eps: list[dict], scenario, fps: int
) -> Any:
    """Render all N drones from a single multi-drone 2D episode in one GIF.

    Mirrors the single-drone 2D animator but draws every drone's
    trajectory + current position with a per-drone palette colour.
    Title reports the joint outcome and per-drone outcomes — same
    convention as the static PNG renderer in
    ``viz._render_episode_multi_2d``.
    """
    import numpy as np

    res = scenario.resolution
    nx, ny = scenario.occupancy.shape
    drones_eps = sorted(drones_eps, key=lambda e: e["meta"].get("drone_id", 0))
    if not drones_eps or not drones_eps[0]["steps"]:
        return None

    # All drones share the same step grid (multi-runner ticks once per global
    # step); use drone 0 to anchor the timeline.
    n_steps = max(len(e["steps"]) for e in drones_eps)
    dt = float(cfg.simulator.get("dt", 0.05))
    frame_indices = frame_indices_for_episode(n_steps, dt, fps)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, nx * res)
    ax.set_ylim(0, ny * res)
    ax.set_aspect("equal")
    ax.imshow(
        scenario._static_occ.T,
        origin="lower",
        extent=(0, nx * res, 0, ny * res),
        cmap="Greys",
        alpha=0.5,
        interpolation="nearest",
    )

    traj_lines: list[Any] = []
    drone_pts: list[Any] = []
    for ep in drones_eps:
        i = int(ep["meta"].get("drone_id", 0))
        color = PALETTE[i % len(PALETTE)]
        name = ep["meta"].get("drone_name", f"d{i}")
        outcome = ep.get("outcome", "?")
        (line,) = ax.plot([], [], "-", color=color, lw=1.3,
                          label=f"{name} ({outcome})", zorder=3)
        (pt,) = ax.plot([], [], "o", color=color, ms=8, zorder=5,
                        mec="black", mew=0.5)
        traj_lines.append(line)
        drone_pts.append(pt)
        if i < len(scenario.drones):
            d = scenario.drones[i]
            ax.plot(d.start[0], d.start[1], "o", color=color, ms=10,
                    mec="black", mew=0.5, zorder=4)
            ax.plot(d.goal[0], d.goal[1], "*", color=color, ms=14,
                    mec="black", mew=0.5, zorder=4)

    dyn_scatter = ax.scatter([], [], s=120, c="dimgray", marker="o",
                             edgecolors="black", zorder=4, label="dynamic")
    title = ax.set_title("")
    ax.legend(loc="lower right", fontsize=7)

    scenario.reseed(drones_eps[0]["meta"]["seed"])
    sim_time_at_step: dict[int, float] = {i: i * dt for i in range(n_steps)}

    def update(idx_in_frames: int):
        i = frame_indices[idx_in_frames]
        if idx_in_frames == 0:
            scenario.reseed(drones_eps[0]["meta"]["seed"])
            scenario._steps_advanced = 0
        target = i
        cur = getattr(scenario, "_steps_advanced", 0)
        for _ in range(cur, target):
            scenario.advance(dt)
        scenario._steps_advanced = target

        for ep, line, pt in zip(drones_eps, traj_lines, drone_pts):
            steps = ep["steps"]
            j = min(i, len(steps) - 1)
            pts = [(s["true_pos"][0], s["true_pos"][1]) for s in steps[: j + 1]]
            if pts:
                tx, ty = zip(*pts)
                line.set_data(tx, ty)
                pt.set_data([tx[-1]], [ty[-1]])

        dyn = scenario.dynamic_obstacles
        if dyn:
            xs = [d["position"][0] for d in dyn]
            ys = [d["position"][1] for d in dyn]
            dyn_scatter.set_offsets(np.column_stack([xs, ys]))
        else:
            dyn_scatter.set_offsets(np.zeros((0, 2)))

        outcomes = [e.get("outcome", "?") for e in drones_eps]
        joint = "all_success" if all(o == "success" for o in outcomes) else "mixed"
        title.set_text(
            f"ep {drones_eps[0]['meta']['episode']:03d}  joint={joint}  "
            f"t={sim_time_at_step[i]:.2f}s"
        )
        return (*traj_lines, *drone_pts, dyn_scatter, title)

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices), interval=1000 / fps, blit=False
    )
    return fig, anim
