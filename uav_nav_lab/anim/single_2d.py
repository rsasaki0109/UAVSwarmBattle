"""Single-drone 2D episode → animated GIF.

Re-walks the scenario step by step (so dynamic obstacles update in
sync) and overlays the recorded trajectory + sensor visibility circle.
"""

from __future__ import annotations

from typing import Any

from ..config import ExperimentConfig
from ._common import frame_indices_for_episode


def animate_episode_2d(plt, animation, cfg: ExperimentConfig, ep: dict, scenario, fps: int) -> Any:
    import numpy as np

    res = scenario.resolution
    nx, ny = scenario.occupancy.shape
    steps = ep["steps"]
    if not steps:
        return None

    dt = float(cfg.simulator.get("dt", 0.05))
    frame_indices = frame_indices_for_episode(len(steps), dt, fps)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, nx * res)
    ax.set_ylim(0, ny * res)
    ax.set_aspect("equal")

    ax.imshow(
        scenario._static_occ.T,  # always show static layer behind everything
        origin="lower",
        extent=(0, nx * res, 0, ny * res),
        cmap="Greys",
        alpha=0.5,
        interpolation="nearest",
    )
    ax.plot(*scenario.start, "go", ms=10, label="start", zorder=4)
    ax.plot(*scenario.goal, "r*", ms=14, label="goal", zorder=4)

    (traj_line,) = ax.plot([], [], "-", color="tab:blue", lw=1.5, label="true", zorder=3)
    (drone_pt,) = ax.plot([], [], "o", color="tab:blue", ms=8, zorder=5)
    dyn_scatter = ax.scatter([], [], s=120, c="tab:red", marker="o",
                             edgecolors="black", zorder=4, label="dynamic")

    sensor_cfg = cfg.sensor
    sensor_range = float(sensor_cfg.get("range", sensor_cfg.get("range_m", 0.0)))
    sensor_circle = None
    if sensor_range > 0 and sensor_cfg.get("type") == "lidar":
        # draw a circle representing visibility around the drone
        from matplotlib.patches import Circle
        sensor_circle = Circle((0, 0), sensor_range, fill=False, color="tab:cyan",
                               lw=1.0, alpha=0.6, zorder=2)
        ax.add_patch(sensor_circle)

    title = ax.set_title("")
    ax.legend(loc="lower right", fontsize=8)

    # We re-step the scenario to recover dynamic obstacle positions per frame.
    scenario.reseed(ep["meta"]["seed"])
    sim_time_at_step: dict[int, float] = {i: i * dt for i in range(len(steps))}

    def update(idx_in_frames: int):
        i = frame_indices[idx_in_frames]
        # walk scenario forward to step i
        # (re-using the same scenario object across frames; reset on first frame)
        if idx_in_frames == 0:
            scenario.reseed(ep["meta"]["seed"])
            scenario._steps_advanced = 0  # bookkeeping
        target = i
        cur = getattr(scenario, "_steps_advanced", 0)
        for k in range(cur, target):
            # feed recorded drone position so a pursuing obstacle reproduces
            scenario.set_targets([steps[min(k, len(steps) - 1)]["true_pos"]])
            scenario.advance(dt)
        scenario._steps_advanced = target

        true_pos = [(s["true_pos"][0], s["true_pos"][1]) for s in steps[: i + 1]]
        if true_pos:
            tx, ty = zip(*true_pos)
            traj_line.set_data(tx, ty)
            drone_pt.set_data([tx[-1]], [ty[-1]])

        dyn = scenario.dynamic_obstacles
        if dyn:
            xs = [d["position"][0] for d in dyn]
            ys = [d["position"][1] for d in dyn]
            dyn_scatter.set_offsets(np.column_stack([xs, ys]))
        else:
            dyn_scatter.set_offsets(np.zeros((0, 2)))

        if sensor_circle is not None and steps:
            sensor_circle.center = (steps[i]["true_pos"][0], steps[i]["true_pos"][1])

        outcome = ep.get("outcome", "?")
        title.set_text(
            f"ep {ep['meta']['episode']:03d}  outcome={outcome}  "
            f"t={sim_time_at_step[i]:.2f}s"
        )
        return traj_line, drone_pt, dyn_scatter, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices), interval=1000 / fps, blit=False
    )
    return fig, anim
