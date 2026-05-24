#!/usr/bin/env python3
"""Render a static mechanism trace for the race-simple phase failure.

The figure compares MPC and GPU MPPI around the deterministic
``p19.8, y=5.5/34.5`` split cell. It uses only existing per-drone JSON
logs and reconstructs dynamic-obstacle positions from the saved config.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from analyze_race_simple_phase_trace import (
    _load_json,
    _load_yaml,
    clearance_to_obstacles,
    min_clearance_for_path,
    nearest,
    obstacle_positions,
)


DEFAULT_RUN_ROOT = Path("results/_race_simple_phase_sweep/p19p8_y5p5_34p5")
DEFAULT_OUT = DEFAULT_RUN_ROOT / "mechanism_trace.png"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--drone", type=int, default=3)
    p.add_argument("--replan-time", type=float, default=29.1)
    p.add_argument("--start-time", type=float, default=28.6)
    p.add_argument("--end-time", type=float, default=29.35)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def load_trace(run_root: Path, planner: str, episode: int, drone: int) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = run_root / planner
    cfg = _load_yaml(run_dir / "config.yaml")
    log = _load_json(run_dir / f"episode_{episode:03d}_drone_{drone:02d}.json")
    return cfg, log


def window_steps(log: dict[str, Any], start_t: float, end_t: float) -> list[dict[str, Any]]:
    return [
        step
        for step in log["steps"]
        if start_t <= float(step["t"]) <= end_t
    ]


def xy(points: list[list[float]]) -> tuple[list[float], list[float]]:
    return [float(p[0]) for p in points], [float(p[1]) for p in points]


def step_points(steps: list[dict[str, Any]], key: str) -> list[list[float]]:
    return [[float(v) for v in step[key]] for step in steps]


def first_collision_step(log: dict[str, Any]) -> dict[str, Any] | None:
    for step in log["steps"]:
        if bool(step.get("collision", False)):
            return step
    return None


def min_step_clearance(cfg: dict[str, Any], steps: list[dict[str, Any]]) -> tuple[float, float]:
    best = math.inf
    best_t = math.nan
    dt = float(cfg["simulator"].get("dt", 0.05))
    for step in steps:
        pos = [float(v) for v in step["true_pos"]]
        t = float(step["t"])
        clearances = clearance_to_obstacles(cfg, pos, t)
        next_clearances = clearance_to_obstacles(cfg, pos, t + dt)
        tight = min(min(clearances), min(next_clearances))
        if tight < best:
            best = tight
            best_t = t
    return best, best_t


def rollout_stats(cfg: dict[str, Any], replan: dict[str, Any]) -> tuple[list[float], int, int, float]:
    rollouts = replan.get("rollouts") or []
    clearances: list[float] = []
    hits = 0
    for path in rollouts:
        clearance, hit = min_clearance_for_path(cfg, path, replan_t=float(replan["t"]))
        clearances.append(clearance)
        hits += int(hit)
    best_idx = int(replan.get("best_rollout_idx", 0))
    best_clearance = clearances[best_idx] if 0 <= best_idx < len(clearances) else math.nan
    return clearances, hits, len(rollouts), best_clearance


def plot_obstacle(
    ax: plt.Axes,
    cfg: dict[str, Any],
    *,
    t: float,
    label: str | None,
    color: str,
    alpha: float,
) -> None:
    drone_radius = float(cfg["simulator"].get("drone_radius", 0.4))
    obstacles = cfg["scenario"].get("dynamic_obstacles", []) or []
    positions = obstacle_positions(cfg, t)
    for idx, (obs, pos) in enumerate(zip(obstacles, positions)):
        x, y = float(pos[0]), float(pos[1])
        obs_radius = float(obs.get("radius", 0.5))
        contact_radius = obs_radius + drone_radius
        circle_label = label if idx == 0 else None
        ax.add_patch(
            Circle(
                (x, y),
                contact_radius,
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                linewidth=1.4,
                label=circle_label,
            )
        )
        ax.add_patch(
            Circle(
                (x, y),
                obs_radius,
                facecolor="none",
                edgecolor=color,
                alpha=min(1.0, alpha + 0.25),
                linewidth=1.0,
            )
        )
        if idx == 0:
            ax.text(x + 0.15, y + 0.25, f"t={t:.2f}", fontsize=8, color=color)


def plot_rollouts(
    ax: plt.Axes,
    cfg: dict[str, Any],
    replan: dict[str, Any],
) -> tuple[float, int, int, float]:
    rollouts = replan.get("rollouts") or []
    clearances, hits, count, best_clearance = rollout_stats(cfg, replan)
    best_idx = int(replan.get("best_rollout_idx", 0))
    for idx, path in enumerate(rollouts):
        xs, ys = xy(path)
        if idx == best_idx:
            continue
        color = "#a3a3a3" if clearances[idx] > 0.0 else "#e57373"
        alpha = 0.12 if clearances[idx] > 0.0 else 0.25
        ax.plot(xs, ys, color=color, alpha=alpha, linewidth=0.9, zorder=1)
    if 0 <= best_idx < len(rollouts):
        xs, ys = xy(rollouts[best_idx])
        ax.plot(
            xs,
            ys,
            color="#f59e0b",
            linewidth=2.5,
            linestyle="-.",
            label=f"GPU selected rollout @ {float(replan['t']):.1f}s",
            zorder=5,
        )
        ax.scatter([xs[0]], [ys[0]], color="#f59e0b", s=28, zorder=6)
    min_rollout_clearance = min(clearances) if clearances else math.nan
    return min_rollout_clearance, hits, count, best_clearance


def set_view_limits(
    ax: plt.Axes,
    point_groups: list[list[list[float]]],
    cfg: dict[str, Any],
    start_t: float,
    end_t: float,
) -> None:
    points: list[list[float]] = []
    for group in point_groups:
        points.extend(group)
    focus_points = list(points)
    for t in [start_t, 28.9, 29.1, 29.3, end_t]:
        for obs_pos in obstacle_positions(cfg, t):
            if not focus_points:
                points.append(obs_pos)
                continue
            nearest_trace_dist = min(
                math.hypot(float(obs_pos[0]) - float(p[0]), float(obs_pos[1]) - float(p[1]))
                for p in focus_points
            )
            if nearest_trace_dist <= 8.0:
                points.append(obs_pos)
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    margin = 1.6
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)


def main() -> int:
    args = parse_args()
    mpc_cfg, mpc_log = load_trace(args.run_root, "mpc", args.episode, args.drone)
    gpu_cfg, gpu_log = load_trace(args.run_root, "gpu_mppi", args.episode, args.drone)
    mpc_steps = window_steps(mpc_log, args.start_time, args.end_time)
    gpu_steps = window_steps(gpu_log, args.start_time, args.end_time)
    if not mpc_steps or not gpu_steps:
        raise ValueError("trace window does not overlap both planner logs")

    replan = nearest(gpu_log["replans"], args.replan_time)
    collision = first_collision_step(gpu_log)
    rollouts = replan.get("rollouts") or []
    best_idx = int(replan.get("best_rollout_idx", 0))
    selected_rollout = rollouts[best_idx] if 0 <= best_idx < len(rollouts) else []

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    ax.set_title("Race-simple phase mechanism: GPU MPPI cuts back into a moving obstacle")

    ref_points = step_points(gpu_steps, "reference_pos")
    ref_x, ref_y = xy(ref_points)
    ax.plot(ref_x, ref_y, color="#525252", linestyle=":", linewidth=2.0, label="reference")

    mpc_points = step_points(mpc_steps, "true_pos")
    gpu_points = step_points(gpu_steps, "true_pos")
    ax.plot(*xy(mpc_points), color="#2563eb", linewidth=2.2, label="MPC actual")
    ax.plot(*xy(gpu_points), color="#dc2626", linewidth=2.4, label="GPU MPPI actual")
    ax.scatter([mpc_points[0][0]], [mpc_points[0][1]], color="#2563eb", marker="o", s=30)
    ax.scatter([gpu_points[0][0]], [gpu_points[0][1]], color="#dc2626", marker="o", s=30)

    rollout_min, rollout_hits, rollout_count, best_rollout_clearance = plot_rollouts(ax, gpu_cfg, replan)

    for idx, t in enumerate([28.7, 28.9, 29.1, 29.25, 29.3]):
        plot_obstacle(
            ax,
            gpu_cfg,
            t=t,
            label="dynamic obstacle contact disk" if idx == 0 else None,
            color="#7c3aed",
            alpha=0.10 + 0.035 * idx,
        )

    if collision is not None:
        pos = [float(v) for v in collision["true_pos"]]
        ax.scatter(
            [pos[0]],
            [pos[1]],
            marker="X",
            s=100,
            color="#991b1b",
            label=f"GPU env collision @ {float(collision['t']):.2f}s",
            zorder=7,
        )

    set_view_limits(
        ax,
        [mpc_points, gpu_points, ref_points, selected_rollout],
        gpu_cfg,
        args.start_time,
        args.end_time,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper left", fontsize=8, frameon=True)

    text = (
        f"selected rollout clearance: {best_rollout_clearance:+.2f} m\n"
        f"visible rollout hits: {rollout_hits}/{rollout_count}\n"
        f"GPU window min clearance: {min_step_clearance(gpu_cfg, gpu_steps)[0]:+.2f} m\n"
        f"MPC window min clearance: {min_step_clearance(mpc_cfg, mpc_steps)[0]:+.2f} m"
    )
    ax.text(
        0.99,
        0.02,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d4d4d4", "alpha": 0.92},
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    gpu_min, gpu_min_t = min_step_clearance(gpu_cfg, gpu_steps)
    mpc_min, mpc_min_t = min_step_clearance(mpc_cfg, mpc_steps)
    print(f"wrote {args.out}")
    print(f"gpu_replan_t={float(replan['t']):.2f}")
    print(
        f"visible_rollouts={rollout_count} hits={rollout_hits} "
        f"min_clearance={rollout_min:+.2f} selected_clearance={best_rollout_clearance:+.2f}"
    )
    print(f"gpu_window_min_clearance={gpu_min:+.2f} at t={gpu_min_t:.2f}")
    print(f"mpc_window_min_clearance={mpc_min:+.2f} at t={mpc_min_t:.2f}")
    if collision is not None:
        print(f"gpu_collision_t={float(collision['t']):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
