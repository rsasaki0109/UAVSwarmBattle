#!/usr/bin/env python3
"""Inspect a race-simple phase-sweep failure trace from existing logs.

This is a read-only companion to ``run_race_simple_phase_sweep.py``. It
reconstructs reflected dynamic-obstacle positions from the saved config,
then prints per-time clearance, command, and visible-rollout collision
statistics for a chosen episode/drone.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


DEFAULT_RUN_ROOT = Path("results/_race_simple_phase_sweep/p19p8_y5p5_34p5")
DEFAULT_TIMES = (28.8, 28.9, 29.0, 29.1, 29.2, 29.25)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping")
    return data


def _reflected_axis(start: float, velocity: float, t: float, upper: float) -> float:
    raw = float(start) + float(velocity) * float(t)
    period = 2.0 * float(upper)
    if period <= 0.0:
        return raw
    phase = raw % period
    if phase > upper:
        return period - phase
    return phase


def obstacle_positions(cfg: dict[str, Any], t: float) -> list[list[float]]:
    size = [float(v) for v in cfg["scenario"]["size"]]
    out: list[list[float]] = []
    for obs in cfg["scenario"].get("dynamic_obstacles", []) or []:
        start = [float(v) for v in obs["start"]]
        velocity = [float(v) for v in obs["velocity"]]
        if bool(obs.get("reflect", True)):
            out.append(
                [
                    _reflected_axis(start[i], velocity[i], t, size[i])
                    for i in range(len(start))
                ]
            )
        else:
            out.append([start[i] + velocity[i] * t for i in range(len(start))])
    return out


def distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def clearance_to_obstacles(
    cfg: dict[str, Any],
    pos: list[float],
    t: float,
) -> list[float]:
    drone_radius = float(cfg["simulator"].get("drone_radius", 0.4))
    obstacles = cfg["scenario"].get("dynamic_obstacles", []) or []
    positions = obstacle_positions(cfg, t)
    clearances: list[float] = []
    for obs, obs_pos in zip(obstacles, positions):
        radius_sum = drone_radius + float(obs.get("radius", 0.5))
        clearances.append(distance(pos, obs_pos) - radius_sum)
    return clearances


def min_clearance_for_path(
    cfg: dict[str, Any],
    path: list[list[float]],
    *,
    replan_t: float,
) -> tuple[float, bool]:
    dt = float(cfg["simulator"].get("dt", 0.05))
    best = float("inf")
    hit = False
    for k, pos in enumerate(path):
        t = replan_t + dt * k
        for clearance in clearance_to_obstacles(cfg, pos, t):
            best = min(best, clearance)
            hit = hit or clearance <= 0.0
    return best, hit


def nearest(items: list[dict[str, Any]], target_t: float) -> dict[str, Any]:
    return min(items, key=lambda row: abs(float(row["t"]) - target_t))


def fmt_vec(v: list[float]) -> str:
    return "(" + ",".join(f"{float(x):+6.2f}" for x in v) + ")"


def summarize_step(
    *,
    cfg: dict[str, Any],
    step: dict[str, Any],
) -> str:
    t = float(step["t"])
    pos = [float(v) for v in step["true_pos"]]
    # Log rows store pre-step position and post-step collision; inspect both
    # adjacent scenario times and report the tighter clearance.
    dt = float(cfg["simulator"].get("dt", 0.05))
    c0 = clearance_to_obstacles(cfg, pos, t)
    c1 = clearance_to_obstacles(cfg, pos, t + dt)
    clearances = [min(a, b) for a, b in zip(c0, c1)]
    return (
        f"t={t:5.2f} pos={fmt_vec(pos)} "
        f"vel={fmt_vec(step['true_vel'])} cmd={fmt_vec(step['cmd'])} "
        f"dyn_clear="
        + ",".join(f"{c:+.2f}" for c in clearances)
        + f" collision={bool(step.get('collision', False))}"
    )


def summarize_rollouts(
    *,
    cfg: dict[str, Any],
    replan: dict[str, Any],
) -> str:
    rollouts = replan.get("rollouts") or []
    if not rollouts:
        return "rollouts=n/a"
    clearances: list[float] = []
    hits = 0
    for path in rollouts:
        clearance, hit = min_clearance_for_path(
            cfg,
            path,
            replan_t=float(replan["t"]),
        )
        clearances.append(clearance)
        hits += int(hit)
    best_idx = int(replan.get("best_rollout_idx", 0))
    best_clearance = clearances[best_idx] if 0 <= best_idx < len(clearances) else math.nan
    return (
        f"rollouts={len(rollouts)} hit={hits}/{len(rollouts)} "
        f"min={min(clearances):+.2f} best_idx={best_idx} "
        f"best_clear={best_clearance:+.2f}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--drone", type=int, default=3)
    p.add_argument(
        "--planner",
        action="append",
        dest="planners",
        help="planner subdir to inspect; repeatable. default: mpc and gpu_mppi",
    )
    p.add_argument(
        "--time",
        action="append",
        dest="times",
        type=float,
        help="time to inspect; repeatable. default: near collision window",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    planners = args.planners or ["mpc", "gpu_mppi"]
    times = args.times or list(DEFAULT_TIMES)
    for planner in planners:
        run_dir = args.run_root / planner
        cfg = _load_yaml(run_dir / "config.yaml")
        log = _load_json(
            run_dir / f"episode_{args.episode:03d}_drone_{args.drone:02d}.json"
        )
        print(f"\n{planner}: outcome={log['outcome']} summary={log['summary']}")
        for t in times:
            step = nearest(log["steps"], t)
            replan = nearest(log["replans"], t)
            print(summarize_step(cfg=cfg, step=step))
            print(f"  replan_t={float(replan['t']):5.2f} {summarize_rollouts(cfg=cfg, replan=replan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
