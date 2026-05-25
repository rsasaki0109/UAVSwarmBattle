#!/usr/bin/env python3
"""Compute metrics for the README race-hero encounter GIF.

The hero GIF is intentionally zoomed into one critical encounter. This
script fixes the quantitative counterpart to that visual: how far the
moving sweeper travels in the GIF window, when the vanilla arm contacts
it, and how much signed clearance the low-temperature counterfactual
keeps in the same window.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from analyze_race_simple_phase_trace import obstacle_positions


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = (
    ROOT / "results/_race_simple_causal_probe/p19p8_y5p0_35p0"
)
DEFAULT_OUT = ROOT / "docs/data/race_hero_encounter_metrics.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected mapping")
    return data


def dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def xy(v: list[float]) -> list[float]:
    return [float(v[0]), float(v[1])]


def first_collision_t(log: dict[str, Any]) -> float | None:
    for step in log.get("steps", []):
        if bool(step.get("collision", False)):
            return float(step["t"])
    return None


def nearest_step(log: dict[str, Any], target_t: float) -> dict[str, Any]:
    return min(log["steps"], key=lambda step: abs(float(step["t"]) - target_t))


def clearance_to_obstacle(
    cfg: dict[str, Any],
    pos: list[float],
    t: float,
    obstacle_idx: int,
) -> float:
    drone_radius = float(cfg["simulator"].get("drone_radius", 0.4))
    obs = cfg["scenario"]["dynamic_obstacles"][obstacle_idx]
    obs_pos = obstacle_positions(cfg, t)[obstacle_idx]
    radius_sum = drone_radius + float(obs.get("radius", 0.5))
    return dist(pos, obs_pos) - radius_sum


def row_clearance_to_obstacle(
    cfg: dict[str, Any],
    step: dict[str, Any],
    obstacle_idx: int,
) -> float:
    # Logs store pre-step position plus post-step collision flag. Evaluate
    # both adjacent scenario times and keep the tighter value.
    pos = [float(v) for v in step["true_pos"]]
    t = float(step["t"])
    dt = float(cfg["simulator"].get("dt", 0.05))
    return min(
        clearance_to_obstacle(cfg, pos, t, obstacle_idx),
        clearance_to_obstacle(cfg, pos, t + dt, obstacle_idx),
    )


def window_min_clearance(
    cfg: dict[str, Any],
    log: dict[str, Any],
    obstacle_idx: int,
    start_t: float,
    end_t: float,
) -> dict[str, Any]:
    steps = [
        step
        for step in log.get("steps", [])
        if start_t <= float(step["t"]) <= end_t
    ]
    best = min(steps, key=lambda step: row_clearance_to_obstacle(cfg, step, obstacle_idx))
    return {
        "t": float(best["t"]),
        "clearance_m": row_clearance_to_obstacle(cfg, best, obstacle_idx),
        "collision_flag": bool(best.get("collision", False)),
    }


def step_snapshot(
    cfg: dict[str, Any],
    log: dict[str, Any],
    obstacle_idx: int,
    target_t: float,
) -> dict[str, Any]:
    step = nearest_step(log, target_t)
    t = float(step["t"])
    pos = [float(v) for v in step["true_pos"]]
    ref = [float(v) for v in step.get("reference_pos", step["true_pos"])]
    obs_pos = obstacle_positions(cfg, t)[obstacle_idx]
    return {
        "t": t,
        "drone_xy": xy(pos),
        "reference_xy": xy(ref),
        "obstacle_xy": xy(obs_pos),
        "clearance_m": clearance_to_obstacle(cfg, pos, t, obstacle_idx),
        "reference_error_m": dist(pos, ref),
        "collision_flag": bool(step.get("collision", False)),
    }


def arm_metrics(
    *,
    label: str,
    run_dir: Path,
    cfg: dict[str, Any],
    episode: int,
    drone: int,
    obstacle_idx: int,
    start_t: float,
    end_t: float,
    snapshot_t: float,
) -> dict[str, Any]:
    log = load_json(run_dir / f"episode_{episode:03d}_drone_{drone:02d}.json")
    return {
        "label": label,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "episode": episode,
        "drone": drone,
        "outcome": log.get("outcome"),
        "first_collision_t": first_collision_t(log),
        "window_min_clearance": window_min_clearance(
            cfg,
            log,
            obstacle_idx,
            start_t,
            end_t,
        ),
        "snapshot": step_snapshot(cfg, log, obstacle_idx, snapshot_t),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--vanilla-tag", default="t1")
    p.add_argument("--low-temp-tag", default="t0p1")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--focus-drone", type=int, default=3)
    p.add_argument("--focus-obstacle", type=int, default=0)
    p.add_argument("--start-step", type=int, default=520)
    p.add_argument("--end-step", type=int, default=632)
    p.add_argument("--snapshot-t", type=float, default=29.30)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.base / args.low_temp_tag / "config.yaml")
    dt = float(cfg["simulator"].get("dt", 0.05))
    start_t = args.start_step * dt
    end_t = args.end_step * dt

    obs_start = obstacle_positions(cfg, start_t)[args.focus_obstacle]
    obs_end = obstacle_positions(cfg, end_t)[args.focus_obstacle]
    arms = [
        arm_metrics(
            label="vanilla-t1.0",
            run_dir=args.base / args.vanilla_tag,
            cfg=cfg,
            episode=args.episode,
            drone=args.focus_drone,
            obstacle_idx=args.focus_obstacle,
            start_t=start_t,
            end_t=end_t,
            snapshot_t=args.snapshot_t,
        ),
        arm_metrics(
            label="low-temp-t0.1",
            run_dir=args.base / args.low_temp_tag,
            cfg=cfg,
            episode=args.episode,
            drone=args.focus_drone,
            obstacle_idx=args.focus_obstacle,
            start_t=start_t,
            end_t=end_t,
            snapshot_t=args.snapshot_t,
        ),
    ]
    low = arms[1]
    vanilla = arms[0]
    result = {
        "source": {
            "config": str((args.base / args.low_temp_tag / "config.yaml").relative_to(ROOT)),
            "gif": "docs/images/compare_race_temperature_avoid.gif",
        },
        "focus": {
            "episode": args.episode,
            "drone": args.focus_drone,
            "obstacle": args.focus_obstacle,
            "window_s": [start_t, end_t],
            "snapshot_t": args.snapshot_t,
            "obstacle_xy_start": xy(obs_start),
            "obstacle_xy_end": xy(obs_end),
            "obstacle_travel_m": dist(obs_start, obs_end),
        },
        "arms": arms,
        "comparison": {
            "low_temp_window_clearance_gain_m": (
                low["window_min_clearance"]["clearance_m"]
                - vanilla["window_min_clearance"]["clearance_m"]
            ),
            "low_temp_snapshot_clearance_gain_m": (
                low["snapshot"]["clearance_m"] - vanilla["snapshot"]["clearance_m"]
            ),
            "vanilla_contacts_but_low_temp_does_not": (
                vanilla["first_collision_t"] is not None
                and low["first_collision_t"] is None
            ),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(
        "hero encounter: "
        f"vanilla contact={vanilla['first_collision_t']} "
        f"low-temp min_clear={low['window_min_clearance']['clearance_m']:+.2f} m "
        f"sweeper_travel={result['focus']['obstacle_travel_m']:.2f} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
