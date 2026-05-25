#!/usr/bin/env python3
"""Run no-sweeper controls for the README race-hero encounter.

The hero GIF can otherwise read as "the low-temperature MPPI happened
to take a different line."  This script reruns the same low-temperature
episode with the scene sweepers removed, then evaluates that ghost
trajectory against the original moving sweeper.  If the no-sweeper
ghost would enter the original safety halo while the moving-sweeper run
does not, the visual has a causal control instead of just a lucky
non-contact.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

from analyze_race_simple_phase_trace import _load_json, _load_yaml, obstacle_positions
from run_race_simple_phase_sweep import ROOT, _write_yaml, run_one, summarize_run


DEFAULT_MOVING_RUN = (
    ROOT / "results/_race_simple_causal_probe/p19p8_y5p0_35p0/t0p1"
)
DEFAULT_OUT_ROOT = ROOT / "results/_race_hero_causality_controls"
DEFAULT_OUT_JSON = ROOT / "docs/data/race_hero_causality_controls.json"


def repo_path(path: Path | str) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def xy(v: list[float]) -> list[float]:
    return [float(v[0]), float(v[1])]


def nearest_step(log: dict[str, Any], target_t: float) -> dict[str, Any]:
    return min(log["steps"], key=lambda step: abs(float(step["t"]) - target_t))


def first_collision_t(log: dict[str, Any]) -> float | None:
    for step in log.get("steps", []):
        if bool(step.get("collision", False)):
            return float(step["t"])
    return None


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
    if not steps:
        return {"t": None, "clearance_m": None, "collision_flag": False}
    best = min(steps, key=lambda step: row_clearance_to_obstacle(cfg, step, obstacle_idx))
    return {
        "t": float(best["t"]),
        "clearance_m": row_clearance_to_obstacle(cfg, best, obstacle_idx),
        "collision_flag": bool(best.get("collision", False)),
    }


def snapshot(
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
        "virtual_clearance_m": clearance_to_obstacle(cfg, pos, t, obstacle_idx),
        "reference_error_m": dist(pos, ref),
        "collision_flag": bool(step.get("collision", False)),
    }


def trajectory_delta(
    a_log: dict[str, Any],
    b_log: dict[str, Any],
    start_t: float,
    end_t: float,
) -> dict[str, Any]:
    samples: list[dict[str, float]] = []
    for a_step in a_log.get("steps", []):
        t = float(a_step["t"])
        if not (start_t <= t <= end_t):
            continue
        b_step = nearest_step(b_log, t)
        delta = dist(
            [float(v) for v in a_step["true_pos"]],
            [float(v) for v in b_step["true_pos"]],
        )
        samples.append({"t": t, "delta_m": delta})
    if not samples:
        return {"samples": 0, "rms_delta_m": None, "max_delta_m": None, "max_delta_t": None}
    rms = math.sqrt(sum(row["delta_m"] ** 2 for row in samples) / len(samples))
    worst = max(samples, key=lambda row: row["delta_m"])
    return {
        "samples": len(samples),
        "rms_delta_m": rms,
        "max_delta_m": worst["delta_m"],
        "max_delta_t": worst["t"],
    }


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    out = {
        key: value
        for key, value in summary.items()
        if key != "collision_details"
    }
    if "run_dir" in out:
        out["run_dir"] = repo_path(out["run_dir"])
    return out


def build_no_sweeper_config(
    moving_cfg: dict[str, Any],
    *,
    cell: str,
    n: int,
    seed: int,
    output_root: Path,
) -> dict[str, Any]:
    cfg = copy.deepcopy(moving_cfg)
    cfg["name"] = f"{moving_cfg.get('name', 'race_hero')}_no_sweeper_control"
    cfg["seed"] = int(seed)
    cfg["num_episodes"] = int(n)
    cfg["scenario"]["dynamic_obstacles"] = []
    cfg.setdefault("output", {})["dir"] = str(output_root / cell / "no_sweeper_t0p1")
    return cfg


def control_metrics(
    *,
    original_cfg: dict[str, Any],
    moving_run: Path,
    control_run: Path,
    control_cfg: dict[str, Any],
    episode: int,
    drone: int,
    obstacle_idx: int,
    start_t: float,
    end_t: float,
    snapshot_t: float,
) -> dict[str, Any]:
    moving_log = _load_json(moving_run / f"episode_{episode:03d}_drone_{drone:02d}.json")
    control_log = _load_json(control_run / f"episode_{episode:03d}_drone_{drone:02d}.json")
    moving_min = window_min_clearance(
        original_cfg,
        moving_log,
        obstacle_idx,
        start_t,
        end_t,
    )
    control_virtual_min = window_min_clearance(
        original_cfg,
        control_log,
        obstacle_idx,
        start_t,
        end_t,
    )
    moving_snap = snapshot(original_cfg, moving_log, obstacle_idx, snapshot_t)
    control_snap = snapshot(original_cfg, control_log, obstacle_idx, snapshot_t)
    snap_delta = dist(
        nearest_step(moving_log, snapshot_t)["true_pos"],
        nearest_step(control_log, snapshot_t)["true_pos"],
    )
    path_delta = trajectory_delta(moving_log, control_log, start_t, end_t)
    moving_clear = moving_min["clearance_m"]
    control_clear = control_virtual_min["clearance_m"]
    supports_causal = (
        moving_clear is not None
        and control_clear is not None
        and moving_clear > 0.0
        and control_clear < 0.0
        and (path_delta["max_delta_m"] or 0.0) >= 0.5
    )
    return {
        "moving_low_temp": {
            "run_dir": repo_path(moving_run),
            "outcome": moving_log.get("outcome"),
            "first_collision_t": first_collision_t(moving_log),
            "window_min_clearance": moving_min,
            "snapshot": moving_snap,
        },
        "control": {
            "label": "no-sweeper low-temp t=0.1",
            "run_dir": repo_path(control_run),
            "config": repo_path(control_run / "config.yaml"),
            "summary": compact_summary(summarize_run(control_run, control_cfg)),
            "outcome": control_log.get("outcome"),
            "first_collision_t": first_collision_t(control_log),
            "virtual_window_min_clearance_to_original_sweeper": control_virtual_min,
            "virtual_snapshot_against_original_sweeper": control_snap,
        },
        "comparison": {
            "trajectory_delta_m": path_delta,
            "snapshot_delta_m": snap_delta,
            "moving_minus_control_virtual_min_clearance_m": (
                moving_clear - control_clear
                if moving_clear is not None and control_clear is not None
                else None
            ),
            "supports_obstacle_caused_detour": supports_causal,
        },
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--moving-run", type=Path, default=DEFAULT_MOVING_RUN)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--scratch-dir", type=Path, default=Path("/tmp/uavnav_race_hero_causality"))
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--summary-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--focus-drone", type=int, default=3)
    p.add_argument("--focus-obstacle", type=int, default=0)
    p.add_argument("--start-step", type=int, default=520)
    p.add_argument("--end-step", type=int, default=632)
    p.add_argument("--snapshot-t", type=float, default=29.30)
    p.add_argument("--summarize-only", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    moving_cfg = _load_yaml(args.moving_run / "config.yaml")
    dt = float(moving_cfg["simulator"].get("dt", 0.05))
    start_t = args.start_step * dt
    end_t = args.end_step * dt
    cell = args.moving_run.parent.name

    control_cfg = build_no_sweeper_config(
        moving_cfg,
        cell=cell,
        n=args.n,
        seed=args.seed,
        output_root=args.output_root,
    )
    config_path = args.scratch_dir / f"{cell}_no_sweeper_t0p1.yaml"
    _write_yaml(config_path, control_cfg)
    control_run = Path(control_cfg["output"]["dir"])
    if not args.summarize_only:
        run_one(config_path, python=str(args.python))

    obs_start = obstacle_positions(moving_cfg, start_t)[args.focus_obstacle]
    obs_end = obstacle_positions(moving_cfg, end_t)[args.focus_obstacle]
    result = {
        "source": {
            "moving_run": repo_path(args.moving_run),
            "control_run": repo_path(control_run),
            "control_config": repo_path(control_run / "config.yaml"),
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
        "metrics": control_metrics(
            original_cfg=moving_cfg,
            moving_run=args.moving_run,
            control_run=control_run,
            control_cfg=control_cfg,
            episode=args.episode,
            drone=args.focus_drone,
            obstacle_idx=args.focus_obstacle,
            start_t=start_t,
            end_t=end_t,
            snapshot_t=args.snapshot_t,
        ),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    metrics = result["metrics"]
    moving = metrics["moving_low_temp"]["window_min_clearance"]["clearance_m"]
    ghost = metrics["control"]["virtual_window_min_clearance_to_original_sweeper"][
        "clearance_m"
    ]
    delta = metrics["comparison"]["trajectory_delta_m"]["max_delta_m"]
    print(f"wrote {args.summary_json}")
    print(
        "race hero control: "
        f"moving_clear={moving:+.2f} m "
        f"no_sweeper_virtual_clear={ghost:+.2f} m "
        f"max_path_delta={delta:.2f} m "
        f"causal={metrics['comparison']['supports_obstacle_caused_detour']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
