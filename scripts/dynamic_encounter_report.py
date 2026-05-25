#!/usr/bin/env python3
"""Build a control-first report for a dynamic-obstacle encounter.

This is intentionally stricter than a GIF renderer.  It evaluates all
arms against one reference moving-obstacle config, so controls such as
`no_obstacle` can still be scored against the original obstacle tube.
The output is meant to decide whether an encounter is real dynamic
avoidance, a weak visual control, or merely a lucky non-contact.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from analyze_race_simple_phase_trace import _load_json, _load_yaml, obstacle_positions
from run_race_simple_phase_sweep import ROOT


DEFAULT_CELL = "p19p8_y5p0_35p0"
DEFAULT_REFERENCE_CONFIG = (
    ROOT / "results/_race_simple_causal_probe" / DEFAULT_CELL / "t0p1/config.yaml"
)
DEFAULT_ARMS = [
    "moving:low-temp-t0.1:results/_race_simple_causal_probe/p19p8_y5p0_35p0/t0p1",
    "no_obstacle:no-sweeper-ghost:results/_race_hero_causality_controls/p19p8_y5p0_35p0/no_sweeper_t0p1",
    "frozen_initial:frozen-at-start:results/_race_hero_control_variants/p19p8_y5p0_35p0/frozen_initial",
    "frozen_encounter:frozen-at-29.30s:results/_race_hero_control_variants/p19p8_y5p0_35p0/frozen_encounter",
    "wrong_velocity:planner-reversed-velocity:results/_race_hero_control_variants/p19p8_y5p0_35p0/wrong_velocity",
    "no_prediction:current-obstacle-only:results/_race_hero_control_variants/p19p8_y5p0_35p0/no_prediction",
    "comparator:vanilla-t1.0:results/_race_simple_causal_probe/p19p8_y5p0_35p0/t1",
]
DEFAULT_OUT = ROOT / f"docs/data/dynamic_encounter_report_{DEFAULT_CELL}.json"


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
    pos = [float(v) for v in step["true_pos"]]
    t = float(step["t"])
    dt = float(cfg["simulator"].get("dt", 0.05))
    return min(
        clearance_to_obstacle(cfg, pos, t, obstacle_idx),
        clearance_to_obstacle(cfg, pos, t + dt, obstacle_idx),
    )


def window_steps(
    log: dict[str, Any],
    start_t: float,
    end_t: float,
) -> list[dict[str, Any]]:
    return [
        step
        for step in log.get("steps", [])
        if start_t <= float(step["t"]) <= end_t
    ]


def window_min_clearance(
    cfg: dict[str, Any],
    log: dict[str, Any],
    obstacle_idx: int,
    start_t: float,
    end_t: float,
) -> dict[str, Any]:
    steps = window_steps(log, start_t, end_t)
    if not steps:
        return {
            "t": None,
            "clearance_m": None,
            "collision_flag": False,
            "first_virtual_hit_t": None,
        }
    best = min(steps, key=lambda step: row_clearance_to_obstacle(cfg, step, obstacle_idx))
    first_hit = next(
        (
            float(step["t"])
            for step in steps
            if row_clearance_to_obstacle(cfg, step, obstacle_idx) <= 0.0
        ),
        None,
    )
    return {
        "t": float(best["t"]),
        "clearance_m": row_clearance_to_obstacle(cfg, best, obstacle_idx),
        "collision_flag": bool(best.get("collision", False)),
        "first_virtual_hit_t": first_hit,
    }


def path_length(steps: list[dict[str, Any]]) -> float:
    total = 0.0
    prev: list[float] | None = None
    for step in steps:
        pos = [float(v) for v in step["true_pos"]]
        if prev is not None:
            total += dist(prev, pos)
        prev = pos
    return total


def reference_tracking(steps: list[dict[str, Any]]) -> dict[str, Any]:
    if not steps:
        return {
            "samples": 0,
            "mean_error_m": None,
            "max_error_m": None,
            "end_error_m": None,
            "path_length_m": None,
        }
    errors = [
        dist(
            [float(v) for v in step["true_pos"]],
            [float(v) for v in step.get("reference_pos", step["true_pos"])],
        )
        for step in steps
    ]
    return {
        "samples": len(steps),
        "mean_error_m": sum(errors) / len(errors),
        "max_error_m": max(errors),
        "end_error_m": errors[-1],
        "path_length_m": path_length(steps),
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
        "clearance_m": clearance_to_obstacle(cfg, pos, t, obstacle_idx),
        "reference_error_m": dist(pos, ref),
        "collision_flag": bool(step.get("collision", False)),
    }


def trajectory_delta(
    ref_log: dict[str, Any],
    arm_log: dict[str, Any],
    start_t: float,
    end_t: float,
) -> dict[str, Any]:
    samples: list[dict[str, float]] = []
    for ref_step in ref_log.get("steps", []):
        t = float(ref_step["t"])
        if not (start_t <= t <= end_t):
            continue
        arm_step = nearest_step(arm_log, t)
        samples.append(
            {
                "t": t,
                "delta_m": dist(
                    [float(v) for v in ref_step["true_pos"]],
                    [float(v) for v in arm_step["true_pos"]],
                ),
            }
        )
    if not samples:
        return {
            "samples": 0,
            "rms_delta_m": None,
            "max_delta_m": None,
            "max_delta_t": None,
        }
    worst = max(samples, key=lambda row: row["delta_m"])
    return {
        "samples": len(samples),
        "rms_delta_m": math.sqrt(
            sum(row["delta_m"] ** 2 for row in samples) / len(samples)
        ),
        "max_delta_m": worst["delta_m"],
        "max_delta_t": worst["t"],
    }


def rollout_min_clearance_to_obstacle(
    cfg: dict[str, Any],
    path: list[list[float]],
    *,
    replan_t: float,
    obstacle_idx: int,
) -> float:
    dt = float(cfg["planner"].get("dt_plan", cfg["simulator"].get("dt", 0.05)))
    best = float("inf")
    for k, pos in enumerate(path):
        best = min(
            best,
            clearance_to_obstacle(cfg, [float(v) for v in pos], replan_t + dt * k, obstacle_idx),
        )
    return best


def rollout_horizon_report(
    cfg: dict[str, Any],
    log: dict[str, Any],
    *,
    obstacle_idx: int,
    start_t: float,
    end_t: float,
    threshold_m: float,
    closest_t: float | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    first_any: float | None = None
    first_selected: float | None = None
    for replan in log.get("replans", []):
        t = float(replan["t"])
        if not (start_t <= t <= end_t):
            continue
        rollouts = replan.get("rollouts") or []
        if not rollouts:
            continue
        clearances = [
            rollout_min_clearance_to_obstacle(
                cfg,
                path,
                replan_t=t,
                obstacle_idx=obstacle_idx,
            )
            for path in rollouts
        ]
        best_idx = int(replan.get("best_rollout_idx", 0))
        selected = clearances[best_idx] if 0 <= best_idx < len(clearances) else None
        row = {
            "t": t,
            "rollout_count": len(rollouts),
            "min_clearance_m": min(clearances),
            "selected_clearance_m": selected,
        }
        rows.append(row)
        if first_any is None and row["min_clearance_m"] <= threshold_m:
            first_any = t
        if (
            first_selected is None
            and selected is not None
            and selected <= threshold_m
        ):
            first_selected = t
    replan_period = float(cfg["planner"].get("replan_period", 0.2))

    def lead(entry_t: float | None) -> dict[str, float | None]:
        if closest_t is None or entry_t is None:
            return {"seconds": None, "replans": None}
        seconds = closest_t - entry_t
        return {"seconds": seconds, "replans": seconds / replan_period}

    return {
        "threshold_m": threshold_m,
        "first_any_rollout_t": first_any,
        "first_selected_rollout_t": first_selected,
        "lead_before_closest_any": lead(first_any),
        "lead_before_closest_selected": lead(first_selected),
        "samples": rows,
    }


def parse_arm(raw: str) -> dict[str, Any]:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "arm must be ROLE:LABEL:PATH, e.g. moving:t0p1:results/run"
        )
    role, label, path = parts
    return {"role": role, "label": label, "path": Path(path)}


def arm_metrics(
    *,
    cfg: dict[str, Any],
    arm: dict[str, Any],
    moving_log: dict[str, Any],
    episode: int,
    drone: int,
    obstacle_idx: int,
    start_t: float,
    end_t: float,
    snapshot_t: float,
) -> dict[str, Any]:
    log = _load_json(arm["path"] / f"episode_{episode:03d}_drone_{drone:02d}.json")
    steps = window_steps(log, start_t, end_t)
    clearance = window_min_clearance(cfg, log, obstacle_idx, start_t, end_t)
    return {
        "role": arm["role"],
        "label": arm["label"],
        "run_dir": repo_path(arm["path"]),
        "outcome": log.get("outcome"),
        "first_collision_t": first_collision_t(log),
        "window_min_clearance_to_reference_obstacle": clearance,
        "snapshot": snapshot(cfg, log, obstacle_idx, snapshot_t),
        "reference_tracking": reference_tracking(steps),
        "path_delta_to_moving": trajectory_delta(moving_log, log, start_t, end_t),
    }


def verdict(
    *,
    arms: list[dict[str, Any]],
    obstacle_travel_m: float,
    thresholds: dict[str, float],
    horizon: dict[str, Any],
) -> dict[str, Any]:
    by_role = {row["role"]: row for row in arms}
    moving = by_role.get("moving")
    no_obstacle = by_role.get("no_obstacle")
    reasons: list[str] = []
    if moving is None:
        return {"class": "insufficient_controls", "reasons": ["missing moving arm"]}
    moving_clear = (
        moving["window_min_clearance_to_reference_obstacle"].get("clearance_m")
    )
    if moving.get("outcome") != "success":
        reasons.append("moving arm did not succeed")
    if moving_clear is None or moving_clear < thresholds["moving_min_clearance_m"]:
        reasons.append("moving arm clearance below target")
    if no_obstacle is None:
        reasons.append("missing no_obstacle control")
    else:
        ghost_clear = no_obstacle[
            "window_min_clearance_to_reference_obstacle"
        ].get("clearance_m")
        delta = no_obstacle["path_delta_to_moving"].get("max_delta_m")
        if ghost_clear is None or ghost_clear >= 0.0:
            reasons.append("no_obstacle control does not enter moving obstacle tube")
        elif ghost_clear > thresholds["no_obstacle_max_clearance_m"]:
            reasons.append("no_obstacle virtual penetration is too small")
        if delta is None or delta < thresholds["path_delta_m"]:
            reasons.append("moving-vs-no_obstacle path delta below target")
    if obstacle_travel_m < thresholds["obstacle_travel_m"]:
        reasons.append("obstacle travel below target")
    lead_replans = (
        horizon.get("lead_before_closest_any", {}).get("replans")
        if horizon
        else None
    )
    if lead_replans is None or lead_replans < thresholds["horizon_lead_replans"]:
        reasons.append("obstacle did not enter rollout horizon early enough")

    missing_strong_controls = [
        role
        for role in ["frozen_initial", "frozen_encounter", "wrong_velocity", "no_prediction"]
        if role not in by_role
    ]
    if missing_strong_controls:
        reasons.append("missing stronger controls: " + ", ".join(missing_strong_controls))

    if not reasons:
        cls = "dynamic_avoidance"
    elif no_obstacle is not None and no_obstacle[
        "window_min_clearance_to_reference_obstacle"
    ].get("clearance_m", 1.0) >= 0.0:
        cls = "lucky_non_contact"
    else:
        cls = "weak_dynamic_avoidance_control"
    return {"class": cls, "reasons": reasons}


def markdown_table(report: dict[str, Any]) -> str:
    lines = [
        "| role | label | outcome | min clearance | first virtual hit | max delta to moving |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm in report["arms"]:
        clearance = arm["window_min_clearance_to_reference_obstacle"]
        delta = arm["path_delta_to_moving"]
        lines.append(
            "| {role} | {label} | {outcome} | {clearance} | {hit} | {delta} |".format(
                role=arm["role"],
                label=arm["label"],
                outcome=arm["outcome"],
                clearance=fmt(clearance.get("clearance_m")),
                hit=fmt(clearance.get("first_virtual_hit_t"), digits=2, signed=False),
                delta=fmt(delta.get("max_delta_m"), signed=False),
            )
        )
    return "\n".join(lines)


def fmt(value: Any, *, digits: int = 2, signed: bool = True) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        sign = "+" if signed else ""
        return f"{value:{sign}.{digits}f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-config", type=Path, default=DEFAULT_REFERENCE_CONFIG)
    p.add_argument("--arm", action="append", type=parse_arm, dest="arms")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--markdown-out", type=Path)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--focus-drone", type=int, default=3)
    p.add_argument("--focus-obstacle", type=int, default=0)
    p.add_argument("--start-step", type=int, default=520)
    p.add_argument("--end-step", type=int, default=632)
    p.add_argument("--snapshot-t", type=float, default=29.30)
    p.add_argument("--horizon-threshold", type=float, default=1.0)
    p.add_argument("--target-moving-clearance", type=float, default=0.25)
    p.add_argument("--target-no-obstacle-clearance", type=float, default=-0.5)
    p.add_argument("--target-path-delta", type=float, default=1.0)
    p.add_argument("--target-obstacle-travel", type=float, default=4.0)
    p.add_argument("--target-horizon-lead-replans", type=float, default=2.0)
    args = p.parse_args()
    args.arms = args.arms or [parse_arm(raw) for raw in DEFAULT_ARMS]
    return args


def main() -> int:
    args = parse_args()
    cfg = _load_yaml(args.reference_config)
    dt = float(cfg["simulator"].get("dt", 0.05))
    start_t = args.start_step * dt
    end_t = args.end_step * dt
    moving_arm = next((arm for arm in args.arms if arm["role"] == "moving"), None)
    if moving_arm is None:
        raise SystemExit("at least one arm with role 'moving' is required")
    moving_log = _load_json(
        moving_arm["path"]
        / f"episode_{args.episode:03d}_drone_{args.focus_drone:02d}.json"
    )

    obs_start = obstacle_positions(cfg, start_t)[args.focus_obstacle]
    obs_end = obstacle_positions(cfg, end_t)[args.focus_obstacle]
    moving_clear = window_min_clearance(
        cfg,
        moving_log,
        args.focus_obstacle,
        start_t,
        end_t,
    )
    horizon = rollout_horizon_report(
        cfg,
        moving_log,
        obstacle_idx=args.focus_obstacle,
        start_t=start_t,
        end_t=end_t,
        threshold_m=args.horizon_threshold,
        closest_t=moving_clear.get("t"),
    )
    arms = [
        arm_metrics(
            cfg=cfg,
            arm=arm,
            moving_log=moving_log,
            episode=args.episode,
            drone=args.focus_drone,
            obstacle_idx=args.focus_obstacle,
            start_t=start_t,
            end_t=end_t,
            snapshot_t=args.snapshot_t,
        )
        for arm in args.arms
    ]
    thresholds = {
        "moving_min_clearance_m": args.target_moving_clearance,
        "no_obstacle_max_clearance_m": args.target_no_obstacle_clearance,
        "path_delta_m": args.target_path_delta,
        "obstacle_travel_m": args.target_obstacle_travel,
        "horizon_lead_replans": args.target_horizon_lead_replans,
    }
    report = {
        "source": {
            "reference_config": repo_path(args.reference_config),
            "script": "scripts/dynamic_encounter_report.py",
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
            "obstacle_speed_mps": dist(obs_start, obs_end) / max(end_t - start_t, 1e-9),
            "obstacle_radius_m": float(
                cfg["scenario"]["dynamic_obstacles"][args.focus_obstacle].get(
                    "radius",
                    0.5,
                )
            ),
        },
        "thresholds": thresholds,
        "rollout_horizon": horizon,
        "arms": arms,
    }
    report["verdict"] = verdict(
        arms=arms,
        obstacle_travel_m=report["focus"]["obstacle_travel_m"],
        thresholds=thresholds,
        horizon=horizon,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = markdown_table(report)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"verdict: {report['verdict']['class']}")
    for reason in report["verdict"]["reasons"]:
        print(f"- {reason}")
    print(f"wrote {args.out}")
    if args.markdown_out:
        print(f"wrote {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
