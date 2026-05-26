#!/usr/bin/env python3
"""Check whether slot-wall failures were visible inside MPPI rollout horizons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_race_simple_phase_trace import _load_json, _load_yaml
from run_race_simple_phase_sweep import ROOT
from race_hero_slot_wall_failure_report import (
    command_limited_next_pos,
    signed_distance_to_aabb,
    static_box,
    voxelized_aabb,
)


DEFAULT_MECHANISM = ROOT / "docs/data/race_hero_slot_wall_failure_mechanism.json"
DEFAULT_OUT = ROOT / "docs/data/race_hero_slot_wall_rollout_horizon_report.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mechanism-report", type=Path, default=DEFAULT_MECHANISM)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--focus-drone", type=int, default=3)
    p.add_argument("--pre-window-s", type=float, default=4.0)
    return p.parse_args(argv)


def episode_log(run_dir: Path, episode: int, drone: int) -> Path:
    return run_dir / f"episode_{episode:03d}_drone_{drone:02d}.json"


def wall_clearance_at_point(
    *,
    pos: list[float],
    aabb: dict[str, list[float]],
    drone_radius: float,
) -> float:
    return signed_distance_to_aabb(pos, aabb) - drone_radius


def rollout_wall_metrics(
    *,
    replan_t: float,
    rollout: list[list[float]],
    aabb: dict[str, list[float]],
    drone_radius: float,
    dt_plan: float,
) -> dict[str, Any]:
    best: tuple[float, int, list[float]] | None = None
    first_hit_t: float | None = None
    for idx, pos in enumerate(rollout):
        clearance = wall_clearance_at_point(
            pos=[float(v) for v in pos],
            aabb=aabb,
            drone_radius=drone_radius,
        )
        if best is None or clearance < best[0]:
            best = (clearance, idx, [float(v) for v in pos])
        if first_hit_t is None and clearance <= 0.0:
            first_hit_t = float(replan_t) + float(dt_plan) * idx
    if best is None:
        return {
            "min_wall_clearance_m": None,
            "min_wall_t": None,
            "first_wall_hit_t": None,
            "min_wall_pos": None,
        }
    clearance, idx, pos = best
    return {
        "min_wall_clearance_m": clearance,
        "min_wall_t": float(replan_t) + float(dt_plan) * idx,
        "first_wall_hit_t": first_hit_t,
        "min_wall_pos": pos,
    }


def summarize_replan(
    *,
    replan: dict[str, Any],
    current_step: dict[str, Any],
    step_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    next_replan_t: float | None,
    collision_t: float,
    aabb: dict[str, list[float]],
    drone_radius: float,
    dt_plan: float,
) -> dict[str, Any]:
    rollouts = replan.get("rollouts") or []
    t = float(replan["t"])
    horizon_end_t = None
    if rollouts:
        horizon_end_t = t + dt_plan * (len(rollouts[0]) - 1)
    best_idx = replan.get("best_rollout_idx")
    best_metrics: dict[str, Any] | None = None
    if best_idx is not None and rollouts:
        idx = min(max(int(best_idx), 0), len(rollouts) - 1)
        best_metrics = rollout_wall_metrics(
            replan_t=t,
            rollout=rollouts[idx],
            aabb=aabb,
            drone_radius=drone_radius,
            dt_plan=dt_plan,
        )
    execution_next: dict[str, Any] | None = None
    if best_idx is not None and rollouts:
        idx = min(max(int(best_idx), 0), len(rollouts) - 1)
        if len(rollouts[idx]) <= 1:
            idx = -1
    if best_idx is not None and rollouts and idx >= 0:
        actual_next = command_limited_next_pos(cfg, current_step)
        rollout_next = [float(v) for v in rollouts[idx][1]]
        actual_clear = wall_clearance_at_point(
            pos=actual_next,
            aabb=aabb,
            drone_radius=drone_radius,
        )
        rollout_clear = wall_clearance_at_point(
            pos=rollout_next,
            aabb=aabb,
            drone_radius=drone_radius,
        )
        execution_next = {
            "actual_next_pos": actual_next,
            "best_rollout_next_pos": rollout_next,
            "actual_next_wall_clearance_m": actual_clear,
            "best_rollout_next_wall_clearance_m": rollout_clear,
            "actual_vs_best_rollout_next_delta_m": sum(
                (actual_next[i] - rollout_next[i]) ** 2 for i in range(len(actual_next))
            )
            ** 0.5,
        }
    execution_segment: dict[str, Any] | None = None
    if best_idx is not None and rollouts:
        idx = min(max(int(best_idx), 0), len(rollouts) - 1)
        segment_end_t = min(
            collision_t,
            next_replan_t if next_replan_t is not None else collision_t,
        )
        samples: list[dict[str, Any]] = []
        for step in step_rows:
            step_t = float(step["t"])
            post_t = step_t + dt_plan
            if not (t <= step_t <= segment_end_t):
                continue
            rollout_idx = int(round((post_t - t) / dt_plan))
            if not (0 <= rollout_idx < len(rollouts[idx])):
                continue
            actual_next = command_limited_next_pos(cfg, step)
            rollout_pos = [float(v) for v in rollouts[idx][rollout_idx]]
            actual_clear = wall_clearance_at_point(
                pos=actual_next,
                aabb=aabb,
                drone_radius=drone_radius,
            )
            rollout_clear = wall_clearance_at_point(
                pos=rollout_pos,
                aabb=aabb,
                drone_radius=drone_radius,
            )
            samples.append(
                {
                    "step_t": step_t,
                    "post_t": post_t,
                    "actual_next_wall_clearance_m": actual_clear,
                    "best_rollout_wall_clearance_m": rollout_clear,
                    "actual_vs_best_rollout_delta_m": sum(
                        (actual_next[i] - rollout_pos[i]) ** 2
                        for i in range(len(actual_next))
                    )
                    ** 0.5,
                    "collision_flag": bool(step.get("collision", False)),
                }
            )
        if samples:
            actual_min = min(samples, key=lambda row: row["actual_next_wall_clearance_m"])
            rollout_min = min(samples, key=lambda row: row["best_rollout_wall_clearance_m"])
            max_delta = max(samples, key=lambda row: row["actual_vs_best_rollout_delta_m"])
            execution_segment = {
                "segment_end_t": segment_end_t,
                "samples": len(samples),
                "actual_min": actual_min,
                "best_rollout_min": rollout_min,
                "max_actual_vs_best_rollout_delta": max_delta,
            }

    rollout_metrics = [
        rollout_wall_metrics(
            replan_t=t,
            rollout=rollout,
            aabb=aabb,
            drone_radius=drone_radius,
            dt_plan=dt_plan,
        )
        for rollout in rollouts
    ]
    hit_metrics = [
        row
        for row in rollout_metrics
        if row["first_wall_hit_t"] is not None
    ]
    min_metrics = [
        row
        for row in rollout_metrics
        if row["min_wall_clearance_m"] is not None
    ]
    visible_min = min(
        (float(row["min_wall_clearance_m"]) for row in min_metrics),
        default=None,
    )
    earliest_visible_hit = min(
        (float(row["first_wall_hit_t"]) for row in hit_metrics),
        default=None,
    )
    return {
        "t": t,
        "horizon_end_t": horizon_end_t,
        "collision_in_horizon": bool(
            horizon_end_t is not None and collision_t <= horizon_end_t
        ),
        "rollouts_logged": len(rollouts),
        "best_rollout_idx": best_idx,
        "best_rollout_wall": best_metrics,
        "execution_next_vs_best_rollout": execution_next,
        "execution_segment_vs_best_rollout": execution_segment,
        "visible_rollout_wall_hits": len(hit_metrics),
        "visible_rollout_min_wall_clearance_m": visible_min,
        "earliest_visible_rollout_wall_hit_t": earliest_visible_hit,
    }


def first_matching(rows: list[dict[str, Any]], predicate: Any) -> dict[str, Any] | None:
    for row in rows:
        if predicate(row):
            return row
    return None


def summarize_episode(
    *,
    gate_run: Path,
    episode_row: dict[str, Any],
    episode: int,
    drone: int,
    pre_window_s: float,
) -> dict[str, Any]:
    cfg = _load_yaml(gate_run / "config.yaml")
    log = _load_json(episode_log(gate_run, episode, drone))
    collision_t = episode_row.get("gate_first_collision_t")
    if collision_t is None:
        return {"episode": episode, "gate_first_collision_t": None}
    collision_t = float(collision_t)
    start_t = max(0.0, collision_t - float(pre_window_s))
    box = static_box(cfg)
    aabb = voxelized_aabb(cfg, box)
    drone_radius = float(cfg["simulator"].get("drone_radius", 0.4))
    dt_plan = float(cfg["planner"].get("dt_plan", cfg["simulator"].get("dt", 0.05)))
    raw_replans = [
        replan
        for replan in log.get("replans", [])
        if start_t <= float(replan["t"]) <= collision_t
    ]
    replans = [
        summarize_replan(
            replan=replan,
            current_step=min(
                log.get("steps", []),
                key=lambda step: abs(float(step["t"]) - float(replan["t"])),
            ),
            step_rows=log.get("steps", []),
            cfg=cfg,
            next_replan_t=(
                float(raw_replans[idx + 1]["t"])
                if idx + 1 < len(raw_replans)
                else None
            ),
            collision_t=collision_t,
            aabb=aabb,
            drone_radius=drone_radius,
            dt_plan=dt_plan,
        )
        for idx, replan in enumerate(raw_replans)
    ]
    first_collision_in_horizon = first_matching(
        replans,
        lambda row: row["collision_in_horizon"],
    )
    first_visible_wall_hit = first_matching(
        replans,
        lambda row: row["visible_rollout_wall_hits"] > 0,
    )
    first_best_wall_hit = first_matching(
        replans,
        lambda row: (row["best_rollout_wall"] or {}).get("first_wall_hit_t")
        is not None,
    )
    last_before_collision = replans[-1] if replans else None
    return {
        "episode": episode,
        "gate_first_collision_t": collision_t,
        "replans_analyzed": len(replans),
        "first_collision_in_horizon": first_collision_in_horizon,
        "first_visible_wall_hit": first_visible_wall_hit,
        "first_best_wall_hit": first_best_wall_hit,
        "last_replan_before_collision": last_before_collision,
    }


def summarize_case(
    *,
    case: dict[str, Any],
    drone: int,
    pre_window_s: float,
) -> dict[str, Any]:
    gate_run = ROOT / case["gate_run"]
    episode_rows = [
        summarize_episode(
            gate_run=gate_run,
            episode_row=row,
            episode=int(row["episode"]),
            drone=drone,
            pre_window_s=pre_window_s,
        )
        for row in case.get("episode_rows", [])
    ]
    return {
        "label": case["label"],
        "gate_run": case["gate_run"],
        "episodes": len(episode_rows),
        "episode_rows": episode_rows,
    }


def compact_replan(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    best = row.get("best_rollout_wall") or {}
    return {
        "t": row["t"],
        "horizon_end_t": row["horizon_end_t"],
        "visible_hits": row["visible_rollout_wall_hits"],
        "visible_min_wall_clearance_m": row[
            "visible_rollout_min_wall_clearance_m"
        ],
        "best_wall_hit_t": best.get("first_wall_hit_t"),
        "best_min_wall_clearance_m": best.get("min_wall_clearance_m"),
        "segment_actual_min_wall_clearance_m": (
            ((row.get("execution_segment_vs_best_rollout") or {}).get("actual_min") or {})
            .get("actual_next_wall_clearance_m")
        ),
        "segment_best_min_wall_clearance_m": (
            ((row.get("execution_segment_vs_best_rollout") or {}).get("best_rollout_min") or {})
            .get("best_rollout_wall_clearance_m")
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mechanism = _load_json(args.mechanism_report)
    report = {
        "source": {
            "script": "scripts/race_hero_slot_wall_rollout_horizon_report.py",
            "mechanism_report": str(args.mechanism_report),
        },
        "focus": {
            "drone": args.focus_drone,
            "pre_window_s": args.pre_window_s,
        },
        "cases": [
            summarize_case(
                case=case,
                drone=args.focus_drone,
                pre_window_s=args.pre_window_s,
            )
            for case in mechanism.get("cases", [])
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("| case | ep | collision | horizon sees | visible wall | best wall | last best min |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for case in report["cases"]:
        for row in case["episode_rows"]:
            horizon = compact_replan(row.get("first_collision_in_horizon"))
            visible = compact_replan(row.get("first_visible_wall_hit"))
            best = compact_replan(row.get("first_best_wall_hit"))
            last = compact_replan(row.get("last_replan_before_collision"))
            print(
                f"| {case['label']} | {row['episode']} | "
                f"{row.get('gate_first_collision_t'):.2f} | "
                f"{horizon['t'] if horizon else 'n/a'} | "
                f"{visible['t'] if visible else 'n/a'} | "
                f"{best['t'] if best else 'n/a'} | "
                f"{last['best_min_wall_clearance_m'] if last else 'n/a'} |"
            )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
