#!/usr/bin/env python3
"""Quantify why slot-wall dynamic-gate boundary runs fail.

The input reports come from either:

* ``race_hero_slot_wall_sweep.py`` with ``base_wall`` / ``gate_wall`` arms, or
* a pair of older control-sweep reports, one for the base wall and one for the
  gate+wall arm.

The output focuses on the failing drone and compares the successful base-wall
trajectory to the failing gate-wall trajectory at the same times.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from analyze_race_simple_phase_trace import _load_json, _load_yaml
from dynamic_encounter_report import (
    first_collision_t,
    nearest_step,
    repo_path,
    row_clearance_to_obstacle,
)
from run_race_simple_phase_sweep import ROOT


DEFAULT_SLOT_REPORTS = [
    ROOT / "docs/data/race_hero_slot_wall_x23_y27p5_sx5_n3.json",
    ROOT / "docs/data/race_hero_slot_wall_x24_y26p5_sx5_n3.json",
]
DEFAULT_PAIRED_REPORTS = [
    (
        "x24_y27p5_sx5_sy2",
        ROOT / "docs/data/race_hero_base_pair_slot_wall_x24_y27p5_n3.json",
        ROOT / "docs/data/race_hero_dynamic_gate_slot_wall_x24_y27p5_n3.json",
    )
]
DEFAULT_OUT = ROOT / "docs/data/race_hero_slot_wall_failure_mechanism.json"


@dataclass(frozen=True)
class Case:
    label: str
    base_run: Path
    gate_run: Path


def dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def xy(pos: list[float]) -> list[float]:
    return [float(pos[0]), float(pos[1])]


def parse_label_path(raw: str) -> tuple[str | None, Path]:
    if ":" not in raw:
        return None, Path(raw)
    label, path = raw.split(":", 1)
    return label, Path(path)


def parse_paired_report(raw: str) -> tuple[str, Path, Path]:
    parts = raw.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "paired report must be LABEL:BASE_REPORT:GATE_REPORT"
        )
    return parts[0], Path(parts[1]), Path(parts[2])


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--slot-report",
        action="append",
        type=parse_label_path,
        help="Slot-wall report path, optionally LABEL:PATH.",
    )
    p.add_argument(
        "--paired-report",
        action="append",
        type=parse_paired_report,
        help="Older reports as LABEL:BASE_REPORT:GATE_REPORT.",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--focus-drone", type=int, default=3)
    p.add_argument("--pre-window-s", type=float, default=4.0)
    p.add_argument(
        "--snapshot-offset",
        action="append",
        type=float,
        default=[-3.0, -2.0, -1.0, -0.5, 0.0],
    )
    p.add_argument("--divergence-threshold", type=float, default=1.0)
    return p.parse_args(argv)


def first_row(report_path: Path) -> dict[str, Any]:
    report = _load_json(report_path)
    rows = report.get("rows") or []
    if not rows:
        raise ValueError(f"{report_path}: no rows")
    return rows[0]


def load_slot_case(label: str | None, report_path: Path) -> Case:
    row = first_row(report_path)
    arms = row.get("arms") or {}
    if "base_wall" not in arms or "gate_wall" not in arms:
        raise ValueError(f"{report_path}: expected base_wall and gate_wall arms")
    wall = row.get("wall") or {}
    return Case(
        label=label or str(wall.get("tag") or report_path.stem),
        base_run=ROOT / arms["base_wall"]["moving"]["run_dir"],
        gate_run=ROOT / arms["gate_wall"]["moving"]["run_dir"],
    )


def load_paired_case(label: str, base_report: Path, gate_report: Path) -> Case:
    base_row = first_row(base_report)
    gate_row = first_row(gate_report)
    return Case(
        label=label,
        base_run=ROOT / base_row["moving"]["run_dir"],
        gate_run=ROOT / gate_row["moving"]["run_dir"],
    )


def load_cases(args: argparse.Namespace) -> list[Case]:
    cases: list[Case] = []
    slot_reports = args.slot_report
    paired_reports = args.paired_report
    if slot_reports is None and paired_reports is None:
        slot_reports = [(None, path) for path in DEFAULT_SLOT_REPORTS]
        paired_reports = list(DEFAULT_PAIRED_REPORTS)
    for label, path in slot_reports or []:
        cases.append(load_slot_case(label, path))
    for label, base_report, gate_report in paired_reports or []:
        cases.append(load_paired_case(label, base_report, gate_report))
    return cases


def episode_log(run_dir: Path, episode: int, drone: int) -> Path:
    return run_dir / f"episode_{episode:03d}_drone_{drone:02d}.json"


def run_episode_count(run_dir: Path, drone: int) -> int:
    return len(sorted(run_dir.glob(f"episode_*_drone_{drone:02d}.json")))


def static_box(cfg: dict[str, Any]) -> dict[str, Any]:
    boxes = ((cfg.get("scenario") or {}).get("obstacles") or {}).get("boxes") or []
    if not boxes:
        raise ValueError("config has no static obstacle boxes")
    box = boxes[0]
    return {
        "center": [float(v) for v in box["center"]],
        "size": [float(v) for v in box["size"]],
    }


def voxelized_aabb(cfg: dict[str, Any], box: dict[str, Any]) -> dict[str, list[float]]:
    resolution = float(cfg["scenario"].get("resolution", 1.0))
    size = np.asarray(cfg["scenario"]["size"], dtype=float)
    center = np.asarray(box["center"], dtype=float)
    extent = np.asarray(box["size"], dtype=float)
    lo_cell = np.floor((center - 0.5 * extent) / resolution)
    hi_cell = np.ceil((center + 0.5 * extent) / resolution) - 1
    lo_cell = np.maximum(lo_cell, 0)
    hi_cell = np.minimum(hi_cell, size - 1)
    lo = lo_cell * resolution
    hi = (hi_cell + 1) * resolution
    return {
        "min": [float(v) for v in lo],
        "max": [float(v) for v in hi],
    }


def signed_distance_to_aabb(pos: list[float], aabb: dict[str, list[float]]) -> float:
    p = np.asarray(pos, dtype=float)
    lo = np.asarray(aabb["min"], dtype=float)
    hi = np.asarray(aabb["max"], dtype=float)
    outside = np.maximum(np.maximum(lo - p, p - hi), 0.0)
    outside_dist = float(np.linalg.norm(outside))
    if outside_dist > 0.0:
        return outside_dist
    return -float(np.min(np.minimum(p - lo, hi - p)))


def wall_clearance(
    cfg: dict[str, Any],
    step: dict[str, Any],
    aabb: dict[str, list[float]],
) -> dict[str, Any]:
    radius = float(cfg["simulator"].get("drone_radius", 0.4))
    dt = float(cfg["simulator"].get("dt", 0.05))
    pos = [float(v) for v in step["true_pos"]]
    vel = [float(v) for v in step.get("true_vel", [0.0, 0.0, 0.0])]
    cmd = [float(v) for v in step.get("cmd", vel)]
    next_vel = command_limited_velocity(cfg, vel, cmd)
    projected = [pos[i] + next_vel[i] * dt for i in range(len(pos))]
    now = signed_distance_to_aabb(pos, aabb) - radius
    next_step = signed_distance_to_aabb(projected, aabb) - radius
    return {
        "current_clearance_m": now,
        "projected_next_clearance_m": next_step,
        "projected_next_pos": projected,
        "projected_next_vel": next_vel,
        "min_clearance_m": min(now, next_step),
    }


def command_limited_velocity(
    cfg: dict[str, Any],
    velocity: list[float],
    command: list[float],
) -> list[float]:
    vel = np.asarray(velocity, dtype=float)
    cmd = np.asarray(command, dtype=float)
    max_accel = float(cfg["simulator"].get("max_accel", 80.0))
    dt = float(cfg["simulator"].get("dt", 0.05))
    dv = cmd - vel
    max_dv = max_accel * dt
    norm = float(np.linalg.norm(dv))
    if norm > max_dv:
        dv = dv * (max_dv / norm)
    return [float(v) for v in (vel + dv)]


def command_limited_next_pos(
    cfg: dict[str, Any],
    step: dict[str, Any],
) -> list[float]:
    dt = float(cfg["simulator"].get("dt", 0.05))
    pos = np.asarray(step["true_pos"], dtype=float)
    vel = [float(v) for v in step.get("true_vel", [0.0, 0.0, 0.0])]
    cmd = [float(v) for v in step.get("cmd", vel)]
    next_vel = np.asarray(command_limited_velocity(cfg, vel, cmd), dtype=float)
    return [float(v) for v in (pos + next_vel * dt)]


def dynamic_min(
    cfg: dict[str, Any],
    step: dict[str, Any],
    indices: range | list[int],
) -> dict[str, Any]:
    best: tuple[float, int] | None = None
    for idx in indices:
        clearance = row_clearance_to_obstacle(cfg, step, idx)
        if best is None or clearance < best[0]:
            best = (clearance, idx)
    if best is None:
        return {"clearance_m": None, "obstacle_idx": None}
    return {"clearance_m": best[0], "obstacle_idx": best[1]}


def window_dynamic_min(
    cfg: dict[str, Any],
    log: dict[str, Any],
    start_t: float,
    end_t: float,
    *,
    first_index: int,
) -> dict[str, Any]:
    obstacles = cfg["scenario"].get("dynamic_obstacles", []) or []
    indices = list(range(first_index, len(obstacles)))
    best: tuple[float, int, dict[str, Any]] | None = None
    for step in log.get("steps", []):
        t = float(step["t"])
        if not (start_t <= t <= end_t):
            continue
        row = dynamic_min(cfg, step, indices)
        if row["clearance_m"] is None:
            continue
        clearance = float(row["clearance_m"])
        if best is None or clearance < best[0]:
            best = (clearance, int(row["obstacle_idx"]), step)
    if best is None:
        return {"clearance_m": None, "t": None, "obstacle_idx": None}
    clearance, idx, step = best
    return {"clearance_m": clearance, "t": float(step["t"]), "obstacle_idx": idx}


def step_delta(base_step: dict[str, Any], gate_step: dict[str, Any]) -> float:
    return dist(
        [float(v) for v in base_step["true_pos"]],
        [float(v) for v in gate_step["true_pos"]],
    )


def first_time(
    *,
    base_log: dict[str, Any],
    gate_log: dict[str, Any],
    start_t: float,
    end_t: float,
    predicate: Any,
) -> float | None:
    for gate_step in gate_log.get("steps", []):
        t = float(gate_step["t"])
        if not (start_t <= t <= end_t):
            continue
        base_step = nearest_step(base_log, t)
        if predicate(base_step, gate_step):
            return t
    return None


def snapshot(
    *,
    base_cfg: dict[str, Any],
    gate_cfg: dict[str, Any],
    base_log: dict[str, Any],
    gate_log: dict[str, Any],
    aabb: dict[str, list[float]],
    t: float,
) -> dict[str, Any]:
    base_step = nearest_step(base_log, t)
    gate_step = nearest_step(gate_log, t)
    obstacles = gate_cfg["scenario"].get("dynamic_obstacles", []) or []
    return {
        "t": float(gate_step["t"]),
        "base_xy": xy(base_step["true_pos"]),
        "gate_xy": xy(gate_step["true_pos"]),
        "gate_projected_next_xy": xy(command_limited_next_pos(gate_cfg, gate_step)),
        "reference_xy": xy(gate_step.get("reference_pos", gate_step["true_pos"])),
        "gate_minus_base_xy": [
            float(gate_step["true_pos"][0]) - float(base_step["true_pos"][0]),
            float(gate_step["true_pos"][1]) - float(base_step["true_pos"][1]),
        ],
        "trajectory_delta_m": step_delta(base_step, gate_step),
        "base_wall_clearance": wall_clearance(base_cfg, base_step, aabb),
        "gate_wall_clearance": wall_clearance(gate_cfg, gate_step, aabb),
        "gate_all_dynamic_min": dynamic_min(gate_cfg, gate_step, range(len(obstacles))),
        "gate_extra_gate_min": dynamic_min(gate_cfg, gate_step, range(2, len(obstacles))),
        "gate_collision_flag": bool(gate_step.get("collision", False)),
    }


def summarize_episode(
    *,
    case: Case,
    episode: int,
    drone: int,
    pre_window_s: float,
    snapshot_offsets: list[float],
    divergence_threshold: float,
) -> dict[str, Any]:
    base_cfg = _load_yaml(case.base_run / "config.yaml")
    gate_cfg = _load_yaml(case.gate_run / "config.yaml")
    box = static_box(gate_cfg)
    aabb = voxelized_aabb(gate_cfg, box)
    base_log = _load_json(episode_log(case.base_run, episode, drone))
    gate_log = _load_json(episode_log(case.gate_run, episode, drone))
    collision_t = first_collision_t(gate_log)
    if collision_t is None:
        return {
            "episode": episode,
            "base_outcome": base_log.get("outcome"),
            "gate_outcome": gate_log.get("outcome"),
            "gate_first_collision_t": None,
        }
    start_t = max(0.0, collision_t - pre_window_s)
    gate_collision_step = nearest_step(gate_log, collision_t)
    base_at_collision = nearest_step(base_log, collision_t)
    gate_projected_next_pos = command_limited_next_pos(gate_cfg, gate_collision_step)
    first_delta_t = first_time(
        base_log=base_log,
        gate_log=gate_log,
        start_t=start_t,
        end_t=collision_t,
        predicate=lambda b, g: step_delta(b, g) >= divergence_threshold,
    )
    first_wall_conflict_t = first_time(
        base_log=base_log,
        gate_log=gate_log,
        start_t=start_t,
        end_t=collision_t,
        predicate=lambda _b, g: wall_clearance(gate_cfg, g, aabb)[
            "min_clearance_m"
        ]
        <= 0.0,
    )
    return {
        "episode": episode,
        "base_outcome": base_log.get("outcome"),
        "gate_outcome": gate_log.get("outcome"),
        "gate_first_collision_t": collision_t,
        "gate_collision_xy": xy(gate_collision_step["true_pos"]),
        "gate_projected_next_xy": xy(gate_projected_next_pos),
        "gate_collision_reference_xy": xy(
            gate_collision_step.get("reference_pos", gate_collision_step["true_pos"])
        ),
        "base_xy_at_gate_collision_t": xy(base_at_collision["true_pos"]),
        "trajectory_delta_at_collision_m": step_delta(
            base_at_collision, gate_collision_step
        ),
        "first_delta_ge_threshold_t": first_delta_t,
        "first_gate_wall_conflict_t": first_wall_conflict_t,
        "gate_wall_clearance_at_collision": wall_clearance(
            gate_cfg, gate_collision_step, aabb
        ),
        "base_wall_clearance_at_gate_collision_t": wall_clearance(
            base_cfg, base_at_collision, aabb
        ),
        "gate_dynamic_min_pre_collision": window_dynamic_min(
            gate_cfg,
            gate_log,
            start_t,
            collision_t,
            first_index=0,
        ),
        "gate_extra_gate_min_pre_collision": window_dynamic_min(
            gate_cfg,
            gate_log,
            start_t,
            collision_t,
            first_index=2,
        ),
        "snapshots": [
            snapshot(
                base_cfg=base_cfg,
                gate_cfg=gate_cfg,
                base_log=base_log,
                gate_log=gate_log,
                aabb=aabb,
                t=collision_t + offset,
            )
            for offset in snapshot_offsets
        ],
    }


def summarize_case(
    *,
    case: Case,
    drone: int,
    pre_window_s: float,
    snapshot_offsets: list[float],
    divergence_threshold: float,
) -> dict[str, Any]:
    gate_cfg = _load_yaml(case.gate_run / "config.yaml")
    box = static_box(gate_cfg)
    aabb = voxelized_aabb(gate_cfg, box)
    episodes = min(
        run_episode_count(case.base_run, drone),
        run_episode_count(case.gate_run, drone),
    )
    episode_rows = [
        summarize_episode(
            case=case,
            episode=episode,
            drone=drone,
            pre_window_s=pre_window_s,
            snapshot_offsets=snapshot_offsets,
            divergence_threshold=divergence_threshold,
        )
        for episode in range(episodes)
    ]
    collisions = [row for row in episode_rows if row["gate_first_collision_t"] is not None]
    return {
        "label": case.label,
        "base_run": repo_path(case.base_run),
        "gate_run": repo_path(case.gate_run),
        "wall": {
            "configured_box": box,
            "voxelized_aabb": aabb,
        },
        "episodes": episodes,
        "gate_collisions": len(collisions),
        "gate_collision_times": [
            row["gate_first_collision_t"] for row in collisions
        ],
        "episode_rows": episode_rows,
    }


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    cases = load_cases(args)
    report = {
        "source": {
            "script": "scripts/race_hero_slot_wall_failure_report.py",
        },
        "focus": {
            "drone": args.focus_drone,
            "pre_window_s": args.pre_window_s,
            "snapshot_offsets_s": args.snapshot_offset,
            "divergence_threshold_m": args.divergence_threshold,
        },
        "cases": [
            summarize_case(
                case=case,
                drone=args.focus_drone,
                pre_window_s=args.pre_window_s,
                snapshot_offsets=args.snapshot_offset,
                divergence_threshold=args.divergence_threshold,
            )
            for case in cases
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("| case | collisions | t first-last | delta@hit | gate wall next | extra gate min |")
    print("|---|---:|---:|---:|---:|---:|")
    for case in report["cases"]:
        rows = [
            row
            for row in case["episode_rows"]
            if row["gate_first_collision_t"] is not None
        ]
        if rows:
            times = [float(row["gate_first_collision_t"]) for row in rows]
            deltas = [float(row["trajectory_delta_at_collision_m"]) for row in rows]
            wall_next = [
                float(row["gate_wall_clearance_at_collision"]["projected_next_clearance_m"])
                for row in rows
            ]
            extra_gate = [
                float(row["gate_extra_gate_min_pre_collision"]["clearance_m"])
                for row in rows
                if row["gate_extra_gate_min_pre_collision"]["clearance_m"] is not None
            ]
            time_label = f"{min(times):.2f}-{max(times):.2f}"
            delta_label = f"{sum(deltas) / len(deltas):.2f}"
            wall_label = f"{sum(wall_next) / len(wall_next):+.2f}"
            extra_label = f"{min(extra_gate):+.2f}" if extra_gate else "n/a"
        else:
            time_label = "n/a"
            delta_label = "n/a"
            wall_label = "n/a"
            extra_label = "n/a"
        print(
            f"| {case['label']} | {case['gate_collisions']}/{case['episodes']} | "
            f"{time_label} | {delta_label} | {wall_label} | {extra_label} |"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
