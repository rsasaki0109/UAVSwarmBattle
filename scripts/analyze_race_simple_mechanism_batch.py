#!/usr/bin/env python3
"""Batch mechanism metrics for race-simple phase-sweep logs.

This turns the single-trace GPU MPPI failure story into measurable rows:

- selected visible-rollout clearance at the pre-contact replan
- actual closed-loop clearance after that replan
- command-y escape-to-obstacle flip timing
- selected-rollout first-step direction vs. actual command-y mismatch
- paired MPC actual clearance over the same episode/drone/time window
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any

from analyze_race_simple_phase_trace import (
    _load_json,
    _load_yaml,
    clearance_to_obstacles,
    min_clearance_for_path,
    nearest,
    obstacle_positions,
)


DEFAULT_ROOT = Path("results/_race_simple_phase_sweep")
DEFAULT_OUT = DEFAULT_ROOT / "mechanism_batch_summary.json"
PLANNERS = ("mpc", "gpu_mppi")
DRONE_LOG_RE = re.compile(r"episode_(?P<episode>\d+)_drone_(?P<drone>\d+)\.json$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument(
        "--cell",
        action="append",
        dest="cells",
        help="cell directory to analyze; repeatable. default: all p19p8_y* cells with MPC and GPU logs",
    )
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    p.add_argument("--scan-start", type=float, default=28.0)
    p.add_argument("--scan-end", type=float, default=30.5)
    p.add_argument(
        "--lead",
        type=float,
        default=0.15,
        help="seconds before event used to select the GPU replan",
    )
    p.add_argument("--follow", type=float, default=0.35)
    p.add_argument("--near-threshold", type=float, default=0.10)
    p.add_argument("--cmd-eps", type=float, default=0.10)
    p.add_argument("--focus-drone", type=int, default=3)
    return p.parse_args()


def parse_drone_log_name(path: Path) -> tuple[int, int] | None:
    m = DRONE_LOG_RE.fullmatch(path.name)
    if m is None:
        return None
    return int(m.group("episode")), int(m.group("drone"))


def list_cells(root: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        cells = [root / name for name in requested]
    else:
        cells = sorted(
            p
            for p in root.iterdir()
            if p.is_dir()
            and p.name.startswith("p19p8_y")
            and all((p / planner / "config.yaml").exists() for planner in PLANNERS)
        )
    missing = [str(p) for p in cells if not p.exists()]
    if missing:
        raise FileNotFoundError("missing cell directories: " + ", ".join(missing))
    return cells


def sign(value: float, eps: float = 0.0) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def row_clearance(cfg: dict[str, Any], step: dict[str, Any]) -> float:
    pos = [float(v) for v in step["true_pos"]]
    t = float(step["t"])
    dt = float(cfg["simulator"].get("dt", 0.05))
    clearances = clearance_to_obstacles(cfg, pos, t)
    next_clearances = clearance_to_obstacles(cfg, pos, t + dt)
    if not clearances:
        return math.inf
    return min(min(clearances), min(next_clearances))


def nearest_obstacle_y_sign(cfg: dict[str, Any], pos: list[float], t: float) -> int:
    positions = obstacle_positions(cfg, t)
    if not positions:
        return 0
    nearest_pos = min(
        positions,
        key=lambda obs_pos: math.sqrt(
            sum((float(a) - float(b)) ** 2 for a, b in zip(pos, obs_pos))
        ),
    )
    return sign(float(nearest_pos[1]) - float(pos[1]))


def command_relation(
    cfg: dict[str, Any],
    step: dict[str, Any],
    *,
    cmd_eps: float,
) -> str:
    pos = [float(v) for v in step["true_pos"]]
    obstacle_y_sign = nearest_obstacle_y_sign(cfg, pos, float(step["t"]))
    cmd_y_sign = sign(float(step["cmd"][1]), cmd_eps)
    if obstacle_y_sign == 0 or cmd_y_sign == 0:
        return "neutral"
    if cmd_y_sign == obstacle_y_sign:
        return "toward"
    return "escape"


def steps_in_window(
    log: dict[str, Any],
    start_t: float,
    end_t: float,
) -> list[dict[str, Any]]:
    return [
        step
        for step in log.get("steps", [])
        if start_t <= float(step["t"]) <= end_t
    ]


def min_clearance_step(
    cfg: dict[str, Any],
    log: dict[str, Any],
    start_t: float,
    end_t: float,
) -> tuple[dict[str, Any] | None, float]:
    steps = steps_in_window(log, start_t, end_t)
    if not steps:
        steps = list(log.get("steps", []))
    if not steps:
        return None, math.nan
    best_step = min(steps, key=lambda step: row_clearance(cfg, step))
    return best_step, row_clearance(cfg, best_step)


def first_env_collision_step(log: dict[str, Any]) -> dict[str, Any] | None:
    for step in log.get("steps", []):
        if bool(step.get("collision", False)):
            return step
    return None


def find_escape_to_toward_flip(
    cfg: dict[str, Any],
    log: dict[str, Any],
    start_t: float,
    end_t: float,
    *,
    cmd_eps: float,
) -> dict[str, Any] | None:
    last_escape: dict[str, Any] | None = None
    for step in steps_in_window(log, start_t, end_t):
        relation = command_relation(cfg, step, cmd_eps=cmd_eps)
        if relation == "escape":
            last_escape = step
        elif relation == "toward" and last_escape is not None:
            return {
                "from_t": float(last_escape["t"]),
                "to_t": float(step["t"]),
                "from_cmd_y": float(last_escape["cmd"][1]),
                "to_cmd_y": float(step["cmd"][1]),
            }
    return None


def actual_window_metrics(
    cfg: dict[str, Any],
    log: dict[str, Any],
    start_t: float,
    end_t: float,
    *,
    cmd_eps: float,
) -> dict[str, Any]:
    steps = steps_in_window(log, start_t, end_t)
    if not steps:
        return {
            "min_clearance_m": None,
            "min_clearance_t": None,
            "cmd_y_at_start": None,
            "cmd_relation_at_start": None,
            "escape_to_toward_flip": None,
        }
    best_step = min(steps, key=lambda step: row_clearance(cfg, step))
    start_step = nearest(log["steps"], start_t)
    return {
        "min_clearance_m": row_clearance(cfg, best_step),
        "min_clearance_t": float(best_step["t"]),
        "cmd_y_at_start": float(start_step["cmd"][1]),
        "cmd_relation_at_start": command_relation(cfg, start_step, cmd_eps=cmd_eps),
        "escape_to_toward_flip": find_escape_to_toward_flip(
            cfg,
            log,
            start_t - 0.4,
            end_t,
            cmd_eps=cmd_eps,
        ),
    }


def selected_rollout_metrics(
    cfg: dict[str, Any],
    replan: dict[str, Any],
    actual_step: dict[str, Any],
    *,
    cmd_eps: float,
) -> dict[str, Any]:
    rollouts = replan.get("rollouts") or []
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
    selected_clearance = (
        clearances[best_idx]
        if 0 <= best_idx < len(clearances)
        else math.nan
    )
    selected_first_dy = math.nan
    selected_first_vy = math.nan
    selected_relation = None
    cmd_mismatch = None
    if 0 <= best_idx < len(rollouts) and len(rollouts[best_idx]) >= 2:
        path = rollouts[best_idx]
        dt = float(cfg["simulator"].get("dt", 0.05))
        selected_first_dy = float(path[1][1]) - float(path[0][1])
        selected_first_vy = selected_first_dy / dt if dt > 0.0 else math.nan
        obstacle_y_sign = nearest_obstacle_y_sign(
            cfg,
            [float(v) for v in path[0]],
            float(replan["t"]),
        )
        rollout_y_sign = sign(selected_first_dy, 1e-6)
        if obstacle_y_sign == 0 or rollout_y_sign == 0:
            selected_relation = "neutral"
        elif rollout_y_sign == obstacle_y_sign:
            selected_relation = "toward"
        else:
            selected_relation = "escape"
        cmd_y_sign = sign(float(actual_step["cmd"][1]), cmd_eps)
        cmd_mismatch = (
            cmd_y_sign != 0
            and rollout_y_sign != 0
            and cmd_y_sign != rollout_y_sign
        )
    return {
        "rollout_count": len(rollouts),
        "rollout_hit_count": hits,
        "rollout_min_clearance_m": min(clearances) if clearances else None,
        "selected_rollout_idx": best_idx,
        "selected_clearance_m": selected_clearance,
        "selected_first_dy_m": selected_first_dy,
        "selected_first_vy_mps": selected_first_vy,
        "selected_relation_to_obstacle": selected_relation,
        "cmd_y_vs_selected_first_dy_mismatch": cmd_mismatch,
    }


def paired_log_path(cell: Path, planner: str, episode: int, drone: int) -> Path:
    return cell / planner / f"episode_{episode:03d}_drone_{drone:02d}.json"


def analyze_gpu_log(
    cell: Path,
    gpu_cfg: dict[str, Any],
    mpc_cfg: dict[str, Any],
    path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    parsed = parse_drone_log_name(path)
    if parsed is None:
        raise ValueError(f"unexpected log filename: {path}")
    episode, drone = parsed
    gpu_log = _load_json(path)
    collision_step = first_env_collision_step(gpu_log)
    if collision_step is not None:
        event_step = collision_step
        event_kind = "gpu_env_collision"
        event_clearance = row_clearance(gpu_cfg, event_step)
    else:
        event_step, event_clearance = min_clearance_step(
            gpu_cfg,
            gpu_log,
            args.scan_start,
            args.scan_end,
        )
        event_kind = "gpu_min_clearance"
    if event_step is None:
        raise ValueError(f"{path}: no steps")

    event_t = float(event_step["t"])
    target_replan_t = max(0.0, event_t - args.lead)
    replan = nearest(gpu_log["replans"], target_replan_t)
    replan_t = float(replan["t"])
    actual_end_t = max(event_t, replan_t + args.follow)
    start_step = nearest(gpu_log["steps"], replan_t)
    gpu_actual = actual_window_metrics(
        gpu_cfg,
        gpu_log,
        replan_t,
        actual_end_t,
        cmd_eps=args.cmd_eps,
    )
    rollout = selected_rollout_metrics(
        gpu_cfg,
        replan,
        start_step,
        cmd_eps=args.cmd_eps,
    )

    mpc_log_path = paired_log_path(cell, "mpc", episode, drone)
    mpc_actual: dict[str, Any]
    mpc_outcome = None
    if mpc_log_path.exists():
        mpc_log = _load_json(mpc_log_path)
        mpc_outcome = mpc_log.get("outcome")
        mpc_actual = actual_window_metrics(
            mpc_cfg,
            mpc_log,
            replan_t,
            actual_end_t,
            cmd_eps=args.cmd_eps,
        )
    else:
        mpc_actual = {
            "min_clearance_m": None,
            "min_clearance_t": None,
            "cmd_y_at_start": None,
            "cmd_relation_at_start": None,
            "escape_to_toward_flip": None,
        }

    selected_clearance = rollout["selected_clearance_m"]
    actual_min = gpu_actual["min_clearance_m"]
    return {
        "cell": cell.name,
        "episode": episode,
        "drone": drone,
        "gpu_outcome": gpu_log.get("outcome"),
        "mpc_outcome": mpc_outcome,
        "event_kind": event_kind,
        "event_t": event_t,
        "event_clearance_m": event_clearance,
        "replan_t": replan_t,
        "actual_window_end_t": actual_end_t,
        "gpu": {
            "rollout": rollout,
            "actual": gpu_actual,
        },
        "mpc": {
            "actual": mpc_actual,
        },
        "mechanism_flags": {
            "selected_rollout_clean": (
                selected_clearance is not None and selected_clearance > 0.0
            ),
            "actual_near_contact": (
                actual_min is not None and actual_min <= args.near_threshold
            ),
            "selected_clean_actual_near": (
                selected_clearance is not None
                and selected_clearance > 0.0
                and actual_min is not None
                and actual_min <= args.near_threshold
            ),
            "gpu_cmd_escape_to_toward_flip": gpu_actual["escape_to_toward_flip"] is not None,
            "mpc_cmd_escape_to_toward_flip": mpc_actual["escape_to_toward_flip"] is not None,
        },
    }


def finite_values(rows: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
    out: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out.append(float(value))
    return out


def mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def min_or_none(values: list[float]) -> float | None:
    return min(values) if values else None


def summarize_cell(rows: list[dict[str, Any]], *, focus_drone: int) -> dict[str, Any]:
    env_rows = [row for row in rows if row["event_kind"] == "gpu_env_collision"]
    focus_rows = [row for row in rows if row["drone"] == focus_drone]

    def counts(source_rows: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "rows": len(source_rows),
            "mpc_success": sum(
                int(row.get("mpc_outcome") == "success")
                for row in source_rows
            ),
            "selected_clean_actual_near": sum(
                int(row["mechanism_flags"]["selected_clean_actual_near"])
                for row in source_rows
            ),
            "differential_clean_actual_near": sum(
                int(
                    row.get("mpc_outcome") == "success"
                    and row["mechanism_flags"]["selected_clean_actual_near"]
                )
                for row in source_rows
            ),
            "gpu_flip": sum(
                int(row["mechanism_flags"]["gpu_cmd_escape_to_toward_flip"])
                for row in source_rows
            ),
            "mpc_flip": sum(
                int(row["mechanism_flags"]["mpc_cmd_escape_to_toward_flip"])
                for row in source_rows
            ),
            "cmd_mismatch": sum(
                int(
                    bool(
                        row["gpu"]["rollout"].get(
                            "cmd_y_vs_selected_first_dy_mismatch"
                        )
                    )
                )
                for row in source_rows
            ),
        }

    return {
        "cell": rows[0]["cell"],
        "all_rows": len(rows),
        "gpu_env_rows": counts(env_rows),
        "focus_drone_rows": counts(focus_rows),
        "env_selected_clearance_mean_m": mean_or_none(
            finite_values(env_rows, ("gpu", "rollout", "selected_clearance_m"))
        ),
        "env_selected_clearance_min_m": min_or_none(
            finite_values(env_rows, ("gpu", "rollout", "selected_clearance_m"))
        ),
        "env_gpu_actual_min_mean_m": mean_or_none(
            finite_values(env_rows, ("gpu", "actual", "min_clearance_m"))
        ),
        "env_gpu_actual_min_min_m": min_or_none(
            finite_values(env_rows, ("gpu", "actual", "min_clearance_m"))
        ),
        "env_mpc_actual_min_mean_m": mean_or_none(
            finite_values(env_rows, ("mpc", "actual", "min_clearance_m"))
        ),
        "env_mpc_actual_min_min_m": min_or_none(
            finite_values(env_rows, ("mpc", "actual", "min_clearance_m"))
        ),
        "focus_gpu_actual_min_mean_m": mean_or_none(
            finite_values(focus_rows, ("gpu", "actual", "min_clearance_m"))
        ),
        "focus_mpc_actual_min_mean_m": mean_or_none(
            finite_values(focus_rows, ("mpc", "actual", "min_clearance_m"))
        ),
    }


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator}/{denominator}"


def print_summary(summaries: list[dict[str, Any]], *, focus_drone: int) -> None:
    print("GPU env-collision mechanism rows")
    print(
        "| cell | env rows | MPC success | diff clean->near | GPU flip | cmd mismatch | "
        "sel clear mean/min | GPU actual mean/min | MPC actual mean/min |"
    )
    print("|---|---:|---:|---:|---:|---:|---|---|---|")
    for item in summaries:
        env = item["gpu_env_rows"]
        print(
            f"| {item['cell']} | {env['rows']} | "
            f"{ratio(env['mpc_success'], env['rows'])} | "
            f"{ratio(env['differential_clean_actual_near'], env['rows'])} | "
            f"{ratio(env['gpu_flip'], env['rows'])} | "
            f"{ratio(env['cmd_mismatch'], env['rows'])} | "
            f"{fmt(item['env_selected_clearance_mean_m'])}/"
            f"{fmt(item['env_selected_clearance_min_m'])} | "
            f"{fmt(item['env_gpu_actual_min_mean_m'])}/"
            f"{fmt(item['env_gpu_actual_min_min_m'])} | "
            f"{fmt(item['env_mpc_actual_min_mean_m'])}/"
            f"{fmt(item['env_mpc_actual_min_min_m'])} |"
        )
    print()
    print(f"Focus-drone {focus_drone} rows")
    print(
        "| cell | rows | clean->near | GPU flip | MPC flip | cmd mismatch | "
        "GPU actual mean | MPC actual mean |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for item in summaries:
        focus = item["focus_drone_rows"]
        print(
            f"| {item['cell']} | {focus['rows']} | "
            f"{ratio(focus['selected_clean_actual_near'], focus['rows'])} | "
            f"{ratio(focus['gpu_flip'], focus['rows'])} | "
            f"{ratio(focus['mpc_flip'], focus['rows'])} | "
            f"{ratio(focus['cmd_mismatch'], focus['rows'])} | "
            f"{fmt(item['focus_gpu_actual_min_mean_m'])} | "
            f"{fmt(item['focus_mpc_actual_min_mean_m'])} |"
        )


def json_clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    return value


def main() -> int:
    args = parse_args()
    cells = list_cells(args.root, args.cells)
    rows: list[dict[str, Any]] = []
    for cell in cells:
        gpu_cfg = _load_yaml(cell / "gpu_mppi" / "config.yaml")
        mpc_cfg = _load_yaml(cell / "mpc" / "config.yaml")
        for path in sorted((cell / "gpu_mppi").glob("episode_*_drone_*.json")):
            rows.append(analyze_gpu_log(cell, gpu_cfg, mpc_cfg, path, args))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["cell"], []).append(row)
    summaries = [
        summarize_cell(grouped[cell], focus_drone=args.focus_drone)
        for cell in sorted(grouped)
    ]
    result = {
        "params": {
            "scan_start": args.scan_start,
            "scan_end": args.scan_end,
            "lead": args.lead,
            "follow": args.follow,
            "near_threshold": args.near_threshold,
            "focus_drone": args.focus_drone,
        },
        "cell_summaries": summaries,
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(json_clean(result), f, indent=2, sort_keys=True)
        f.write("\n")
    print_summary(summaries, focus_drone=args.focus_drone)
    print()
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
