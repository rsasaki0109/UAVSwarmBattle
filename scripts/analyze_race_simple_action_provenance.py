#!/usr/bin/env python3
"""Inspect GPU MPPI action provenance for a race-simple split-cell rerun."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from analyze_race_simple_phase_trace import (
    _load_json,
    _load_yaml,
    min_clearance_for_path,
    nearest,
    obstacle_positions,
)


DEFAULT_RUN_DIR = Path(
    "results/_race_simple_action_provenance/p19p8_y5p5_34p5/gpu_mppi"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--drone", type=int, default=3)
    p.add_argument("--lead", type=float, default=0.15)
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="summary JSON path; default: RUN_DIR/action_provenance_summary.json",
    )
    return p.parse_args()


def first_env_collision_step(log: dict[str, Any]) -> dict[str, Any] | None:
    for step in log.get("steps", []):
        if bool(step.get("collision", False)):
            return step
    return None


def vec_norm(values: list[float]) -> float:
    return math.sqrt(sum(float(v) ** 2 for v in values))


def vec_diff_norm(a: list[float], b: list[float]) -> float:
    return vec_norm([float(x) - float(y) for x, y in zip(a, b)])


def fmt_optional(value: float | None, fmt: str) -> str:
    return "n/a" if value is None else format(value, fmt)


def sign(value: float, eps: float = 1e-9) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


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


def action_relation(
    cfg: dict[str, Any],
    pos: list[float],
    t: float,
    action: list[float],
) -> str:
    obstacle_y_sign = nearest_obstacle_y_sign(cfg, pos, t)
    action_y_sign = sign(float(action[1])) if len(action) > 1 else 0
    if obstacle_y_sign == 0 or action_y_sign == 0:
        return "neutral"
    if action_y_sign == obstacle_y_sign:
        return "toward"
    return "escape"


def selected_visible_rollout_clearance(
    cfg: dict[str, Any],
    replan: dict[str, Any],
) -> float | None:
    rollouts = replan.get("rollouts") or []
    if not rollouts:
        return None
    best_idx = int(replan.get("best_rollout_idx", 0))
    if not 0 <= best_idx < len(rollouts):
        return None
    clearance, _ = min_clearance_for_path(
        cfg,
        rollouts[best_idx],
        replan_t=float(replan["t"]),
    )
    return clearance


def provenance_replan(log: dict[str, Any], target_t: float) -> dict[str, Any]:
    with_provenance = [
        row
        for row in log.get("replans", [])
        if (row.get("planner_meta") or {}).get("action_provenance")
    ]
    if not with_provenance:
        raise ValueError(
            "no action provenance found; rerun with --gpu-log-action-provenance"
        )
    return nearest(with_provenance, target_t)


def main() -> int:
    args = parse_args()
    out_json = args.out_json or (args.run_dir / "action_provenance_summary.json")
    cfg = _load_yaml(args.run_dir / "config.yaml")
    log = _load_json(
        args.run_dir / f"episode_{args.episode:03d}_drone_{args.drone:02d}.json"
    )
    collision_step = first_env_collision_step(log)
    if collision_step is None:
        raise ValueError("selected log has no env collision step")
    event_t = float(collision_step["t"])
    replan = provenance_replan(log, event_t - args.lead)
    provenance = replan["planner_meta"]["action_provenance"]
    replan_t = float(replan["t"])
    step = nearest(log["steps"], replan_t)
    pos = [float(v) for v in step["true_pos"]]
    cmd = [float(v) for v in step["cmd"]]
    selected_clearance = selected_visible_rollout_clearance(cfg, replan)

    actions = {
        "cmd": cmd,
        "chosen": provenance["chosen_action"],
        "softmax": provenance["softmax_action"],
        "argmax_weight": provenance["argmax_weight_action"],
        "argmin": provenance["argmin_action"],
    }
    action_rows = {
        name: {
            "action": values,
            "speed": vec_norm(values),
            "relation_to_nearest_obstacle_y": action_relation(
                cfg,
                pos,
                replan_t,
                values,
            ),
        }
        for name, values in actions.items()
    }
    top_rows = []
    for row in provenance.get("top_weighted_actions", []):
        action = row["action"]
        top_rows.append(
            {
                **row,
                "relation_to_nearest_obstacle_y": action_relation(
                    cfg,
                    pos,
                    replan_t,
                    action,
                ),
            }
        )

    summary = {
        "run_dir": str(args.run_dir),
        "episode": args.episode,
        "drone": args.drone,
        "outcome": log.get("outcome"),
        "event_t": event_t,
        "replan_t": replan_t,
        "action_source": provenance["action_source"],
        "cmd_chosen_diff_norm": vec_diff_norm(cmd, provenance["chosen_action"]),
        "chosen_softmax_diff_norm": vec_diff_norm(
            provenance["chosen_action"],
            provenance["softmax_action"],
        ),
        "selected_visible_rollout_clearance_m": selected_clearance,
        "weight_entropy": provenance["weight_entropy"],
        "weight_max": provenance["weight_max"],
        "weight_mass_by_action_y_sign": provenance["weight_mass_by_action_y_sign"],
        "actions": action_rows,
        "top_weighted_actions": top_rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"run={args.run_dir}")
    print(f"outcome={log.get('outcome')} event_t={event_t:.2f} replan_t={replan_t:.2f}")
    print(
        f"source={summary['action_source']} "
        f"cmd_vs_chosen={summary['cmd_chosen_diff_norm']:.3e} "
        f"chosen_vs_softmax={summary['chosen_softmax_diff_norm']:.3e}"
    )
    print(
        "selected_visible_rollout_clearance="
        f"{fmt_optional(selected_clearance, '+.2f')} m "
        f"weight_max={provenance['weight_max']:.3f} "
        f"entropy={provenance['weight_entropy']:.2f}"
    )
    mass = provenance["weight_mass_by_action_y_sign"]
    print(
        "weight_mass_y "
        f"positive={fmt_optional(mass['positive'], '.3f')} "
        f"negative={fmt_optional(mass['negative'], '.3f')}"
    )
    print("actions:")
    for name, row in action_rows.items():
        action = row["action"]
        print(
            f"  {name:<13} y={float(action[1]):+6.2f} "
            f"speed={row['speed']:.2f} "
            f"relation={row['relation_to_nearest_obstacle_y']}"
        )
    print("top weighted actions:")
    for row in top_rows[:5]:
        action = row["action"]
        print(
            f"  #{row['rank']} idx={row['sample_idx']} "
            f"w={row['weight']:.3f} cost={row['cost']:+.1f} "
            f"y={float(action[1]):+6.2f} "
            f"relation={row['relation_to_nearest_obstacle_y']}"
        )
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
