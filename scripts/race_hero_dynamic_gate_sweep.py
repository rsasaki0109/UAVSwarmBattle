#!/usr/bin/env python3
"""Sweep dynamic-gate width/speed variants for the race hero cell."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from race_hero_control_sweep import (
    DEFAULT_BASE_CONFIG,
    DEFAULT_NO_OBSTACLE_RUN,
    Candidate,
    ExtraObstacle,
    candidate_config,
    candidate_passes,
    compact_summary,
    existing_log,
    fmt,
    min_candidate_clearance,
    moving_metrics,
    parse_float_list,
    repo_path,
    tag_float,
)
from analyze_race_simple_phase_trace import _load_json, _load_yaml
from run_race_simple_phase_sweep import ROOT, _write_yaml, run_one, summarize_run


DEFAULT_OUT_ROOT = ROOT / "results/_race_hero_dynamic_gate_sweep"
DEFAULT_OUT_JSON = ROOT / "docs/data/race_hero_dynamic_gate_width_speed_sweep.json"
DEFAULT_CANDIDATE = Candidate(19.8, 3.5, 13.0, 1.5, 1.75)


@dataclass(frozen=True)
class GateVariant:
    separation: float
    gate_speed: float
    center_y: float
    x: float
    z: float
    radius: float
    encounter_t: float

    @property
    def tag(self) -> str:
        return (
            f"gap{tag_float(self.separation)}"
            f"_vy{tag_float(self.gate_speed)}"
            f"_t{tag_float(self.encounter_t)}"
        )

    def extra_obstacles(self) -> tuple[ExtraObstacle, ExtraObstacle]:
        lower_y = self.center_y - 0.5 * self.separation
        upper_y = self.center_y + 0.5 * self.separation
        lower_start = lower_y - self.gate_speed * self.encounter_t
        upper_start = upper_y + self.gate_speed * self.encounter_t
        return (
            ExtraObstacle(
                start=(self.x, lower_start, self.z),
                velocity=(0.0, self.gate_speed, 0.0),
                radius=self.radius,
                reflect=False,
            ),
            ExtraObstacle(
                start=(self.x, upper_start, self.z),
                velocity=(0.0, -self.gate_speed, 0.0),
                radius=self.radius,
                reflect=False,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        lower_y = self.center_y - 0.5 * self.separation
        upper_y = self.center_y + 0.5 * self.separation
        return {
            "tag": self.tag,
            "separation_m": self.separation,
            "gate_speed_mps": self.gate_speed,
            "center_y": self.center_y,
            "x": self.x,
            "z": self.z,
            "radius_m": self.radius,
            "encounter_t": self.encounter_t,
            "target_y_at_encounter": [lower_y, upper_y],
            "extra_obstacles": [
                {
                    "start": [float(v) for v in extra.start],
                    "velocity": [float(v) for v in extra.velocity],
                    "radius": float(extra.radius),
                    "reflect": bool(extra.reflect),
                }
                for extra in self.extra_obstacles()
            ],
        }


def parse_candidate(raw: str) -> Candidate:
    values = [float(part) for part in raw.split(",") if part.strip()]
    if len(values) != 5:
        raise argparse.ArgumentTypeError(
            "candidate must be PERIOD,Y_LOW,Y_HIGH,SPEED,RADIUS"
        )
    return Candidate(*values)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    p.add_argument("--no-obstacle-run", type=Path, default=DEFAULT_NO_OBSTACLE_RUN)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--scratch-dir", type=Path, default=Path("/tmp/uavnav_race_gate_sweep"))
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--candidate", type=parse_candidate, default=DEFAULT_CANDIDATE)
    p.add_argument("--separation", type=parse_float_list, default=[2.8, 2.4, 2.0, 1.6])
    p.add_argument("--gate-speed", type=parse_float_list, default=[0.32, 0.48])
    p.add_argument("--center-y", type=float, default=31.5)
    p.add_argument("--x", type=float, default=24.5)
    p.add_argument("--z", type=float, default=7.0)
    p.add_argument("--gate-radius", type=float, default=1.75)
    p.add_argument("--encounter-t", type=parse_float_list, default=[28.5])
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--safety-margin", type=float, default=0.8)
    p.add_argument("--w-obs", type=float, default=500.0)
    p.add_argument("--w-reach-time", type=float, default=1000.0)
    p.add_argument("--w-clean-ctg", type=float, default=100.0)
    p.add_argument("--fallback-to-argmin", action="store_true", default=True)
    p.add_argument("--fallback-commit-steps", type=int, default=3)
    p.add_argument("--score-collision-after-goal", action="store_true", default=True)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--focus-drone", type=int, default=3)
    p.add_argument("--focus-obstacle", type=int, default=-1)
    p.add_argument("--start-step", type=int, default=520)
    p.add_argument("--end-step", type=int, default=632)
    p.add_argument("--top-moving", type=int, default=4)
    p.add_argument("--screen-only", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--rerun-existing", action="store_true")
    p.add_argument("--target-moving-clearance", type=float, default=0.25)
    p.add_argument("--target-no-obstacle-clearance", type=float, default=-0.5)
    p.add_argument("--target-path-delta", type=float, default=1.0)
    p.add_argument("--min-conflicting-obstacles", type=int, default=4)
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    base_cfg = _load_yaml(args.base_config)
    candidate = args.candidate
    dt = float(base_cfg["simulator"].get("dt", 0.05))
    start_t = args.start_step * dt
    end_t = args.end_step * dt
    no_obstacle_log = _load_json(
        args.no_obstacle_run
        / f"episode_{args.episode:03d}_drone_{args.focus_drone:02d}.json"
    )
    variants = [
        GateVariant(
            separation=float(separation),
            gate_speed=float(gate_speed),
            center_y=float(args.center_y),
            x=float(args.x),
            z=float(args.z),
            radius=float(args.gate_radius),
            encounter_t=float(encounter_t),
        )
        for encounter_t in args.encounter_t
        for separation in args.separation
        for gate_speed in args.gate_speed
    ]

    rows: list[dict[str, Any]] = []
    for variant in variants:
        cfg = candidate_config(
            base_cfg,
            candidate,
            temperature=args.temperature,
            safety_margin=args.safety_margin,
            w_obs=args.w_obs,
            w_reach_time=args.w_reach_time,
            w_clean_ctg=args.w_clean_ctg,
            fallback_to_argmin=args.fallback_to_argmin,
            fallback_commit_steps=args.fallback_commit_steps,
            dynamic_branch_sampling=False,
            dynamic_branch_extra_radius=None,
            dynamic_branch_lateral_gain=None,
            dynamic_branch_speeds=None,
            dynamic_branch_max_obstacles=None,
            score_collision_after_goal=args.score_collision_after_goal,
            n=args.n,
            seed=args.seed,
            output_root=args.output_root / variant.tag,
            extra_obstacles=variant.extra_obstacles(),
            static_boxes=(),
        )
        ghost = min_candidate_clearance(
            cfg=cfg,
            log=no_obstacle_log,
            obstacle_idx=args.focus_obstacle,
            start_t=start_t,
            end_t=end_t,
        )
        rows.append(
            {
                "gate": variant.to_dict(),
                "candidate": {
                    "tag": candidate.tag,
                    "period": candidate.period,
                    "y_low": candidate.y_low,
                    "y_high": candidate.y_high,
                    "speed": candidate.speed,
                    "radius": candidate.radius,
                },
                "screen_selected": False,
                "moving_config": cfg,
                "no_obstacle": ghost,
            }
        )

    rows.sort(key=sort_key)
    selected = rows[: max(0, int(args.top_moving))]
    for row in selected:
        row["screen_selected"] = True

    if not args.screen_only:
        for row in selected:
            cfg = copy.deepcopy(row["moving_config"])
            run_dir = Path(cfg["output"]["dir"])
            config_path = args.scratch_dir / f"{row['gate']['tag']}_moving.yaml"
            _write_yaml(config_path, cfg)
            if (
                not args.summarize_only
                and (
                    args.rerun_existing
                    or not existing_log(
                        run_dir,
                        episode=args.episode,
                        drone=args.focus_drone,
                    )
                )
            ):
                run_one(config_path, python=str(args.python))
            if existing_log(run_dir, episode=args.episode, drone=args.focus_drone):
                row["moving"] = moving_metrics(
                    cfg=cfg,
                    moving_run=run_dir,
                    no_obstacle_log=no_obstacle_log,
                    episode=args.episode,
                    drone=args.focus_drone,
                    obstacle_idx=args.focus_obstacle,
                    start_t=start_t,
                    end_t=end_t,
                )
                row["moving"]["summary"] = compact_summary(summarize_run(run_dir, cfg))

    thresholds = {
        "moving_min_clearance_m": args.target_moving_clearance,
        "no_obstacle_max_clearance_m": args.target_no_obstacle_clearance,
        "path_delta_m": args.target_path_delta,
        "min_conflicting_obstacles": args.min_conflicting_obstacles,
    }
    public_rows: list[dict[str, Any]] = []
    for row in rows:
        out = {key: value for key, value in row.items() if key != "moving_config"}
        out["passes_thresholds"] = candidate_passes(
            out,
            target_no_obstacle_clearance=args.target_no_obstacle_clearance,
            target_moving_clearance=args.target_moving_clearance,
            target_path_delta=args.target_path_delta,
            min_conflicting_obstacles=args.min_conflicting_obstacles,
        )
        public_rows.append(out)

    report = {
        "source": {
            "script": "scripts/race_hero_dynamic_gate_sweep.py",
            "base_config": repo_path(args.base_config),
            "no_obstacle_run": repo_path(args.no_obstacle_run),
        },
        "focus": {
            "episode": args.episode,
            "drone": args.focus_drone,
            "obstacle": args.focus_obstacle,
            "window_s": [start_t, end_t],
        },
        "thresholds": thresholds,
        "planner_variant": {
            "temperature": args.temperature,
            "safety_margin": args.safety_margin,
            "w_obs": args.w_obs,
            "w_reach_time": args.w_reach_time,
            "w_clean_ctg": args.w_clean_ctg,
            "fallback_to_argmin": args.fallback_to_argmin,
            "fallback_commit_steps": args.fallback_commit_steps,
            "score_collision_after_goal": args.score_collision_after_goal,
        },
        "rows": public_rows,
        "survivors": [
            row["gate"]["tag"]
            for row in public_rows
            if row["passes_thresholds"]
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        "| rank | selected | gate | ghost | moving | clear | delta | joint | pass |"
    )
    print("|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(public_rows[: max(10, args.top_moving)], start=1):
        moving = row.get("moving") or {}
        moving_clear = (moving.get("window_min_clearance") or {}).get("clearance_m")
        delta = (moving.get("path_delta_to_no_obstacle") or {}).get("max_delta_m")
        summary = moving.get("summary") or {}
        joint = summary.get("joint_success")
        episodes = summary.get("episodes")
        joint_label = "n/a" if joint is None else f"{joint}/{episodes}"
        print(
            f"| {rank} | {str(row['screen_selected']).lower()} | "
            f"{row['gate']['tag']} | {fmt(row['no_obstacle'].get('clearance_m'))} | "
            f"{moving.get('outcome', 'n/a')} | {fmt(moving_clear)} | "
            f"{fmt(delta, signed=False)} | {joint_label} | "
            f"{str(row['passes_thresholds']).lower()} |"
        )
    print(f"survivors: {', '.join(report['survivors']) if report['survivors'] else 'none'}")
    print(f"wrote {args.out}")
    return 0


def sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    per_obstacle = row["no_obstacle"].get("per_obstacle_min_clearance") or []
    gate_conflict = 0.0
    for obstacle in per_obstacle:
        idx = int(obstacle.get("obstacle_idx", -1))
        if idx >= 2 and obstacle.get("clearance_m") is not None:
            gate_conflict += min(0.0, float(obstacle["clearance_m"]))
    ghost = row["no_obstacle"].get("clearance_m")
    if ghost is None:
        ghost = math.inf
    return (gate_conflict, float(ghost), float(row["gate"]["separation_m"]))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
