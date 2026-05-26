#!/usr/bin/env python3
"""Sweep static slot-wall variants against the dynamic-gate race cell.

The goal is to distinguish three cases:

* wall_too_blunt: the static wall alone breaks the base paired-sweeper scene.
* base_wall_failure: the static wall breaks the base scene but the gate scene
  survives, so the wall perturbs the baseline in a different way.
* still_solved: the base wall and extra dynamic-gate wall both succeed.
* gate_wall_boundary: the base wall succeeds, but the extra dynamic gate plus
  the same wall fails.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analyze_race_simple_phase_trace import _load_json, _load_yaml
from race_hero_control_sweep import (
    DEFAULT_BASE_CONFIG,
    DEFAULT_NO_OBSTACLE_RUN,
    Candidate,
    StaticBox,
    candidate_config,
    compact_summary,
    existing_log,
    fmt,
    min_candidate_clearance,
    moving_metrics,
    parse_float_list,
    repo_path,
    tag_float,
)
from race_hero_dynamic_gate_sweep import gate_obstacles
from run_race_simple_phase_sweep import ROOT, _write_yaml, run_one, summarize_run


DEFAULT_OUT_ROOT = ROOT / "results/_race_hero_slot_wall_sweep"
DEFAULT_OUT_JSON = ROOT / "docs/data/race_hero_slot_wall_sweep.json"
DEFAULT_CANDIDATE = Candidate(19.8, 3.5, 13.0, 1.5, 1.75)


@dataclass(frozen=True)
class WallVariant:
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float

    @property
    def tag(self) -> str:
        return (
            f"x{tag_float(self.center_x)}"
            f"_y{tag_float(self.center_y)}"
            f"_sx{tag_float(self.size_x)}"
            f"_sy{tag_float(self.size_y)}"
        )

    def static_box(self) -> StaticBox:
        return StaticBox(
            center=(self.center_x, self.center_y, self.center_z),
            size=(self.size_x, self.size_y, self.size_z),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "center": [self.center_x, self.center_y, self.center_z],
            "size": [self.size_x, self.size_y, self.size_z],
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
    p.add_argument("--scratch-dir", type=Path, default=Path("/tmp/uavnav_slot_wall_sweep"))
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--candidate", type=parse_candidate, default=DEFAULT_CANDIDATE)
    p.add_argument("--wall-center-x", type=parse_float_list, default=[24.0])
    p.add_argument("--wall-center-y", type=parse_float_list, default=[27.5])
    p.add_argument("--wall-center-z", type=float, default=7.0)
    p.add_argument("--wall-size-x", type=parse_float_list, default=[5.0])
    p.add_argument("--wall-size-y", type=parse_float_list, default=[2.0])
    p.add_argument("--wall-size-z", type=float, default=14.0)
    p.add_argument("--gate-separation", type=float, default=0.8)
    p.add_argument("--gate-speed", type=float, default=0.64)
    p.add_argument("--gate-center-y", type=float, default=31.5)
    p.add_argument("--gate-x", type=float, default=24.5)
    p.add_argument("--gate-z", type=float, default=7.0)
    p.add_argument("--gate-radius", type=float, default=1.75)
    p.add_argument("--gate-encounter-t", type=float, default=28.5)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--safety-margin", type=float, default=0.8)
    p.add_argument("--w-obs", type=float, default=500.0)
    p.add_argument("--w-reach-time", type=float, default=1000.0)
    p.add_argument("--w-clean-ctg", type=float, default=100.0)
    p.add_argument(
        "--inflate",
        type=int,
        help="Override the GPU MPPI static obstacle inflation in grid cells.",
    )
    p.add_argument("--fallback-to-argmin", action="store_true", default=True)
    p.add_argument("--fallback-commit-steps", type=int, default=3)
    p.add_argument("--score-collision-after-goal", action="store_true", default=True)
    p.add_argument(
        "--rollout-max-accel",
        type=float,
        help="Opt into acceleration-limited GPU MPPI rollout dynamics.",
    )
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--focus-drone", type=int, default=3)
    p.add_argument("--focus-obstacle", type=int, default=-1)
    p.add_argument("--start-step", type=int, default=520)
    p.add_argument("--end-step", type=int, default=632)
    p.add_argument("--screen-only", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--rerun-existing", action="store_true")
    return p.parse_args(argv)


def build_walls(args: argparse.Namespace) -> list[WallVariant]:
    walls: list[WallVariant] = []
    for center_x in args.wall_center_x:
        for center_y in args.wall_center_y:
            for size_x in args.wall_size_x:
                for size_y in args.wall_size_y:
                    walls.append(
                        WallVariant(
                            center_x=float(center_x),
                            center_y=float(center_y),
                            center_z=float(args.wall_center_z),
                            size_x=float(size_x),
                            size_y=float(size_y),
                            size_z=float(args.wall_size_z),
                        )
                    )
    return walls


def arm_config(
    *,
    args: argparse.Namespace,
    base_cfg: dict[str, Any],
    wall: WallVariant,
    arm: str,
    extra_gate: bool,
) -> dict[str, Any]:
    extra_obstacles = ()
    if extra_gate:
        extra_obstacles = gate_obstacles(
            x=args.gate_x,
            center_y=args.gate_center_y,
            separation=args.gate_separation,
            gate_speed=args.gate_speed,
            z=args.gate_z,
            radius=args.gate_radius,
            encounter_t=args.gate_encounter_t,
        )
    return candidate_config(
        base_cfg,
        args.candidate,
        temperature=args.temperature,
        safety_margin=args.safety_margin,
        w_obs=args.w_obs,
        w_reach_time=args.w_reach_time,
        w_clean_ctg=args.w_clean_ctg,
        inflate=args.inflate,
        fallback_to_argmin=args.fallback_to_argmin,
        fallback_commit_steps=args.fallback_commit_steps,
        dynamic_branch_sampling=False,
        dynamic_branch_extra_radius=None,
        dynamic_branch_lateral_gain=None,
        dynamic_branch_speeds=None,
        dynamic_branch_max_obstacles=None,
        score_collision_after_goal=args.score_collision_after_goal,
        rollout_max_accel=args.rollout_max_accel,
        n=args.n,
        seed=args.seed,
        output_root=args.output_root / wall.tag / arm,
        extra_obstacles=extra_obstacles,
        static_boxes=(wall.static_box(),),
    )


def run_arm(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    no_obstacle_log: dict[str, Any],
    wall: WallVariant,
    arm: str,
    start_t: float,
    end_t: float,
) -> dict[str, Any]:
    ghost = min_candidate_clearance(
        cfg=cfg,
        log=no_obstacle_log,
        obstacle_idx=args.focus_obstacle,
        start_t=start_t,
        end_t=end_t,
    )
    row: dict[str, Any] = {
        "no_obstacle": ghost,
        "run_config": repo_path(Path(cfg["output"]["dir"]) / "config.yaml"),
    }
    run_dir = Path(cfg["output"]["dir"])
    config_path = args.scratch_dir / f"{wall.tag}_{arm}.yaml"
    if not args.screen_only:
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
    return row


def joint_success(arm: dict[str, Any]) -> int | None:
    summary = (arm.get("moving") or {}).get("summary") or {}
    value = summary.get("joint_success")
    return None if value is None else int(value)


def classify(row: dict[str, Any], episodes: int) -> str:
    base_joint = joint_success(row["arms"]["base_wall"])
    gate_joint = joint_success(row["arms"]["gate_wall"])
    if base_joint is None or gate_joint is None:
        return "not_run"
    if base_joint < episodes and gate_joint < episodes:
        return "wall_too_blunt"
    if base_joint < episodes:
        return "base_wall_failure"
    if gate_joint < episodes:
        return "gate_wall_boundary"
    return "still_solved"


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    base_cfg = _load_yaml(args.base_config)
    no_obstacle_log = _load_json(
        args.no_obstacle_run
        / f"episode_{args.episode:03d}_drone_{args.focus_drone:02d}.json"
    )
    dt = float(base_cfg["simulator"].get("dt", 0.05))
    start_t = args.start_step * dt
    end_t = args.end_step * dt

    rows: list[dict[str, Any]] = []
    for wall in build_walls(args):
        arms: dict[str, Any] = {}
        for arm, extra_gate in (("base_wall", False), ("gate_wall", True)):
            cfg = arm_config(
                args=args,
                base_cfg=copy.deepcopy(base_cfg),
                wall=wall,
                arm=arm,
                extra_gate=extra_gate,
            )
            arms[arm] = run_arm(
                args=args,
                cfg=cfg,
                no_obstacle_log=no_obstacle_log,
                wall=wall,
                arm=arm,
                start_t=start_t,
                end_t=end_t,
            )
        row = {"wall": wall.to_dict(), "arms": arms}
        row["classification"] = classify(row, int(args.n))
        rows.append(row)

    report = {
        "source": {
            "script": "scripts/race_hero_slot_wall_sweep.py",
            "base_config": repo_path(args.base_config),
            "no_obstacle_run": repo_path(args.no_obstacle_run),
        },
        "focus": {
            "episode": args.episode,
            "drone": args.focus_drone,
            "obstacle": args.focus_obstacle,
            "window_s": [start_t, end_t],
        },
        "gate": {
            "x": args.gate_x,
            "center_y": args.gate_center_y,
            "separation_m": args.gate_separation,
            "gate_speed_mps": args.gate_speed,
            "z": args.gate_z,
            "radius_m": args.gate_radius,
            "encounter_t": args.gate_encounter_t,
        },
        "planner_variant": {
            "temperature": args.temperature,
            "safety_margin": args.safety_margin,
            "w_obs": args.w_obs,
            "w_reach_time": args.w_reach_time,
            "w_clean_ctg": args.w_clean_ctg,
            "inflate": args.inflate,
            "fallback_to_argmin": args.fallback_to_argmin,
            "fallback_commit_steps": args.fallback_commit_steps,
            "score_collision_after_goal": args.score_collision_after_goal,
            "rollout_max_accel": args.rollout_max_accel,
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("| wall | base joint | gate joint | gate clear | gate delta | class |")
    print("|---|---:|---:|---:|---:|---|")
    for row in rows:
        base = row["arms"]["base_wall"]
        gate = row["arms"]["gate_wall"]
        base_summary = (base.get("moving") or {}).get("summary") or {}
        gate_summary = (gate.get("moving") or {}).get("summary") or {}
        gate_moving = gate.get("moving") or {}
        gate_clear = (gate_moving.get("window_min_clearance") or {}).get("clearance_m")
        gate_delta = (gate_moving.get("path_delta_to_no_obstacle") or {}).get(
            "max_delta_m"
        )
        print(
            f"| {row['wall']['tag']} | "
            f"{base_summary.get('joint_success', 'n/a')}/{base_summary.get('episodes', 'n/a')} | "
            f"{gate_summary.get('joint_success', 'n/a')}/{gate_summary.get('episodes', 'n/a')} | "
            f"{fmt(gate_clear)} | {fmt(gate_delta, signed=False)} | "
            f"{row['classification']} |"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
