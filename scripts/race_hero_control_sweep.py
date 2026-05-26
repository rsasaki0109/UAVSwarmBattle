#!/usr/bin/env python3
"""Screen race-hero dynamic-obstacle cells before rendering a GIF.

The first pass is intentionally cheap: reuse one no-obstacle low-temp
trajectory and score it against many hypothetical moving-sweeper tubes.
Only the deepest virtual conflicts are then rerun with the actual moving
obstacles enabled.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analyze_race_simple_phase_trace import _load_json, _load_yaml
from dynamic_encounter_report import (
    first_collision_t,
    path_length,
    repo_path,
    row_clearance_to_obstacle,
    trajectory_delta,
    window_min_clearance,
)
from run_race_simple_phase_sweep import (
    ROOT,
    _set_dynamic_obstacles,
    _write_yaml,
    run_one,
    summarize_run,
)


DEFAULT_BASE_CONFIG = (
    ROOT / "results/_race_simple_causal_probe/p19p8_y5p0_35p0/t0p1/config.yaml"
)
DEFAULT_NO_OBSTACLE_RUN = (
    ROOT
    / "results/_race_hero_causality_controls/p19p8_y5p0_35p0/no_sweeper_t0p1"
)
DEFAULT_OUT_ROOT = ROOT / "results/_race_hero_control_sweep"
DEFAULT_OUT_JSON = ROOT / "docs/data/race_hero_control_sweep.json"


@dataclass(frozen=True)
class Candidate:
    period: float
    y_low: float
    y_high: float
    speed: float
    radius: float

    @property
    def tag(self) -> str:
        return (
            f"p{tag_float(self.period)}"
            f"_y{tag_float(self.y_low)}_{tag_float(self.y_high)}"
            f"_v{tag_float(self.speed)}_r{tag_float(self.radius)}"
        )


@dataclass(frozen=True)
class ExtraObstacle:
    start: tuple[float, float, float]
    velocity: tuple[float, float, float]
    radius: float
    reflect: bool = True


@dataclass(frozen=True)
class StaticBox:
    center: tuple[float, float, float]
    size: tuple[float, float, float]


def tag_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def temp_tag(value: float) -> str:
    return f"t{value:g}".replace(".", "p")


def parse_float_list(raw: str) -> list[float]:
    return [float(part) for part in raw.split(",") if part.strip()]


def parse_candidate(raw: str) -> Candidate:
    parts = [float(part) for part in raw.split(",")]
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "candidate must be PERIOD,Y_LOW,Y_HIGH,SPEED,RADIUS"
        )
    return Candidate(*parts)


def parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {raw}")


def parse_extra_obstacle(raw: str) -> ExtraObstacle:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) not in {7, 8}:
        raise argparse.ArgumentTypeError(
            "extra obstacle must be X,Y,Z,VX,VY,VZ,RADIUS[,REFLECT]"
        )
    values = [float(part) for part in parts[:7]]
    reflect = parse_bool(parts[7]) if len(parts) == 8 else True
    return ExtraObstacle(
        start=(values[0], values[1], values[2]),
        velocity=(values[3], values[4], values[5]),
        radius=values[6],
        reflect=reflect,
    )


def parse_static_box(raw: str) -> StaticBox:
    values = [float(part) for part in raw.split(",") if part.strip()]
    if len(values) != 6:
        raise argparse.ArgumentTypeError(
            "static box must be CENTER_X,CENTER_Y,CENTER_Z,SIZE_X,SIZE_Y,SIZE_Z"
        )
    return StaticBox(
        center=(values[0], values[1], values[2]),
        size=(values[3], values[4], values[5]),
    )


def default_candidates(args: argparse.Namespace) -> list[Candidate]:
    rows: list[Candidate] = []
    for period in args.period:
        for y_low in args.y_low:
            y_high = 40.0 - y_low
            for speed in args.speed:
                for radius in args.radius:
                    rows.append(
                        Candidate(
                            period=float(period),
                            y_low=float(y_low),
                            y_high=float(y_high),
                            speed=float(speed),
                            radius=float(radius),
                        )
                    )
    return rows


def candidate_config(
    base_cfg: dict[str, Any],
    candidate: Candidate,
    *,
    temperature: float,
    safety_margin: float | None,
    w_obs: float | None,
    w_reach_time: float | None,
    w_clean_ctg: float | None,
    inflate: int | None,
    fallback_to_argmin: bool,
    fallback_commit_steps: int | None,
    dynamic_branch_sampling: bool,
    dynamic_branch_extra_radius: float | None,
    dynamic_branch_lateral_gain: float | None,
    dynamic_branch_speeds: tuple[float, ...] | None,
    dynamic_branch_max_obstacles: int | None,
    score_collision_after_goal: bool,
    rollout_max_accel: float | None,
    n: int,
    seed: int,
    output_root: Path,
    extra_obstacles: tuple[ExtraObstacle, ...],
    static_boxes: tuple[StaticBox, ...],
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["name"] = f"race_hero_control_sweep_{candidate.tag}_{temp_tag(temperature)}"
    cfg["seed"] = int(seed)
    cfg["num_episodes"] = int(n)
    scenario = cfg["scenario"]
    scenario["period"] = float(candidate.period)
    _set_dynamic_obstacles(
        cfg,
        y_low=candidate.y_low,
        y_high=candidate.y_high,
        speed=candidate.speed,
        radius=candidate.radius,
    )
    append_extra_obstacles(cfg, extra_obstacles)
    append_static_boxes(cfg, static_boxes)
    dt = float(cfg["simulator"].get("dt", 0.05))
    n_loops = int(scenario.get("n_loops", 2))
    cfg["simulator"]["max_steps"] = int(round(float(candidate.period) * n_loops / dt))
    planner = cfg.setdefault("planner", {})
    planner["type"] = "gpu_mppi"
    planner["temperature"] = float(temperature)
    if inflate is not None:
        planner["inflate"] = int(inflate)
    if safety_margin is not None:
        planner["safety_margin"] = float(safety_margin)
    if w_obs is not None:
        planner["w_obs"] = float(w_obs)
    if w_reach_time is not None:
        planner["w_reach_time"] = float(w_reach_time)
    if w_clean_ctg is not None:
        planner["w_clean_ctg"] = float(w_clean_ctg)
    planner["fallback_to_argmin"] = bool(fallback_to_argmin)
    if fallback_commit_steps is not None:
        planner["fallback_commit_steps"] = int(fallback_commit_steps)
    planner["dynamic_branch_sampling"] = bool(dynamic_branch_sampling)
    if dynamic_branch_extra_radius is not None:
        planner["dynamic_branch_extra_radius"] = float(dynamic_branch_extra_radius)
    if dynamic_branch_lateral_gain is not None:
        planner["dynamic_branch_lateral_gain"] = float(dynamic_branch_lateral_gain)
    if dynamic_branch_speeds is not None:
        planner["dynamic_branch_speeds"] = [float(v) for v in dynamic_branch_speeds]
    if dynamic_branch_max_obstacles is not None:
        planner["dynamic_branch_max_obstacles"] = int(dynamic_branch_max_obstacles)
    planner["score_collision_after_goal"] = bool(score_collision_after_goal)
    if rollout_max_accel is not None:
        planner["rollout_max_accel"] = float(rollout_max_accel)
    planner["mode_aware_sampling"] = False
    planner["log_action_provenance"] = True
    cfg.setdefault("output", {})["dir"] = str(
        output_root
        / candidate.tag
        / planner_variant_tag(
            temperature=temperature,
            safety_margin=safety_margin,
            w_obs=w_obs,
            w_reach_time=w_reach_time,
            w_clean_ctg=w_clean_ctg,
            inflate=inflate,
            fallback_to_argmin=fallback_to_argmin,
            fallback_commit_steps=fallback_commit_steps,
            dynamic_branch_sampling=dynamic_branch_sampling,
            dynamic_branch_extra_radius=dynamic_branch_extra_radius,
            dynamic_branch_lateral_gain=dynamic_branch_lateral_gain,
            dynamic_branch_speeds=dynamic_branch_speeds,
            dynamic_branch_max_obstacles=dynamic_branch_max_obstacles,
            score_collision_after_goal=score_collision_after_goal,
            rollout_max_accel=rollout_max_accel,
            extra_obstacles=extra_obstacles,
            static_boxes=static_boxes,
        )
    )
    return cfg


def append_static_boxes(
    cfg: dict[str, Any],
    static_boxes: tuple[StaticBox, ...],
) -> None:
    if not static_boxes:
        return
    obstacles = cfg["scenario"].setdefault("obstacles", {})
    obstacles["type"] = "none"
    boxes = obstacles.setdefault("boxes", [])
    for box in static_boxes:
        boxes.append(
            {
                "center": [float(v) for v in box.center],
                "size": [float(v) for v in box.size],
            }
        )


def append_extra_obstacles(
    cfg: dict[str, Any],
    extra_obstacles: tuple[ExtraObstacle, ...],
) -> None:
    dyn = cfg["scenario"].setdefault("dynamic_obstacles", [])
    for extra in extra_obstacles:
        dyn.append(
            {
                "start": [float(v) for v in extra.start],
                "velocity": [float(v) for v in extra.velocity],
                "reflect": bool(extra.reflect),
                "radius": float(extra.radius),
            }
        )


def extra_obstacle_tag(extra_obstacles: tuple[ExtraObstacle, ...]) -> str | None:
    if not extra_obstacles:
        return None
    parts: list[str] = []
    for idx, extra in enumerate(extra_obstacles, start=1):
        sx, sy, sz = extra.start
        vx, vy, vz = extra.velocity
        parts.append(
            "x"
            f"{tag_float(sx)}y{tag_float(sy)}z{tag_float(sz)}"
            f"v{tag_float(vx)}_{tag_float(vy)}_{tag_float(vz)}"
            f"r{tag_float(extra.radius)}"
            f"{'ref' if extra.reflect else 'lin'}"
        )
    return "extra" + "-".join(parts)


def static_box_tag(static_boxes: tuple[StaticBox, ...]) -> str | None:
    if not static_boxes:
        return None
    parts: list[str] = []
    for box in static_boxes:
        cx, cy, cz = box.center
        sx, sy, sz = box.size
        parts.append(
            f"c{tag_float(cx)}_{tag_float(cy)}_{tag_float(cz)}"
            f"s{tag_float(sx)}_{tag_float(sy)}_{tag_float(sz)}"
        )
    return "box" + "-".join(parts)


def planner_variant_tag(
    *,
    temperature: float,
    safety_margin: float | None,
    w_obs: float | None,
    w_reach_time: float | None,
    w_clean_ctg: float | None,
    inflate: int | None,
    fallback_to_argmin: bool,
    fallback_commit_steps: int | None,
    dynamic_branch_sampling: bool,
    dynamic_branch_extra_radius: float | None,
    dynamic_branch_lateral_gain: float | None,
    dynamic_branch_speeds: tuple[float, ...] | None,
    dynamic_branch_max_obstacles: int | None,
    score_collision_after_goal: bool,
    rollout_max_accel: float | None,
    extra_obstacles: tuple[ExtraObstacle, ...] = (),
    static_boxes: tuple[StaticBox, ...] = (),
) -> str:
    parts = [f"moving_{temp_tag(temperature)}"]
    if fallback_to_argmin:
        parts.append("argmin")
    if dynamic_branch_sampling:
        parts.append("dynbranch")
    if score_collision_after_goal:
        parts.append("postgoal")
    if rollout_max_accel is not None:
        parts.append(f"rma{tag_float(rollout_max_accel)}")
    if safety_margin is not None:
        parts.append(f"sm{tag_float(safety_margin)}")
    if w_obs is not None:
        parts.append(f"wobs{tag_float(w_obs)}")
    if w_reach_time is not None:
        parts.append(f"wrt{tag_float(w_reach_time)}")
    if w_clean_ctg is not None:
        parts.append(f"wclean{tag_float(w_clean_ctg)}")
    if inflate is not None:
        parts.append(f"inf{int(inflate)}")
    if fallback_commit_steps is not None:
        parts.append(f"fc{int(fallback_commit_steps)}")
    if dynamic_branch_extra_radius is not None:
        parts.append(f"dbr{tag_float(dynamic_branch_extra_radius)}")
    if dynamic_branch_lateral_gain is not None:
        parts.append(f"dbl{tag_float(dynamic_branch_lateral_gain)}")
    if dynamic_branch_max_obstacles is not None:
        parts.append(f"dbo{int(dynamic_branch_max_obstacles)}")
    if dynamic_branch_speeds is not None:
        speed_tag = "-".join(tag_float(v) for v in dynamic_branch_speeds)
        parts.append(f"dbs{speed_tag}")
    extra_tag = extra_obstacle_tag(extra_obstacles)
    if extra_tag:
        parts.append(extra_tag)
    box_tag = static_box_tag(static_boxes)
    if box_tag:
        parts.append(box_tag)
    return "_".join(parts)


def existing_log(run_dir: Path, *, episode: int, drone: int) -> bool:
    return (run_dir / f"episode_{episode:03d}_drone_{drone:02d}.json").exists()


def min_candidate_clearance(
    *,
    cfg: dict[str, Any],
    log: dict[str, Any],
    obstacle_idx: int,
    start_t: float,
    end_t: float,
) -> dict[str, Any]:
    if obstacle_idx < 0:
        clearance = window_min_clearance_all_obstacles(
            cfg,
            log,
            start_t,
            end_t,
        )
    else:
        clearance = window_min_clearance(cfg, log, obstacle_idx, start_t, end_t)
        clearance["obstacle_idx"] = int(obstacle_idx)
        clearance["obstacle_scope"] = "single"
    steps = [
        step
        for step in log.get("steps", [])
        if start_t <= float(step["t"]) <= end_t
    ]
    clearance["window_path_length_m"] = path_length(steps)
    return clearance


def window_min_clearance_all_obstacles(
    cfg: dict[str, Any],
    log: dict[str, Any],
    start_t: float,
    end_t: float,
) -> dict[str, Any]:
    obstacles = cfg["scenario"].get("dynamic_obstacles", []) or []
    steps = [
        step
        for step in log.get("steps", [])
        if start_t <= float(step["t"]) <= end_t
    ]
    if not steps or not obstacles:
        return {
            "t": None,
            "clearance_m": None,
            "collision_flag": False,
            "first_virtual_hit_t": None,
            "obstacle_idx": None,
            "obstacle_scope": "all",
            "obstacle_count": len(obstacles),
        }

    per_obstacle: list[dict[str, Any]] = []
    for idx in range(len(obstacles)):
        obs_best: tuple[float, dict[str, Any]] | None = None
        obs_first_hit: float | None = None
        for step in steps:
            t = float(step["t"])
            clearance = row_clearance_to_obstacle(cfg, step, idx)
            if obs_best is None or clearance < obs_best[0]:
                obs_best = (clearance, step)
            if obs_first_hit is None and clearance <= 0.0:
                obs_first_hit = t
        if obs_best is not None:
            clearance, step = obs_best
            per_obstacle.append(
                {
                    "obstacle_idx": idx,
                    "t": float(step["t"]),
                    "clearance_m": clearance,
                    "collision_flag": bool(step.get("collision", False)),
                    "first_virtual_hit_t": obs_first_hit,
                }
            )

    best: tuple[float, dict[str, Any], int] | None = None
    first_hit: float | None = None
    for step in steps:
        t = float(step["t"])
        for idx in range(len(obstacles)):
            clearance = row_clearance_to_obstacle(cfg, step, idx)
            if best is None or clearance < best[0]:
                best = (clearance, step, idx)
            if first_hit is None and clearance <= 0.0:
                first_hit = t
    if best is None:
        return {
            "t": None,
            "clearance_m": None,
            "collision_flag": False,
            "first_virtual_hit_t": None,
            "obstacle_idx": None,
            "obstacle_scope": "all",
            "obstacle_count": len(obstacles),
        }
    clearance, step, idx = best
    return {
        "t": float(step["t"]),
        "clearance_m": clearance,
        "collision_flag": bool(step.get("collision", False)),
        "first_virtual_hit_t": first_hit,
        "obstacle_idx": idx,
        "obstacle_scope": "all",
        "obstacle_count": len(obstacles),
        "per_obstacle_min_clearance": per_obstacle,
    }


def moving_metrics(
    *,
    cfg: dict[str, Any],
    moving_run: Path,
    no_obstacle_log: dict[str, Any],
    episode: int,
    drone: int,
    obstacle_idx: int,
    start_t: float,
    end_t: float,
) -> dict[str, Any]:
    log = _load_json(moving_run / f"episode_{episode:03d}_drone_{drone:02d}.json")
    return {
        "run_dir": repo_path(moving_run),
        "outcome": log.get("outcome"),
        "first_collision_t": first_collision_t(log),
        "window_min_clearance": min_candidate_clearance(
            cfg=cfg,
            log=log,
            obstacle_idx=obstacle_idx,
            start_t=start_t,
            end_t=end_t,
        ),
        "path_delta_to_no_obstacle": trajectory_delta(
            log,
            no_obstacle_log,
            start_t,
            end_t,
        ),
        "summary": compact_summary(summarize_run(moving_run, cfg)),
    }


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    out = {key: value for key, value in summary.items() if key != "collision_details"}
    if "run_dir" in out:
        out["run_dir"] = repo_path(out["run_dir"])
    return out


def candidate_passes(
    row: dict[str, Any],
    *,
    target_no_obstacle_clearance: float,
    target_moving_clearance: float,
    target_path_delta: float,
    min_conflicting_obstacles: int,
) -> bool:
    moving = row.get("moving")
    if not moving:
        return False
    ghost_clear = row["no_obstacle"]["clearance_m"]
    ghost_per_obstacle = row["no_obstacle"].get("per_obstacle_min_clearance") or []
    if ghost_per_obstacle:
        ghost_conflicts = sum(
            1
            for obstacle in ghost_per_obstacle
            if obstacle.get("clearance_m") is not None
            and float(obstacle["clearance_m"]) <= target_no_obstacle_clearance
        )
    else:
        ghost_conflicts = (
            1
            if ghost_clear is not None
            and ghost_clear <= target_no_obstacle_clearance
            else 0
        )
    moving_clear = moving["window_min_clearance"]["clearance_m"]
    path_delta = moving["path_delta_to_no_obstacle"]["max_delta_m"]
    return (
        row["screen_selected"]
        and moving["outcome"] == "success"
        and ghost_clear is not None
        and ghost_clear <= target_no_obstacle_clearance
        and ghost_conflicts >= min_conflicting_obstacles
        and moving_clear is not None
        and moving_clear >= target_moving_clearance
        and path_delta is not None
        and path_delta >= target_path_delta
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    p.add_argument("--no-obstacle-run", type=Path, default=DEFAULT_NO_OBSTACLE_RUN)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--scratch-dir", type=Path, default=Path("/tmp/uavnav_race_hero_sweep"))
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--candidate", action="append", type=parse_candidate)
    p.add_argument("--period", type=parse_float_list, default=[19.8])
    p.add_argument(
        "--y-low",
        type=parse_float_list,
        default=[3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
    )
    p.add_argument("--speed", type=parse_float_list, default=[1.0, 1.5, 2.0])
    p.add_argument("--radius", type=parse_float_list, default=[1.0, 1.25, 1.5])
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--safety-margin", type=float)
    p.add_argument("--w-obs", type=float)
    p.add_argument(
        "--w-reach-time",
        type=float,
        help="Clean-reach tie-break penalty per rollout step before reaching the short race goal.",
    )
    p.add_argument(
        "--w-clean-ctg",
        type=float,
        help="Clean-reach tie-break penalty for average cost-to-go after reaching the short race goal.",
    )
    p.add_argument(
        "--inflate",
        type=int,
        help="Override the GPU MPPI static obstacle inflation in grid cells.",
    )
    p.add_argument("--fallback-to-argmin", action="store_true")
    p.add_argument("--fallback-commit-steps", type=int)
    p.add_argument("--dynamic-branch-sampling", action="store_true")
    p.add_argument("--dynamic-branch-extra-radius", type=float)
    p.add_argument("--dynamic-branch-lateral-gain", type=float)
    p.add_argument("--dynamic-branch-speeds", type=parse_float_list)
    p.add_argument("--dynamic-branch-max-obstacles", type=int)
    p.add_argument(
        "--extra-obstacle",
        action="append",
        type=parse_extra_obstacle,
        help="Append a dynamic obstacle as X,Y,Z,VX,VY,VZ,RADIUS[,REFLECT].",
    )
    p.add_argument(
        "--static-box",
        action="append",
        type=parse_static_box,
        help="Append a static box as CENTER_X,CENTER_Y,CENTER_Z,SIZE_X,SIZE_Y,SIZE_Z.",
    )
    p.add_argument("--score-collision-after-goal", action="store_true")
    p.add_argument(
        "--rollout-max-accel",
        type=float,
        help="Opt into acceleration-limited GPU MPPI rollout dynamics.",
    )
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--focus-drone", type=int, default=3)
    p.add_argument(
        "--focus-obstacle",
        type=int,
        default=0,
        help="Obstacle index to score; use -1 to score the closest of all obstacles.",
    )
    p.add_argument("--start-step", type=int, default=520)
    p.add_argument("--end-step", type=int, default=632)
    p.add_argument("--top-moving", type=int, default=4)
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--screen-only", action="store_true")
    p.add_argument("--rerun-existing", action="store_true")
    p.add_argument("--target-moving-clearance", type=float, default=0.25)
    p.add_argument("--target-no-obstacle-clearance", type=float, default=-0.5)
    p.add_argument("--target-path-delta", type=float, default=1.0)
    p.add_argument(
        "--min-conflicting-obstacles",
        type=int,
        default=1,
        help=(
            "Minimum number of obstacles whose no-obstacle ghost clearance "
            "must be at or below --target-no-obstacle-clearance."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    base_cfg = _load_yaml(args.base_config)
    extra_obstacles = tuple(args.extra_obstacle or ())
    static_boxes = tuple(args.static_box or ())
    dt = float(base_cfg["simulator"].get("dt", 0.05))
    start_t = args.start_step * dt
    end_t = args.end_step * dt
    no_obstacle_log = _load_json(
        args.no_obstacle_run
        / f"episode_{args.episode:03d}_drone_{args.focus_drone:02d}.json"
    )
    candidates = args.candidate or default_candidates(args)

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        cfg = candidate_config(
            base_cfg,
            candidate,
            temperature=args.temperature,
            safety_margin=args.safety_margin,
            w_obs=args.w_obs,
            w_reach_time=args.w_reach_time,
            w_clean_ctg=args.w_clean_ctg,
            inflate=args.inflate,
            fallback_to_argmin=args.fallback_to_argmin,
            fallback_commit_steps=args.fallback_commit_steps,
            dynamic_branch_sampling=args.dynamic_branch_sampling,
            dynamic_branch_extra_radius=args.dynamic_branch_extra_radius,
            dynamic_branch_lateral_gain=args.dynamic_branch_lateral_gain,
            dynamic_branch_speeds=(
                tuple(args.dynamic_branch_speeds)
                if args.dynamic_branch_speeds is not None
                else None
            ),
            dynamic_branch_max_obstacles=args.dynamic_branch_max_obstacles,
            score_collision_after_goal=args.score_collision_after_goal,
            rollout_max_accel=args.rollout_max_accel,
            n=args.n,
            seed=args.seed,
            output_root=args.output_root,
            extra_obstacles=extra_obstacles,
            static_boxes=static_boxes,
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

    rows.sort(key=lambda row: float(row["no_obstacle"].get("clearance_m", math.inf)))
    selected = rows[: max(0, int(args.top_moving))]
    for row in selected:
        row["screen_selected"] = True

    if not args.screen_only:
        for row in selected:
            cfg = copy.deepcopy(row["moving_config"])
            candidate = row["candidate"]
            run_dir = Path(cfg["output"]["dir"])
            config_path = args.scratch_dir / f"{candidate['tag']}_moving.yaml"
            _write_yaml(config_path, cfg)
            if (
                not args.summarize_only
                and (args.rerun_existing or not existing_log(
                    run_dir,
                    episode=args.episode,
                    drone=args.focus_drone,
                ))
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
            "script": "scripts/race_hero_control_sweep.py",
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
            "inflate": args.inflate,
            "fallback_to_argmin": args.fallback_to_argmin,
            "fallback_commit_steps": args.fallback_commit_steps,
            "dynamic_branch_sampling": args.dynamic_branch_sampling,
            "dynamic_branch_extra_radius": args.dynamic_branch_extra_radius,
            "dynamic_branch_lateral_gain": args.dynamic_branch_lateral_gain,
            "dynamic_branch_speeds": args.dynamic_branch_speeds,
            "dynamic_branch_max_obstacles": args.dynamic_branch_max_obstacles,
            "rollout_max_accel": args.rollout_max_accel,
            "extra_obstacles": [
                {
                    "start": [float(v) for v in extra.start],
                    "velocity": [float(v) for v in extra.velocity],
                    "radius": float(extra.radius),
                    "reflect": bool(extra.reflect),
                }
                for extra in extra_obstacles
            ],
            "static_boxes": [
                {
                    "center": [float(v) for v in box.center],
                    "size": [float(v) for v in box.size],
                }
                for box in static_boxes
            ],
            "score_collision_after_goal": args.score_collision_after_goal,
        },
        "rows": public_rows,
        "survivors": [
            row["candidate"]["tag"]
            for row in public_rows
            if row["passes_thresholds"]
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        "| rank | selected | candidate | ghost clear | moving | moving clear | delta | pass |"
    )
    print("|---:|---:|---|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(public_rows[: max(10, args.top_moving)], start=1):
        moving = row.get("moving") or {}
        moving_clear = (moving.get("window_min_clearance") or {}).get("clearance_m")
        delta = (moving.get("path_delta_to_no_obstacle") or {}).get("max_delta_m")
        print(
            f"| {rank} | {str(row['screen_selected']).lower()} | "
            f"{row['candidate']['tag']} | "
            f"{fmt(row['no_obstacle'].get('clearance_m'))} | "
            f"{moving.get('outcome', 'n/a')} | "
            f"{fmt(moving_clear)} | {fmt(delta, signed=False)} | "
            f"{str(row['passes_thresholds']).lower()} |"
        )
    print(f"survivors: {', '.join(report['survivors']) if report['survivors'] else 'none'}")
    print(f"wrote {args.out}")
    return 0


def fmt(value: Any, *, signed: bool = True) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        sign = "+" if signed else ""
        return f"{value:{sign}.2f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
