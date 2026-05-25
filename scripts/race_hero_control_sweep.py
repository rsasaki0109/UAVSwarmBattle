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
    fallback_to_argmin: bool,
    fallback_commit_steps: int | None,
    n: int,
    seed: int,
    output_root: Path,
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
    dt = float(cfg["simulator"].get("dt", 0.05))
    n_loops = int(scenario.get("n_loops", 2))
    cfg["simulator"]["max_steps"] = int(round(float(candidate.period) * n_loops / dt))
    planner = cfg.setdefault("planner", {})
    planner["type"] = "gpu_mppi"
    planner["temperature"] = float(temperature)
    if safety_margin is not None:
        planner["safety_margin"] = float(safety_margin)
    if w_obs is not None:
        planner["w_obs"] = float(w_obs)
    planner["fallback_to_argmin"] = bool(fallback_to_argmin)
    if fallback_commit_steps is not None:
        planner["fallback_commit_steps"] = int(fallback_commit_steps)
    planner["mode_aware_sampling"] = False
    planner["log_action_provenance"] = True
    cfg.setdefault("output", {})["dir"] = str(
        output_root
        / candidate.tag
        / planner_variant_tag(
            temperature=temperature,
            safety_margin=safety_margin,
            w_obs=w_obs,
            fallback_to_argmin=fallback_to_argmin,
            fallback_commit_steps=fallback_commit_steps,
        )
    )
    return cfg


def planner_variant_tag(
    *,
    temperature: float,
    safety_margin: float | None,
    w_obs: float | None,
    fallback_to_argmin: bool,
    fallback_commit_steps: int | None,
) -> str:
    parts = [f"moving_{temp_tag(temperature)}"]
    if fallback_to_argmin:
        parts.append("argmin")
    if safety_margin is not None:
        parts.append(f"sm{tag_float(safety_margin)}")
    if w_obs is not None:
        parts.append(f"wobs{tag_float(w_obs)}")
    if fallback_commit_steps is not None:
        parts.append(f"fc{int(fallback_commit_steps)}")
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
    clearance = window_min_clearance(cfg, log, obstacle_idx, start_t, end_t)
    steps = [
        step
        for step in log.get("steps", [])
        if start_t <= float(step["t"]) <= end_t
    ]
    clearance["window_path_length_m"] = path_length(steps)
    return clearance


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
) -> bool:
    moving = row.get("moving")
    if not moving:
        return False
    ghost_clear = row["no_obstacle"]["clearance_m"]
    moving_clear = moving["window_min_clearance"]["clearance_m"]
    path_delta = moving["path_delta_to_no_obstacle"]["max_delta_m"]
    return (
        row["screen_selected"]
        and moving["outcome"] == "success"
        and ghost_clear is not None
        and ghost_clear <= target_no_obstacle_clearance
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
    p.add_argument("--fallback-to-argmin", action="store_true")
    p.add_argument("--fallback-commit-steps", type=int)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--focus-drone", type=int, default=3)
    p.add_argument("--focus-obstacle", type=int, default=0)
    p.add_argument("--start-step", type=int, default=520)
    p.add_argument("--end-step", type=int, default=632)
    p.add_argument("--top-moving", type=int, default=4)
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--screen-only", action="store_true")
    p.add_argument("--rerun-existing", action="store_true")
    p.add_argument("--target-moving-clearance", type=float, default=0.25)
    p.add_argument("--target-no-obstacle-clearance", type=float, default=-0.5)
    p.add_argument("--target-path-delta", type=float, default=1.0)
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    base_cfg = _load_yaml(args.base_config)
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
            fallback_to_argmin=args.fallback_to_argmin,
            fallback_commit_steps=args.fallback_commit_steps,
            n=args.n,
            seed=args.seed,
            output_root=args.output_root,
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
    }
    public_rows: list[dict[str, Any]] = []
    for row in rows:
        out = {key: value for key, value in row.items() if key != "moving_config"}
        out["passes_thresholds"] = candidate_passes(
            out,
            target_no_obstacle_clearance=args.target_no_obstacle_clearance,
            target_moving_clearance=args.target_moving_clearance,
            target_path_delta=args.target_path_delta,
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
            "fallback_to_argmin": args.fallback_to_argmin,
            "fallback_commit_steps": args.fallback_commit_steps,
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
