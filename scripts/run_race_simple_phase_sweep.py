#!/usr/bin/env python3
"""Generate, run, and summarize race-simple dynamic-obstacle phase probes.

The post-1646e11 race-simple retune showed a non-floor pilot, but the
period/radius/speed knobs were discontinuous and the observed MPC loss looked
phase/peer dominated. This helper keeps follow-up probes reproducible without
adding one-off YAMLs to examples/.

Example:
    python scripts/run_race_simple_phase_sweep.py --n 3
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MPC_BASE = ROOT / "examples/exp_race_simple_retuned_n5_mpc.yaml"
DEFAULT_GPU_BASE = ROOT / "examples/exp_race_simple_retuned_n5_gpu_mppi.yaml"


def _period_tag(period: float) -> str:
    return f"p{period:g}".replace(".", "p")


def _variant_tag(period: float, y_low: float, y_high: float) -> str:
    return (
        f"{_period_tag(period)}_y"
        f"{str(y_low).replace('.', 'p')}_{str(y_high).replace('.', 'p')}"
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _set_dynamic_obstacles(
    cfg: dict[str, Any],
    *,
    y_low: float,
    y_high: float,
    speed: float,
    radius: float,
) -> None:
    dyn = cfg["scenario"].setdefault("dynamic_obstacles", [])
    if len(dyn) != 2:
        raise ValueError("race-simple phase sweep expects exactly 2 dynamic obstacles")

    dyn[0]["start"] = [20.0, float(y_low), 7.0]
    dyn[0]["velocity"] = [0.0, float(speed), 0.0]
    dyn[0]["radius"] = float(radius)
    dyn[0]["reflect"] = True

    dyn[1]["start"] = [20.0, float(y_high), 7.0]
    dyn[1]["velocity"] = [0.0, -float(speed), 0.0]
    dyn[1]["radius"] = float(radius)
    dyn[1]["reflect"] = True


def build_config(
    base: dict[str, Any],
    *,
    planner: str,
    period: float,
    y_low: float,
    y_high: float,
    n: int,
    seed: int,
    radius: float,
    speed: float,
    output_root: Path,
    gpu_log_action_provenance: bool = False,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    tag = _variant_tag(period, y_low, y_high)
    cfg["name"] = f"race_simple_phase_{tag}_{planner}"
    cfg["seed"] = int(seed)
    cfg["num_episodes"] = int(n)

    scenario = cfg["scenario"]
    scenario["period"] = float(period)
    _set_dynamic_obstacles(
        cfg,
        y_low=y_low,
        y_high=y_high,
        speed=speed,
        radius=radius,
    )

    dt = float(cfg["simulator"].get("dt", 0.05))
    n_loops = int(scenario.get("n_loops", 2))
    cfg["simulator"]["max_steps"] = int(round(float(period) * n_loops / dt))
    cfg.setdefault("output", {})["dir"] = str(output_root / tag / planner)
    if planner == "gpu_mppi" and gpu_log_action_provenance:
        cfg.setdefault("planner", {})["log_action_provenance"] = True
    return cfg


def _reflected_position(
    start: list[float],
    velocity: list[float],
    t: float,
    world_size: list[float],
    *,
    reflect: bool,
) -> list[float]:
    pos: list[float] = []
    for p0, v, upper in zip(start, velocity, world_size):
        raw = float(p0) + float(v) * float(t)
        if reflect:
            period = 2.0 * float(upper)
            if period > 0.0:
                raw = raw % period
                if raw > float(upper):
                    raw = period - raw
        pos.append(raw)
    return pos


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _min_dynamic_clearance(drone_log: dict[str, Any], cfg: dict[str, Any]) -> float | None:
    dyn = cfg.get("scenario", {}).get("dynamic_obstacles", []) or []
    if not dyn:
        return None
    steps = drone_log.get("steps", []) or []
    if not steps:
        return None

    world_size = [float(v) for v in cfg.get("scenario", {}).get("size", [])]
    if len(world_size) < 3:
        return None
    dt = float(cfg.get("simulator", {}).get("dt", 0.05))
    drone_radius = float(cfg.get("simulator", {}).get("drone_radius", 0.4))
    best = float("inf")

    for step in steps:
        pos = step.get("true_pos")
        if not isinstance(pos, list):
            continue
        # The log row stores the pre-step position and post-step collision flag.
        # Check both adjacent scenario times as a cheap nearest-contact proxy.
        for t in (float(step.get("t", 0.0)), float(step.get("t", 0.0)) + dt):
            for obs in dyn:
                obs_pos = _reflected_position(
                    obs["start"],
                    obs["velocity"],
                    t,
                    world_size,
                    reflect=bool(obs.get("reflect", True)),
                )
                clearance = _distance(pos, obs_pos) - (
                    float(obs.get("radius", 0.5)) + drone_radius
                )
                best = min(best, clearance)
    return best if math.isfinite(best) else None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_run(run_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    n_eps = int(cfg.get("num_episodes", 0))
    n_drones = int(cfg.get("scenario", {}).get("n_drones", 4))
    counts: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    min_clearance: float | None = None

    for ep in range(n_eps):
        joint = _load_json(run_dir / f"episode_{ep:03d}_joint.json")
        if joint is None:
            counts["missing_joint"] += 1
        elif joint.get("outcome") == "success":
            counts["joint_success"] += 1

        for drone_idx in range(n_drones):
            drone = _load_json(run_dir / f"episode_{ep:03d}_drone_{drone_idx:02d}.json")
            if drone is None:
                counts["missing_drone"] += 1
                continue
            outcome = str(drone.get("outcome", "unknown"))
            if outcome == "success":
                counts["drone_success"] += 1
                continue
            if outcome == "timeout":
                counts["timeout"] += 1
                continue
            if outcome != "collision":
                counts[f"other_{outcome}"] += 1
                continue

            step_collision = any(bool(s.get("collision", False)) for s in drone.get("steps", []))
            source = "env" if step_collision else "peer"
            counts[f"{source}_collision"] += 1
            clearance = _min_dynamic_clearance(drone, cfg)
            if clearance is not None:
                min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
            details.append(
                {
                    "episode": ep,
                    "drone": drone_idx,
                    "source": source,
                    "final_t": float(drone.get("summary", {}).get("final_t", 0.0)),
                    "min_dynamic_clearance_m": clearance,
                }
            )

    total_drone_eps = n_eps * n_drones
    return {
        "run_dir": str(run_dir),
        "episodes": n_eps,
        "n_drones": n_drones,
        "joint_success": int(counts["joint_success"]),
        "drone_success": int(counts["drone_success"]),
        "drone_total": total_drone_eps,
        "env_collision": int(counts["env_collision"]),
        "peer_collision": int(counts["peer_collision"]),
        "timeout": int(counts["timeout"]),
        "missing_joint": int(counts["missing_joint"]),
        "missing_drone": int(counts["missing_drone"]),
        "min_dynamic_clearance_m": min_clearance,
        "collision_details": details,
    }


def run_one(config_path: Path, python: str) -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [python, "-m", "uav_nav_lab.cli", "run", str(config_path)]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _parse_y_pair(raw: str) -> tuple[float, float]:
    try:
        a, b = raw.split(",", 1)
        return float(a), float(b)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected LOW,HIGH, e.g. 5,35") from exc


def _format_clearance(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2f}"


def print_table(rows: list[dict[str, Any]]) -> None:
    print()
    print(
        "variant             planner   joint  drone    env peer timeout min_dyn  run_dir"
    )
    print(
        "------------------- -------- ------- -------- --- ---- ------- -------- ------------------------------"
    )
    for row in rows:
        summary = row["summary"]
        print(
            f"{row['variant']:<19} {row['planner']:<8} "
            f"{summary['joint_success']:>2}/{summary['episodes']:<4} "
            f"{summary['drone_success']:>3}/{summary['drone_total']:<4} "
            f"{summary['env_collision']:>3} {summary['peer_collision']:>4} "
            f"{summary['timeout']:>7} "
            f"{_format_clearance(summary['min_dynamic_clearance_m']):>8} "
            f"{summary['run_dir']}"
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=3, help="episodes per planner/variant")
    p.add_argument("--seed", type=int, default=42, help="base seed")
    p.add_argument(
        "--period",
        action="append",
        dest="periods",
        type=float,
        help="period to test; repeatable. default: 19.5 and 19.8",
    )
    p.add_argument(
        "--y-pair",
        action="append",
        dest="y_pairs",
        type=_parse_y_pair,
        help="dynamic-obstacle start y pair LOW,HIGH; repeatable. default: 5,35 7,33 9,31",
    )
    p.add_argument("--radius", type=float, default=1.0, help="dynamic obstacle radius")
    p.add_argument("--speed", type=float, default=1.5, help="absolute dynamic obstacle y speed")
    p.add_argument(
        "--planner",
        action="append",
        dest="planners",
        choices=("mpc", "gpu_mppi"),
        help="planner to run; repeatable. default: both",
    )
    p.add_argument("--base-mpc", type=Path, default=DEFAULT_MPC_BASE)
    p.add_argument("--base-gpu", type=Path, default=DEFAULT_GPU_BASE)
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for uav-nav runs; useful when torch is installed outside .venv",
    )
    p.add_argument(
        "--scratch-dir",
        type=Path,
        default=Path("/tmp/uavnav_race_simple_phase_sweep"),
        help="where generated YAMLs are written",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/_race_simple_phase_sweep",
        help="root for generated result directories",
    )
    p.add_argument(
        "--summarize-only",
        action="store_true",
        help="skip uav-nav runs and summarize existing result directories",
    )
    p.add_argument(
        "--gpu-log-action-provenance",
        action="store_true",
        help="enable compact GPU MPPI action-source logging in replan JSON",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    periods = args.periods or [19.5, 19.8]
    y_pairs = args.y_pairs or [(5.0, 35.0), (7.0, 33.0), (9.0, 31.0)]
    planners = args.planners or ["mpc", "gpu_mppi"]
    bases = {
        "mpc": _load_yaml(args.base_mpc),
        "gpu_mppi": _load_yaml(args.base_gpu),
    }

    rows: list[dict[str, Any]] = []
    for period in periods:
        for y_low, y_high in y_pairs:
            variant = _variant_tag(period, y_low, y_high)
            for planner in planners:
                cfg = build_config(
                    bases[planner],
                    planner=planner,
                    period=period,
                    y_low=y_low,
                    y_high=y_high,
                    n=args.n,
                    seed=args.seed,
                    radius=args.radius,
                    speed=args.speed,
                    output_root=args.output_root,
                    gpu_log_action_provenance=bool(args.gpu_log_action_provenance),
                )
                config_path = args.scratch_dir / f"{variant}_{planner}.yaml"
                _write_yaml(config_path, cfg)
                if not args.summarize_only:
                    run_one(config_path, python=str(args.python))
                run_dir = Path(cfg["output"]["dir"])
                rows.append(
                    {
                        "variant": variant,
                        "planner": planner,
                        "config": str(config_path),
                        "summary": summarize_run(run_dir, cfg),
                    }
                )

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "phase_sweep_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print_table(rows)
    print(f"\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
