#!/usr/bin/env python3
"""Run stronger controls for the README race-hero dynamic obstacle cell.

These arms keep the same seed, planner, and reference race setup as the
low-temperature moving-obstacle run, but ablate the dynamic cue:

- frozen_initial: obstacles are fixed at their t=0 positions.
- frozen_encounter: obstacles are fixed at the encounter-time positions.
- wrong_velocity: the simulator keeps true motion, but the planner sees
  reversed dynamic-obstacle velocities through the perfect sensor.
- no_prediction: the planner sees current obstacles but does not forecast
  their motion.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

from analyze_race_simple_phase_trace import _load_yaml, obstacle_positions
from run_race_simple_phase_sweep import ROOT, _write_yaml, run_one, summarize_run


DEFAULT_MOVING_RUN = (
    ROOT / "results/_race_simple_causal_probe/p19p8_y5p0_35p0/t0p1"
)
DEFAULT_OUT_ROOT = ROOT / "results/_race_hero_control_variants"
DEFAULT_SUMMARY = ROOT / "docs/data/race_hero_control_variants.json"
VARIANTS = ("frozen_initial", "frozen_encounter", "wrong_velocity", "no_prediction")


def repo_path(path: Path | str) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    out = {key: value for key, value in summary.items() if key != "collision_details"}
    if "run_dir" in out:
        out["run_dir"] = repo_path(out["run_dir"])
    return out


def freeze_dynamic_obstacles(cfg: dict[str, Any], t: float) -> list[dict[str, Any]]:
    positions = obstacle_positions(cfg, t)
    frozen: list[dict[str, Any]] = []
    for obs, pos in zip(cfg["scenario"].get("dynamic_obstacles", []) or [], positions):
        frozen.append(
            {
                "start": [float(v) for v in pos],
                "velocity": [0.0 for _ in pos],
                "reflect": False,
                "radius": float(obs.get("radius", 0.5)),
            }
        )
    return frozen


def build_variant_config(
    moving_cfg: dict[str, Any],
    *,
    variant: str,
    cell: str,
    n: int,
    seed: int,
    output_root: Path,
    freeze_encounter_t: float,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")

    cfg = copy.deepcopy(moving_cfg)
    cfg["name"] = f"{moving_cfg.get('name', 'race_hero')}_{variant}_control"
    cfg["seed"] = int(seed)
    cfg["num_episodes"] = int(n)
    cfg.setdefault("output", {})["dir"] = str(output_root / cell / variant)

    if variant == "frozen_initial":
        cfg["scenario"]["dynamic_obstacles"] = freeze_dynamic_obstacles(cfg, 0.0)
    elif variant == "frozen_encounter":
        cfg["scenario"]["dynamic_obstacles"] = freeze_dynamic_obstacles(
            cfg,
            float(freeze_encounter_t),
        )
    elif variant == "wrong_velocity":
        cfg.setdefault("sensor", {})["type"] = "perfect"
        cfg["sensor"]["dynamic_velocity_scale"] = -1.0
    elif variant == "no_prediction":
        cfg.setdefault("planner", {})["use_prediction"] = False

    return cfg


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--moving-run", type=Path, default=DEFAULT_MOVING_RUN)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--scratch-dir", type=Path, default=Path("/tmp/uavnav_race_hero_controls"))
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--variant", action="append", choices=VARIANTS)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze-encounter-t", type=float, default=29.30)
    p.add_argument("--summarize-only", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    moving_cfg = _load_yaml(args.moving_run / "config.yaml")
    cell = args.moving_run.parent.name
    variants = args.variant or list(VARIANTS)

    rows: list[dict[str, Any]] = []
    for variant in variants:
        cfg = build_variant_config(
            moving_cfg,
            variant=variant,
            cell=cell,
            n=args.n,
            seed=args.seed,
            output_root=args.output_root,
            freeze_encounter_t=args.freeze_encounter_t,
        )
        config_path = args.scratch_dir / f"{cell}_{variant}.yaml"
        _write_yaml(config_path, cfg)
        if not args.summarize_only:
            run_one(config_path, python=str(args.python))
        run_dir = Path(cfg["output"]["dir"])
        rows.append(
            {
                "variant": variant,
                "run_dir": repo_path(run_dir),
                "config": repo_path(run_dir / "config.yaml"),
                "summary": compact_summary(summarize_run(run_dir, cfg)),
            }
        )

    report = {
        "source": {
            "moving_run": repo_path(args.moving_run),
            "script": "scripts/race_hero_control_variants.py",
        },
        "freeze_encounter_t": float(args.freeze_encounter_t),
        "variants": rows,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("| variant | joint | drone | env | min dyn | run_dir |")
    print("|---|---:|---:|---:|---:|---|")
    for row in rows:
        summary = row["summary"]
        min_dyn = summary.get("min_dynamic_clearance_m")
        min_dyn_text = "n/a" if min_dyn is None else f"{float(min_dyn):+.2f}"
        print(
            f"| {row['variant']} | "
            f"{summary['joint_success']}/{summary['episodes']} | "
            f"{summary['drone_success']}/{summary['drone_total']} | "
            f"{summary['env_collision']} | {min_dyn_text} | {row['run_dir']} |"
        )
    print(f"wrote {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
