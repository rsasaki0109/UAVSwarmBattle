#!/usr/bin/env python3
"""Temperature counterfactual for the race-simple softmax failure.

The phase-sweep/provenance work isolates a split cell where vanilla GPU
MPPI sees an escape rollout but emits the softmax-averaged command back
toward a moving obstacle. This script reruns the same cell while only
changing the MPPI softmax temperature, then writes a compact tracked
summary and figure.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_race_simple_phase_trace import (
    _load_json,
    _load_yaml,
    clearance_to_obstacles,
    min_clearance_for_path,
    nearest,
)
from run_race_simple_phase_sweep import (
    DEFAULT_GPU_BASE,
    ROOT,
    _set_dynamic_obstacles,
    _variant_tag,
    _write_yaml,
    run_one,
    summarize_run,
)


DEFAULT_OUT_ROOT = ROOT / "results/_race_simple_temperature_counterfactual"
DEFAULT_SUMMARY = ROOT / "docs/data/race_simple_temperature_counterfactual.json"
DEFAULT_FIGURE = ROOT / "docs/images/race_simple_temperature_counterfactual.png"
DEFAULT_VANILLA_RUN = ROOT / "results/_race_simple_phase_sweep/p19p8_y5p5_34p5/gpu_mppi"
DEFAULT_VANILLA_PROVENANCE = (
    ROOT
    / "results/_race_simple_action_provenance/p19p8_y5p5_34p5/gpu_mppi/action_provenance_summary.json"
)


def parse_y_pair(raw: str) -> tuple[float, float]:
    try:
        a, b = raw.split(",", 1)
        return float(a), float(b)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected LOW,HIGH, e.g. 5.5,34.5") from exc


def temp_tag(value: float) -> str:
    return f"t{value:g}".replace(".", "p")


def fmt_optional(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def repo_path(path: Path | str) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    out = {
        key: value
        for key, value in summary.items()
        if key != "collision_details"
    }
    if "run_dir" in out:
        out["run_dir"] = repo_path(out["run_dir"])
    return out


def build_config(
    base: dict[str, Any],
    *,
    period: float,
    y_low: float,
    y_high: float,
    speed: float,
    radius: float,
    temperature: float,
    n: int,
    seed: int,
    output_root: Path,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cell = _variant_tag(period, y_low, y_high)
    tag = temp_tag(temperature)
    cfg["name"] = f"race_simple_temperature_counterfactual_{cell}_{tag}"
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

    planner = cfg.setdefault("planner", {})
    planner["type"] = "gpu_mppi"
    planner["temperature"] = float(temperature)
    planner["fallback_to_argmin"] = False
    planner["mode_aware_sampling"] = False
    planner["log_action_provenance"] = True
    cfg.setdefault("output", {})["dir"] = str(output_root / cell / tag)
    return cfg


def first_env_collision_t(log: dict[str, Any]) -> float | None:
    for step in log.get("steps", []):
        if bool(step.get("collision", False)):
            return float(step["t"])
    return None


def row_clearance(cfg: dict[str, Any], step: dict[str, Any]) -> float:
    pos = [float(v) for v in step["true_pos"]]
    t = float(step["t"])
    dt = float(cfg["simulator"].get("dt", 0.05))
    clearances = clearance_to_obstacles(cfg, pos, t)
    next_clearances = clearance_to_obstacles(cfg, pos, t + dt)
    if not clearances:
        return math.inf
    return min(min(clearances), min(next_clearances))


def actual_window_min(
    cfg: dict[str, Any],
    log: dict[str, Any],
    start_t: float,
    end_t: float,
) -> dict[str, Any]:
    steps = [
        step
        for step in log.get("steps", [])
        if start_t <= float(step["t"]) <= end_t
    ]
    if not steps:
        return {"clearance_m": None, "t": None}
    best = min(steps, key=lambda step: row_clearance(cfg, step))
    return {"clearance_m": row_clearance(cfg, best), "t": float(best["t"])}


def selected_visible_rollout_clearance(
    cfg: dict[str, Any],
    replan: dict[str, Any],
) -> float | None:
    rollouts = replan.get("rollouts") or []
    best_idx = int(replan.get("best_rollout_idx", 0))
    if not 0 <= best_idx < len(rollouts):
        return None
    clearance, _ = min_clearance_for_path(
        cfg,
        rollouts[best_idx],
        replan_t=float(replan["t"]),
    )
    return clearance


def vec_y(values: list[float] | None) -> float | None:
    if values is None or len(values) < 2:
        return None
    return float(values[1])


def probe_metrics(
    run_dir: Path,
    *,
    episode: int,
    drone: int,
    probe_t: float,
    follow: float,
) -> dict[str, Any]:
    cfg = _load_yaml(run_dir / "config.yaml")
    log = _load_json(run_dir / f"episode_{episode:03d}_drone_{drone:02d}.json")
    replan = nearest(log["replans"], probe_t)
    step = nearest(log["steps"], float(replan["t"]))
    provenance = (replan.get("planner_meta") or {}).get("action_provenance") or {}
    return {
        "episode": int(episode),
        "drone": int(drone),
        "outcome": log.get("outcome"),
        "env_collision_t": first_env_collision_t(log),
        "probe_t": float(probe_t),
        "replan_t": float(replan["t"]),
        "cmd_y_mps": vec_y([float(v) for v in step.get("cmd", [])]),
        "chosen_y_mps": vec_y(provenance.get("chosen_action")),
        "softmax_y_mps": vec_y(provenance.get("softmax_action")),
        "argmin_y_mps": vec_y(provenance.get("argmin_action")),
        "argmax_weight_y_mps": vec_y(provenance.get("argmax_weight_action")),
        "action_source": provenance.get("action_source"),
        "weight_max": provenance.get("weight_max"),
        "weight_entropy": provenance.get("weight_entropy"),
        "weight_mass_by_action_y_sign": provenance.get("weight_mass_by_action_y_sign"),
        "selected_visible_rollout_clearance_m": selected_visible_rollout_clearance(
            cfg,
            replan,
        ),
        "actual_window_min": actual_window_min(
            cfg,
            log,
            float(replan["t"]),
            float(replan["t"]) + float(follow),
        ),
    }


def merge_provenance_summary(probe: dict[str, Any], provenance_path: Path) -> dict[str, Any]:
    if not provenance_path.exists():
        return probe
    with provenance_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    actions = summary.get("actions") or {}
    for key, action_name in [
        ("cmd_y_mps", "cmd"),
        ("chosen_y_mps", "chosen"),
        ("softmax_y_mps", "softmax"),
        ("argmin_y_mps", "argmin"),
        ("argmax_weight_y_mps", "argmax_weight"),
    ]:
        row = actions.get(action_name) or {}
        y_value = vec_y(row.get("action"))
        if y_value is not None:
            probe[key] = y_value
    for key in [
        "action_source",
        "weight_max",
        "weight_entropy",
        "weight_mass_by_action_y_sign",
        "selected_visible_rollout_clearance_m",
    ]:
        if key in summary:
            probe[key] = summary[key]
    return probe


def existing_vanilla_row(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.vanilla_run_dir
    cfg = _load_yaml(run_dir / "config.yaml")
    summary = summarize_run(run_dir, cfg)
    probe = probe_metrics(
        run_dir,
        episode=args.probe_episode,
        drone=args.probe_drone,
        probe_t=args.probe_t,
        follow=args.follow,
    )
    probe = merge_provenance_summary(probe, args.vanilla_provenance_json)
    return {
        "label": "t=1",
        "tag": "t1",
        "temperature": 1.0,
        "config": repo_path(run_dir / "config.yaml"),
        "run_dir": repo_path(run_dir),
        "source": "existing_phase_sweep_baseline",
        "summary": compact_summary(summary),
        "probe": probe,
    }


def run_or_summarize(args: argparse.Namespace) -> dict[str, Any]:
    base = _load_yaml(args.base_gpu)
    y_low, y_high = args.y_pair
    cell = _variant_tag(args.period, y_low, y_high)
    rows: list[dict[str, Any]] = []
    if args.use_existing_vanilla:
        rows.append(existing_vanilla_row(args))
    for temperature in args.temperatures:
        cfg = build_config(
            base,
            period=args.period,
            y_low=y_low,
            y_high=y_high,
            speed=args.speed,
            radius=args.radius,
            temperature=temperature,
            n=args.n,
            seed=args.seed,
            output_root=args.output_root,
        )
        tag = temp_tag(temperature)
        config_path = args.scratch_dir / f"{cell}_{tag}.yaml"
        _write_yaml(config_path, cfg)
        if not args.summarize_only:
            run_one(config_path, python=str(args.python))
        run_dir = Path(cfg["output"]["dir"])
        summary = summarize_run(run_dir, cfg)
        rows.append(
            {
                "label": f"t={temperature:g}",
                "tag": tag,
                "temperature": float(temperature),
                "config": repo_path(run_dir / "config.yaml"),
                "run_dir": repo_path(run_dir),
                "source": "fresh_run",
                "summary": compact_summary(summary),
                "probe": probe_metrics(
                    run_dir,
                    episode=args.probe_episode,
                    drone=args.probe_drone,
                    probe_t=args.probe_t,
                    follow=args.follow,
                ),
            }
        )

    return {
        "cell": cell,
        "params": {
            "n": int(args.n),
            "seed": int(args.seed),
            "period": float(args.period),
            "y_pair": [float(y_low), float(y_high)],
            "speed": float(args.speed),
            "radius": float(args.radius),
            "probe_episode": int(args.probe_episode),
            "probe_drone": int(args.probe_drone),
            "probe_t": float(args.probe_t),
            "follow": float(args.follow),
        },
        "arms": rows,
    }


def render_figure(result: dict[str, Any], out: Path) -> None:
    arms = sorted(result["arms"], key=lambda row: float(row["temperature"]), reverse=True)
    labels = [row["label"] for row in arms]
    joint_pct = [
        100.0 * row["summary"]["joint_success"] / row["summary"]["episodes"]
        for row in arms
    ]
    drone_pct = [
        100.0 * row["summary"]["drone_success"] / row["summary"]["drone_total"]
        for row in arms
    ]
    env_counts = [row["summary"]["env_collision"] for row in arms]
    peer_counts = [row["summary"]["peer_collision"] for row in arms]
    chosen_y = [row["probe"].get("chosen_y_mps") for row in arms]
    argmin_y = [row["probe"].get("argmin_y_mps") for row in arms]
    window_clearance = [
        (row["probe"].get("actual_window_min") or {}).get("clearance_m")
        for row in arms
    ]

    x = list(range(len(arms)))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), gridspec_kw={"wspace": 0.32})

    ax = axes[0]
    width = 0.36
    ax.bar([i - width / 2 for i in x], joint_pct, width, label="joint success", color="#2563eb")
    ax.bar([i + width / 2 for i in x], drone_pct, width, label="drone success", color="#10b981")
    for i, row in enumerate(arms):
        ax.text(
            i,
            max(joint_pct[i], drone_pct[i]) + 3.0,
            f"env {env_counts[i]}\npeer {peer_counts[i]}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, 112)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("success (%)")
    ax.set_title("Closed-loop outcome")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    ax.axhline(0.0, color="#525252", linewidth=0.8)
    ax.bar([i - width / 2 for i in x], chosen_y, width, label="chosen/softmax cmd y", color="#dc2626")
    ax.bar([i + width / 2 for i in x], argmin_y, width, label="argmin rollout y", color="#f59e0b")
    ax2 = ax.twinx()
    ax2.plot(x, window_clearance, "o-", color="#059669", linewidth=2.0, label="window min clearance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("probe y velocity [m/s]")
    ax2.set_ylabel("actual min clearance [m]")
    ax.set_title(f"Probe at t={result['params']['probe_t']:.1f}s, drone {result['params']['probe_drone']}")
    ax.grid(axis="y", alpha=0.25)
    handles, labels_left = ax.get_legend_handles_labels()
    handles2, labels_right = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels_left + labels_right, loc="lower right", fontsize=8)

    fig.suptitle(
        "Race-simple split cell: lowering MPPI temperature tests whether softmax aggregation causes contact",
        fontsize=10,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def print_table(result: dict[str, Any]) -> None:
    print("Race-simple temperature counterfactual")
    print("| arm | joint | drone | env | peer | min_dyn | cmd_y | argmin_y | window_min |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(result["arms"], key=lambda item: float(item["temperature"]), reverse=True):
        summary = row["summary"]
        probe = row["probe"]
        actual = probe.get("actual_window_min") or {}
        print(
            f"| {row['label']} | "
            f"{summary['joint_success']}/{summary['episodes']} | "
            f"{summary['drone_success']}/{summary['drone_total']} | "
            f"{summary['env_collision']} | {summary['peer_collision']} | "
            f"{fmt_optional(summary['min_dynamic_clearance_m'])} | "
            f"{fmt_optional(probe.get('chosen_y_mps'))} | "
            f"{fmt_optional(probe.get('argmin_y_mps'))} | "
            f"{fmt_optional(actual.get('clearance_m'))} |"
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--period", type=float, default=19.8)
    p.add_argument("--y-pair", type=parse_y_pair, default=(5.5, 34.5))
    p.add_argument("--speed", type=float, default=1.5)
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument(
        "--temperature",
        action="append",
        dest="temperatures",
        type=float,
        help="fresh temperature arm; repeatable. default: 0.001",
    )
    p.add_argument("--base-gpu", type=Path, default=DEFAULT_GPU_BASE)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--scratch-dir", type=Path, default=Path("/tmp/uavnav_race_simple_temp_cf"))
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    p.add_argument("--vanilla-run-dir", type=Path, default=DEFAULT_VANILLA_RUN)
    p.add_argument(
        "--vanilla-provenance-json",
        type=Path,
        default=DEFAULT_VANILLA_PROVENANCE,
    )
    p.add_argument(
        "--no-existing-vanilla",
        action="store_false",
        dest="use_existing_vanilla",
        help="do not include the existing n=10 vanilla baseline",
    )
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--probe-episode", type=int, default=0)
    p.add_argument("--probe-drone", type=int, default=3)
    p.add_argument("--probe-t", type=float, default=29.1)
    p.add_argument("--follow", type=float, default=0.35)
    args = p.parse_args(argv)
    args.temperatures = args.temperatures or [0.001]
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    result = run_or_summarize(args)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    render_figure(result, args.figure)
    print_table(result)
    print(f"wrote {args.summary_json}")
    print(f"wrote {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
