#!/usr/bin/env python3
"""Compare spatial agreement between two uav-nav run directories.

This is meant for backend parity checks where the planner/scenario/sensor
are identical and only the simulator bridge differs, e.g. direct AirSim
RPC versus AirSim routed through ROS 2.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _episode_files(run_dir: Path) -> list[Path]:
    files = sorted(
        p for p in Path(run_dir).glob("episode_*.json")
        if not p.name.endswith("_joint.json")
    )
    if not files:
        raise FileNotFoundError(f"no episode_*.json files under {run_dir}")
    return files


def _load_episode(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _positions(ep: dict[str, Any]) -> np.ndarray:
    steps = ep.get("steps") or []
    if not steps:
        return np.zeros((0, 0), dtype=float)
    return np.asarray([s["true_pos"] for s in steps], dtype=float)


def _path_length(pos: np.ndarray) -> float:
    if pos.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())


def _resample_by_index(pos: np.ndarray, n: int) -> np.ndarray:
    if pos.shape[0] == 0 or n <= 0:
        return np.zeros((0, 0), dtype=float)
    if pos.shape[0] == n:
        return pos
    src = np.linspace(0.0, 1.0, pos.shape[0])
    dst = np.linspace(0.0, 1.0, n)
    cols = [np.interp(dst, src, pos[:, i]) for i in range(pos.shape[1])]
    return np.stack(cols, axis=1)


def compare_runs(
    left_dir: Path,
    right_dir: Path,
    *,
    final_pos_threshold_m: float = 0.75,
    rms_threshold_m: float = 1.0,
    path_length_threshold_m: float = 2.0,
) -> dict[str, Any]:
    left_files = _episode_files(left_dir)
    right_files = _episode_files(right_dir)
    if len(left_files) != len(right_files):
        raise ValueError(
            f"episode count mismatch: {left_dir} has {len(left_files)}, "
            f"{right_dir} has {len(right_files)}"
        )

    episodes: list[dict[str, Any]] = []
    for idx, (lf, rf) in enumerate(zip(left_files, right_files)):
        left = _load_episode(lf)
        right = _load_episode(rf)
        lpos = _positions(left)
        rpos = _positions(right)
        if lpos.shape[1:] != rpos.shape[1:]:
            raise ValueError(
                f"episode {idx}: dimension mismatch {lpos.shape} vs {rpos.shape}"
            )

        n = max(1, min(lpos.shape[0], rpos.shape[0]))
        lrs = _resample_by_index(lpos, n)
        rrs = _resample_by_index(rpos, n)
        rms = float(math.sqrt(float(np.mean(np.sum((lrs - rrs) ** 2, axis=1)))))
        final_delta = float(np.linalg.norm(lpos[-1] - rpos[-1]))
        path_delta = abs(_path_length(lpos) - _path_length(rpos))

        episodes.append(
            {
                "episode": idx,
                "left_file": str(lf),
                "right_file": str(rf),
                "left_outcome": left.get("outcome"),
                "right_outcome": right.get("outcome"),
                "left_steps": int(lpos.shape[0]),
                "right_steps": int(rpos.shape[0]),
                "final_position_delta_m": final_delta,
                "rms_position_delta_m": rms,
                "path_length_delta_m": float(path_delta),
            }
        )

    max_final = max(e["final_position_delta_m"] for e in episodes)
    max_rms = max(e["rms_position_delta_m"] for e in episodes)
    max_path = max(e["path_length_delta_m"] for e in episodes)
    outcomes_match = all(e["left_outcome"] == e["right_outcome"] for e in episodes)
    passed = (
        outcomes_match
        and max_final <= final_pos_threshold_m
        and max_rms <= rms_threshold_m
        and max_path <= path_length_threshold_m
    )
    return {
        "left_dir": str(left_dir),
        "right_dir": str(right_dir),
        "n_episodes": len(episodes),
        "passed": passed,
        "thresholds": {
            "final_position_delta_m": final_pos_threshold_m,
            "rms_position_delta_m": rms_threshold_m,
            "path_length_delta_m": path_length_threshold_m,
        },
        "max_final_position_delta_m": max_final,
        "max_rms_position_delta_m": max_rms,
        "max_path_length_delta_m": max_path,
        "outcomes_match": outcomes_match,
        "episodes": episodes,
    }


def _format_report(report: dict[str, Any]) -> str:
    status = "PASS" if report["passed"] else "FAIL"
    lines = [
        f"[spatial-compare] {status}",
        f"  left:  {report['left_dir']}",
        f"  right: {report['right_dir']}",
        f"  episodes: {report['n_episodes']}",
        f"  outcomes match: {report['outcomes_match']}",
        (
            "  max deltas: "
            f"final={report['max_final_position_delta_m']:.3f} m, "
            f"rms={report['max_rms_position_delta_m']:.3f} m, "
            f"path={report['max_path_length_delta_m']:.3f} m"
        ),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("left_run")
    p.add_argument("right_run")
    p.add_argument("--final-pos-threshold-m", type=float, default=0.75)
    p.add_argument("--rms-threshold-m", type=float, default=1.0)
    p.add_argument("--path-length-threshold-m", type=float, default=2.0)
    p.add_argument("--json-out", type=Path)
    args = p.parse_args(argv)

    report = compare_runs(
        Path(args.left_run),
        Path(args.right_run),
        final_pos_threshold_m=args.final_pos_threshold_m,
        rms_threshold_m=args.rms_threshold_m,
        path_length_threshold_m=args.path_length_threshold_m,
    )
    print(_format_report(report))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
