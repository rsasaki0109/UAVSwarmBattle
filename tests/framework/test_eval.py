"""Evaluator metrics: Wilson CI, summary fields, spatial-run comparisons."""

from __future__ import annotations

import json  # noqa: F401
import subprocess  # noqa: F401
import sys  # noqa: F401
from pathlib import Path  # noqa: F401

import numpy as np  # noqa: F401
import pytest  # noqa: F401

from uav_nav_lab.cli import build_parser, main  # noqa: F401
from uav_nav_lab.config import ExperimentConfig  # noqa: F401
from uav_nav_lab.eval import evaluate_run  # noqa: F401
from uav_nav_lab.planner import PLANNER_REGISTRY  # noqa: F401
from uav_nav_lab.runner import expand_sweep, run_experiment  # noqa: F401

from tests._helpers import EXAMPLES, _basic_cfg, _require_mplot3d  # noqa: F401


def test_compare_spatial_runs_accepts_matching_episode_logs(tmp_path: Path) -> None:
    from scripts.compare_spatial_runs import compare_runs

    def write_ep(run_dir: Path, name: str, xs: list[float]) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        steps = [
            {
                "t": i * 0.05,
                "true_pos": [x, x + 1.0, 2.0],
                "true_vel": [1.0, 1.0, 0.0],
                "observed_pos": [x, x + 1.0, 2.0],
                "cmd": [1.0, 1.0, 0.0],
                "collision": False,
                "goal_reached": False,
            }
            for i, x in enumerate(xs)
        ]
        (run_dir / name).write_text(
            json.dumps(
                {
                    "meta": {"episode": 0, "seed": 42},
                    "outcome": "success",
                    "summary": {"final_t": len(xs) * 0.05},
                    "replans": [],
                    "steps": steps,
                }
            ),
            encoding="utf-8",
        )

    left = tmp_path / "direct"
    right = tmp_path / "ros2"
    write_ep(left, "episode_000.json", [0.0, 1.0, 2.0, 3.0])
    write_ep(right, "episode_000.json", [0.02, 1.02, 2.02, 3.02])

    report = compare_runs(left, right)
    assert report["passed"] is True
    assert report["max_final_position_delta_m"] < 0.1


def test_wilson_ci_bounds() -> None:
    """Wilson interval should: (1) bracket the point estimate, (2) widen at
    small N, (3) stay inside [0,1] even at boundary outcomes (0/N or N/N)."""
    from uav_nav_lab.eval.metrics import _wilson

    p, lo, hi = _wilson(3, 5)
    assert lo <= p <= hi
    assert lo > 0 and hi < 1
    assert (hi - lo) > 0.4  # N=5 is wide

    _, lo, hi = _wilson(50, 100)
    assert (hi - lo) < 0.2  # N=100 is much tighter

    _, lo, hi = _wilson(0, 5)  # boundary
    assert lo == 0.0 and 0 < hi < 1
    _, lo, hi = _wilson(5, 5)  # other boundary
    assert hi == 1.0 and 0 < lo < 1


def test_summary_includes_ci(tmp_path: Path) -> None:
    cfg = _basic_cfg()
    run_dir = run_experiment(cfg, tmp_path / "ci_run")
    summary = evaluate_run(run_dir)
    assert "success_ci95" in summary
    lo, hi = summary["success_ci95"]
    assert 0.0 <= lo <= summary["success_rate"] <= hi <= 1.0
    assert "ci_lo" in summary["avg_speed"]
    assert "sem" in summary["avg_speed"]
