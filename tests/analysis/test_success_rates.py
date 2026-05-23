"""Tests for uav_nav_lab.analysis.success_rates.

Covers the recurring research-script patterns: missing dirs return
sentinel empty values, n_eps caps the loaded count, the percentage
Wilson wrapper aligns with the underlying stdlib stats.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from uav_nav_lab.analysis import (
    joint_outcomes,
    joint_success_rate,
    wilson,
)


def _write_joint(path: Path, seed: int, outcome: str) -> None:
    path.write_text(json.dumps({
        "meta": {"seed": seed},
        "outcome": outcome,
        "per_drone_outcomes": [outcome],
    }))


def test_joint_outcomes_missing_dir_returns_empty(tmp_path: Path):
    assert joint_outcomes(tmp_path / "does_not_exist") == []


def test_joint_outcomes_returns_booleans_sorted_by_seed(tmp_path: Path):
    _write_joint(tmp_path / "episode_002_joint.json", seed=44, outcome="success")
    _write_joint(tmp_path / "episode_000_joint.json", seed=42, outcome="collision")
    _write_joint(tmp_path / "episode_001_joint.json", seed=43, outcome="success")
    assert joint_outcomes(tmp_path) == [False, True, True]


def test_joint_outcomes_n_eps_caps_count(tmp_path: Path):
    for i, outcome in enumerate(["success", "collision", "success", "success"]):
        _write_joint(tmp_path / f"episode_{i:03d}_joint.json",
                     seed=40 + i, outcome=outcome)
    assert joint_outcomes(tmp_path, n_eps=2) == [True, False]


def test_joint_success_rate_missing_dir_returns_nan_sentinel():
    rate, lo, hi, k, n = joint_success_rate("/tmp/nonexistent_uav_test_dir")
    assert math.isnan(rate) and math.isnan(lo) and math.isnan(hi)
    assert k == 0 and n == 0


def test_joint_success_rate_matches_wilson_percentage(tmp_path: Path):
    outcomes = ["success"] * 7 + ["collision"] * 3
    for i, oc in enumerate(outcomes):
        _write_joint(tmp_path / f"episode_{i:03d}_joint.json",
                     seed=42 + i, outcome=oc)
    rate, lo, hi, k, n = joint_success_rate(tmp_path)
    p_exp, lo_exp, hi_exp = wilson(7, 10)
    assert rate == pytest.approx(p_exp * 100.0)
    assert lo == pytest.approx(lo_exp * 100.0)
    assert hi == pytest.approx(hi_exp * 100.0)
    assert k == 7 and n == 10


def test_joint_success_rate_chunked_layout(tmp_path: Path):
    for seed, oc in [(7, "success"), (8, "collision"), (9, "success")]:
        sub = tmp_path / f"seed_{seed:03d}"
        sub.mkdir()
        _write_joint(sub / "episode_000_joint.json", seed=seed, outcome=oc)
    rate, _, _, k, n = joint_success_rate(tmp_path, layout="chunked")
    assert (k, n) == (2, 3)
    assert rate == pytest.approx(2.0 / 3.0 * 100.0)
