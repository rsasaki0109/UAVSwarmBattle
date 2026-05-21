"""Characterization tests for uav_nav_lab.analysis.joint_stats.

Mirrors the inputs the paired-analysis scripts actually pass — Wilson
CI on small N, mcnemar symmetry, both JSON layouts (flat + chunked).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from uav_nav_lab.analysis import (
    binom_pmf,
    load_joint_episodes,
    mcnemar_exact_p,
    wilson,
)


def test_wilson_zero_n_returns_zeros():
    assert wilson(0, 0) == (0.0, 0.0, 0.0)


def test_wilson_known_value_50_of_100():
    # k=50, n=100, z=1.96  → p̂=0.5, CI ≈ [0.404, 0.596]
    p, lo, hi = wilson(50, 100)
    assert p == pytest.approx(0.5)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)


def test_wilson_extremes_clamped_to_unit_interval():
    # All-success: lower bound > 0, upper bound clamped to 1
    _, lo, hi = wilson(10, 10)
    assert lo > 0.0
    assert hi == pytest.approx(1.0)
    # All-failure: lower bound clamped to 0, upper > 0
    _, lo, hi = wilson(0, 10)
    assert lo == pytest.approx(0.0)
    assert hi > 0.0


def test_binom_pmf_distribution_sums_to_one():
    n = 8
    total = sum(binom_pmf(k, n, 0.3) for k in range(n + 1))
    assert total == pytest.approx(1.0)


def test_mcnemar_no_disagreement_returns_one():
    assert mcnemar_exact_p(0, 0) == 1.0


def test_mcnemar_is_symmetric_in_b_and_c():
    # The two-sided exact test must not depend on which side "won"
    assert mcnemar_exact_p(7, 2) == pytest.approx(mcnemar_exact_p(2, 7))


def test_mcnemar_balanced_split_gives_high_p_value():
    # Equal off-diagonals → no evidence of asymmetry, p should be ≈ 1
    assert mcnemar_exact_p(5, 5) == pytest.approx(1.0)


def test_mcnemar_large_imbalance_gives_small_p_value():
    # 10 vs 0 is the maximum imbalance — exact p = 2 × (1/2)^10
    p = mcnemar_exact_p(10, 0)
    assert p == pytest.approx(2.0 * (0.5 ** 10))


def _write_joint(path: Path, *, seed: int, outcome: str, per_drone, final_t=None):
    payload = {
        "meta": {"seed": seed},
        "outcome": outcome,
        "per_drone_outcomes": per_drone,
    }
    if final_t is not None:
        payload["final_t"] = final_t
    path.write_text(json.dumps(payload))


def test_load_joint_episodes_flat_layout(tmp_path: Path):
    _write_joint(tmp_path / "episode_002_joint.json", seed=42, outcome="success",
                 per_drone=["success", "success"], final_t=3.5)
    _write_joint(tmp_path / "episode_000_joint.json", seed=40, outcome="collision",
                 per_drone=["collision", "success"], final_t=1.0)
    _write_joint(tmp_path / "episode_001_joint.json", seed=41, outcome="timeout",
                 per_drone=["success", "timeout"])

    rows = load_joint_episodes(tmp_path, layout="flat")
    assert [r["seed"] for r in rows] == [40, 41, 42]  # sorted by seed
    assert [r["joint"] for r in rows] == [False, False, True]
    assert rows[0]["per_drone"] == [False, True]
    assert rows[1]["final_t"] is None  # absent → None preserved
    assert rows[2]["final_t"] == 3.5


def test_load_joint_episodes_chunked_layout(tmp_path: Path):
    for seed, outcome in [(7, "success"), (8, "collision")]:
        sub = tmp_path / f"seed_{seed:03d}"
        sub.mkdir()
        _write_joint(sub / "episode_000_joint.json", seed=seed, outcome=outcome,
                     per_drone=["success", outcome], final_t=2.0)

    rows = load_joint_episodes(tmp_path, layout="chunked")
    assert [r["seed"] for r in rows] == [7, 8]
    assert rows[0]["joint"] is True
    assert rows[1]["joint"] is False
    assert rows[1]["per_drone"] == [True, False]


def test_load_joint_episodes_empty_dir_returns_empty(tmp_path: Path):
    assert load_joint_episodes(tmp_path, layout="flat") == []
    assert load_joint_episodes(tmp_path, layout="chunked") == []
