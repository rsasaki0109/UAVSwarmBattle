"""WarmupSelectMPPIPlanner unit tests.

The planner runs episode 0 at warmup_temperature, accumulates per-replan
top-2 and chosen-vs-goal angles, and at the start of episode 1 maps
the means through the N+P rule to pick a fixed temperature for the
remaining episodes.

Tests cover the four selection branches by directly priming the warmup
buffers and triggering a second reset(), plus an end-to-end check that
the first reset() leaves the planner at warmup_temperature with empty
buffers."""

from __future__ import annotations

import numpy as np
import pytest

from uav_nav_lab.planner import PLANNER_REGISTRY
from uav_nav_lab.planner.warmup_select_mppi import WarmupSelectMPPIPlanner


def _make() -> WarmupSelectMPPIPlanner:
    return PLANNER_REGISTRY.get("warmup_select_mppi").from_config(
        {
            "horizon": 10,
            "n_samples": 8,
            "max_speed": 5.0,
            "warmup_temperature": 1.0,
            "uniform_temperature": 10.0,
            "argmin_temperature": 0.1,
            "appl_cut": 50.0,
            "choice_cut": 12.5,
        }
    )


def test_registered_under_expected_name() -> None:
    cls = PLANNER_REGISTRY.get("warmup_select_mppi")
    assert cls is WarmupSelectMPPIPlanner


def test_first_reset_arms_warmup_pass() -> None:
    p = _make()
    p.reset()
    assert p.temperature == pytest.approx(1.0)
    assert p._episode_idx == 0
    assert p._selected_reason == "warmup_pass"
    assert p._warm_top2 == []
    assert p._warm_cvg == []


def test_warmup_episode_collects_angles_from_plan() -> None:
    """During episode 0, plan() must populate the warmup buffers from
    the parent MPPI's _last_* diagnostics."""
    p = _make()
    p.reset()
    occ = np.zeros((30, 30), dtype=bool)
    p.plan(np.array([2.0, 2.0]), np.array([20.0, 20.0]), occ)
    p.plan(np.array([3.0, 3.0]), np.array([20.0, 20.0]), occ)
    assert len(p._warm_top2) == 2
    assert len(p._warm_cvg) == 2


def test_second_reset_with_aligned_warmup_selects_uniform() -> None:
    """top-2 low + chosen-vs-goal small → prior is correct → uniform
    temperature."""
    p = _make()
    p.reset()
    p._warm_top2 = [30.0, 32.0, 28.0]
    p._warm_cvg = [5.0, 6.0, 4.0]
    p.reset()
    assert p.temperature == pytest.approx(10.0)
    assert "prior_aligned" in p._selected_reason


def test_second_reset_with_misaligned_warmup_selects_argmin() -> None:
    """top-2 low + chosen-vs-goal large → prior misses → argmin
    temperature."""
    p = _make()
    p.reset()
    p._warm_top2 = [30.0, 32.0]
    p._warm_cvg = [20.0, 22.0]
    p.reset()
    assert p.temperature == pytest.approx(0.1)
    assert "prior_misses" in p._selected_reason


def test_second_reset_with_chaotic_top2_falls_back_to_warmup() -> None:
    """top-2 large → rule N/A → keep warmup_temperature rather than
    pretend we have signal."""
    p = _make()
    p.reset()
    p._warm_top2 = [80.0, 85.0, 90.0]
    p._warm_cvg = [5.0, 5.0, 5.0]  # would have selected uniform but for top-2
    p.reset()
    assert p.temperature == pytest.approx(1.0)
    assert "chaotic" in p._selected_reason


def test_second_reset_with_no_warmup_samples_falls_back_to_warmup() -> None:
    """If episode 0 had zero replans (degenerate edge case) the planner
    cannot select — keep warmup_temperature."""
    p = _make()
    p.reset()
    p.reset()
    assert p.temperature == pytest.approx(1.0)
    assert p._selected_reason == "no_warmup_samples_fallback"


def test_selection_persists_across_later_resets() -> None:
    """Episode >=2 must not re-select — the warmup pass is the only
    decision point."""
    p = _make()
    p.reset()
    p._warm_top2 = [30.0]
    p._warm_cvg = [5.0]
    p.reset()  # ep 1 — selects uniform
    selected_temp = p.temperature
    assert selected_temp == pytest.approx(10.0)
    p.reset()  # ep 2 — temperature must remain uniform
    p.reset()  # ep 3 — same
    assert p.temperature == pytest.approx(selected_temp)


def test_plan_meta_carries_warmup_select_diagnostics() -> None:
    p = _make()
    p.reset()
    occ = np.zeros((30, 30), dtype=bool)
    plan = p.plan(np.array([2.0, 2.0]), np.array([20.0, 20.0]), occ)
    info = plan.meta["warmup_select"]
    assert info["episode_idx"] == 0
    assert info["temperature"] == pytest.approx(1.0)
    assert info["selected_reason"] == "warmup_pass"
    assert info["n_warmup_samples"] == 1
