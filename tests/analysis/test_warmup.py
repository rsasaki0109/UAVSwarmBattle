"""Tests for uav_nav_lab.analysis.warmup.diagnose_warmup.

Uses a small canonical YAML (intersection_v1_noisy30 warmup_select_mppi)
as the regression fixture — it is fast enough for unit testing
(~1-2 s per episode) and exercises the multi-drone runner path that
all other diagnose_warmup callers depend on.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from uav_nav_lab.analysis import WarmupDiagnostic, diagnose_warmup

YAML = "examples/exp_intersection_v1_noisy30_warmup_select_mppi_n20.yaml"


@pytest.fixture(autouse=True)
def _clear_sessions():
    from uav_nav_lab.planner.warmup_select_mppi import _SHARED_SESSIONS
    _SHARED_SESSIONS.clear()
    yield
    _SHARED_SESSIONS.clear()


def test_diagnose_warmup_ep0_returns_pooled_signal():
    diag = diagnose_warmup(YAML, episodes=1)
    assert isinstance(diag, WarmupDiagnostic)
    assert diag.n_drones == 2
    assert diag.n_samples > 0
    # The intersection_v1 warmup signal is well-known from T:
    # pooled cvg ~ 9.2°, top2 ~ 29° (small drift across runs OK).
    assert 5.0 < diag.cvg_mean < 15.0
    assert 15.0 < diag.top2_mean < 50.0
    # No auto-pick fires when episodes=1
    assert diag.selected_temperature is None
    assert diag.selected_reason is None


def test_diagnose_warmup_ep1_fires_autopick():
    diag = diagnose_warmup(YAML, episodes=2)
    # intersection_v1 has low cvg → uniform t=10 per N+P
    assert diag.selected_temperature == pytest.approx(10.0)
    assert "prior_aligned" in (diag.selected_reason or "")


def test_diagnose_warmup_series_length_matches_n_samples():
    diag = diagnose_warmup(YAML, episodes=1)
    assert len(diag.top2_series) == diag.n_samples
    assert len(diag.cvg_series) == diag.n_samples


def test_diagnose_warmup_max_is_at_least_mean():
    diag = diagnose_warmup(YAML, episodes=1)
    # Trivial invariant — but if max ever drops below mean we have a
    # statistic computation bug worth catching.
    assert diag.top2_max >= diag.top2_mean - 1e-6
    assert diag.cvg_max >= diag.cvg_mean - 1e-6


def test_diagnose_warmup_does_not_leak_session():
    # After the function returns, the module-level singleton should be
    # either cleared by the next call or carry only this run's data.
    # Either way, a second call must produce a self-consistent result.
    from uav_nav_lab.planner.warmup_select_mppi import _SHARED_SESSIONS
    diag1 = diagnose_warmup(YAML, episodes=1)
    n1 = diag1.n_samples
    diag2 = diagnose_warmup(YAML, episodes=1)
    n2 = diag2.n_samples
    # diagnose_warmup clears _SHARED_SESSIONS at entry → samples don't
    # accumulate across calls.
    assert n2 == n1
