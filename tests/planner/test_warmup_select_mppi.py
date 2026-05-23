"""WarmupSelectMPPIPlanner unit tests.

The planner runs episode 0 at warmup_temperature, accumulates per-replan
top-2 and chosen-vs-goal angles, and at the start of episode 1 maps
the means through the N+P rule to pick a fixed temperature for the
remaining episodes.

Tests cover the four selection branches by directly priming the warmup
buffers and triggering a second reset(). Per-drone (local) selection is
exercised with share_warmup=False; the default-shared pooling is
exercised separately in the trailing block of tests."""

from __future__ import annotations

import numpy as np
import pytest

from uav_nav_lab.planner import PLANNER_REGISTRY
from uav_nav_lab.planner.warmup_select_mppi import (
    WarmupSelectMPPIPlanner,
    _SHARED_SESSIONS,
)


def _make_local() -> WarmupSelectMPPIPlanner:
    """Per-drone selection (share_warmup=False) — for unit-testing the
    single-instance branches."""
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
            "share_warmup": False,
        }
    )


def _make_shared(key: str) -> WarmupSelectMPPIPlanner:
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
            "share_warmup": True,
            "share_warmup_key": key,
        }
    )


@pytest.fixture(autouse=True)
def _clear_sessions():
    """Each test starts with an empty shared-session registry."""
    _SHARED_SESSIONS.clear()
    yield
    _SHARED_SESSIONS.clear()


def test_registered_under_expected_name() -> None:
    cls = PLANNER_REGISTRY.get("warmup_select_mppi")
    assert cls is WarmupSelectMPPIPlanner


def test_first_reset_arms_warmup_pass() -> None:
    p = _make_local()
    p.reset()
    assert p.temperature == pytest.approx(1.0)
    assert p._episode_idx == 0
    assert p._selected_reason == "warmup_pass"
    assert p._warm_top2 == []
    assert p._warm_cvg == []


def test_warmup_episode_collects_angles_from_plan(empty_grid_30) -> None:
    """During episode 0, plan() must populate the warmup buffers from
    the parent MPPI's _last_* diagnostics."""
    p = _make_local()
    p.reset()
    occ = empty_grid_30
    p.plan(np.array([2.0, 2.0]), np.array([20.0, 20.0]), occ)
    p.plan(np.array([3.0, 3.0]), np.array([20.0, 20.0]), occ)
    assert len(p._warm_top2) == 2
    assert len(p._warm_cvg) == 2


def test_second_reset_with_aligned_warmup_selects_uniform() -> None:
    p = _make_local()
    p.reset()
    p._warm_top2 = [30.0, 32.0, 28.0]
    p._warm_cvg = [5.0, 6.0, 4.0]
    p.reset()
    assert p.temperature == pytest.approx(10.0)
    assert "prior_aligned" in p._selected_reason


def test_second_reset_with_misaligned_warmup_selects_argmin() -> None:
    p = _make_local()
    p.reset()
    p._warm_top2 = [30.0, 32.0]
    p._warm_cvg = [20.0, 22.0]
    p.reset()
    assert p.temperature == pytest.approx(0.1)
    assert "prior_misses" in p._selected_reason


def test_second_reset_with_chaotic_top2_falls_back_to_warmup() -> None:
    p = _make_local()
    p.reset()
    p._warm_top2 = [80.0, 85.0, 90.0]
    p._warm_cvg = [5.0, 5.0, 5.0]
    p.reset()
    assert p.temperature == pytest.approx(1.0)
    assert "chaotic" in p._selected_reason


def test_second_reset_with_no_warmup_samples_falls_back_to_warmup() -> None:
    p = _make_local()
    p.reset()
    p.reset()
    assert p.temperature == pytest.approx(1.0)
    assert p._selected_reason == "no_warmup_samples_fallback"


def test_selection_persists_across_later_resets() -> None:
    p = _make_local()
    p.reset()
    p._warm_top2 = [30.0]
    p._warm_cvg = [5.0]
    p.reset()
    selected_temp = p.temperature
    assert selected_temp == pytest.approx(10.0)
    p.reset()
    p.reset()
    assert p.temperature == pytest.approx(selected_temp)


def test_plan_meta_carries_warmup_select_diagnostics(empty_grid_30) -> None:
    p = _make_local()
    p.reset()
    occ = empty_grid_30
    plan = p.plan(np.array([2.0, 2.0]), np.array([20.0, 20.0]), occ)
    info = plan.meta["warmup_select"]
    assert info["episode_idx"] == 0
    assert info["temperature"] == pytest.approx(1.0)
    assert info["selected_reason"] == "warmup_pass"
    assert info["n_warmup_samples"] == 1


def test_share_warmup_defaults_to_true() -> None:
    """from_config without share_warmup → share_warmup_key is set."""
    p = PLANNER_REGISTRY.get("warmup_select_mppi").from_config(
        {"horizon": 10, "n_samples": 8, "max_speed": 5.0}
    )
    assert p._share_warmup_key == "_default"


# ---- shared-warmup pooling ----


def test_shared_members_register_against_same_session() -> None:
    a = _make_shared("pool_a")
    b = _make_shared("pool_a")
    a.reset()
    b.reset()
    assert "pool_a" in _SHARED_SESSIONS
    # Both members clear the same session in ep 0 reset (idempotent).
    assert _SHARED_SESSIONS["pool_a"].top2 == []


def test_shared_pooled_means_override_per_drone_signal() -> None:
    """Two drones in the same session — one with cvg=5 (would select
    uniform alone), one with cvg=25 (would select argmin alone). Pooled
    mean cvg=15 → falls in the argmin region by the choice_cut, so BOTH
    drones must adopt argmin, NOT split the decision per-drone (which is
    exactly the v1 drift the share fixes)."""
    a = _make_shared("pool_b")
    b = _make_shared("pool_b")
    a.reset()
    b.reset()
    # Manually plant pooled session data — this is what plan() would
    # have accumulated across both drones' ep 0 replans.
    sess = _SHARED_SESSIONS["pool_b"]
    sess.top2.extend([30.0, 30.0])
    sess.cvg.extend([5.0, 25.0])
    a.reset()
    b.reset()
    assert a.temperature == pytest.approx(0.1)
    assert b.temperature == pytest.approx(0.1)
    assert a._selected_reason == b._selected_reason
    assert "pooled" in a._selected_reason


def test_shared_pooled_aligned_selects_uniform() -> None:
    """Two drones, pooled cvg=4 → uniform."""
    a = _make_shared("pool_c")
    b = _make_shared("pool_c")
    a.reset()
    b.reset()
    sess = _SHARED_SESSIONS["pool_c"]
    sess.top2.extend([20.0, 25.0, 22.0])
    sess.cvg.extend([3.0, 5.0, 4.0])
    a.reset()
    b.reset()
    assert a.temperature == pytest.approx(10.0)
    assert b.temperature == pytest.approx(10.0)


def test_shared_first_member_to_select_writes_session() -> None:
    """The first member through reset(ep 1) computes the decision and
    stores it; the second member adopts without recomputing."""
    a = _make_shared("pool_d")
    b = _make_shared("pool_d")
    a.reset()
    b.reset()
    sess = _SHARED_SESSIONS["pool_d"]
    sess.top2.extend([30.0])
    sess.cvg.extend([3.0])
    a.reset()  # ep 1 — A picks
    assert sess.selected_temperature == pytest.approx(10.0)
    # Mutate session means after A picked — B should still adopt A's pick
    sess.cvg[:] = [99.0]
    b.reset()
    assert b.temperature == pytest.approx(10.0)


def test_shared_session_resets_between_runs() -> None:
    """Two consecutive runs sharing the default key — the second run's
    ep 0 reset must wipe the first run's selection."""
    a = _make_shared("pool_e")
    a.reset()
    a._warm_top2 = [30.0]
    a._warm_cvg = [5.0]
    _SHARED_SESSIONS["pool_e"].top2.extend([30.0])
    _SHARED_SESSIONS["pool_e"].cvg.extend([5.0])
    a.reset()  # ep 1 — selects uniform
    assert _SHARED_SESSIONS["pool_e"].selected_temperature == pytest.approx(10.0)
    # New planner with same key (e.g. a second experiment) starts fresh
    b = _make_shared("pool_e")
    b.reset()  # ep 0 — must clear session
    assert _SHARED_SESSIONS["pool_e"].selected_temperature is None
    assert _SHARED_SESSIONS["pool_e"].top2 == []


def test_different_keys_do_not_pool() -> None:
    a = _make_shared("pool_f")
    b = _make_shared("pool_g")
    a.reset()
    b.reset()
    _SHARED_SESSIONS["pool_f"].top2.append(30.0)
    _SHARED_SESSIONS["pool_f"].cvg.append(5.0)
    _SHARED_SESSIONS["pool_g"].top2.append(30.0)
    _SHARED_SESSIONS["pool_g"].cvg.append(25.0)
    a.reset()
    b.reset()
    assert a.temperature == pytest.approx(10.0)
    assert b.temperature == pytest.approx(0.1)


def test_shared_writes_to_session_during_plan(empty_grid_30) -> None:
    """plan() during ep 0 must push samples into the shared session AND
    the local buffer."""
    p = _make_shared("pool_h")
    p.reset()
    occ = empty_grid_30
    p.plan(np.array([2.0, 2.0]), np.array([20.0, 20.0]), occ)
    p.plan(np.array([3.0, 3.0]), np.array([20.0, 20.0]), occ)
    assert len(_SHARED_SESSIONS["pool_h"].top2) == 2
    assert len(_SHARED_SESSIONS["pool_h"].cvg) == 2
    assert len(p._warm_top2) == 2
