"""CVaR-MPPI risk-aware planner tests.

Two layers:
  1. cvar_costs() — the pure tail-averaging math.
  2. CVaRMPPIPlanner — registry wiring, reduction to vanilla MPPI in the
     degenerate cases, and the risk-aversion behaviour that is the whole
     point: under prediction noise, a low risk_alpha steers away from
     actions with a bad worst-case tail.
"""

from __future__ import annotations

import numpy as np
import pytest

from uav_nav_lab.planner import PLANNER_REGISTRY
from uav_nav_lab.planner.cvar_mppi.cvar import cvar_costs
from uav_nav_lab.planner.cvar_mppi.planner import CVaRMPPIPlanner
from uav_nav_lab.planner.mppi import MPPIPlanner


# --- cvar_costs ----------------------------------------------------------

def test_cvar_alpha_one_is_mean():
    c = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 10.0, 0.0]])
    out = cvar_costs(c, risk_alpha=1.0)
    assert np.allclose(out, c.mean(axis=1))


def test_cvar_small_alpha_is_worst_case():
    c = np.array([[1.0, 2.0, 3.0, 100.0]])
    # alpha → smallest tail = the single worst (largest) cost
    out = cvar_costs(c, risk_alpha=0.01)
    assert out[0] == 100.0


def test_cvar_quarter_tail_averages_worst_quarter():
    # 8 scenarios, alpha 0.25 → worst 2 averaged
    c = np.array([[0, 1, 2, 3, 4, 5, 6, 10]], dtype=float)
    out = cvar_costs(c, risk_alpha=0.25)
    assert out[0] == pytest.approx((6 + 10) / 2)


def test_cvar_monotonic_in_alpha():
    rng = np.random.default_rng(0)
    c = rng.normal(size=(5, 50))
    prev = None
    for a in (1.0, 0.5, 0.25, 0.1, 0.02):
        v = cvar_costs(c, risk_alpha=a)
        if prev is not None:
            # smaller alpha = focus on worse tail = cost cannot decrease
            assert np.all(v >= prev - 1e-9)
        prev = v


def test_cvar_rejects_bad_alpha():
    c = np.zeros((2, 4))
    with pytest.raises(ValueError):
        cvar_costs(c, risk_alpha=0.0)
    with pytest.raises(ValueError):
        cvar_costs(c, risk_alpha=1.5)


# --- planner wiring ------------------------------------------------------

def test_cvar_mppi_registered():
    assert "cvar_mppi" in PLANNER_REGISTRY.names()


def test_cvar_mppi_from_config_defaults():
    p = PLANNER_REGISTRY.get("cvar_mppi").from_config(
        {"max_speed": 5.0, "n_scenarios": 8, "risk_alpha": 0.3}
    )
    assert isinstance(p, CVaRMPPIPlanner)
    assert p.n_scenarios == 8
    assert p.risk_alpha == 0.3
    assert p.max_speed == 5.0


def test_cvar_no_dynamics_matches_mppi(empty_grid_30):
    """With no dynamic obstacles, CVaR-MPPI scoring is deterministic and must
    reduce to vanilla MPPI exactly (same rollout/cost/softmax stack)."""
    occ = empty_grid_30
    obs = np.array([2.0, 2.0])
    goal = np.array([20.0, 20.0])
    args = dict(max_speed=5.0, horizon=20, n_samples=16, inflate=0, temperature=1.0)

    mppi = MPPIPlanner(**args)
    mppi.reset()
    a_mppi = mppi.plan(obs, goal, occ).target_velocity

    cvar = CVaRMPPIPlanner(n_scenarios=16, risk_alpha=0.1, **args)
    cvar.reset()
    a_cvar = cvar.plan(obs, goal, occ).target_velocity

    assert np.allclose(a_mppi, a_cvar, atol=1e-9)


def test_cvar_alpha_one_matches_mppi_under_dynamics(empty_grid_30):
    """risk_alpha=1.0 averages every scenario. With zero noise the scenarios
    are identical, so the mean equals the nominal → identical to MPPI even
    with a dynamic obstacle present."""
    occ = empty_grid_30
    obs = np.array([2.0, 2.0])
    goal = np.array([26.0, 26.0])
    dyn = [{"position": [14.0, 14.0], "velocity": [0.0, 0.0], "radius": 1.0}]
    args = dict(max_speed=5.0, horizon=20, n_samples=16, inflate=0, temperature=1.0)

    mppi = MPPIPlanner(**args)
    mppi.reset()
    a_mppi = mppi.plan(obs, goal, occ, dynamic_obstacles=dyn).target_velocity

    cvar = CVaRMPPIPlanner(n_scenarios=8, risk_alpha=1.0, pred_noise_std=0.0, **args)
    cvar.reset()
    a_cvar = cvar.plan(obs, goal, occ, dynamic_obstacles=dyn).target_velocity

    assert np.allclose(a_mppi, a_cvar, atol=1e-9)


def test_cvar_meta_reports_risk_params(empty_grid_30):
    occ = empty_grid_30
    dyn = [{"position": [14.0, 14.0], "velocity": [1.0, 0.0], "radius": 1.0}]
    cvar = CVaRMPPIPlanner(
        n_scenarios=12, risk_alpha=0.25, pred_noise_std=0.5,
        max_speed=5.0, horizon=20, n_samples=16, inflate=0,
    )
    cvar.reset()
    plan = cvar.plan(np.array([2.0, 2.0]), np.array([26.0, 26.0]), occ,
                     dynamic_obstacles=dyn)
    assert plan.meta["planner"] == "cvar_mppi"
    assert plan.meta["risk_alpha"] == 0.25
    assert plan.meta["n_scenarios"] == 12


def test_cvar_is_more_conservative_than_mppi():
    """The behavioural claim: facing a fast crossing obstacle under prediction
    noise, the risk-averse planner does not steer more toward the threat than
    risk-neutral MPPI.

    The obstacle is below the straight line and sweeping up; a risk-averse
    planner should not dive further down (toward the threat's path) than the
    neutral one.
    """
    occ = np.zeros((40, 40), dtype=bool)
    obs = np.array([2.0, 20.0])
    goal = np.array([38.0, 20.0])
    dyn = [{"position": [16.0, 14.0], "velocity": [0.0, 6.0], "radius": 1.5}]
    args = dict(max_speed=6.0, horizon=30, n_samples=24, inflate=0,
                temperature=0.5, safety_margin=0.5)

    mppi = MPPIPlanner(**args)
    mppi.reset()
    a_neutral = mppi.plan(obs, goal, occ, dynamic_obstacles=dyn).target_velocity

    cvar = CVaRMPPIPlanner(n_scenarios=24, risk_alpha=0.1, pred_noise_std=1.0,
                           base_seed=1, **args)
    cvar.reset()
    a_averse = cvar.plan(obs, goal, occ, dynamic_obstacles=dyn).target_velocity

    # risk-averse should not steer more downward (toward the rising threat).
    assert a_averse[1] >= a_neutral[1] - 1e-6
