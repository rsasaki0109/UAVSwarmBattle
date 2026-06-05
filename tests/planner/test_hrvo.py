"""HRVO planner: hybrid reciprocal velocity-obstacle property checks (Snape 2011)."""
import numpy as np
import pytest

from uav_nav_lab.planner import PLANNER_REGISTRY


def _hrvo(**kw):
    cfg = {"max_speed": 5.0, "radius": 0.4, "safety_margin": 0.1, "time_horizon": 2.0,
           "neighbor_dist": 15.0, "goal_radius": 1.5, "time_step": 0.05}
    cfg.update(kw)
    p = PLANNER_REGISTRY.get("hrvo").from_config(cfg)
    p.reset()
    return p


def test_registered():
    assert "hrvo" in PLANNER_REGISTRY.names()


def test_no_neighbours_heads_to_goal():
    p = _hrvo()
    p.set_current_state(np.array([5.0, 25.0]), np.zeros(2))
    v = p.plan(np.array([5.0, 25.0]), np.array([45.0, 25.0]), None, dynamic_obstacles=[]).target_velocity
    np.testing.assert_allclose(v, [5.0, 0.0], atol=1e-6)


def test_inside_goal_radius_stops():
    p = _hrvo()
    v = p.plan(np.array([45.0, 25.0]), np.array([45.5, 25.0]), None, dynamic_obstacles=[]).target_velocity
    np.testing.assert_allclose(v, [0.0, 0.0], atol=1e-9)


def test_speed_capped():
    p = _hrvo()
    p.set_current_state(np.array([5.0, 25.0]), np.array([5.0, 0.0]))
    v = p.plan(np.array([5.0, 25.0]), np.array([45.0, 25.0]), None,
               dynamic_obstacles=[{"position": [12.0, 25.0], "velocity": [-5.0, 0.0], "radius": 0.4}]).target_velocity
    assert float(np.linalg.norm(v)) <= 5.0 + 1e-6


def test_head_on_deflects():
    p = _hrvo()
    p.set_current_state(np.array([5.0, 25.0]), np.array([5.0, 0.0]))
    v = p.plan(np.array([5.0, 25.0]), np.array([45.0, 25.0]), None,
               dynamic_obstacles=[{"position": [10.0, 25.0], "velocity": [-5.0, 0.0], "radius": 0.4}]).target_velocity
    assert abs(v[1]) > 1e-3 or v[0] < 5.0 - 1e-3


def test_side_commitment_is_consistent():
    # HRVO should commit to the same side across two near-identical steps (no flip),
    # given the agent already has a lateral velocity favouring one side.
    p = _hrvo()
    obs = [{"position": [10.0, 25.0], "velocity": [-5.0, 0.0], "radius": 0.4}]
    p.set_current_state(np.array([5.0, 25.0]), np.array([4.8, -1.3]))  # favouring -y
    v1 = p.plan(np.array([5.0, 25.0]), np.array([45.0, 25.0]), None, dynamic_obstacles=obs).target_velocity
    p.set_current_state(np.array([5.05, 24.95]), np.array(v1))
    v2 = p.plan(np.array([5.05, 24.95]), np.array([45.0, 25.0]), None, dynamic_obstacles=obs).target_velocity
    assert np.sign(v1[1]) == np.sign(v2[1])  # same side both steps


def test_three_d_rejected():
    p = _hrvo()
    with pytest.raises(ValueError):
        p.plan(np.zeros(3), np.array([1.0, 0.0, 1.0]), None, dynamic_obstacles=[])


def test_pairwise_bias_tilts_toward_convention():
    # With a neighbour ahead, pairwise_bias should tilt the chosen velocity to a
    # consistent side vs the unbiased planner (the right-of-way convention).
    obs = [{"position": [12.0, 25.0], "velocity": [0.0, 0.0], "radius": 0.4}]
    base = _hrvo()
    base.set_current_state(np.array([5.0, 25.0]), np.array([5.0, 0.0]))
    vb = base.plan(np.array([5.0, 25.0]), np.array([45.0, 25.0]), None, dynamic_obstacles=obs).target_velocity
    conv = _hrvo(pairwise_bias=0.6, pairwise_radius=6.0)
    conv.set_current_state(np.array([5.0, 25.0]), np.array([5.0, 0.0]))
    vc = conv.plan(np.array([5.0, 25.0]), np.array([45.0, 25.0]), None, dynamic_obstacles=obs).target_velocity
    # convention tilts to the neighbour's right (clockwise => negative y here)
    assert vc[1] < vb[1] - 1e-3


def test_lateral_bias_zero_is_stock():
    # A zero convention must be identical to omitting it.
    obs = [{"position": [10.0, 25.0], "velocity": [-5.0, 0.0], "radius": 0.4}]
    a = _hrvo()
    a.set_current_state(np.array([5.0, 25.0]), np.array([5.0, 0.0]))
    va = a.plan(np.array([5.0, 25.0]), np.array([45.0, 25.0]), None, dynamic_obstacles=obs).target_velocity
    b = _hrvo(lateral_bias=0.0, pairwise_bias=0.0)
    b.set_current_state(np.array([5.0, 25.0]), np.array([5.0, 0.0]))
    vb = b.plan(np.array([5.0, 25.0]), np.array([45.0, 25.0]), None, dynamic_obstacles=obs).target_velocity
    np.testing.assert_allclose(va, vb, atol=1e-9)
