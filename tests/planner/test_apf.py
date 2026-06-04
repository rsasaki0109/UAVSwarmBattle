"""APF planner: artificial-potential-field property checks (Khatib 1986)."""
import numpy as np
import pytest

from uav_nav_lab.planner import PLANNER_REGISTRY


def _apf(**kw):
    cfg = {"max_speed": 5.0, "radius": 0.4, "k_att": 1.0, "k_rep": 6.0,
           "influence_dist": 4.0, "time_step": 0.05, "goal_radius": 1.5}
    cfg.update(kw)
    return PLANNER_REGISTRY.get("apf").from_config(cfg)


def test_registered():
    assert "apf" in PLANNER_REGISTRY.names()


def test_no_neighbors_goes_straight_at_cruise():
    p = _apf()
    v = p.plan(np.array([5.0, 25.0]), np.array([45.0, 25.0]), None,
               dynamic_obstacles=[]).target_velocity
    np.testing.assert_allclose(v, [5.0, 0.0], atol=1e-6)


def test_inside_goal_radius_stops():
    p = _apf()
    v = p.plan(np.array([45.0, 25.0]), np.array([45.5, 25.0]), None,
               dynamic_obstacles=[]).target_velocity
    np.testing.assert_allclose(v, [0.0, 0.0], atol=1e-9)


def test_speed_capped():
    p = _apf()
    v = p.plan(np.array([5.0, 25.0]), np.array([45.0, 25.0]), None,
               dynamic_obstacles=[{"position": [7.0, 25.0], "radius": 0.4}]).target_velocity
    assert float(np.linalg.norm(v)) <= 5.0 + 1e-6


def test_close_peer_repels():
    # A peer straight ahead and close must push the velocity away from it
    # (negative x component) — the repulsive field dominates near contact.
    p = _apf()
    v = p.plan(np.array([5.0, 25.0]), np.array([45.0, 25.0]), None,
               dynamic_obstacles=[{"position": [6.0, 25.0], "radius": 0.4}]).target_velocity
    assert v[0] < 0.0


def test_pairwise_bias_does_not_tilt_a_lone_agent():
    p = _apf(pairwise_bias=0.5, pairwise_radius=8.0)
    v = p.plan(np.array([5.0, 25.0]), np.array([45.0, 25.0]), None,
               dynamic_obstacles=[]).target_velocity
    np.testing.assert_allclose(v, [5.0, 0.0], atol=1e-6)


def test_3d_supported():
    p = _apf()
    v = p.plan(np.array([5.0, 25.0, 8.0]), np.array([45.0, 25.0, 8.0]), None,
               dynamic_obstacles=[]).target_velocity
    assert v.shape == (3,)
    np.testing.assert_allclose(v, [5.0, 0.0, 0.0], atol=1e-5)


def test_four_d_rejected():
    p = _apf()
    with pytest.raises(ValueError):
        p.plan(np.zeros(4), np.ones(4), None, dynamic_obstacles=[])
