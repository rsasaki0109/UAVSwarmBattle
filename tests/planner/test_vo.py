"""VO planner: velocity-obstacle property checks (Fiorini & Shiller 1998)."""
import numpy as np
import pytest

from uav_nav_lab.planner import PLANNER_REGISTRY


def _vo(**kw):
    cfg = {"max_speed": 5.0, "radius": 0.4, "safety_margin": 0.1, "time_horizon": 2.0,
           "neighbor_dist": 15.0, "goal_radius": 1.5, "time_step": 0.05}
    cfg.update(kw)
    p = PLANNER_REGISTRY.get("vo").from_config(cfg)
    p.reset()
    return p


def test_registered():
    assert "vo" in PLANNER_REGISTRY.names()


def test_no_neighbours_heads_to_goal():
    p = _vo()
    v = p.plan(np.array([5.0, 25.0]), np.array([45.0, 25.0]), None, dynamic_obstacles=[]).target_velocity
    np.testing.assert_allclose(v, [5.0, 0.0], atol=1e-6)


def test_inside_goal_radius_stops():
    p = _vo()
    v = p.plan(np.array([45.0, 25.0]), np.array([45.5, 25.0]), None, dynamic_obstacles=[]).target_velocity
    np.testing.assert_allclose(v, [0.0, 0.0], atol=1e-9)


def test_speed_capped():
    p = _vo()
    v = p.plan(np.array([5.0, 25.0]), np.array([45.0, 25.0]), None,
               dynamic_obstacles=[{"position": [12.0, 25.0], "velocity": [-5.0, 0.0], "radius": 0.4}]).target_velocity
    assert float(np.linalg.norm(v)) <= 5.0 + 1e-6


def test_head_on_deflects():
    p = _vo()
    v = p.plan(np.array([5.0, 25.0]), np.array([45.0, 25.0]), None,
               dynamic_obstacles=[{"position": [10.0, 25.0], "velocity": [-5.0, 0.0], "radius": 0.4}]).target_velocity
    assert abs(v[1]) > 1e-3 or v[0] < 5.0 - 1e-3


def test_three_d_rejected():
    p = _vo()
    with pytest.raises(ValueError):
        p.plan(np.zeros(3), np.array([1.0, 0.0, 1.0]), None, dynamic_obstacles=[])
