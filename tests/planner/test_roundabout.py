"""Roundabout (Merry-Go-Round) planner property checks."""
import numpy as np

from uav_nav_lab.planner import PLANNER_REGISTRY


def _rab(**kw):
    cfg = {"max_speed": 5.0, "center": [25.0, 25.0], "ring_radius": 20.0,
           "exit_angle": 0.35, "time_step": 0.05, "goal_radius": 1.5}
    cfg.update(kw)
    return PLANNER_REGISTRY.get("roundabout").from_config(cfg)


def test_registered():
    assert "roundabout" in PLANNER_REGISTRY.names()


def test_inside_goal_radius_stops():
    p = _rab()
    v = p.plan(np.array([5.0, 25.0]), np.array([5.5, 25.0]), None).target_velocity
    np.testing.assert_allclose(v, [0.0, 0.0], atol=1e-9)


def test_orbits_counter_clockwise():
    # On the ring at angle 0 (east of centre) with the goal at the antipode
    # (west), the drone must steer CCW = +y (tangent), not straight at the goal.
    p = _rab()
    pos = np.array([45.0, 25.0])      # centre+ (20,0)
    goal = np.array([5.0, 25.0])      # antipode
    v = p.plan(pos, goal, None).target_velocity
    assert v[1] > 1e-3                # moving +y (CCW), not -x toward goal
    assert abs(float(np.linalg.norm(v)) - 5.0) < 1e-6


def test_exits_to_goal_when_aligned():
    # On the ring at a bearing just CCW-before the goal's bearing (within
    # exit_angle) but well outside goal_radius -> head straight to goal.
    import math
    p = _rab()
    a = 3.0  # goal bearing is pi (~3.14); CCW gap pi-3.0 = 0.14 < exit_angle
    pos = np.array([25.0 + 20.0 * math.cos(a), 25.0 + 20.0 * math.sin(a)])
    goal = np.array([5.0, 25.0])
    assert np.linalg.norm(goal - pos) > 1.5     # outside goal radius
    v = p.plan(pos, goal, None).target_velocity
    assert float(v @ (goal - pos)) > 0.0        # points toward the goal


def test_speed_capped():
    p = _rab()
    v = p.plan(np.array([40.0, 30.0]), np.array([10.0, 20.0]), None).target_velocity
    assert float(np.linalg.norm(v)) <= 5.0 + 1e-6
