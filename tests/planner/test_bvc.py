"""BVC planner: Buffered Voronoi Cell property checks (Zhou et al., 2017)."""
import numpy as np
import pytest

from uav_nav_lab.planner import PLANNER_REGISTRY


def _bvc(**kw):
    cfg = {"max_speed": 5.0, "radius": 0.4, "time_step": 0.1,
           "neighbor_dist": 15.0, "goal_radius": 1.5, "safety_margin": 0.1}
    cfg.update(kw)
    p = PLANNER_REGISTRY.get("bvc").from_config(cfg)
    p.reset()
    return p


def test_registered():
    assert "bvc" in PLANNER_REGISTRY.names()


def test_no_neighbors_goes_straight():
    p = _bvc()
    v = p.plan(np.array([5.0, 25.0]), np.array([45.0, 25.0]), None,
               dynamic_obstacles=[]).target_velocity
    np.testing.assert_allclose(v, [5.0, 0.0], atol=1e-6)


def test_inside_goal_radius_stops():
    p = _bvc()
    v = p.plan(np.array([45.0, 25.0]), np.array([45.5, 25.0]), None,
               dynamic_obstacles=[]).target_velocity
    np.testing.assert_allclose(v, [0.0, 0.0], atol=1e-9)


def test_speed_never_exceeds_max():
    p = _bvc()
    dyn = [{"position": [9.0, 25.0], "radius": 0.4}]
    v = p.plan(np.array([5.0, 25.0]), np.array([45.0, 25.0]), None,
               dynamic_obstacles=dyn).target_velocity
    assert float(np.linalg.norm(v)) <= 5.0 + 1e-6


def test_stays_inside_buffered_voronoi_cell():
    """The defining BVC safety property: the commanded next position must stay
    strictly on the ego's side of every buffered bisector — i.e. it can never
    cross into a peer's buffer. Here a peer sits straight ahead on the goal
    line; the ego must stop short of the buffered boundary, not drive through."""
    pos = np.array([5.0, 25.0])
    peer = np.array([6.5, 25.0])  # only 1.5 m ahead
    p = _bvc()
    plan = p.plan(pos, np.array([45.0, 25.0]), None,
                  dynamic_obstacles=[{"position": peer, "radius": 0.4}])
    nxt = pos + plan.target_velocity * 0.1
    a = peer - pos
    r_buf = 0.4 + 0.4 + 0.1
    b = float(a @ (0.5 * (pos + peer))) - r_buf * float(np.hypot(*a))
    assert float(a @ nxt) <= b + 1e-6  # next pos stays inside the buffered cell


def test_pairwise_bias_does_not_tilt_a_lone_agent():
    p = _bvc(pairwise_bias=5.0, pairwise_radius=8.0)
    v = p.plan(np.array([5.0, 25.0]), np.array([45.0, 25.0]), None,
               dynamic_obstacles=[]).target_velocity
    np.testing.assert_allclose(v, [5.0, 0.0], atol=1e-6)


def test_three_d_rejected():
    p = _bvc()
    with pytest.raises(ValueError):
        p.plan(np.zeros(3), np.array([1.0, 0.0, 1.0]), None, dynamic_obstacles=[])
