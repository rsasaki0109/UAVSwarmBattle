"""ORCA planner: algorithm-property checks (van den Berg 2011 / RVO2)."""
import numpy as np
import pytest

from uav_nav_lab.planner import PLANNER_REGISTRY
from uav_nav_lab.planner.orca import ORCAPlanner


def _orca(**kw):
    cfg = {"max_speed": 5.0, "radius": 0.4, "time_horizon": 2.0,
           "neighbor_dist": 15.0, "goal_radius": 1.5}
    cfg.update(kw)
    p = PLANNER_REGISTRY.get("orca").from_config(cfg)
    p.reset()
    return p


def test_registered():
    assert "orca" in PLANNER_REGISTRY.names()
    assert isinstance(PLANNER_REGISTRY.get("orca").from_config({}), ORCAPlanner)


def test_no_neighbors_returns_preferred_velocity():
    p = _orca()
    p.set_current_state(np.zeros(2), np.zeros(2))
    plan = p.plan(np.zeros(2), np.array([10.0, 0.0]), None, dynamic_obstacles=[])
    # straight to goal at max_speed
    np.testing.assert_allclose(plan.target_velocity, [5.0, 0.0], atol=1e-6)


def test_inside_goal_radius_stops():
    p = _orca()
    p.set_current_state(np.zeros(2), np.zeros(2))
    plan = p.plan(np.zeros(2), np.array([1.0, 0.0]), None, dynamic_obstacles=[])
    np.testing.assert_allclose(plan.target_velocity, [0.0, 0.0], atol=1e-9)


def test_speed_never_exceeds_max():
    p = _orca()
    p.set_current_state(np.zeros(2), np.array([5.0, 0.0]))
    # a cluster of neighbours on a collision course
    dyn = [{"position": [4.0, dy], "velocity": [-5.0, 0.0], "radius": 0.4}
           for dy in (-0.3, 0.0, 0.3)]
    plan = p.plan(np.zeros(2), np.array([20.0, 0.0]), None, dynamic_obstacles=dyn)
    assert np.linalg.norm(plan.target_velocity) <= 5.0 + 1e-6


def test_head_on_neighbor_induces_avoidance():
    # Peer dead ahead, closing head-on. ORCA must deflect off the pure
    # straight-to-goal preferred velocity (non-zero lateral component or slow-down).
    p = _orca()
    p.set_current_state(np.zeros(2), np.array([5.0, 0.0]))
    dyn = [{"position": [3.0, 0.0], "velocity": [-5.0, 0.0], "radius": 0.4}]
    plan = p.plan(np.zeros(2), np.array([20.0, 0.0]), None, dynamic_obstacles=dyn)
    v = plan.target_velocity
    # avoidance = the velocity is no longer the clean (5,0) preferred velocity
    assert not np.allclose(v, [5.0, 0.0], atol=1e-3)
    assert plan.meta["n_lines"] == 1


def test_offset_neighbor_deflects_laterally():
    # Peer slightly off to the +y side on a crossing course -> ego should gain
    # a -y (away) component.
    p = _orca()
    p.set_current_state(np.zeros(2), np.array([5.0, 0.0]))
    dyn = [{"position": [3.0, 0.5], "velocity": [-5.0, 0.0], "radius": 0.4}]
    plan = p.plan(np.zeros(2), np.array([20.0, 0.0]), None, dynamic_obstacles=dyn)
    assert plan.target_velocity[1] < -1e-3  # veers away from the +y peer


def test_distant_neighbor_ignored():
    p = _orca(neighbor_dist=5.0)
    p.set_current_state(np.zeros(2), np.array([5.0, 0.0]))
    dyn = [{"position": [50.0, 0.0], "velocity": [-5.0, 0.0], "radius": 0.4}]
    plan = p.plan(np.zeros(2), np.array([100.0, 0.0]), None, dynamic_obstacles=dyn)
    assert plan.meta["n_lines"] == 0
    np.testing.assert_allclose(plan.target_velocity, [5.0, 0.0], atol=1e-6)


def test_lateral_bias_tilts_preferred_velocity_right():
    # Heading +x with no neighbours: a right-of-way bias must tilt the
    # preferred velocity to the ego's right = -y, at full cruise speed.
    p = _orca(lateral_bias=0.3)
    p.set_current_state(np.zeros(2), np.zeros(2))
    plan = p.plan(np.zeros(2), np.array([10.0, 0.0]), None, dynamic_obstacles=[])
    v = plan.target_velocity
    assert v[1] < -1e-3            # veers right (-y)
    assert v[0] > 0.0              # still makes forward progress
    np.testing.assert_allclose(np.linalg.norm(v), 5.0, atol=1e-6)  # speed preserved


def test_zero_lateral_bias_is_stock_orca():
    p = _orca(lateral_bias=0.0)
    p.set_current_state(np.zeros(2), np.zeros(2))
    plan = p.plan(np.zeros(2), np.array([10.0, 0.0]), None, dynamic_obstacles=[])
    np.testing.assert_allclose(plan.target_velocity, [5.0, 0.0], atol=1e-6)


def test_three_d_rejected():
    p = _orca()
    p.set_current_state(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError):
        p.plan(np.zeros(3), np.array([1.0, 0.0, 1.0]), None, dynamic_obstacles=[])


def test_pairwise_bias_does_not_tilt_a_lone_agent():
    # The defining contrast with the GLOBAL lateral_bias: with no neighbour the
    # pairwise tilt vanishes, so a lone agent flies straight to goal (this is
    # why pairwise cannot over-rotate into an orbit). Compare with
    # test_lateral_bias_tilts_preferred_velocity_right, where a lone agent IS
    # tilted by the global rule.
    p = _orca(pairwise_bias=10.0, pairwise_radius=8.0)
    p.set_current_state(np.zeros(2), np.zeros(2))
    plan = p.plan(np.zeros(2), np.array([10.0, 0.0]), None, dynamic_obstacles=[])
    np.testing.assert_allclose(plan.target_velocity, [5.0, 0.0], atol=1e-6)


def test_pairwise_bias_tilts_toward_neighbour_side():
    # With a neighbour ahead, the pairwise tilt veers to the ego's right of the
    # bearing to it (-y here), at cruise speed.
    p = _orca(pairwise_bias=10.0, pairwise_radius=8.0)
    p.set_current_state(np.zeros(2), np.array([5.0, 0.0]))
    plan = p.plan(np.zeros(2), np.array([10.0, 0.0]), None,
                  dynamic_obstacles=[{"position": [6.0, 0.0],
                                      "velocity": [-5.0, 0.0], "radius": 0.4}])
    v = plan.target_velocity
    assert v[1] < -1e-3
    assert np.linalg.norm(v) <= 5.0 + 1e-6


def test_zero_pairwise_bias_is_stock_orca():
    p = _orca(pairwise_bias=0.0)
    p.set_current_state(np.zeros(2), np.zeros(2))
    plan = p.plan(np.zeros(2), np.array([10.0, 0.0]), None,
                  dynamic_obstacles=[{"position": [6.0, 0.0],
                                      "velocity": [-5.0, 0.0], "radius": 0.4}])
    stock = _orca(pairwise_bias=0.0)
    stock.set_current_state(np.zeros(2), np.zeros(2))
    plan2 = stock.plan(np.zeros(2), np.array([10.0, 0.0]), None,
                       dynamic_obstacles=[{"position": [6.0, 0.0],
                                           "velocity": [-5.0, 0.0], "radius": 0.4}])
    np.testing.assert_allclose(plan.target_velocity, plan2.target_velocity, atol=1e-9)
