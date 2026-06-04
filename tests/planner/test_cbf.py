"""CBF planner: control-barrier-function QP safety-filter property checks."""
import numpy as np
import pytest

from uav_nav_lab.planner import PLANNER_REGISTRY


def _cbf(**kw):
    cfg = {"max_speed": 5.0, "radius": 0.4, "time_step": 0.1, "alpha": 2.0,
           "neighbor_dist": 15.0, "goal_radius": 1.5, "safety_margin": 0.1}
    cfg.update(kw)
    p = PLANNER_REGISTRY.get("cbf").from_config(cfg)
    p.reset()
    return p


def test_registered():
    assert "cbf" in PLANNER_REGISTRY.names()


def test_no_neighbors_goes_straight():
    p = _cbf()
    p.set_current_state(np.array([5.0, 25.0]), np.zeros(2))
    v = p.plan(np.array([5.0, 25.0]), np.array([45.0, 25.0]), None,
               dynamic_obstacles=[]).target_velocity
    np.testing.assert_allclose(v, [5.0, 0.0], atol=1e-6)


def test_inside_goal_radius_stops():
    p = _cbf()
    v = p.plan(np.array([45.0, 25.0]), np.array([45.5, 25.0]), None,
               dynamic_obstacles=[]).target_velocity
    np.testing.assert_allclose(v, [0.0, 0.0], atol=1e-9)


def test_speed_never_exceeds_max():
    p = _cbf()
    p.set_current_state(np.array([5.0, 25.0]), np.array([5.0, 0.0]))
    dyn = [{"position": [9.0, 25.5], "velocity": [-5.0, 0.0], "radius": 0.4}]
    v = p.plan(np.array([5.0, 25.0]), np.array([45.0, 25.0]), None,
               dynamic_obstacles=dyn).target_velocity
    assert float(np.linalg.norm(v)) <= 5.0 + 1e-6


def test_cbf_constraint_satisfied():
    """The barrier half-plane (p_j-p_i)·v <= (p_j-p_i)·v_j + (alpha/2)·h must
    hold for the chosen velocity — the agent never commits to a velocity that
    violates the discrete CBF condition for an approaching neighbour."""
    pos = np.array([5.0, 25.0])
    peer = np.array([7.0, 25.0])
    v_peer = np.array([-5.0, 0.0])
    p = _cbf(reciprocal=False)  # full responsibility = strict single-agent CBF
    p.set_current_state(pos, np.array([5.0, 0.0]))
    v = p.plan(pos, np.array([45.0, 25.0]), None,
               dynamic_obstacles=[{"position": peer, "velocity": v_peer,
                                   "radius": 0.4}]).target_velocity
    a = peer - pos
    r_safe = 0.4 + 0.4 + 0.1
    h = float(a @ a) - r_safe ** 2
    b = float(a @ v_peer) + 0.5 * 2.0 * h
    assert float(a @ v) <= b + 1e-6


def test_pairwise_bias_does_not_tilt_a_lone_agent():
    p = _cbf(pairwise_bias=5.0, pairwise_radius=8.0)
    p.set_current_state(np.array([5.0, 25.0]), np.zeros(2))
    v = p.plan(np.array([5.0, 25.0]), np.array([45.0, 25.0]), None,
               dynamic_obstacles=[]).target_velocity
    np.testing.assert_allclose(v, [5.0, 0.0], atol=1e-6)


def test_three_d_rejected():
    p = _cbf()
    p.set_current_state(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError):
        p.plan(np.zeros(3), np.array([1.0, 0.0, 1.0]), None, dynamic_obstacles=[])
