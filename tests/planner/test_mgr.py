"""Merry-Go-Round planner: triggered, locally-negotiated roundabout property checks.

The roundabout must (a) stay dormant — plain CBF — when there is no deadlock, so
it cannot harm unstructured traffic; (b) engage only when the ego is braked to a
stop with a peer close ahead; (c) tier its ring radius with the cluster size; and
(d) keep the CBF safety guarantee while orbiting.
"""
import numpy as np

from uav_nav_lab.planner import PLANNER_REGISTRY


def _mgr(**kw):
    cfg = {"max_speed": 5.0, "radius": 0.4, "time_step": 0.1, "alpha": 2.0,
           "neighbor_dist": 15.0, "goal_radius": 1.5, "safety_margin": 0.1,
           "detect_dist": 5.0, "stall_frac": 0.5, "ring_min": 4.0,
           "trigger_persist": 1}
    cfg.update(kw)
    p = PLANNER_REGISTRY.get("mgr").from_config(cfg)
    p.reset()
    return p


def test_registered():
    assert "mgr" in PLANNER_REGISTRY.names()


def test_lone_agent_goes_straight():
    """No peers -> never triggers -> plain straight-to-goal at cruise speed."""
    p = _mgr()
    p.set_current_state(np.array([5.0, 25.0]), np.zeros(2))
    v = p.plan(np.array([5.0, 25.0]), np.array([45.0, 25.0]), None,
               dynamic_obstacles=[]).target_velocity
    np.testing.assert_allclose(v, [5.0, 0.0], atol=1e-6)
    assert p._mode == "free"


def test_moving_with_peer_ahead_does_not_trigger():
    """Cruising at full speed (not stalled) must NOT engage the roundabout, even
    with a peer ahead — only a genuine stall is a deadlock."""
    p = _mgr()
    pos = np.array([5.0, 25.0])
    p.set_current_state(pos, np.array([5.0, 0.0]))  # full speed toward goal
    dyn = [{"position": [8.0, 25.0], "velocity": [0.0, 0.0], "radius": 0.4}]
    p.plan(pos, np.array([45.0, 25.0]), None, dynamic_obstacles=dyn)
    assert p._mode == "free"


def test_stalled_no_peer_ahead_does_not_trigger():
    """A stalled agent with the only peer BEHIND it is not deadlocked."""
    p = _mgr()
    pos = np.array([25.0, 25.0])
    p.set_current_state(pos, np.zeros(2))  # stalled
    dyn = [{"position": [22.0, 25.0], "velocity": [0.0, 0.0], "radius": 0.4}]  # behind
    p.plan(pos, np.array([45.0, 25.0]), None, dynamic_obstacles=dyn)
    assert p._mode == "free"


def test_deadlock_triggers_orbit():
    """Stalled with a peer close ahead -> engage the roundabout (orbit mode), and
    pick a ring centre near the ego/peer midpoint."""
    p = _mgr()
    pos = np.array([25.0, 25.0])
    p.set_current_state(pos, np.zeros(2))  # braked to a stop
    dyn = [{"position": [26.5, 25.0], "velocity": [0.0, 0.0], "radius": 0.4}]  # ahead, close
    p.plan(pos, np.array([45.0, 25.0]), None, dynamic_obstacles=dyn)
    assert p._mode == "orbit"
    assert p._center is not None
    # centre is the centroid of {ego, peer}
    np.testing.assert_allclose(p._center, [25.75, 25.0], atol=1e-6)


def test_capacity_radius_grows_with_cluster():
    """A bigger conflict cluster must demand a bigger ring (overflow-to-radius)."""
    p = _mgr(ring_min=1.0, ring_gap=2.0)
    p._cluster_n = 3
    small = p._ring_radius()
    p._cluster_n = 40
    big = p._ring_radius()
    assert big > small
    # below capacity the floor holds
    p._cluster_n = 1
    assert p._ring_radius() == 1.0


def test_orbit_velocity_is_tangential():
    """While orbiting, the nominal points along the CCW tangent, not at the goal."""
    p = _mgr()
    p._mode = "orbit"
    p._center = np.array([26.0, 25.0])
    pos = np.array([22.0, 25.0])  # to the left of centre
    v = p._orbit_velocity(pos, np.array([45.0, 25.0]))
    # ego is at angle 180deg from centre; CCW tangent there points -y (downward)
    assert v[1] < -1.0
    assert abs(float(np.linalg.norm(v)) - 5.0) < 1e-6


def test_speed_never_exceeds_max_while_orbiting():
    p = _mgr()
    pos = np.array([25.0, 25.0])
    p.set_current_state(pos, np.zeros(2))
    dyn = [{"position": [26.5, 25.0], "velocity": [0.0, 0.0], "radius": 0.4}]
    v = p.plan(pos, np.array([45.0, 25.0]), None, dynamic_obstacles=dyn).target_velocity
    assert float(np.linalg.norm(v)) <= 5.0 + 1e-6
    assert p._mode == "orbit"


def test_debounce_requires_persistent_stall():
    """With trigger_persist>1 a single stalled frame must NOT engage; only a
    sustained stall does (filters transient stalls in dense traffic)."""
    p = _mgr(trigger_persist=3)
    pos = np.array([25.0, 25.0])
    goal = np.array([45.0, 25.0])
    dyn = [{"position": [26.5, 25.0], "velocity": [0.0, 0.0], "radius": 0.4}]
    p.set_current_state(pos, np.zeros(2))
    p.plan(pos, goal, None, dynamic_obstacles=dyn)
    assert p._mode == "free"  # 1 stall < 3
    p.plan(pos, goal, None, dynamic_obstacles=dyn)
    assert p._mode == "free"  # 2 stalls < 3
    p.plan(pos, goal, None, dynamic_obstacles=dyn)
    assert p._mode == "orbit"  # 3rd sustained stall engages


def test_transient_stall_resets_debounce():
    """A stall that clears resets the counter — it cannot accumulate across a
    gap, so only a CONTINUOUS stall trips the roundabout."""
    p = _mgr(trigger_persist=3)
    pos = np.array([25.0, 25.0])
    goal = np.array([45.0, 25.0])
    stalled = [{"position": [26.5, 25.0], "velocity": [0.0, 0.0], "radius": 0.4}]
    p.set_current_state(pos, np.zeros(2))
    p.plan(pos, goal, None, dynamic_obstacles=stalled)
    p.plan(pos, goal, None, dynamic_obstacles=stalled)
    # moving again (not stalled) clears the counter
    p.set_current_state(pos, np.array([5.0, 0.0]))
    p.plan(pos, goal, None, dynamic_obstacles=stalled)
    assert p._stall_count == 0 and p._mode == "free"


def test_reset_clears_mode():
    p = _mgr()
    p._mode = "orbit"
    p._center = np.array([1.0, 2.0])
    p.reset()
    assert p._mode == "free"
    assert p._center is None
