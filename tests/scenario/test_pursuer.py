"""Tests for intelligent (pursuing) dynamic obstacles.

Covers the three motion policies on `_DynamicObstacle`:
  - linear   : constant velocity, ignores drones (regression guard)
  - pursue   : steers toward the nearest drone (lead pursuit)
  - intercept: leads a moving drone via finite-difference velocity estimate
and the `set_targets` plumbing on `GridWorldScenario`.
"""

from __future__ import annotations

import numpy as np

from uav_nav_lab.scenario.grid_world import GridWorldScenario, _DynamicObstacle


def _grid(dynamic, size=(50, 50)):
    cfg = {
        "size": size,
        "start": (1.0, 1.0),
        "goal": (size[0] - 2, size[1] - 2),
        "obstacles": {"type": "none"},
        "dynamic_obstacles": dynamic,
    }
    return GridWorldScenario.from_config(cfg)


def _dist(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def test_linear_ignores_targets():
    """A linear obstacle integrates constant velocity regardless of targets."""
    scn = _grid([{"start": [10.0, 10.0], "velocity": [1.0, 0.0], "reflect": False}])
    dt = 0.1
    for _ in range(10):
        scn.set_targets([np.array([10.0, 10.0])])  # right on top — must be ignored
        scn.advance(dt)
    pos = scn.dynamic_obstacles[0]["position"]
    # pure linear: x = 10 + 1.0 * (10*0.1) = 11.0, y unchanged
    # (approx compare — repeated += 0.1 accumulates float error)
    assert abs(pos[0] - 11.0) < 1e-9
    assert abs(pos[1] - 10.0) < 1e-9


def test_pursue_converges_on_stationary_target():
    """A pursuer monotonically closes distance to a stationary drone."""
    scn = _grid(
        [{"start": [0.0, 0.0], "velocity": [1.0, 0.0], "policy": "pursue",
          "speed": 2.0, "reflect": False}]
    )
    target = np.array([20.0, 15.0])  # distance 25 from origin
    dt = 0.1
    prev = _dist(scn.dynamic_obstacles[0]["position"], target)
    # 200 steps * speed 2.0 * dt 0.1 = 40 units of travel — comfortably > 25.
    for _ in range(200):
        scn.set_targets([target])
        scn.advance(dt)
        cur = _dist(scn.dynamic_obstacles[0]["position"], target)
        # require monotonic approach only while still closing; once it
        # arrives it oscillates within one step (speed*dt) of the target.
        if prev > 1.0:
            assert cur <= prev + 1e-9  # never moves away while far
        prev = cur
    assert prev < 1.0  # actually catches it


def test_pursue_picks_nearest_of_several_targets():
    scn = _grid(
        [{"start": [0.0, 0.0], "policy": "pursue", "speed": 3.0, "reflect": False}]
    )
    far = np.array([40.0, 40.0])
    near = np.array([5.0, 0.0])
    dt = 0.1
    for _ in range(40):
        scn.set_targets([far, near])
        scn.advance(dt)
    pos = scn.dynamic_obstacles[0]["position"]
    assert _dist(pos, near) < _dist(pos, far)


def test_intercept_beats_pursue_on_fleeing_target():
    """Lead pursuit (intercept) catches a fleeing target at least as well as
    naive pursuit, which always aims at the target's current (stale) cell."""
    common = {"start": [0.0, 0.0], "speed": 2.0, "reflect": False}
    scn_p = _grid([{**common, "policy": "pursue"}])
    scn_i = _grid([{**common, "policy": "intercept"}])
    target = np.array([6.0, 0.0])
    tvel = np.array([1.6, 0.6])  # fleeing diagonally, slower than pursuer
    dt = 0.1
    for _ in range(60):
        target = target + tvel * dt
        scn_p.set_targets([target])
        scn_i.set_targets([target])
        scn_p.advance(dt)
        scn_i.advance(dt)
    d_p = _dist(scn_p.dynamic_obstacles[0]["position"], target)
    d_i = _dist(scn_i.dynamic_obstacles[0]["position"], target)
    assert d_i <= d_p + 1e-9


def test_turn_rate_caps_heading_change():
    """With a small turn_rate the heading cannot snap instantly to the target."""
    ob = _DynamicObstacle(
        pos0=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),  # heading +x
        reflect=False,
        policy="pursue",
        speed=1.0,
        turn_rate=0.2,  # rad/s
    )
    ob.reset()
    dt = 0.1
    before = ob.vel.copy()
    ob.step(dt, (50.0, 50.0), targets=[np.array([0.0, 10.0])])  # target straight up
    after = ob.vel
    cos_a = float(
        np.dot(before, after) / (np.linalg.norm(before) * np.linalg.norm(after))
    )
    angle = float(np.arccos(np.clip(cos_a, -1.0, 1.0)))
    assert angle <= 0.2 * dt + 1e-6


def test_pursuer_coasts_without_targets():
    """A pursuer with no targets fed just coasts on its current velocity."""
    scn = _grid(
        [{"start": [5.0, 5.0], "velocity": [1.0, 0.0], "policy": "pursue",
          "speed": 1.0, "reflect": False}]
    )
    dt = 0.1
    scn.advance(dt)  # no set_targets call → coast
    pos = scn.dynamic_obstacles[0]["position"]
    assert abs(pos[0] - (5.0 + 1.0 * dt)) < 1e-9


def test_intercept_target_switch_does_not_fling_aim():
    """When the nearest target switches between steps, the intercept lead must
    not difference two distinct targets' positions into a bogus huge velocity.

    Step 1: target A near the obstacle is the nearest; the obstacle records it.
    Step 2: a far target B suddenly becomes the nearest (A removed). Without the
    target-identity guard, tvel = (B - A)/dt would be a fabricated ~100+ m/s
    velocity and the aim would be flung far past B. With the guard, the step
    falls back to pure pursuit (aim = B), so the obstacle moves a bounded
    distance no larger than its cruise speed * dt toward B.
    """
    ob = _DynamicObstacle(
        pos0=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        reflect=False,
        policy="intercept",
        speed=2.0,
    )
    ob.reset()
    dt = 0.1
    a = np.array([1.0, 0.0])      # near A (nearest on step 1)
    b = np.array([0.0, 30.0])     # far B
    ob.step(dt, (100.0, 100.0), targets=[a, b])  # nearest = A (idx 0)
    pos_before = ob.pos.copy()
    # A vanishes; B is now the only / nearest target → nearest_idx changes.
    ob.step(dt, (100.0, 100.0), targets=[b])
    moved = float(np.linalg.norm(ob.pos - pos_before))
    # pure-pursuit bound: cruise speed (2.0) * dt (0.1) = 0.2 m, plus a little
    # slack. A fling would move many metres.
    assert moved <= 2.0 * dt + 1e-6
