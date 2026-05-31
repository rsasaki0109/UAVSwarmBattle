"""Game-theoretic (best-response) predictor tests.

The predictor's contract:
  - a peer carrying a `goal` is predicted to head straight for that goal at
    constant speed (current speed, or configured peer_speed), clamped so it
    does not overshoot;
  - an obstacle without a `goal` falls back to constant-velocity;
  - registry wiring works.
"""

from __future__ import annotations

import numpy as np

from uav_nav_lab.predictor import PREDICTOR_REGISTRY, build_predictor
from uav_nav_lab.predictor.game_theoretic import GameTheoreticPredictor


def test_registered():
    assert "game_theoretic" in PREDICTOR_REGISTRY.names()
    assert isinstance(build_predictor({"type": "game_theoretic"}),
                      GameTheoreticPredictor)


def test_no_goal_falls_back_to_constant_velocity():
    p = GameTheoreticPredictor()
    dyn = [{"position": [0.0, 0.0], "velocity": [1.0, 0.0]}]  # no goal
    dts = np.array([1.0, 2.0, 3.0])
    out = p.predict(dyn, dts)
    assert out.shape == (1, 3, 2)
    # straight line along +x at speed 1
    assert np.allclose(out[0, :, 0], [1.0, 2.0, 3.0])
    assert np.allclose(out[0, :, 1], 0.0)


def test_goal_redirects_toward_goal_not_velocity():
    """Peer moving +x but with a goal due north should be predicted heading
    north, not continuing east."""
    p = GameTheoreticPredictor()
    dyn = [{"position": [0.0, 0.0], "velocity": [2.0, 0.0],  # heading east
            "goal": [0.0, 10.0]}]                            # but goal is north
    dts = np.array([1.0, 2.0])
    out = p.predict(dyn, dts)
    # speed = |velocity| = 2; direction = +y → positions (0,2), (0,4)
    assert np.allclose(out[0, 0], [0.0, 2.0])
    assert np.allclose(out[0, 1], [0.0, 4.0])


def test_goal_does_not_overshoot():
    p = GameTheoreticPredictor()
    dyn = [{"position": [0.0, 0.0], "velocity": [5.0, 0.0], "goal": [3.0, 0.0]}]
    dts = np.array([1.0, 2.0, 10.0])  # would travel 5, 10, 50 but goal is at 3
    out = p.predict(dyn, dts)
    # clamps to the goal distance; never past x=3
    assert out[0, 0, 0] == 3.0  # 5*1 clamped to 3
    assert out[0, 2, 0] == 3.0  # holds at goal
    assert np.all(out[0, :, 0] <= 3.0 + 1e-9)


def test_peer_speed_overrides_current_speed():
    p = GameTheoreticPredictor(peer_speed=1.0)
    dyn = [{"position": [0.0, 0.0], "velocity": [9.0, 0.0], "goal": [100.0, 0.0]}]
    dts = np.array([1.0, 2.0])
    out = p.predict(dyn, dts)
    # uses peer_speed=1, not the observed 9
    assert np.allclose(out[0, :, 0], [1.0, 2.0])


def test_stationary_peer_with_goal_uses_no_speed_is_ballistic():
    """A peer at rest (zero velocity) and no configured speed has undefined
    cruise speed → falls back to constant velocity (stays put)."""
    p = GameTheoreticPredictor()
    dyn = [{"position": [1.0, 1.0], "velocity": [0.0, 0.0], "goal": [9.0, 9.0]}]
    dts = np.array([1.0, 5.0])
    out = p.predict(dyn, dts)
    assert np.allclose(out[0, 0], [1.0, 1.0])
    assert np.allclose(out[0, 1], [1.0, 1.0])


def test_mixed_goal_and_ballistic_obstacles():
    p = GameTheoreticPredictor()
    dyn = [
        {"position": [0.0, 0.0], "velocity": [1.0, 0.0], "goal": [0.0, 5.0]},
        {"position": [10.0, 0.0], "velocity": [-1.0, 0.0]},  # ballistic
    ]
    dts = np.array([1.0])
    out = p.predict(dyn, dts)
    assert np.allclose(out[0, 0], [0.0, 1.0])   # redirected north
    assert np.allclose(out[1, 0], [9.0, 0.0])   # ballistic west
