"""Constant-turn-rate (CTR) predictor tests.

Contract:
  - a straight-moving obstacle (no rotation between calls) forecasts exactly as
    constant-velocity — CTR is a no-op on non-curving traffic;
  - an obstacle whose velocity rotates at a steady ω is rolled forward along the
    matching circular arc;
  - ω is estimated across calls via nearest-neighbour association (stateful),
    and a freshly seen obstacle forecasts as constant-velocity for one step;
  - reset() clears the rotation history;
  - ndim != 2 falls back to constant velocity;
  - registry wiring works.
"""

from __future__ import annotations

import numpy as np

from uav_nav_lab.predictor import PREDICTOR_REGISTRY, build_predictor
from uav_nav_lab.predictor.constant_turn import ConstantTurnPredictor


def _cv(p0, v, dts):
    p0, v, dts = np.asarray(p0), np.asarray(v), np.asarray(dts)
    return p0[None, :] + dts[:, None] * v[None, :]


def test_registered():
    assert "constant_turn" in PREDICTOR_REGISTRY.names()
    assert isinstance(build_predictor({"type": "constant_turn"}),
                      ConstantTurnPredictor)


def test_first_sighting_is_constant_velocity():
    p = ConstantTurnPredictor(dt=0.1)
    dyn = [{"position": [0.0, 0.0], "velocity": [1.0, 0.0]}]
    dts = np.array([1.0, 2.0, 3.0])
    out = p.predict(dyn, dts)
    assert out.shape == (1, 3, 2)
    assert np.allclose(out[0], _cv([0.0, 0.0], [1.0, 0.0], dts))


def test_straight_line_stays_constant_velocity():
    """Two calls with an unrotated velocity → ω estimate is 0 → CV forecast."""
    p = ConstantTurnPredictor(dt=0.1)
    dts = np.array([1.0, 2.0])
    p.predict([{"position": [0.0, 0.0], "velocity": [1.0, 0.0]}], dts)
    out = p.predict([{"position": [0.1, 0.0], "velocity": [1.0, 0.0]}], dts)
    assert np.allclose(out[0], _cv([0.1, 0.0], [1.0, 0.0], dts))


def test_steady_rotation_rolls_along_arc():
    """Velocity rotates +0.2 rad over dt=0.1 → ω=2 rad/s; the next forecast must
    trace the analytic constant-turn arc, not a straight line."""
    dt = 0.1
    omega = 2.0
    speed = 1.0
    p = ConstantTurnPredictor(dt=dt)
    th_prev = 0.0
    th_cur = omega * dt  # velocity heading advanced by ω·dt
    v_prev = speed * np.array([np.cos(th_prev), np.sin(th_prev)])
    v_cur = speed * np.array([np.cos(th_cur), np.sin(th_cur)])
    p.predict([{"position": [0.0, 0.0], "velocity": v_prev}], np.array([dt]))
    p0 = np.array([0.3, 0.05])
    dts = np.array([0.5, 1.0, 1.5])
    out = p.predict([{"position": p0, "velocity": v_cur}], dts)
    # analytic arc from p0 with heading th_cur and turn rate ω
    exp = np.empty((len(dts), 2))
    for i, t in enumerate(dts):
        exp[i, 0] = p0[0] + (speed / omega) * (np.sin(th_cur + omega * t) - np.sin(th_cur))
        exp[i, 1] = p0[1] - (speed / omega) * (np.cos(th_cur + omega * t) - np.cos(th_cur))
    assert np.allclose(out[0], exp)
    # and it must differ from the straight-line CV forecast
    assert not np.allclose(out[0], _cv(p0, v_cur, dts), atol=1e-3)


def test_reset_clears_history():
    p = ConstantTurnPredictor(dt=0.1)
    th = 0.2
    v0 = np.array([1.0, 0.0])
    v1 = np.array([np.cos(th), np.sin(th)])
    p.predict([{"position": [0.0, 0.0], "velocity": v0}], np.array([0.1]))
    p.reset()
    # after reset the next call is a first sighting again → CV
    dts = np.array([1.0, 2.0])
    out = p.predict([{"position": [0.0, 0.0], "velocity": v1}], dts)
    assert np.allclose(out[0], _cv([0.0, 0.0], v1, dts))


def test_unassociated_jump_is_constant_velocity():
    """If the obstacle teleports past the association gate, it is treated as a
    new track (no rotation history) → CV, not a spurious arc."""
    p = ConstantTurnPredictor(dt=0.1, association_threshold=1.0)
    p.predict([{"position": [0.0, 0.0], "velocity": [1.0, 0.0]}], np.array([0.1]))
    dts = np.array([1.0, 2.0])
    far = [{"position": [50.0, 50.0], "velocity": [0.0, 1.0]}]  # well past the gate
    out = p.predict(far, dts)
    assert np.allclose(out[0], _cv([50.0, 50.0], [0.0, 1.0], dts))


def test_max_turn_rate_clamps():
    p = ConstantTurnPredictor(dt=0.1, max_turn_rate=1.0)
    # rotate velocity by ~1.5 rad over dt=0.1 → raw ω=15, clamped to 1.0
    v0 = np.array([1.0, 0.0])
    v1 = np.array([np.cos(1.5), np.sin(1.5)])
    p.predict([{"position": [0.0, 0.0], "velocity": v0}], np.array([0.1]))
    dts = np.array([1.0])
    out = p.predict([{"position": [0.0, 0.0], "velocity": v1}], dts)
    exp = np.empty((1, 2))
    om, th0, s = 1.0, float(np.arctan2(v1[1], v1[0])), 1.0
    exp[0, 0] = (s / om) * (np.sin(th0 + om) - np.sin(th0))
    exp[0, 1] = -(s / om) * (np.cos(th0 + om) - np.cos(th0))
    assert np.allclose(out[0], exp)


def test_3d_falls_back_to_constant_velocity():
    p = ConstantTurnPredictor(dt=0.1)
    dyn = [{"position": [0.0, 0.0, 0.0], "velocity": [1.0, 1.0, 1.0]}]
    dts = np.array([1.0, 2.0])
    out = p.predict(dyn, dts)
    assert out.shape == (1, 2, 3)
    assert np.allclose(out[0], _cv([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], dts))


def test_empty_obstacles():
    p = ConstantTurnPredictor()
    out = p.predict([], np.array([1.0, 2.0]))
    assert out.shape == (0, 2, 0)
