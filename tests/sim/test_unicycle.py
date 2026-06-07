"""Unit tests for the non-holonomic unicycle simulator (dummy_unicycle)."""

from __future__ import annotations

import numpy as np
import pytest

from uav_nav_lab.scenario import SCENARIO_REGISTRY
from uav_nav_lab.sim import SIM_REGISTRY


def _sim(turn_rate_max, *, goal=(40.0, 10.0), start=(10.0, 10.0), max_accel=100.0):
    scn = SCENARIO_REGISTRY.get("grid_world").from_config(
        {"size": [50, 50], "start": list(start), "goal": list(goal),
         "obstacles": {"type": "none"}}
    )
    sim = SIM_REGISTRY.get("dummy_unicycle").from_config(
        {"dt": 0.1, "max_steps": 100, "max_accel": max_accel,
         "turn_rate_max": turn_rate_max}, scn
    )
    sim.reset(seed=0)
    return sim


def test_registered():
    assert SIM_REGISTRY.get("dummy_unicycle") is not None


def test_heading_initialised_toward_goal():
    sim = _sim(3.0, goal=(40.0, 10.0))   # goal due +x of start
    assert abs(sim._heading) < 1e-6


def test_cannot_strafe_sideways():
    """Commanding a velocity perpendicular to the heading must NOT produce
    sideways motion on the first step — the drone turns (rate-limited) and
    keeps moving mostly along its current heading."""
    sim = _sim(0.5, goal=(40.0, 10.0))   # heading +x
    state, _ = sim.step(np.array([0.0, 5.0]))   # command straight +y
    vx, vy = state.velocity
    assert vx > vy            # still mostly forward, not sideways
    assert abs(sim._heading) <= 0.5 * 0.1 + 1e-9   # turned at most turn_rate*dt


def test_high_turn_rate_approaches_holonomic():
    """With a very high turn rate the drone can immediately point at the
    command, so the velocity aligns with the commanded direction."""
    sim = _sim(100.0, goal=(40.0, 10.0))
    state, _ = sim.step(np.array([0.0, 5.0]))   # command straight +y
    vx, vy = state.velocity
    assert vy > abs(vx)       # now points toward the command


def test_forward_speed_accel_limited():
    sim = _sim(100.0, max_accel=10.0)
    state, _ = sim.step(np.array([5.0, 0.0]))
    assert np.linalg.norm(state.velocity) <= 10.0 * 0.1 + 1e-9


def test_three_d_scenario_rejected():
    scn = SCENARIO_REGISTRY.get("voxel_world").from_config(
        {"size": [20, 20, 20], "start": [1.0, 1.0, 1.0], "goal": [10.0, 10.0, 10.0],
         "obstacles": {"type": "none"}}
    )
    with pytest.raises(ValueError):
        SIM_REGISTRY.get("dummy_unicycle").from_config({"dt": 0.05}, scn)
