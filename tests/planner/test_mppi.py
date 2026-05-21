"""MPPI planner unit tests."""

from __future__ import annotations

import json  # noqa: F401
import subprocess  # noqa: F401
import sys  # noqa: F401
from pathlib import Path  # noqa: F401

import numpy as np  # noqa: F401
import pytest  # noqa: F401

from uav_nav_lab.cli import build_parser, main  # noqa: F401
from uav_nav_lab.config import ExperimentConfig  # noqa: F401
from uav_nav_lab.eval import evaluate_run  # noqa: F401
from uav_nav_lab.planner import PLANNER_REGISTRY  # noqa: F401
from uav_nav_lab.runner import expand_sweep, run_experiment  # noqa: F401

from tests._helpers import EXAMPLES, _basic_cfg, _require_mplot3d  # noqa: F401


def test_sample_unit_directions_3d() -> None:
    from uav_nav_lab.planner._grid import sample_unit_directions

    base = np.array([1.0, 0.0, 0.0])
    dirs = sample_unit_directions(3, 16, base)
    assert dirs.shape == (16, 3)
    norms = np.linalg.norm(dirs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    # first sample is the goal direction
    assert np.allclose(dirs[0], base)


def test_mppi_low_temperature_collapses_to_argmin() -> None:
    """As temperature → 0 the softmax weights become a one-hot at the
    argmin sample, so MPPI's chosen action should converge to MPC's. We
    verify with both planners on the same scenario at temperature=0.001 —
    the goal-direction sample dominates because the reach-goal bonus
    (-1e6) gives it a cost ~1e6 lower than any other."""
    from uav_nav_lab.planner.mpc import SamplingMPCPlanner
    from uav_nav_lab.planner.mppi import MPPIPlanner

    occ = np.zeros((30, 30), dtype=bool)
    obs = np.array([2.0, 2.0])
    goal = np.array([20.0, 20.0])

    args = dict(max_speed=5.0, horizon=20, n_samples=16, inflate=0)
    mpc = SamplingMPCPlanner(**args)
    mpc.reset()
    mpc_action = mpc.plan(obs, goal, occ).target_velocity

    mppi = MPPIPlanner(temperature=0.001, **args)
    mppi.reset()
    mppi_action = mppi.plan(obs, goal, occ).target_velocity

    # Both should pick the goal-direction action at ~max_speed.
    assert np.allclose(mppi_action, mpc_action, atol=0.1)


def test_mppi_high_temperature_attenuates_speed() -> None:
    """As temperature → ∞ the softmax weights flatten to uniform, so the
    chosen action converges to the *mean* of all sample directions. For
    an n=16 set of evenly-spread 2D directions the mean magnitude collapses
    toward zero — this is MPPI's signature failure mode (and why the
    framework's default temperature=10 sits in a deliberate middle range)."""
    from uav_nav_lab.planner.mppi import MPPIPlanner

    occ = np.zeros((30, 30), dtype=bool)
    obs = np.array([2.0, 2.0])
    goal = np.array([20.0, 20.0])

    low = MPPIPlanner(max_speed=5.0, horizon=20, n_samples=16, temperature=0.1, inflate=0)
    low.reset()
    speed_low = float(np.linalg.norm(low.plan(obs, goal, occ).target_velocity))

    high = MPPIPlanner(max_speed=5.0, horizon=20, n_samples=16, temperature=1e6, inflate=0)
    high.reset()
    speed_high = float(np.linalg.norm(high.plan(obs, goal, occ).target_velocity))

    # Low temp keeps the chosen sample's full speed; high temp dilutes it.
    assert speed_low > speed_high + 1.0


def test_mppi_invalid_temperature_raises() -> None:
    """Zero or negative temperature would make the softmax explode (divide
    by zero in the exp argument). Construction must reject it loudly."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    with pytest.raises(ValueError, match="temperature must be"):
        PLANNER_REGISTRY.get("mppi").from_config({"temperature": 0.0})
    with pytest.raises(ValueError, match="temperature must be"):
        PLANNER_REGISTRY.get("mppi").from_config({"temperature": -1.0})


def test_mppi_meta_carries_softmax_diagnostics() -> None:
    """weight_max ∈ [1/n, 1] and weight_entropy ∈ [0, log(n)] tell the
    user how concentrated MPPI's selection was. These appear in the plan
    metadata for downstream eval / sweep filtering."""
    from uav_nav_lab.planner import PLANNER_REGISTRY
    import math

    p = PLANNER_REGISTRY.get("mppi").from_config(
        {"horizon": 20, "n_samples": 16, "max_speed": 5.0, "temperature": 5.0}
    )
    occ = np.zeros((30, 30), dtype=bool)
    plan = p.plan(np.array([2.0, 2.0]), np.array([20.0, 20.0]), occ)
    assert plan.meta["planner"] == "mppi"
    assert 1.0 / 16 <= plan.meta["weight_max"] <= 1.0 + 1e-6
    assert 0.0 <= plan.meta["weight_entropy"] <= math.log(16) + 1e-6
