"""MPC-CHOMP hybrid planner unit tests."""

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


def test_mpc_chomp_smooths_mpc_rollout_corners(
    planner_registry, empty_grid_20
) -> None:
    """The MPC rollout is a piecewise-straight constant-velocity prediction;
    CHOMP smoothing should reduce the trajectory's second-difference norm
    versus the raw MPC waypoints (acceleration profile), without breaking
    the start position. We use an obstacle-free world so the only force on
    the optimiser is the smoothness term — the smoothed path stays on the
    same direct route as the raw MPC rollout."""
    obs = np.array([2.0, 2.0])
    goal = np.array([18.0, 18.0])
    occ = empty_grid_20

    mpc = planner_registry.get("mpc").from_config(
        {"horizon": 30, "dt_plan": 0.05, "n_samples": 16, "max_speed": 5.0}
    )
    raw = mpc.plan(obs, goal, occ)

    hybrid = planner_registry.get("mpc_chomp").from_config(
        {
            "max_speed": 5.0,
            "n_smooth_iters": 30,
            "w_smooth": 1.0,
            "w_obs": 0.0,
            "mpc": {
                "horizon": 30,
                "dt_plan": 0.05,
                "n_samples": 16,
                "max_speed": 5.0,
            },
        }
    )
    smoothed = hybrid.plan(obs, goal, occ)

    # target_velocity must be cleared so the runner pure-pursues the
    # smoothed waypoints instead of the constant rollout velocity.
    assert smoothed.target_velocity is None
    assert smoothed.meta["smoothed"] is True
    # Same horizon-length output as the raw MPC rollout.
    assert smoothed.waypoints.shape == raw.waypoints.shape
    # Smoothness improved (lower second-difference norm).
    raw_acc = float(np.linalg.norm(np.diff(raw.waypoints, n=2, axis=0)))
    sm_acc = float(np.linalg.norm(np.diff(smoothed.waypoints, n=2, axis=0)))
    assert sm_acc <= raw_acc + 1e-6


def test_mpc_chomp_clears_target_velocity_for_pure_pursuit(
    planner_registry, empty_grid_20
) -> None:
    """Even when the underlying MPC sets target_velocity (its normal mode),
    the wrapper must clear it so the runner falls back to pure-pursuit on
    the smoothed waypoints. Otherwise the smoothing is purely cosmetic."""
    occ = empty_grid_20
    raw = planner_registry.get("mpc").from_config(
        {"horizon": 20, "dt_plan": 0.05, "n_samples": 8, "max_speed": 5.0}
    ).plan(np.array([2.0, 2.0]), np.array([18.0, 18.0]), occ)
    # Sanity: raw MPC does set target_velocity.
    assert raw.target_velocity is not None

    hybrid = planner_registry.get("mpc_chomp").from_config(
        {
            "max_speed": 5.0,
            "n_smooth_iters": 5,
            "mpc": {"horizon": 20, "n_samples": 8, "max_speed": 5.0},
        }
    )
    plan = hybrid.plan(np.array([2.0, 2.0]), np.array([18.0, 18.0]), occ)
    assert plan.target_velocity is None


def test_mpc_chomp_short_rollout_passthrough(planner_registry, empty_grid_20) -> None:
    """When the MPC rollout has fewer than 3 waypoints there's nothing to
    smooth — wrapper should pass the plan through unchanged (preserving
    target_velocity) so the goal-reach behaviour is identical to plain MPC.
    Regression guard against trying to invert a 1×1 interior Hessian."""
    occ = empty_grid_20
    # horizon=2 ⇒ MPC returns at most 2 waypoints (rollout[1:] of length 2),
    # which is below the wrapper's smoothing threshold.
    hybrid = planner_registry.get("mpc_chomp").from_config(
        {
            "max_speed": 5.0,
            "mpc": {
                "horizon": 2,
                "dt_plan": 0.05,
                "n_samples": 8,
                "max_speed": 5.0,
            },
        }
    )
    plan = hybrid.plan(np.array([1.0, 1.0]), np.array([18.0, 18.0]), occ)
    assert plan.waypoints.shape[0] == 2
    assert plan.meta["smoothed"] is False
    # Pass-through must preserve MPC's target_velocity (no pure-pursuit
    # fallback when there's nothing to smooth).
    assert plan.target_velocity is not None


def test_mpc_chomp_registry_and_from_config_round_trip(planner_registry) -> None:
    """Registration + from_config wiring smoke test."""
    cls = planner_registry.get("mpc_chomp")
    p = cls.from_config(
        {
            "max_speed": 7.0,
            "n_smooth_iters": 5,
            "w_obs": 3.0,
            "mpc": {"horizon": 10, "n_samples": 4, "max_speed": 7.0},
        }
    )
    assert p.max_speed == 7.0
    assert p.n_smooth_iters == 5
    assert p.w_obs == 3.0
    assert p.output == "waypoints"  # default


def test_mpc_chomp_velocity_profile_output_shape_and_dt(
    planner_registry, empty_grid_20
) -> None:
    """`output: velocity_profile` must emit a (T, ndim) profile aligned to
    the MPC's dt_plan grid: one velocity per smoothed waypoint, each
    representing the displacement traveled in one dt_plan tick. T equals
    the smoothed waypoint count (forward differences from the obs+wps stack
    yield exactly len(waypoints) velocities)."""
    occ = empty_grid_20
    p = planner_registry.get("mpc_chomp").from_config(
        {
            "max_speed": 5.0,
            "n_smooth_iters": 5,
            "output": "velocity_profile",
            "mpc": {"horizon": 20, "dt_plan": 0.05, "n_samples": 8, "max_speed": 5.0},
        }
    )
    plan = p.plan(np.array([1.0, 1.0]), np.array([18.0, 18.0]), occ)
    assert plan.velocity_profile is not None
    assert plan.profile_dt == pytest.approx(0.05)
    assert plan.velocity_profile.shape == (plan.waypoints.shape[0], 2)
    # Must clear target_velocity so the runner picks the profile path.
    assert plan.target_velocity is None
    # Profile speeds within the planner's max_speed (no negative-step glitch).
    speeds = np.linalg.norm(plan.velocity_profile, axis=1)
    assert float(speeds.max()) <= 5.0 + 1e-6


def test_mpc_chomp_velocity_profile_invalid_output_raises(planner_registry) -> None:
    """Typo in `output` must fail loud at construction, mirroring the
    `init` validation in the inner CHOMP planner."""
    with pytest.raises(ValueError, match="output must be"):
        planner_registry.get("mpc_chomp").from_config(
            {"output": "spline", "mpc": {"horizon": 10, "n_samples": 4}}
        )


def test_follow_plan_velocity_profile_indexed_by_elapsed_time() -> None:
    """The runner's `_follow_plan` must pick the profile bin that matches
    `t_since_replan / profile_dt` and clip past the end. This is the
    machinery that lets a smoothing planner drive a varying velocity
    instead of one constant target_velocity."""
    from uav_nav_lab.planner.base import Plan
    from uav_nav_lab.runner.experiment import _follow_plan

    profile = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    plan = Plan(
        waypoints=np.zeros((3, 2)),
        velocity_profile=profile,
        profile_dt=0.1,
    )
    obs = np.array([0.0, 0.0])
    # t=0 → bin 0
    assert np.allclose(_follow_plan(plan, obs, max_speed=5.0, t_since_replan=0.0),
                       [1.0, 0.0])
    # t=0.15 → bin 1 (floor)
    assert np.allclose(_follow_plan(plan, obs, max_speed=5.0, t_since_replan=0.15),
                       [0.0, 1.0])
    # t past end → clip to last bin
    assert np.allclose(_follow_plan(plan, obs, max_speed=5.0, t_since_replan=10.0),
                       [-1.0, 0.0])
    # max_speed cap still applies (profile entry is unit norm; cap=0.5 halves it)
    assert np.allclose(_follow_plan(plan, obs, max_speed=0.5, t_since_replan=0.0),
                       [0.5, 0.0])


def test_mpc_chomp_w_action_jump_meta_and_round_trip(
    planner_registry, empty_grid_20
) -> None:
    """w_action_jump round-trips through from_config, surfaces in meta on
    velocity_profile emit, and is accepted as a non-zero default. (The
    knob's *empirical effect* on smoothness is documented in the YAML
    header as a negative result — it modifies x[1] but the smoothness
    Hessian's neighbour coupling reverses the gain at sample 1.)"""
    p = planner_registry.get("mpc_chomp").from_config(
        {
            "max_speed": 5.0,
            "w_action_jump": 0.5,
            "output": "velocity_profile",
            "mpc": {"horizon": 20, "dt_plan": 0.05, "n_samples": 8, "max_speed": 5.0},
        }
    )
    assert p.w_action_jump == 0.5
    occ = empty_grid_20
    plan = p.plan(np.array([2.0, 2.0]), np.array([18.0, 18.0]), occ)
    assert plan.meta["w_action_jump"] == 0.5


def test_mpc_chomp_w_action_jump_only_active_after_first_replan(
    planner_registry, empty_grid_20
) -> None:
    """Episode-start replan has prev_emitted=None so the jump cost is
    inactive — otherwise the first plan of every episode would be biased
    by stale state. Tests the dispatch guard and the reset() hook clearing
    the cache."""
    occ = empty_grid_20
    p = planner_registry.get("mpc_chomp").from_config(
        {
            "max_speed": 5.0,
            "n_smooth_iters": 5,
            "w_action_jump": 100.0,  # huge — would dominate if active
            "output": "velocity_profile",
            "mpc": {"horizon": 20, "dt_plan": 0.05, "n_samples": 8, "max_speed": 5.0},
        }
    )
    plan1 = p.plan(np.array([2.0, 2.0]), np.array([18.0, 18.0]), occ)
    # First plan: prev was None, cache populated at emit time.
    assert p._prev_emitted_velocity is not None
    assert np.allclose(p._prev_emitted_velocity, plan1.velocity_profile[0])
    # reset() must clear the cache so a new episode starts unbiased.
    p.reset()
    assert p._prev_emitted_velocity is None


def test_follow_plan_velocity_profile_takes_priority_over_target_velocity() -> None:
    """If both velocity_profile and target_velocity are set on the same
    Plan, the profile must win — it's the more-specific signal. Regression
    guard for the dispatch order in `_follow_plan`."""
    from uav_nav_lab.planner.base import Plan
    from uav_nav_lab.runner.experiment import _follow_plan

    plan = Plan(
        waypoints=np.zeros((1, 2)),
        target_velocity=np.array([5.0, 0.0]),
        velocity_profile=np.array([[0.0, 3.0]]),
        profile_dt=0.1,
    )
    cmd = _follow_plan(plan, np.array([0.0, 0.0]), max_speed=10.0, t_since_replan=0.0)
    assert np.allclose(cmd, [0.0, 3.0])
