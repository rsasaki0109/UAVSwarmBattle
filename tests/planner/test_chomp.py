"""CHOMP planner unit tests."""

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


def test_chomp_open_world_returns_essentially_straight_line() -> None:
    """With no obstacles the smoothness term dominates and CHOMP should
    converge to the straight-line trajectory (zero second-difference).
    Endpoints stay pinned at start / goal exactly."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    chomp = PLANNER_REGISTRY.get("chomp").from_config(
        {"n_waypoints": 20, "n_iters": 50}
    )
    occ = np.zeros((20, 20), dtype=bool)
    plan = chomp.plan(np.array([1.0, 1.0]), np.array([18.0, 18.0]), occ)
    assert plan.meta["status"] == "ok"
    # Endpoints clamped to start / goal exactly.
    assert np.allclose(plan.waypoints[0], [1.0, 1.0])
    assert np.allclose(plan.waypoints[-1], [18.0, 18.0])
    # Straight line ⇒ second differences ≈ 0.
    assert float(np.linalg.norm(np.diff(plan.waypoints, n=2, axis=0))) < 1e-6


def test_chomp_routes_around_a_horizontal_bar() -> None:
    """A straight-line init from (2, 8) to (18, 12) crosses a horizontal
    bar at y=10 (x ∈ [5, 14]). CHOMP's local optimisation should detour
    *under* the bar (lower y) and report `status=ok`. Asymmetric start /
    goal y-coordinates break the symmetry that traps the box test."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    chomp = PLANNER_REGISTRY.get("chomp").from_config(
        {"n_waypoints": 30, "n_iters": 100}
    )
    occ = np.zeros((20, 20), dtype=bool)
    occ[5:15, 10] = True
    plan = chomp.plan(np.array([2.0, 8.0]), np.array([18.0, 12.0]), occ)
    assert plan.meta["status"] == "ok"
    # No waypoint inside the raw obstacle.
    cells = np.clip(np.round(plan.waypoints).astype(int), 0, 19)
    assert int(occ[tuple(cells.T)].sum()) == 0
    # The detour reaches y ≤ 9 — actually goes below the bar.
    assert float(plan.waypoints[:, 1].min()) <= 9.0


def test_chomp_reports_local_minimum_when_init_cannot_escape() -> None:
    """Symmetric box: start (2, 15), goal (28, 15), box at x∈[10,19],
    y∈[10,19]. The straight-line init is symmetric across y=15 so the
    obstacle gradient cancels in the y-axis — CHOMP cannot decide to go
    up or down. The planner should detect this and return
    `status=local_minimum`, not silently produce a colliding plan."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    chomp = PLANNER_REGISTRY.get("chomp").from_config(
        {"n_waypoints": 30, "n_iters": 100}
    )
    occ = np.zeros((30, 30), dtype=bool)
    occ[10:20, 10:20] = True
    plan = chomp.plan(np.array([2.0, 15.0]), np.array([28.0, 15.0]), occ)
    assert plan.meta["status"] == "local_minimum"


def test_chomp_smoothness_hessian_inverse_keeps_step_stable_at_n50() -> None:
    """Plain GD on K diverges around n≈20 because λ_max(K) ≳ 16. The
    M⁻¹-preconditioned step should stay bounded for n=50 + 200 iters with
    a high obstacle weight. Regression guard against re-introducing the
    `K @ x` raw-gradient bug."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    chomp = PLANNER_REGISTRY.get("chomp").from_config(
        {"n_waypoints": 50, "n_iters": 200, "w_obs": 10.0}
    )
    occ = np.zeros((30, 30), dtype=bool)
    occ[14, 14] = True  # single-cell obstacle near the path
    plan = chomp.plan(np.array([2.0, 14.0]), np.array([28.0, 14.0]), occ)
    # No waypoint blew outside a generous box around the world.
    assert float(np.abs(plan.waypoints).max()) < 100.0
    # And no NaN / Inf.
    assert np.all(np.isfinite(plan.waypoints))


def test_chomp_init_rrt_escapes_box_that_traps_straight_init() -> None:
    """The symmetric box scenario locks straight-line init in a saddle
    (see test_chomp_reports_local_minimum_when_init_cannot_escape).
    `init: rrt` uses an RRT path as the warm start, which detours around
    the box, so CHOMP smooths a *collision-free* trajectory and reports
    `status=ok` — the whole point of the feature."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    occ = np.zeros((30, 30), dtype=bool)
    occ[10:20, 10:20] = True

    chomp = PLANNER_REGISTRY.get("chomp").from_config(
        {
            "n_waypoints": 30,
            "n_iters": 100,
            "init": "rrt",
            "rrt_max_samples": 1000,
            "rrt_seed": 42,
            "rrt_goal_bias": 0.2,
        }
    )
    chomp.reset()
    plan = chomp.plan(np.array([2.0, 15.0]), np.array([28.0, 15.0]), occ)
    assert plan.meta["status"] == "ok"
    assert plan.meta["init"] == "rrt"
    cells = np.clip(np.round(plan.waypoints).astype(int), 0, 29)
    assert int(occ[tuple(cells.T)].sum()) == 0


def test_chomp_init_rrt_falls_back_to_straight_when_rrt_fails() -> None:
    """When the inner RRT exhausts max_samples without finding a path,
    CHOMP must keep working — fall back to straight-line init and report
    `init=rrt_fallback_straight` so the failure is observable in the
    plan log without crashing the run."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    occ = np.zeros((30, 30), dtype=bool)
    occ[10:20, 10:20] = True

    chomp = PLANNER_REGISTRY.get("chomp").from_config(
        {
            "n_waypoints": 30,
            "n_iters": 50,
            "init": "rrt",
            "rrt_max_samples": 5,  # nowhere near enough to find a path
            "rrt_seed": 0,
        }
    )
    plan = chomp.plan(np.array([2.0, 15.0]), np.array([28.0, 15.0]), occ)
    assert plan.meta["init"] == "rrt_fallback_straight"
    assert plan.meta["status"] == "local_minimum"


def test_chomp_resample_polyline_uniform_arc_length() -> None:
    """The internal arc-length resampler should turn a 3-vertex L-shape
    (segments 10 + 6 = 16 m) into n equally-spaced points along the
    polyline. Index 10 of 17 sits exactly at the joint (10/16 of arc
    length)."""
    from uav_nav_lab.planner.chomp import _resample_polyline

    wps = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 6.0]])
    out = _resample_polyline(wps, 17)
    assert out.shape == (17, 2)
    assert np.allclose(out[0], wps[0])
    assert np.allclose(out[-1], wps[-1])
    assert np.allclose(out[10], np.array([10.0, 0.0]))
    # Single-vertex degenerate case returns the start repeated.
    out2 = _resample_polyline(np.array([[3.0, 4.0]]), 5)
    assert out2.shape == (5, 2)
    assert np.allclose(out2, np.tile([3.0, 4.0], (5, 1)))


def test_chomp_init_invalid_value_raises() -> None:
    """Typo in the init field should fail loud at construction, not
    silently default — guards `init: 'RRT'` / `init: 'random'` from
    looking like they did something."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    with pytest.raises(ValueError, match="init must be"):
        PLANNER_REGISTRY.get("chomp").from_config({"init": "random"})


def test_chomp_registry_and_from_config_round_trip() -> None:
    """Registration + from_config wiring smoke test."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    cls = PLANNER_REGISTRY.get("chomp")
    chomp = cls.from_config(
        {
            "max_speed": 7.0,
            "n_waypoints": 25,
            "n_iters": 10,
            "learning_rate": 0.1,
            "w_obs": 3.0,
            "epsilon": 1.5,
            "resolution": 0.5,
            "inflate": 1,
        }
    )
    assert chomp.max_speed == 7.0
    assert chomp.n_waypoints == 25
    assert chomp.n_iters == 10
    assert chomp.epsilon == 1.5
