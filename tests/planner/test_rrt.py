"""RRT / RRT* planner unit tests."""

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


def test_rrt_star_returns_shorter_path_than_rrt_on_open_world(
    planner_registry, empty_grid_30
) -> None:
    """RRT* rewiring should produce a path no longer than plain RRT on
    average. We compare on a wide-open world where rewiring has clear
    headroom (zigzag RRT path → near-straight RRT* path)."""
    occ = empty_grid_30
    start = np.array([2.0, 2.0])
    goal = np.array([28.0, 28.0])

    rrt = planner_registry.get("rrt").from_config(
        {"step_size": 1.5, "goal_tolerance": 1.0, "max_samples": 800, "seed": 1}
    )
    rrt_star = planner_registry.get("rrt_star").from_config(
        {
            "step_size": 1.5,
            "rewire_radius": 4.0,
            "goal_tolerance": 1.0,
            "max_samples": 800,
            "seed": 1,
        }
    )
    p_rrt = rrt.plan(start, goal, occ)
    p_star = rrt_star.plan(start, goal, occ)
    assert p_rrt.meta["status"] == "ok"
    assert p_star.meta["status"] == "ok"

    def path_len(wps):
        return float(np.sum(np.linalg.norm(np.diff(wps, axis=0), axis=1)))

    # On open ground RRT* should not be longer than RRT (it's allowed to
    # be equal in the rare case both find the same path).
    assert path_len(p_star.waypoints) <= path_len(p_rrt.waypoints) + 1e-6
    # And it should report a path_cost in its metadata.
    assert "path_cost" in p_star.meta


def test_rrt_planner_finds_path_around_a_wall(planner_registry, empty_grid_20) -> None:
    """RRT should find *some* path around a wall and reach goal_tolerance."""
    rrt_cls = planner_registry.get("rrt")
    rrt = rrt_cls.from_config(
        {
            "max_speed": 10.0,
            "step_size": 2.0,
            "goal_tolerance": 1.5,
            "max_samples": 1000,
            "goal_bias": 0.2,
            "seed": 42,
        }
    )
    occ = empty_grid_20
    occ[10, 5:15] = True  # wall down the middle with two openings
    plan = rrt.plan(np.array([2.0, 10.0]), np.array([18.0, 10.0]), occ)
    assert plan.meta["status"] == "ok"
    assert plan.waypoints.shape[0] >= 2
    # last waypoint must land within goal_tolerance of the goal
    last = plan.waypoints[-1]
    assert float(np.linalg.norm(last - np.array([18.0, 10.0]))) <= 1.5
