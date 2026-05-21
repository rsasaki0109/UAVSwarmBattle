"""LiDAR sensor + recorder summarisation."""

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


def test_lidar_dynamics_filtered_by_range() -> None:
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    lidar_cls = SENSOR_REGISTRY.get("lidar")
    s = lidar_cls.from_config({"range": 5.0, "delay": 0.0, "memory": False})
    s.reset(seed=0)
    dyn = [
        {"position": [3.0, 0.0], "velocity": [0.0, 0.0], "radius": 0.5},  # in range
        {"position": [50.0, 0.0], "velocity": [0.0, 0.0], "radius": 0.5},  # out of range
    ]
    seen = s.observe_dynamics(0.0, np.array([0.0, 0.0]), dyn)
    assert len(seen) == 1
    assert seen[0]["position"] == [3.0, 0.0]


def test_lidar_sensor_partial_map() -> None:
    """Lidar should only mark obstacles within `range` of the drone, and
    accumulate them across observations when memory=True."""
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    occ = np.zeros((20, 20), dtype=bool)
    occ[5, 5] = True   # near start
    occ[15, 15] = True  # far from start

    sensor_cls = SENSOR_REGISTRY.get("lidar")
    s = sensor_cls.from_config({"range": 4.0, "delay": 0.0, "resolution": 1.0, "memory": True})
    s.reset(seed=0)

    seen0 = s.observe_map(0.0, np.array([5.0, 5.0]), occ)
    assert seen0[5, 5]            # close obstacle is visible
    assert not seen0[15, 15]      # distant one is not

    # drone moves close to the far obstacle; both should now be in memory
    seen1 = s.observe_map(0.1, np.array([15.0, 15.0]), occ)
    assert seen1[5, 5] and seen1[15, 15]


def test_recorder_summarizes_lidar_points_into_step_row() -> None:
    """When a sim backend populates state.extra['lidar_points'] with
    name-keyed (N, 3) arrays, EpisodeRecorder.log_step should surface
    {name: count} into the step row so episode JSONs show that lidar
    was actually being polled. Full clouds stay in memory only."""
    from uav_nav_lab.recorder import EpisodeRecorder

    rec = EpisodeRecorder(episode_index=0, seed=0)
    pos = np.array([1.0, 2.0])
    cloud_a = np.zeros((42, 3))
    cloud_b = np.zeros((7, 3))

    # Step 1: lidar populated → counts persisted.
    rec.log_step(
        t=0.0, true_pos=pos, true_vel=pos, observed_pos=pos, cmd=pos,
        info={"collision": False, "goal_reached": False},
        sim_extra={"lidar_points": {"FrontLidar": cloud_a, "RearLidar": cloud_b}},
    )
    # Step 2: empty extra dict → no lidar key in row.
    rec.log_step(
        t=0.05, true_pos=pos, true_vel=pos, observed_pos=pos, cmd=pos,
        info={"collision": False, "goal_reached": False},
        sim_extra={},
    )
    # Step 3: extra carries something else but no lidar → no lidar key.
    rec.log_step(
        t=0.10, true_pos=pos, true_vel=pos, observed_pos=pos, cmd=pos,
        info={"collision": False, "goal_reached": False},
        sim_extra={"depth_image": "ignored"},
    )

    assert rec.steps[0]["lidar_points"] == {"FrontLidar": 42, "RearLidar": 7}
    assert "lidar_points" not in rec.steps[1]
    assert "lidar_points" not in rec.steps[2]
