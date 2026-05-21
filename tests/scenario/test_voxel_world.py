"""voxel_world / multi_drone_voxel scenario builders."""

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


def test_voxel_world_accepts_explicit_box_obstacles() -> None:
    from uav_nav_lab.scenario import SCENARIO_REGISTRY

    voxel_cls = SCENARIO_REGISTRY.get("voxel_world")
    sc = voxel_cls.from_config(
        {
            "size": [8, 8, 8],
            "start": [1.0, 1.0, 1.0],
            "goal": [7.0, 7.0, 7.0],
            "obstacles": {
                "type": "none",
                "boxes": [
                    {"min": [2, 2, 2], "max": [3, 4, 5]},
                    {"center": [6.0, 2.0, 2.0], "size": [2.0, 2.0, 2.0]},
                ],
            },
        }
    )

    assert sc.occupancy[2:4, 2:5, 2:6].all()
    assert sc.occupancy[5:7, 1:3, 1:3].all()
    assert not sc.occupancy[0, 0, 0]


def test_voxel_world_dynamic_obstacles_advance_and_collide() -> None:
    """voxel_world should support dynamic obstacles symmetrically with grid_world."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY

    voxel_cls = SCENARIO_REGISTRY.get("voxel_world")
    sc = voxel_cls.from_config(
        {
            "size": [20, 20, 10],
            "start": [2.0, 2.0, 5.0],
            "goal": [17.0, 17.0, 5.0],
            "resolution": 1.0,
            "obstacles": {"type": "none"},
            "dynamic_obstacles": [
                {"start": [10.0, 10.0, 5.0], "velocity": [1.0, 0.0, 0.0], "radius": 0.5},
            ],
        }
    )
    # property reflects current state
    assert len(sc.dynamic_obstacles) == 1
    assert sc.dynamic_obstacles[0]["position"][0] == pytest.approx(10.0)
    # advance moves the obstacle linearly
    sc.advance(1.0)
    assert sc.dynamic_obstacles[0]["position"][0] == pytest.approx(11.0)
    # collision check uses sphere-sphere distance against true position
    assert sc.is_collision(np.array([11.0, 10.0, 5.0]), radius=0.4)
    assert not sc.is_collision(np.array([15.0, 15.0, 5.0]), radius=0.4)


def test_voxel_world_random_layer_obstacles_stay_on_requested_z() -> None:
    """random_layer supports dense same-altitude AirSim latency sweeps
    without committing hundreds of explicit obstacle cells to YAML."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY

    voxel_cls = SCENARIO_REGISTRY.get("voxel_world")
    sc = voxel_cls.from_config(
        {
            "size": [30, 30, 8],
            "start": [2.0, 2.0, 4.0],
            "goal": [27.0, 27.0, 4.0],
            "resolution": 1.0,
            "obstacles": {"type": "random_layer", "count": 80, "seed": 7, "z": 4},
        }
    )

    occupied = np.argwhere(sc.occupancy)
    assert occupied.shape[0] == 80
    assert set(occupied[:, 2].tolist()) == {4}
    assert not sc.occupancy[2, 2, 4]
    assert not sc.occupancy[27, 27, 4]


def test_multi_drone_voxel_scenario_constructs_3d() -> None:
    """multi_drone_voxel registers, validates 3D drone coords, and exposes
    n_drones + ndim==3 so the multi runner can iterate over drones the same
    way it does for multi_drone_grid."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY

    cls = SCENARIO_REGISTRY.get("multi_drone_voxel")
    sc = cls.from_config({
        "size": [20, 20, 8],
        "obstacles": {"type": "none"},
        "drones": [
            {"name": "a", "start": [2.0, 10.0, 4.0], "goal": [18.0, 10.0, 4.0]},
            {"name": "b", "start": [18.0, 10.0, 4.0], "goal": [2.0, 10.0, 4.0]},
        ],
    })
    assert sc.n_drones == 2
    assert sc.ndim == 3
    assert sc.start.shape == (3,)
    # 2D coords on a drone must be rejected at config-load time
    with pytest.raises(ValueError):
        cls.from_config({
            "size": [20, 20, 8],
            "obstacles": {"type": "none"},
            "drones": [{"name": "bad", "start": [2.0, 10.0], "goal": [18.0, 10.0]}],
        })
