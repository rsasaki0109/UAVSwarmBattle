"""pointcloud_occupancy sensor unit tests."""

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


def test_pointcloud_occupancy_marks_world_cells_from_local_points() -> None:
    """Drone at world (5, 5); lidar reports local points (1, 0, 0) and
    (-1, 1, 0) → world (6, 5) and (4, 6) → cells [6][5] and [4][6]
    flipped on the (10, 10) occupancy grid. Other cells stay free."""
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    cls = SENSOR_REGISTRY.get("pointcloud_occupancy")
    sensor = cls.from_config({"resolution": 1.0, "memory": True})
    sensor.reset()

    occ_shape = (10, 10)
    base = np.zeros(occ_shape, dtype=bool)
    pos = np.array([5.0, 5.0])
    cloud = np.array([[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0]])
    out = sensor.observe_map(
        t=0.0, true_position=pos, true_obstacle_map=base,
        sim_extra={"lidar_points": {"FrontLidar": cloud}},
    )
    assert out.shape == occ_shape
    assert out[6, 5]
    assert out[4, 6]
    # exactly 2 cells set
    assert out.sum() == 2


def test_pointcloud_occupancy_memory_accumulates_then_clears_when_off() -> None:
    """memory=True sweeps OR together; memory=False keeps only the latest."""
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    cls = SENSOR_REGISTRY.get("pointcloud_occupancy")
    base = np.zeros((10, 10), dtype=bool)
    pos = np.array([5.0, 5.0])

    s_mem = cls.from_config({"resolution": 1.0, "memory": True})
    s_mem.reset()
    s_mem.observe_map(0.0, pos, base, sim_extra={"lidar_points": {"L": np.array([[1.0, 0.0, 0.0]])}})
    out = s_mem.observe_map(0.05, pos, base, sim_extra={"lidar_points": {"L": np.array([[-1.0, 0.0, 0.0]])}})
    assert out.sum() == 2  # both points retained

    s_nom = cls.from_config({"resolution": 1.0, "memory": False})
    s_nom.reset()
    s_nom.observe_map(0.0, pos, base, sim_extra={"lidar_points": {"L": np.array([[1.0, 0.0, 0.0]])}})
    out = s_nom.observe_map(0.05, pos, base, sim_extra={"lidar_points": {"L": np.array([[-1.0, 0.0, 0.0]])}})
    assert out.sum() == 1  # only the latest sweep


def test_pointcloud_occupancy_handles_empty_or_missing_extras() -> None:
    """No sim_extra / no lidar_points / unknown lidar name → empty grid
    (or accumulated memory). Should not crash and should not pretend to
    see ground-truth occupancy."""
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    cls = SENSOR_REGISTRY.get("pointcloud_occupancy")
    sensor = cls.from_config({"resolution": 1.0, "memory": True})
    sensor.reset()
    base = np.zeros((10, 10), dtype=bool)
    pos = np.array([5.0, 5.0])

    assert sensor.observe_map(0.0, pos, base, sim_extra=None).sum() == 0
    assert sensor.observe_map(0.0, pos, base, sim_extra={}).sum() == 0
    assert sensor.observe_map(0.0, pos, base, sim_extra={"lidar_points": {}}).sum() == 0
    # malformed point cloud (not (N, 3)) → silently skipped
    assert sensor.observe_map(
        0.0, pos, base, sim_extra={"lidar_points": {"L": np.array([1.0, 2.0])}}
    ).sum() == 0


def test_pointcloud_occupancy_3d_grid_uses_z_component() -> None:
    """In 3D scenarios the sensor should index into 3D occupancy with
    the world-frame z component, not silently project to 2D."""
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    cls = SENSOR_REGISTRY.get("pointcloud_occupancy")
    sensor = cls.from_config({"resolution": 1.0, "memory": True})
    sensor.reset()
    base = np.zeros((10, 10, 8), dtype=bool)
    pos = np.array([5.0, 5.0, 4.0])
    # local (1, 0, 1) → world (6, 5, 5) → cell [6][5][5]
    out = sensor.observe_map(
        0.0, pos, base,
        sim_extra={"lidar_points": {"L": np.array([[1.0, 0.0, 1.0]])}},
    )
    assert out.shape == (10, 10, 8)
    assert out[6, 5, 5]
    assert out.sum() == 1


def test_pointcloud_occupancy_inflate_dilates_each_hit() -> None:
    """`inflate: 1` should add a 1-cell ring around every hit cell
    (cross-shaped via separable shifts; 4 neighbors in 2D)."""
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    cls = SENSOR_REGISTRY.get("pointcloud_occupancy")
    sensor = cls.from_config({"resolution": 1.0, "memory": True, "inflate": 1})
    sensor.reset()
    base = np.zeros((10, 10), dtype=bool)
    pos = np.array([5.0, 5.0])
    # Single point → cell [6, 5] + 4-cell ring around it.
    out = sensor.observe_map(
        0.0, pos, base,
        sim_extra={"lidar_points": {"L": np.array([[1.0, 0.0, 0.0]])}},
    )
    assert out[6, 5]
    assert out[5, 5] and out[7, 5] and out[6, 4] and out[6, 6]
    assert out.sum() == 5  # center + 4 neighbors, no diagonals
