"""depth_image_occupancy sensor unit tests."""

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


def test_depth_image_occupancy_projects_pixel_to_correct_world_cell() -> None:
    """Drone at world (5, 5), 2D occupancy. A single non-sky pixel at
    column 40 row 24 with depth 3 m, intrinsics fx=fy=32, cx=32, cy=24:
        camera frame: x = (40-32) * 3 / 32 = 0.75, y = 0, z = 3
    With identity rotation that's body (0.75, 0, 3); projecting to 2D
    occupancy (xy) with resolution 1.0 lands the hit at cell (5, 5)
    (5 + 0.75 → floor → 5). All other cells stay free."""
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    sensor = SENSOR_REGISTRY.get("depth_image_occupancy").from_config(
        {"resolution": 1.0, "memory": True, "stride": 1, "max_depth": 10.0}
    )
    sensor.reset()
    depth = np.full((48, 64), 100.0, dtype=np.float32)  # everything is sky
    depth[24, 40] = 3.0
    base = np.zeros((20, 20), dtype=bool)
    out = sensor.observe_map(
        t=0.0, true_position=np.array([5.0, 5.0]), true_obstacle_map=base,
        sim_extra={"depth_images": {"front": {
            "depth": depth,
            "intrinsics": {"fx": 32.0, "fy": 32.0, "cx": 32.0, "cy": 24.0},
        }}},
    )
    assert out.sum() == 1
    assert out[5, 5]


def test_depth_image_occupancy_drops_out_of_range_pixels() -> None:
    """`max_depth: M` should drop pixels reporting d > M (sky / no-return).
    With max_depth=5 m and a depth image of all 8 m, the resulting
    occupancy must stay empty — no false positives from saturated pixels."""
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    sensor = SENSOR_REGISTRY.get("depth_image_occupancy").from_config(
        {"resolution": 1.0, "memory": True, "stride": 1, "max_depth": 5.0}
    )
    sensor.reset()
    depth = np.full((10, 10), 8.0, dtype=np.float32)  # all beyond max_depth
    base = np.zeros((20, 20), dtype=bool)
    out = sensor.observe_map(
        0.0, np.array([10.0, 10.0]), base,
        sim_extra={"depth_images": {"front": {
            "depth": depth,
            "intrinsics": {"fx": 5.0, "fy": 5.0, "cx": 5.0, "cy": 5.0},
        }}},
    )
    assert out.sum() == 0


def test_depth_image_occupancy_handles_missing_or_malformed_payload() -> None:
    """No sim_extra / no depth_images / missing intrinsics / wrong-shape
    depth array should all return the (memory-accumulated, possibly
    empty) grid without crashing — same forgiving behaviour as
    pointcloud_occupancy."""
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    sensor = SENSOR_REGISTRY.get("depth_image_occupancy").from_config(
        {"resolution": 1.0, "memory": True}
    )
    sensor.reset()
    base = np.zeros((10, 10), dtype=bool)
    pos = np.array([5.0, 5.0])

    assert sensor.observe_map(0.0, pos, base, sim_extra=None).sum() == 0
    assert sensor.observe_map(0.0, pos, base, sim_extra={}).sum() == 0
    assert sensor.observe_map(0.0, pos, base, sim_extra={"depth_images": {}}).sum() == 0
    # Missing intrinsics → silently skip.
    assert sensor.observe_map(
        0.0, pos, base, sim_extra={"depth_images": {"f": {"depth": np.ones((4, 4), np.float32)}}}
    ).sum() == 0
    # 1D depth array → silently skip.
    assert sensor.observe_map(
        0.0, pos, base, sim_extra={"depth_images": {"f": {
            "depth": np.ones(16, np.float32),
            "intrinsics": {"fx": 4.0, "fy": 4.0, "cx": 2.0, "cy": 2.0},
        }}},
    ).sum() == 0


def test_depth_image_occupancy_stride_subsamples_for_compute() -> None:
    """`stride: 2` halves the pixel grid in each axis. A 4x4 depth image
    with all pixels valid should mark *fewer* cells when stride=2 vs
    stride=1 (cost-vs-coverage tradeoff)."""
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    cls = SENSOR_REGISTRY.get("depth_image_occupancy")
    base = np.zeros((30, 30), dtype=bool)
    pos = np.array([15.0, 15.0])
    # 16 unique depths so each pixel projects to a distinct (X, Y).
    depth = np.linspace(1.0, 5.0, 16, dtype=np.float32).reshape(4, 4)
    payload = {"f": {
        "depth": depth,
        "intrinsics": {"fx": 4.0, "fy": 4.0, "cx": 2.0, "cy": 2.0},
    }}
    s1 = cls.from_config({"resolution": 0.5, "memory": True, "stride": 1, "max_depth": 10.0})
    s1.reset()
    s2 = cls.from_config({"resolution": 0.5, "memory": True, "stride": 2, "max_depth": 10.0})
    s2.reset()
    n1 = int(s1.observe_map(0.0, pos, base, sim_extra={"depth_images": payload}).sum())
    n2 = int(s2.observe_map(0.0, pos, base, sim_extra={"depth_images": payload}).sum())
    assert n1 > 0
    assert n2 > 0
    assert n2 < n1  # subsampling marks fewer cells
