"""Dynamic-obstacle motion in voxel scenarios."""

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


def test_dynamic_obstacle_motion(tmp_path: Path) -> None:
    """A dynamic obstacle should appear in occupancy at the right cells over
    time, and the lidar memory should clear when it moves out of range."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    cfg = {
        "size": [20, 20],
        "start": [2.0, 2.0],
        "goal": [18.0, 18.0],
        "obstacles": {"type": "none"},
        "dynamic_obstacles": [
            {"start": [10.0, 10.0], "velocity": [5.0, 0.0], "reflect": False, "radius": 0.6}
        ],
    }
    scn = SCENARIO_REGISTRY.get("grid_world").from_config(cfg)
    occ_t0 = scn.occupancy.copy()
    assert occ_t0[10, 10]
    scn.advance(1.0)  # move 5 cells in x
    assert scn.occupancy[15, 10]
    assert not scn.occupancy[10, 10]  # the dynamic obstacle is no longer here

    # lidar memory: cell (10,10) was seen as obstacle, then we observe again
    # with obstacle gone — should be cleared from memory.
    lidar_cls = SENSOR_REGISTRY.get("lidar")
    lidar = lidar_cls.from_config({"range": 5.0, "delay": 0.0, "memory": True})
    lidar.reset(seed=0)
    seen0 = lidar.observe_map(0.0, np.array([10.0, 10.0]), occ_t0)
    assert seen0[10, 10]
    seen1 = lidar.observe_map(1.0, np.array([10.0, 10.0]), scn.occupancy)
    assert not seen1[10, 10]


def test_rect_obstacles_fill_blocks() -> None:
    """`obstacles.rects` fills inclusive [x0,y0,x1,y1] blocks — two rects leave
    a passable slit between them (the pinch-corridor layout)."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY

    cfg = {
        "size": [40, 40],
        "start": [2.0, 20.0],
        "goal": [38.0, 20.0],
        "obstacles": {"type": "cells", "rects": [[18, 0, 22, 16], [18, 23, 22, 39]]},
        "dynamic_obstacles": [],
    }
    scn = SCENARIO_REGISTRY.get("grid_world").from_config(cfg)
    occ = scn._static_occ
    # walls present
    assert occ[18, 0] and occ[22, 16] and occ[20, 39]
    # slit y=17..22 at x=20 is free (drone can pass)
    assert [y for y in range(40) if not occ[20, y]] == list(range(17, 23))
    # out-of-band columns untouched
    assert not occ[10, 10] and not occ[30, 30]


def test_rect_bounds_are_clamped() -> None:
    """A rect spilling past the grid is clamped, not an index error."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY

    cfg = {
        "size": [10, 10],
        "start": [0.0, 0.0],
        "goal": [9.0, 9.0],
        "obstacles": {"type": "cells", "rects": [[-5, -5, 2, 2]]},
        "dynamic_obstacles": [],
    }
    scn = SCENARIO_REGISTRY.get("grid_world").from_config(cfg)
    occ = scn._static_occ
    assert occ[0, 0] and occ[2, 2]
    assert not occ[3, 3]


def test_obstacle_jitter_is_seeded_and_reproducible() -> None:
    """start_jitter/vel_jitter diversify the spawn per episode (so the seed
    actually varies the scenario), yet a re-used seed reproduces it exactly."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY

    cfg = {
        "size": [40, 40],
        "start": [2.0, 20.0],
        "goal": [38.0, 20.0],
        "obstacles": {"type": "none"},
        "dynamic_obstacles": [
            {"start": [18.0, 8.0], "velocity": [0.0, 7.0], "radius": 1.8,
             "start_jitter": 3.0, "vel_jitter": 1.5}
        ],
    }
    scn = SCENARIO_REGISTRY.get("grid_world").from_config(cfg)
    # without reseed: nominal (no jitter applied)
    assert scn._dynamic[0].pos.tolist() == [18.0, 8.0]

    scn.reseed(7)
    pos7 = scn._dynamic[0].pos.copy()
    vel7 = scn._dynamic[0].vel.copy()
    scn.reseed(8)
    pos8 = scn._dynamic[0].pos.copy()
    # different seeds → different spawn (the whole point)
    assert not np.allclose(pos7, pos8)
    # same seed → identical (replay/GIF reproducibility)
    scn.reseed(7)
    assert np.allclose(scn._dynamic[0].pos, pos7)
    assert np.allclose(scn._dynamic[0].vel, vel7)


def test_zero_jitter_keeps_obstacle_deterministic() -> None:
    """With no jitter configured, reseed must NOT move the obstacle — existing
    fixed-layout scenarios stay byte-identical across episodes."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY

    cfg = {
        "size": [40, 40],
        "start": [2.0, 20.0],
        "goal": [38.0, 20.0],
        "obstacles": {"type": "none"},
        "dynamic_obstacles": [
            {"start": [18.0, 8.0], "velocity": [0.0, 7.0], "radius": 1.8}
        ],
    }
    scn = SCENARIO_REGISTRY.get("grid_world").from_config(cfg)
    scn.reseed(7)
    assert scn._dynamic[0].pos.tolist() == [18.0, 8.0]
    scn.reseed(99)
    assert scn._dynamic[0].pos.tolist() == [18.0, 8.0]
