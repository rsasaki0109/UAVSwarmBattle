"""Dummy (in-process) simulator unit tests."""

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


def test_dummy_sim_wind_blows_drone() -> None:
    """A drone with zero command + non-zero wind should drift along the wind."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim import SIM_REGISTRY

    scn_cfg = {
        "size": [50, 50], "start": [10.0, 10.0], "goal": [40.0, 40.0],
        "obstacles": {"type": "none"},
    }
    scn = SCENARIO_REGISTRY.get("grid_world").from_config(scn_cfg)
    sim_cfg = {
        "dt": 0.1, "max_steps": 100, "max_accel": 100.0,
        "disturbance": {"wind": [3.0, 0.0]},
    }
    sim = SIM_REGISTRY.get("dummy_2d").from_config(sim_cfg, scn)
    sim.reset(seed=0)
    initial_x = sim.state.position[0]
    for _ in range(10):
        sim.step(np.array([0.0, 0.0]))  # zero velocity command
    drift = sim.state.position[0] - initial_x
    # 3 m/s wind for 1.0s should produce ~3m of drift
    assert 2.5 < drift < 3.5


def test_dummy_sim_synthetic_lidar_emits_in_range_obstacle_cells() -> None:
    """When `synthetic_perception.lidar_range > 0`, dummy sim should
    populate `state.extra["lidar_points"]["omni"]` with vehicle-local
    points for every occupied cell within the configured range — letting
    the same dummy world drive `pointcloud_occupancy` for ablations
    that don't require AirSim / ROS 2."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim import SIM_REGISTRY

    scn = SCENARIO_REGISTRY.get("grid_world").from_config(
        {"size": [20, 20], "start": [10.0, 10.0], "goal": [18.0, 10.0],
         "obstacles": {"type": "none"}}
    )
    sim = SIM_REGISTRY.get("dummy_2d").from_config(
        {"dt": 0.05, "synthetic_perception": {"lidar_range": 8.0}}, scn,
    )
    # reset() reseeds the scenario which clears the obstacle grid; plant
    # AFTER reset so the test obstacles survive.
    sim.reset(seed=0)
    scn.occupancy[15, 10] = True   # 5.5 m from drone at (10, 10)
    scn.occupancy[18, 18] = True   # ~11.4 m, beyond 8 m range
    state, _ = sim.step(np.array([0.0, 0.0]))
    cloud = state.extra["lidar_points"]["omni"]
    assert cloud.shape[1] == 3
    assert cloud.shape[0] == 1                # only the in-range cell
    # Vehicle-local: world (15.5, 10.5) − drone (10.0, 10.0) ≈ (5.5, 0.5, 0).
    # Drone position drifts slightly during step() so allow ±1.0 tolerance.
    assert 4.5 < float(cloud[0, 0]) < 6.5
    assert abs(float(cloud[0, 1])) < 1.5


def test_dummy_sim_synthetic_depth_camera_projects_forward_obstacles() -> None:
    """`synthetic_perception.depth: {fov_deg, width, height, max_depth}`
    should populate `state.extra["depth_images"]["front"]` with a
    pinhole depth image plus intrinsics. Pixels with no obstacle stay
    at `max_depth + 1` so the depth_image_occupancy sensor's cap drops
    them; pixels with an obstacle in front carry the cell's depth."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim import SIM_REGISTRY

    scn = SCENARIO_REGISTRY.get("grid_world").from_config(
        {"size": [20, 20], "start": [5.0, 10.0], "goal": [18.0, 10.0],
         "obstacles": {"type": "none"}}
    )
    sim = SIM_REGISTRY.get("dummy_2d").from_config(
        {"dt": 0.05, "synthetic_perception": {
            "depth": {"fov_deg": 90, "width": 16, "height": 12, "max_depth": 6.0}
        }}, scn,
    )
    sim.reset(seed=0)
    # Single obstacle directly ahead of drone: world (10, 10), depth ≈ 5.5 m.
    scn.occupancy[10, 10] = True
    state, _ = sim.step(np.array([0.0, 0.0]))
    payload = state.extra["depth_images"]["front"]
    depth = payload["depth"]
    assert depth.shape == (12, 16)
    # The obstacle's centre projects to image centre (drone faces +x,
    # obstacle at (10.5, 10.5) is bearing 0° + slight offset). Closest
    # depth in the image should be ~5.5 m (cell centre at x=10.5, drone x=5).
    assert 5.0 < float(depth.min()) < 6.0
    # Non-hit pixels stay at the sentinel (max_depth + 1).
    assert float(depth.max()) == pytest.approx(7.0)
    # Intrinsics: 90° FOV on 16-wide → fx = 8 / tan(45°) = 8.
    assert payload["intrinsics"]["fx"] == pytest.approx(8.0)
    # And R_cam_to_body so the depth_image_occupancy sensor reverses
    # the bridge's world→cam projection correctly.
    assert payload["R_cam_to_body"].shape == (3, 3)


def test_dummy_sim_synthetic_perception_round_trip_through_sensors() -> None:
    """Both pointcloud_occupancy and depth_image_occupancy fed by the
    same dummy sim should mark the obstacle cell that's directly in
    front of the drone — proves the camera-frame rotation in the
    synthetic depth payload is consistent with the sensor's reverse
    projection."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sensor import SENSOR_REGISTRY
    from uav_nav_lab.sim import SIM_REGISTRY

    scn = SCENARIO_REGISTRY.get("grid_world").from_config(
        {"size": [20, 20], "start": [5.0, 10.0], "goal": [18.0, 10.0],
         "obstacles": {"type": "none"}}
    )
    sim = SIM_REGISTRY.get("dummy_2d").from_config(
        {"dt": 0.05, "synthetic_perception": {
            "lidar_range": 8.0,
            "depth": {"fov_deg": 90, "width": 32, "height": 24, "max_depth": 8.0},
        }}, scn,
    )
    sim.reset(seed=0)
    scn.occupancy[10, 10] = True
    state, _ = sim.step(np.array([0.0, 0.0]))

    pc = SENSOR_REGISTRY.get("pointcloud_occupancy").from_config(
        {"resolution": 1.0, "memory": False}
    )
    pc.reset()
    pc_occ = pc.observe_map(0.05, state.position, scn.occupancy, sim_extra=state.extra)

    dp = SENSOR_REGISTRY.get("depth_image_occupancy").from_config(
        {"resolution": 1.0, "memory": False, "stride": 1, "max_depth": 8.0}
    )
    dp.reset()
    dp_occ = dp.observe_map(0.05, state.position, scn.occupancy, sim_extra=state.extra)

    # Both sensors should mark cell (10, 10) — the obstacle itself.
    assert pc_occ[10, 10]
    assert dp_occ[10, 10]
