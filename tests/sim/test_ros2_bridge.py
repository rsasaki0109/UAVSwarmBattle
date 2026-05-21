"""ROS 2 bridge unit tests against mock adapters."""

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


def test_ros2_bridge_step_round_trips_enu_via_mock_adapter() -> None:
    """Verify the ROS 2 bridge's publish-spin-read plumbing against an
    injected mock adapter — no rclpy install required."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.ros2_bridge import Ros2Bridge

    grid_cls = SCENARIO_REGISTRY.get("grid_world")
    sc = grid_cls.from_config(
        {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0], "obstacles": {"type": "none"}}
    )

    class FakeAdapter:
        def __init__(self) -> None:
            self.commands: list[tuple[float, float, float]] = []
            self.teleports: list[np.ndarray] = []
            # Canned ENU pose / velocity returned from /odom on every tick.
            self._pose = np.array([3.0, 4.0, 1.0])
            self._vel = np.array([0.5, 0.6, 0.0])
            self._collision = False

        def publish_velocity(self, vx: float, vy: float, vz: float) -> None:
            self.commands.append((vx, vy, vz))

        def latest_pose_velocity(self):
            return (self._pose.copy(), self._vel.copy())

        def latest_collision(self) -> bool:
            return self._collision

        def tick(self, _timeout_s: float) -> None:
            pass

        def teleport(self, pos_enu: np.ndarray) -> None:
            self.teleports.append(np.asarray(pos_enu).copy())

    fake = FakeAdapter()
    bridge = Ros2Bridge(dt=0.05, scenario=sc, adapter=fake)
    state = bridge.reset()
    assert state.position.shape[0] == 2
    # reset() teleports the (3D-padded) start pose.
    assert len(fake.teleports) == 1
    assert np.allclose(fake.teleports[0][:2], np.array([1.0, 1.0]))
    # Initial state taken from the canned odom (ENU pass-through, no flip).
    assert np.allclose(state.position, np.array([3.0, 4.0]))

    # ENU velocity (1, 2) → adapter sees (1, 2, 0). No frame flip vs AirSim's NED.
    out_state, info = bridge.step(np.array([1.0, 2.0]))
    last = fake.commands[-1]
    assert last[0] == 1.0
    assert last[1] == 2.0
    assert last[2] == 0.0  # 2D scenario → vz = 0
    assert np.allclose(out_state.position, np.array([3.0, 4.0]))
    assert info.collision is False


def test_ros2_bridge_can_convert_ned_wrappers_via_mock_adapter() -> None:
    """AirSim's ROS wrapper defaults to NED odometry / velocity commands.
    Ros2Bridge should still expose ENU state to the framework."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.ros2_bridge import Ros2Bridge

    voxel_cls = SCENARIO_REGISTRY.get("voxel_world")
    sc = voxel_cls.from_config(
        {
            "size": [10, 10, 10],
            "start": [1.0, 2.0, 3.0],
            "goal": [9.0, 8.0, 7.0],
            "obstacles": {"type": "none"},
        }
    )

    class FakeNedAdapter:
        def __init__(self) -> None:
            self.commands: list[tuple[float, float, float]] = []
            self.teleports: list[np.ndarray] = []
            self._pose_ned = np.array([4.0, 3.0, -2.0])
            self._vel_ned = np.array([0.6, 0.5, -0.1])

        def publish_velocity(self, vx: float, vy: float, vz: float) -> None:
            self.commands.append((vx, vy, vz))

        def latest_pose_velocity(self):
            return (self._pose_ned.copy(), self._vel_ned.copy())

        def latest_collision(self) -> bool:
            return False

        def tick(self, _timeout_s: float) -> None:
            pass

        def teleport(self, pos_ros: np.ndarray) -> None:
            self.teleports.append(np.asarray(pos_ros).copy())

    fake = FakeNedAdapter()
    bridge = Ros2Bridge(dt=0.05, scenario=sc, adapter=fake, frame="ned")
    state = bridge.reset()

    # start [1, 2, 3] ENU is sent to the wrapper as [2, 1, -3] NED.
    assert np.allclose(fake.teleports[0], np.array([2.0, 1.0, -3.0]))
    # Odom [4, 3, -2] NED is surfaced to the framework as [3, 4, 2] ENU.
    assert np.allclose(state.position, np.array([3.0, 4.0, 2.0]))
    assert np.allclose(state.velocity, np.array([0.5, 0.6, 0.1]))

    bridge.step(np.array([1.0, 2.0, 3.0]))
    assert fake.commands[-1] == pytest.approx((2.0, 1.0, -3.0))


def test_ros2_bridge_surfaces_lidar_camera_via_mock_adapter() -> None:
    """When `lidars`/`cameras` topics are configured, Ros2Bridge.step() should
    populate `state.extra['lidar_points'][topic]` and `['camera_images'][topic]`
    from the adapter — same keys as AirSimBridge so the pointcloud_occupancy
    sensor and `uav-nav video` CLI consume both backends transparently."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.ros2_bridge import Ros2Bridge

    grid_cls = SCENARIO_REGISTRY.get("grid_world")
    sc = grid_cls.from_config(
        {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0], "obstacles": {"type": "none"}}
    )

    class FakeAdapter:
        def __init__(self) -> None:
            self._pose = np.array([2.0, 3.0, 1.0])
            self._vel = np.array([0.0, 0.0, 0.0])
            self._clouds = {
                "/front_lidar": np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.5]], dtype=np.float32),
                "/rear_lidar": np.array([[-1.0, -1.0, 0.0]], dtype=np.float32),
            }
            self._images = {
                "/front_camera/image_raw": b"PNG_FRONT",
                "/down_camera/image_raw": b"PNG_DOWN",
            }

        def publish_velocity(self, vx, vy, vz):  # noqa: ARG002
            pass

        def latest_pose_velocity(self):
            return (self._pose.copy(), self._vel.copy())

        def latest_collision(self) -> bool:
            return False

        def tick(self, _timeout_s: float) -> None:
            pass

        def teleport(self, _pos_enu: np.ndarray) -> None:
            pass

        def latest_lidar_clouds(self):
            return {k: v.copy() for k, v in self._clouds.items()}

        def latest_camera_images(self):
            return dict(self._images)

    fake = FakeAdapter()
    bridge = Ros2Bridge(
        dt=0.05,
        scenario=sc,
        lidars=["/front_lidar", "/rear_lidar"],
        cameras=["/front_camera/image_raw", "/down_camera/image_raw"],
        adapter=fake,
    )
    bridge.reset()
    out_state, _ = bridge.step(np.array([0.0, 0.0]))

    clouds = out_state.extra["lidar_points"]
    assert set(clouds.keys()) == {"/front_lidar", "/rear_lidar"}
    assert clouds["/front_lidar"].shape == (2, 3)
    assert clouds["/rear_lidar"].shape == (1, 3)
    # Pass-through: bridge does NOT flip frames for ROS 2 (REP-103 is ENU).
    assert np.allclose(clouds["/front_lidar"][0], np.array([1.0, 0.0, 0.0]))

    cams = out_state.extra["camera_images"]
    assert cams["/front_camera/image_raw"] == b"PNG_FRONT"
    assert cams["/down_camera/image_raw"] == b"PNG_DOWN"


def test_ros2_bridge_omits_extras_when_lidars_cameras_not_configured() -> None:
    """If neither lidars nor cameras are configured the bridge must not
    poll those adapter methods — keeps the no-sensor case lightweight and
    means adapters that don't implement them still work."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.ros2_bridge import Ros2Bridge

    grid_cls = SCENARIO_REGISTRY.get("grid_world")
    sc = grid_cls.from_config(
        {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0], "obstacles": {"type": "none"}}
    )

    class MinimalAdapter:
        # Deliberately omits latest_lidar_clouds / latest_camera_images.
        def publish_velocity(self, *_args): pass
        def latest_pose_velocity(self):
            return (np.zeros(3), np.zeros(3))
        def latest_collision(self): return False
        def tick(self, _t): pass
        def teleport(self, _p): pass

    bridge = Ros2Bridge(dt=0.05, scenario=sc, adapter=MinimalAdapter())
    bridge.reset()
    out_state, _ = bridge.step(np.array([0.0, 0.0]))
    assert "lidar_points" not in out_state.extra
    assert "camera_images" not in out_state.extra


def test_ros2_bridge_sim_time_advances_state_t_via_clock_not_wall() -> None:
    """When `use_sim_time: true`, Ros2Bridge.step() should advance
    `state.t` based on the adapter's `wait_for_sim_time_advance` return
    value rather than the wall-clock dt. Confirms the bridge prefers
    the sim's `/clock` over its own wall-clock counter — the load-bearing
    behaviour for PX4-SITL fast-forward."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.ros2_bridge import Ros2Bridge

    grid_cls = SCENARIO_REGISTRY.get("grid_world")
    sc = grid_cls.from_config(
        {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0], "obstacles": {"type": "none"}}
    )

    class FakeSimTimeAdapter:
        """Mock adapter whose sim clock is driven by the bridge's wait
        calls rather than by wall-clock — `wait_for_sim_time_advance`
        instantly jumps the clock to the requested target. Lets us check
        the bridge wired sim-time through end-to-end without sleeping."""
        def __init__(self) -> None:
            self._sim_t = 0.0
            self.wait_targets: list[float] = []
            self.wait_timeouts: list[float] = []
            # Intentionally no `tick` method on this adapter so the test
            # would fail loudly if the bridge fell back to wall-clock.

        def publish_velocity(self, *_args): pass
        def latest_pose_velocity(self):
            return (np.zeros(3), np.zeros(3))
        def latest_collision(self): return False
        def teleport(self, _p): pass
        def latest_sim_time(self): return self._sim_t

        def wait_for_sim_time_advance(self, *, target_time, wall_timeout):
            self.wait_targets.append(float(target_time))
            self.wait_timeouts.append(float(wall_timeout))
            # Simulate the sim's clock jumping forward to the target.
            self._sim_t = float(target_time)
            return self._sim_t

        # reset() still needs to spin once for the first odom — provide
        # tick() but assert it's only called from reset(), never from step().
        def tick(self, _t): pass

    fake = FakeSimTimeAdapter()
    bridge = Ros2Bridge(
        dt=0.05, scenario=sc, adapter=fake,
        use_sim_time=True, sim_time_wall_timeout=2.0,
    )
    bridge.reset()
    out_state_a, _ = bridge.step(np.array([0.0, 0.0]))
    out_state_b, _ = bridge.step(np.array([0.0, 0.0]))
    # state.t tracks sim-time, not the wall-clock the test ran at.
    assert out_state_a.t == pytest.approx(0.05)
    assert out_state_b.t == pytest.approx(0.10)
    # Targets re-anchor on the previously-observed clock value, not on
    # accumulated dt — protects against drift over a long episode.
    assert fake.wait_targets == [pytest.approx(0.05), pytest.approx(0.10)]
    # The configured wall-clock safety timeout reaches the adapter.
    assert all(t == pytest.approx(2.0) for t in fake.wait_timeouts)


def test_ros2_bridge_sim_time_falls_back_to_wall_clock_for_legacy_adapters() -> None:
    """A user-supplied mock adapter that doesn't implement
    `wait_for_sim_time_advance` should keep working — bridge should
    silently fall back to wall-clock `tick()` rather than crashing.
    Lets users opt into use_sim_time without re-writing existing
    test fakes."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.ros2_bridge import Ros2Bridge

    grid_cls = SCENARIO_REGISTRY.get("grid_world")
    sc = grid_cls.from_config(
        {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0], "obstacles": {"type": "none"}}
    )

    class WallClockOnlyAdapter:
        # Note absence of wait_for_sim_time_advance / latest_sim_time.
        def __init__(self) -> None:
            self.tick_calls = 0
        def publish_velocity(self, *_args): pass
        def latest_pose_velocity(self):
            return (np.zeros(3), np.zeros(3))
        def latest_collision(self): return False
        def teleport(self, _p): pass
        def tick(self, _t): self.tick_calls += 1

    fake = WallClockOnlyAdapter()
    bridge = Ros2Bridge(dt=0.1, scenario=sc, adapter=fake, use_sim_time=True)
    bridge.reset()
    out, _ = bridge.step(np.array([0.0, 0.0]))
    # tick() was called twice (once in reset, once in step's wall fallback).
    assert fake.tick_calls == 2
    # Wall-clock fallback advances state.t by the configured dt.
    assert out.t == pytest.approx(0.1)


def test_ros2_bridge_sim_time_disabled_uses_wall_clock_dt() -> None:
    """Default mode (`use_sim_time=False`) must keep the original
    behaviour — `state.t` advances by `dt` regardless of any sim-time
    methods on the adapter. Regression guard against accidentally
    routing default runs through the sim-time path."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.ros2_bridge import Ros2Bridge

    grid_cls = SCENARIO_REGISTRY.get("grid_world")
    sc = grid_cls.from_config(
        {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0], "obstacles": {"type": "none"}}
    )

    class AmbiguousAdapter:
        # Implements *both* tick and the sim-time methods; the bridge
        # should still pick wall-clock when use_sim_time is off.
        def __init__(self) -> None:
            self.wait_calls = 0
            self.tick_calls = 0
        def publish_velocity(self, *_args): pass
        def latest_pose_velocity(self):
            return (np.zeros(3), np.zeros(3))
        def latest_collision(self): return False
        def teleport(self, _p): pass
        def tick(self, _t): self.tick_calls += 1
        def latest_sim_time(self): return 999.0
        def wait_for_sim_time_advance(self, **_kw):
            self.wait_calls += 1
            return 999.0

    fake = AmbiguousAdapter()
    bridge = Ros2Bridge(dt=0.05, scenario=sc, adapter=fake)  # use_sim_time defaults False
    bridge.reset()
    out, _ = bridge.step(np.array([0.0, 0.0]))
    assert fake.wait_calls == 0  # sim-time path NEVER taken
    assert fake.tick_calls == 2  # one in reset, one in step
    assert out.t == pytest.approx(0.05)
