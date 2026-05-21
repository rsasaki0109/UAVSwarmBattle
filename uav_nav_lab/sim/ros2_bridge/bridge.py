"""ROS 2 bridge — wires the framework's ``SimInterface`` to a ROS 2 stack.

Not exercised in CI (would need rclpy + a sim like Gazebo / Ignition /
PX4-SITL via MAVROS) but the adapter is mockable, so the publish-spin-
read plumbing is unit-tested via an injected fake adapter.

Run a real ROS 2 + sim stack publishing /odom and accepting /cmd_vel,
then:

    source /opt/ros/jazzy/setup.bash
    uav-nav run examples/exp_ros2.yaml

Contract:
  - reset(seed)        → re-seeds the scenario, optionally teleports via
                          adapter.teleport(...), spins once so the first
                          /odom message lands. Falls back to scenario.start
                          if no odom arrives within the dt window.
  - step(velocity_cmd) → publish a velocity command on cmd_topic, advance time by
                          ``dt`` (wall-clock by default; sim-time if
                          ``use_sim_time`` is enabled — see below), then
                          read latest pose/velocity back. Collision flag
                          comes from /collision (or False if the topic is
                          unconfigured). Optional LiDAR / camera readouts
                          populate ``state.extra`` mirroring the AirSim
                          bridge — see below.
  - state              → ENU pose / velocity from the latest odom message.
  - obstacle_map       → comes from the scenario; this bridge does not
                          ingest a ROS occupancy grid.

Coordinate frames:
  - Framework: ENU (east-north-up, +z up).
  - ROS 2:     ENU per REP-103 by default. Set ``frame: ned`` for wrappers
    that publish NED odometry / consume NED velocity commands, such as
    AirSim's default ROS wrapper topics.

Topology:
  - publish:   cmd_topic        geometry_msgs/Twist or airsim_interfaces/VelCmd
  - subscribe: odom_topic       nav_msgs/Odometry      (true pose+velocity)
  - subscribe: collision_topic  std_msgs/Bool          (optional)
  - subscribe: lidars[*]        sensor_msgs/PointCloud2 (optional)
  - subscribe: cameras[*]       sensor_msgs/Image       (optional)
  - subscribe: clock_topic      rosgraph_msgs/Clock    (only if use_sim_time)

Sim-time:
  - Default (use_sim_time=False) — each step() does one ``spin_once`` with
    a wall-clock timeout of ``dt``. Fine for real-time sims (Gazebo at
    rate 1×, real robots) but coupled to wall-clock — PX4-SITL
    fast-forward at 8× wall-clock would still tick the bridge at
    wall-clock dt, defeating the point.
  - use_sim_time=True — bridge spins until ``/clock`` has advanced by
    ``dt`` of sim-time, with a wall-clock safety timeout so a paused or
    crashed sim does not deadlock the runner. The runner's own
    ``state.t`` then tracks sim-time and PX4-SITL fast-forward speeds the
    experiment up by the sim's accelaration factor.

LiDAR / camera readouts mirror the AirSim bridge so the same
``pointcloud_occupancy`` sensor and ``uav-nav video`` CLI work
transparently across backends:
  - state.extra["lidar_points"][topic]  = (N, 3) ENU point cloud
  - state.extra["camera_images"][topic] = compressed PNG bytes

The adapter abstraction (``_Ros2Adapter`` duck-typed surface) keeps the
bridge testable without rclpy installed. :class:`_RclpyAdapter` (in
:mod:`.adapter`) is the production implementation; tests inject a fake
adapter that records publishes and returns canned odom / lidar /
camera data.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..base import SIM_REGISTRY, SimInterface, SimState, SimStepInfo
from .adapter import _RclpyAdapter
from .coords import _enu_to_ned, _ned_to_enu


@SIM_REGISTRY.register("ros2")
class Ros2Bridge(SimInterface):
    def __init__(
        self,
        dt: float,
        scenario: Any,
        cmd_topic: str = "/cmd_vel",
        odom_topic: str = "/odom",
        collision_topic: str | None = None,
        goal_radius: float = 1.5,
        max_steps: int = 2000,
        lidars: list[str] | None = None,
        cameras: list[str] | None = None,
        frame: str = "enu",
        cmd_msg_type: str = "twist",
        use_sim_time: bool = False,
        clock_topic: str = "/clock",
        sim_time_wall_timeout: float = 5.0,
        adapter: Any = None,
    ) -> None:
        self.dt = float(dt)
        self.scenario = scenario
        self.cmd_topic = cmd_topic
        self.odom_topic = odom_topic
        self.collision_topic = collision_topic
        self.goal_radius = float(goal_radius)
        self.max_steps = int(max_steps)
        # PointCloud2 topics to subscribe to. Each step's latest message
        # lands at state.extra["lidar_points"][topic] as (N, 3) ENU points.
        self.lidars: list[str] = list(lidars or [])
        # sensor_msgs/Image topics. Each step's latest frame is encoded
        # to PNG bytes and stashed at state.extra["camera_images"][topic].
        self.cameras: list[str] = list(cameras or [])
        self.frame = str(frame).lower()
        if self.frame not in ("enu", "ned"):
            raise ValueError("ros2 simulator frame must be 'enu' or 'ned'")
        self.cmd_msg_type = str(cmd_msg_type).lower()
        if self.cmd_msg_type not in ("twist", "airsim_vel_cmd"):
            raise ValueError("ros2 cmd_msg_type must be 'twist' or 'airsim_vel_cmd'")
        # When True, the bridge advances time by waiting for `/clock` to
        # advance by `dt` rather than ticking wall-clock. This lets
        # PX4-SITL fast-forward (and Gazebo `--lockstep`) speed the
        # experiment up by the same factor as the sim.
        self.use_sim_time = bool(use_sim_time)
        self.clock_topic = str(clock_topic)
        # Hard upper bound (wall-clock) on each sim-time wait — protects
        # the runner from deadlocking if the sim pauses or crashes.
        self.sim_time_wall_timeout = float(sim_time_wall_timeout)
        # `adapter` lets tests inject a fake; in production the real
        # rclpy-backed adapter is created lazily on first reset/step.
        self._adapter: Any = adapter
        self._state: SimState | None = None
        self._step_count = 0
        # Sim-time anchor: the `/clock` value observed at reset(); each
        # step() waits for clock to advance by another dt past the last
        # anchor so jitter in the sim's clock publish rate doesn't drift.
        self._sim_time_anchor: float | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], scenario: Any) -> "Ros2Bridge":
        return cls(
            dt=float(cfg.get("dt", 0.05)),
            scenario=scenario,
            cmd_topic=str(cfg.get("cmd_topic", "/cmd_vel")),
            odom_topic=str(cfg.get("odom_topic", "/odom")),
            collision_topic=cfg.get("collision_topic"),
            goal_radius=float(cfg.get("goal_radius", 1.5)),
            max_steps=int(cfg.get("max_steps", 2000)),
            lidars=[str(t) for t in (cfg.get("lidars") or [])],
            cameras=[str(t) for t in (cfg.get("cameras") or [])],
            frame=str(cfg.get("frame", "enu")),
            cmd_msg_type=str(cfg.get("cmd_msg_type", "twist")),
            use_sim_time=bool(cfg.get("use_sim_time", False)),
            clock_topic=str(cfg.get("clock_topic", "/clock")),
            sim_time_wall_timeout=float(cfg.get("sim_time_wall_timeout", 5.0)),
        )

    def _ensure_adapter(self) -> Any:
        if self._adapter is not None:
            return self._adapter
        self._adapter = _RclpyAdapter(
            cmd_topic=self.cmd_topic,
            odom_topic=self.odom_topic,
            collision_topic=self.collision_topic,
            lidar_topics=self.lidars,
            camera_topics=self.cameras,
            clock_topic=self.clock_topic if self.use_sim_time else None,
            cmd_msg_type=self.cmd_msg_type,
        )
        return self._adapter

    def _from_ros_frame(self, vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=float)
        return _ned_to_enu(v) if self.frame == "ned" else v

    def _to_ros_frame(self, vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=float)
        return _enu_to_ned(v) if self.frame == "ned" else v

    def _advance_time(self, adapter: Any) -> float:
        """Wait one `dt` of time and return the actual sim-time observed.

        Wall-clock mode: single `spin_once(timeout_sec=dt)`; returns dt.
        Sim-time mode: ask the adapter to spin until `/clock` advances by
        dt past the last anchor (protected by `sim_time_wall_timeout`),
        and return the actual advance — which may be slightly larger
        than dt depending on the sim's clock publish granularity."""
        if not self.use_sim_time:
            adapter.tick(self.dt)
            return self.dt

        wait = getattr(adapter, "wait_for_sim_time_advance", None)
        if not callable(wait):
            # Adapter does not implement sim-time waits — fall back to
            # wall-clock so the bridge still progresses. Real rclpy
            # adapters always implement it; this branch is for legacy
            # mock adapters in tests.
            adapter.tick(self.dt)
            return self.dt

        # First step after reset: anchor on the current sim clock.
        if self._sim_time_anchor is None:
            self._sim_time_anchor = float(adapter.latest_sim_time() or 0.0)

        target = self._sim_time_anchor + self.dt
        actual = float(wait(target_time=target, wall_timeout=self.sim_time_wall_timeout))
        # Re-anchor on the *actual* time we ended up at, so jitter in
        # one step doesn't compound across a long episode.
        advance = max(0.0, actual - self._sim_time_anchor)
        self._sim_time_anchor = actual
        return advance

    def reset(
        self,
        *,
        seed: int | None = None,
        initial_position: np.ndarray | None = None,
    ) -> SimState:
        adapter = self._ensure_adapter()
        if seed is not None:
            self.scenario.reseed(seed)
        if initial_position is not None:
            start = np.asarray(initial_position, dtype=float)
        else:
            start = np.asarray(self.scenario.start, dtype=float)
        ndim = self.scenario.ndim
        # Optional teleport (Gazebo set_entity_state, Ignition /world/.../set_pose).
        # Silently no-op if the adapter does not implement it.
        teleport = getattr(adapter, "teleport", None)
        if callable(teleport):
            pos3 = np.zeros(3)
            pos3[:ndim] = start[:ndim]
            teleport(self._to_ros_frame(pos3))
        # Reset the sim-time anchor — set on the first step() rather than
        # here so any odom arriving during the wall-clock spin below
        # doesn't get attributed to a sim-time advance.
        self._sim_time_anchor = None
        # Spin briefly (wall-clock — sim-time mode kicks in from step())
        # so the first odom arrives. Fall back to scenario.start if
        # nothing lands within the dt window — the run can still progress.
        adapter.tick(self.dt)
        latest = adapter.latest_pose_velocity()
        if latest is not None:
            pos_enu, vel_enu = (self._from_ros_frame(v) for v in latest)
            self._state = SimState(
                t=0.0,
                position=np.asarray(pos_enu, dtype=float)[:ndim].copy(),
                velocity=np.asarray(vel_enu, dtype=float)[:ndim].copy(),
            )
        else:
            self._state = SimState(
                t=0.0, position=start[:ndim].copy(), velocity=np.zeros(ndim)
            )
        self._step_count = 0
        return self._state.copy()

    def step(self, command: np.ndarray) -> tuple[SimState, SimStepInfo]:
        assert self._state is not None, "call reset() first"
        adapter = self._ensure_adapter()
        v = np.asarray(command, dtype=float)
        v3 = np.zeros(3)
        v3[: min(3, v.size)] = v[:3]  # 2D scenarios pad vz=0
        ros_v = self._to_ros_frame(v3)
        adapter.publish_velocity(float(ros_v[0]), float(ros_v[1]), float(ros_v[2]))
        # Wall-clock vs sim-time: `_advance_time` returns the elapsed time
        # of this step. In sim-time mode the value tracks `/clock` exactly
        # so a fast-forwarded sim makes `state.t` advance at the sim's
        # clock rate, not the runner's wall-clock rate.
        elapsed = self._advance_time(adapter)
        latest = adapter.latest_pose_velocity()
        ndim = self.scenario.ndim
        if latest is not None:
            pos_enu, vel_enu = (self._from_ros_frame(v) for v in latest)
            self._state.position = np.asarray(pos_enu, dtype=float)[:ndim].copy()
            self._state.velocity = np.asarray(vel_enu, dtype=float)[:ndim].copy()
        # else: keep previous state (sensor dropout); planner replan handles it.
        self._state.t += elapsed
        self._step_count += 1
        # Mirror AirSim bridge: surface latest sensor side-channel data
        # under the same state.extra keys so consumers (pointcloud_occupancy
        # sensor, uav-nav video CLI) work transparently across backends.
        if self.lidars:
            clouds = adapter.latest_lidar_clouds() if hasattr(adapter, "latest_lidar_clouds") else {}
            if clouds:
                self._state.extra["lidar_points"] = dict(clouds)
        if self.cameras:
            imgs = adapter.latest_camera_images() if hasattr(adapter, "latest_camera_images") else {}
            if imgs:
                self._state.extra["camera_images"] = dict(imgs)
        collision = bool(adapter.latest_collision())
        goal = np.asarray(self.scenario.goal, dtype=float)
        goal_reached = bool(
            np.linalg.norm(self._state.position - goal[:ndim]) <= self.goal_radius
        )
        truncated = self._step_count >= self.max_steps
        return self._state.copy(), SimStepInfo(
            collision=collision, goal_reached=goal_reached, truncated=truncated
        )

    @property
    def state(self) -> SimState:
        assert self._state is not None
        return self._state.copy()

    @property
    def goal(self) -> np.ndarray:
        return np.asarray(self.scenario.goal, dtype=float)

    @property
    def obstacle_map(self) -> Any:
        return self.scenario.occupancy
