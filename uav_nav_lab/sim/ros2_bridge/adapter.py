"""Production rclpy adapter — owns the ROS node + subs + pub.

Lazy-imports ``rclpy`` so :mod:`uav_nav_lab.sim.ros2_bridge` loads
cleanly without ROS sourced (CI / lightweight environments). Tests
inject a fake adapter that mimics the duck-typed surface used by
:class:`uav_nav_lab.sim.ros2_bridge.bridge.Ros2Bridge`, so this module
is only exercised end-to-end in a real ROS 2 environment — hence the
``pragma: no cover`` on the class body.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .messages import _decode_pointcloud2, _encode_image_to_png


class _RclpyAdapter:  # pragma: no cover
    """Production adapter — owns an rclpy node, a Twist publisher, and
    Odometry / Bool / PointCloud2 / Image subscriptions. Lazy-imports
    rclpy + PIL so the bridge module imports cleanly without ROS sourced
    or PIL installed."""

    def __init__(
        self,
        cmd_topic: str,
        odom_topic: str,
        collision_topic: str | None,
        lidar_topics: list[str] | None = None,
        camera_topics: list[str] | None = None,
        clock_topic: str | None = None,
        cmd_msg_type: str = "twist",
    ) -> None:
        try:
            import rclpy  # type: ignore[import-not-found]
            from geometry_msgs.msg import Twist  # type: ignore[import-not-found]
            from nav_msgs.msg import Odometry  # type: ignore[import-not-found]
            from std_msgs.msg import Bool  # type: ignore[import-not-found]
            from sensor_msgs.msg import Image, PointCloud2  # type: ignore[import-not-found]
            from rosgraph_msgs.msg import Clock  # type: ignore[import-not-found]
            from rclpy.qos import QoSPresetProfiles  # type: ignore[import-not-found]
        except ImportError as e:
            raise SystemExit(
                "rclpy is not on PYTHONPATH. Source ROS 2 (e.g. "
                "`source /opt/ros/jazzy/setup.bash`) before running this experiment."
            ) from e

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self._cmd_msg_type = str(cmd_msg_type).lower()
        if self._cmd_msg_type == "twist":
            self._CmdMsg = Twist
        elif self._cmd_msg_type == "airsim_vel_cmd":
            try:
                from airsim_interfaces.msg import VelCmd  # type: ignore[import-not-found]
            except ImportError:
                try:
                    from airsim_ros_pkgs.msg import VelCmd  # type: ignore[import-not-found]
                except ImportError as e:
                    raise SystemExit(
                        "AirSim ROS2 VelCmd message is not on PYTHONPATH. "
                        "Source the AirSim ROS2 workspace before running this experiment."
                    ) from e
            self._CmdMsg = VelCmd
        else:
            raise ValueError("cmd_msg_type must be 'twist' or 'airsim_vel_cmd'")
        self._node = rclpy.create_node("uav_nav_lab_ros2_bridge")
        self._cmd_pub = self._node.create_publisher(self._CmdMsg, cmd_topic, 10)
        self._latest_odom: Any = None
        self._latest_collision: bool = False
        self._latest_clouds: dict[str, np.ndarray] = {}
        self._latest_images: dict[str, bytes] = {}
        self._latest_sim_time: float | None = None
        # SENSOR_DATA QoS: best-effort + small depth, matches typical odom /
        # lidar / camera publishers (Gazebo, ardupilot_gz, PX4-SITL via MAVROS).
        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        self._node.create_subscription(Odometry, odom_topic, self._on_odom, sensor_qos)
        if collision_topic is not None:
            self._node.create_subscription(Bool, collision_topic, self._on_collision, 10)
        for topic in (lidar_topics or []):
            self._node.create_subscription(
                PointCloud2, topic,
                lambda msg, t=topic: self._on_pointcloud(t, msg),
                sensor_qos,
            )
        for topic in (camera_topics or []):
            self._node.create_subscription(
                Image, topic,
                lambda msg, t=topic: self._on_image(t, msg),
                sensor_qos,
            )
        if clock_topic is not None:
            # `/clock` uses RELIABLE QoS by ROS 2 convention; SENSOR_DATA
            # would silently drop messages and break sim-time anchoring.
            self._node.create_subscription(Clock, clock_topic, self._on_clock, 10)

    def _on_odom(self, msg: Any) -> None:
        self._latest_odom = msg

    def _on_collision(self, msg: Any) -> None:
        self._latest_collision = bool(msg.data)

    def _on_pointcloud(self, topic: str, msg: Any) -> None:
        self._latest_clouds[topic] = _decode_pointcloud2(msg)

    def _on_image(self, topic: str, msg: Any) -> None:
        self._latest_images[topic] = _encode_image_to_png(msg)

    def _on_clock(self, msg: Any) -> None:
        # rosgraph_msgs/Clock has a single field `clock` of type
        # builtin_interfaces/Time { sec, nanosec }.
        self._latest_sim_time = float(msg.clock.sec) + float(msg.clock.nanosec) * 1e-9

    def publish_velocity(self, vx: float, vy: float, vz: float) -> None:
        msg = self._CmdMsg()
        target = msg.twist if self._cmd_msg_type == "airsim_vel_cmd" else msg
        target.linear.x = float(vx)
        target.linear.y = float(vy)
        target.linear.z = float(vz)
        self._cmd_pub.publish(msg)

    def latest_pose_velocity(self) -> tuple[np.ndarray, np.ndarray] | None:
        msg = self._latest_odom
        if msg is None:
            return None
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        return (
            np.array([p.x, p.y, p.z]),
            np.array([v.x, v.y, v.z]),
        )

    def latest_collision(self) -> bool:
        return self._latest_collision

    def latest_lidar_clouds(self) -> dict[str, np.ndarray]:
        return dict(self._latest_clouds)

    def latest_camera_images(self) -> dict[str, bytes]:
        return dict(self._latest_images)

    def tick(self, timeout_s: float) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=float(timeout_s))

    def latest_sim_time(self) -> float | None:
        return self._latest_sim_time

    def wait_for_sim_time_advance(
        self, *, target_time: float, wall_timeout: float
    ) -> float:
        """Spin until `/clock` reaches `target_time` or `wall_timeout`
        wall-clock seconds elapse, whichever comes first. Returns the
        actual sim-time observed at exit (may exceed `target_time` if
        the clock publish granularity is coarse, or fall short of it on
        timeout). The bridge re-anchors on the returned value, so a
        short overshoot does not compound across the episode."""
        import time

        start = time.monotonic()
        # Use a small per-spin slice so a paused sim still notices wall_timeout.
        slice_s = min(0.05, max(1e-3, wall_timeout / 100.0))
        while True:
            cur = self._latest_sim_time
            if cur is not None and cur >= target_time:
                return float(cur)
            if (time.monotonic() - start) >= wall_timeout:
                return float(cur if cur is not None else target_time)
            self._rclpy.spin_once(self._node, timeout_sec=slice_s)

    def teleport(self, pos_enu: np.ndarray) -> None:  # noqa: ARG002
        # Sim-specific service call (Gazebo set_entity_state, Ignition
        # /world/.../set_pose). Left as a no-op in the generic adapter;
        # subclass and override if you need it.
        return None
