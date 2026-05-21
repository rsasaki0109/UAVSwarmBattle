"""AirSimBridge — wires the framework's `SimInterface` to Microsoft AirSim.

Not exercised in CI (would need an AirSim server) but the AirSim Python
client is mockable, so the logic that converts between AirSim's NED
convention and the framework's ENU is unit-tested via the
``coords`` helpers and the bridge's instance methods against an
injected fake client.

Run a real AirSim instance, then::

    pip install airsim
    uav-nav run examples/exp_airsim.yaml

Contract:
  - reset(seed)        → resets the AirSim world, teleports to start (in NED)
  - step(velocity_cmd) → simPause → moveByVelocity for `dt` → simContinueForTime
                          → read kinematics back. The pause/continue dance
                          is what gives the experiment runner deterministic
                          fast-forward instead of real-time wall clock.
  - state              → ENU pose / velocity converted from NED kinematics
  - obstacle_map       → comes from the scenario; AirSim has no occupancy grid

Coordinate frames:
  - Framework: ENU (east-north-up, +z up).
  - AirSim:    NED (north-east-down, +z down).
  - We map (x, y, z)_ENU = (y, x, -z)_NED.

LiDAR / camera / depth configuration: see the package docstring in
``__init__``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..base import SIM_REGISTRY, SimInterface, SimState, SimStepInfo
from . import obstacles as _obstacles
from . import sensors as _sensors
from .coords import _enu_to_ned, _ned_pointcloud_to_enu, _ned_to_enu


@SIM_REGISTRY.register("airsim")
class AirSimBridge(SimInterface):
    def __init__(
        self,
        dt: float,
        scenario: Any,
        host: str = "127.0.0.1",
        port: int = 41451,
        vehicle: str = "Drone1",
        goal_radius: float = 1.5,
        max_steps: int = 2000,
        lidars: list[str] | None = None,
        cameras: list[Mapping[str, Any]] | None = None,
        depths: list[Mapping[str, Any]] | None = None,
        static_obstacles: list[Mapping[str, Any]] | None = None,
        settle_after_reset: float = 0.0,
        settle_after_teleport: float = 0.0,
        client: Any = None,
        wind: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.dt = float(dt)
        self.scenario = scenario
        self.host = host
        self.port = port
        self.vehicle = vehicle
        self.goal_radius = float(goal_radius)
        self.max_steps = int(max_steps)
        self.wind = tuple(float(v) for v in wind)
        self.lidars: list[str] = list(lidars or [])
        self.cameras: list[dict[str, str]] = [
            {"name": str(c["name"]), "image_type": str(c.get("image_type", "scene"))}
            for c in (cameras or [])
        ]
        self.depths: list[dict[str, Any]] = [
            {
                "name": str(d["name"]),
                "image_type": str(d.get("image_type", "depth_planar")),
                "fov_deg": float(d.get("fov_deg", 90.0)),
                "width": int(d.get("width", 256)),
                "height": int(d.get("height", 144)),
            }
            for d in (depths or [])
        ]
        self.static_obstacles = [
            _obstacles.normalise_static_obstacle(i, dict(spec))
            for i, spec in enumerate(static_obstacles or [])
        ]
        self.settle_after_reset = float(settle_after_reset)
        self.settle_after_teleport = float(settle_after_teleport)
        self._client: Any = client
        self._state: SimState | None = None
        self._step_count = 0
        # Multi-drone runner sets this to False on every bridge except
        # sim 0. AirSim has a single shared physics clock — only one
        # bridge per global tick should call simContinueForTime, or the
        # world advances N×dt instead of dt.
        self._advance_scenario: bool = True

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], scenario: Any) -> "AirSimBridge":
        lidars_cfg = cfg.get("lidars", []) or []
        cameras_cfg = cfg.get("cameras", []) or []
        depths_cfg = cfg.get("depths", []) or []
        static_obstacles_cfg = cfg.get("static_obstacles", []) or []
        wind_raw = cfg.get("wind", ()) or ()
        wind_tuple = (
            (float(wind_raw[0]), float(wind_raw[1]), float(wind_raw[2]))
            if len(wind_raw) >= 2
            else (0.0, 0.0, 0.0)
        )
        return cls(
            dt=float(cfg.get("dt", 0.05)),
            scenario=scenario,
            host=str(cfg.get("host", "127.0.0.1")),
            port=int(cfg.get("port", 41451)),
            vehicle=str(cfg.get("vehicle", "Drone1")),
            goal_radius=float(cfg.get("goal_radius", 1.5)),
            max_steps=int(cfg.get("max_steps", 2000)),
            lidars=[str(name) for name in lidars_cfg],
            cameras=list(cameras_cfg),
            depths=list(depths_cfg),
            static_obstacles=list(static_obstacles_cfg),
            settle_after_reset=float(cfg.get("settle_after_reset", 0.0)),
            settle_after_teleport=float(cfg.get("settle_after_teleport", 0.0)),
            wind=wind_tuple,
        )

    # ---- Compatibility shims --------------------------------------------------
    # These preserve the pre-split external surface: existing callers (tests,
    # debugging code) hit the class-level helpers. New code should reach for
    # the free functions in ``obstacles`` / ``sensors`` directly.

    @staticmethod
    def _normalise_static_obstacle(idx: int, spec: dict[str, Any]) -> dict[str, Any]:
        return _obstacles.normalise_static_obstacle(idx, spec)

    @staticmethod
    def _intrinsics_from_fov(fov_deg: float, width: int, height: int) -> dict[str, float]:
        return _sensors.intrinsics_from_fov(fov_deg, width, height)

    def _build_image_requests(self) -> list[Any]:
        return _sensors.build_image_requests(self.cameras)

    def _build_depth_requests(self) -> list[Any]:
        return _sensors.build_depth_requests(self.depths)

    # ---- Client lifecycle ----------------------------------------------------

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import airsim  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise SystemExit(
                "airsim package is not installed. Install with `pip install airsim` "
                "and start an AirSim server before running this experiment."
            ) from e
        self._client = airsim.MultirotorClient(ip=self.host, port=self.port)
        self._client.confirmConnection()
        self._client.enableApiControl(True, self.vehicle)
        self._client.armDisarm(True, self.vehicle)
        return self._client

    # ---- Reset / step --------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        initial_position: np.ndarray | None = None,
    ) -> SimState:
        client = self._ensure_client()
        if seed is not None:
            self.scenario.reseed(seed)
        # `client.reset()` is global in AirSim — only the master bridge
        # (sim 0) should call it; passive sims would clobber sim 0's
        # already-teleported drone every time they reset.
        if self._advance_scenario:
            client.reset()
            # Pause IMMEDIATELY after reset so physics cannot tick while
            # drones are still at their settings.json spawn pose. Without
            # this, AirSim runs physics for ``settle_after_reset`` seconds
            # with the drones on the ground, which registers a ground
            # collision that survives the teleport.
            if hasattr(client, "simPause"):
                client.simPause(True)
            # Set global wind via API (settings.json Wind not supported by
            # all AirSim builds). ENU → NED.
            try:
                import airsim  # type: ignore[import-not-found]
                w = self.wind
                wind_ned = airsim.Vector3r(
                    float(w[1]), float(w[0]), float(-w[2]) if len(w) > 2 else 0.0
                )
                client.simSetWind(wind_ned)
            except Exception:
                pass
            _obstacles.sync_static_obstacles(client, self.static_obstacles)
            self._dyn_obstacle_names = _obstacles.sync_dynamic_obstacles_initial(
                client, self.scenario
            )
            if hasattr(client, "simPause"):
                import time as _time
                _time.sleep(self.settle_after_reset)
        client.enableApiControl(True, self.vehicle)
        client.armDisarm(True, self.vehicle)
        if initial_position is not None:
            start = np.asarray(initial_position, dtype=float)
        else:
            start = np.asarray(self.scenario.start, dtype=float)
        ned_start = _enu_to_ned(start)
        if hasattr(client, "simSetVehiclePose"):
            try:
                import airsim  # type: ignore[import-not-found]
                pose = airsim.Pose(
                    airsim.Vector3r(
                        float(ned_start[0]), float(ned_start[1]), float(ned_start[2])
                    ),
                    airsim.to_quaternion(0.0, 0.0, 0.0),
                )
                client.simSetVehiclePose(
                    pose, ignore_collision=True, vehicle_name=self.vehicle
                )
                if hasattr(client, "simPause"):
                    import time as _time
                    _time.sleep(self.settle_after_teleport)
            except ImportError:  # pragma: no cover
                pass
        # Pause AirSim before returning so the drone holds its teleported
        # pose during the (possibly multi-second) first replan. step()
        # will simPause(False) → moveByVelocity → simContinueForTime(dt)
        # → simPause(True). Without this pause, an armed multirotor at
        # altitude can drift / fall during long planner waits and trigger
        # a t=0 collision.
        if hasattr(client, "simPause"):
            client.simPause(True)
        ndim = self.scenario.ndim
        self._state = SimState(
            t=0.0, position=start[:ndim].copy(), velocity=np.zeros(ndim)
        )
        self._step_count = 0
        return self._state.copy()

    def step_command(self, command: np.ndarray) -> None:
        """Queue a velocity command. If master, also handle
        simPause(False) → moveByVelocityAsync → simContinueForTime(dt) →
        simPause(True). State is NOT read back — call
        :meth:`step_readback` after *all* bridges have issued their
        commands so readbacks see the fully-advanced physics tick.
        """
        assert self._state is not None, "call reset() first"
        client = self._ensure_client()
        v = np.asarray(command, dtype=float)
        v3 = np.zeros(3)
        v3[: min(3, v.size)] = v[:3]
        v_ned = _enu_to_ned(v3)
        if self._advance_scenario and hasattr(client, "simPause"):
            client.simPause(False)
        _future = client.moveByVelocityAsync(
            float(v_ned[0]),
            float(v_ned[1]),
            float(v_ned[2]),
            self.dt,
            vehicle_name=self.vehicle,
        )
        if self._advance_scenario:
            # Advance the scenario's authoritative clock (dynamic obstacle
            # positions) and mirror to AirSim so collision detection sees
            # the cubes at their post-tick positions.
            if hasattr(self.scenario, "advance"):
                self.scenario.advance(self.dt)
            _obstacles.update_dynamic_obstacle_poses(
                client, getattr(self, "_dyn_obstacle_names", []), self.scenario
            )
            if hasattr(client, "simContinueForTime"):
                client.simContinueForTime(self.dt)
            elif hasattr(client, "simPause"):  # pragma: no cover
                client.simPause(True)
            if hasattr(client, "simPause"):
                client.simPause(True)

    def step_readback(self) -> tuple[SimState, SimStepInfo]:
        """Read kinematics, sensors and collision after the physics tick."""
        client = self._ensure_client()
        kin = client.getMultirotorState(
            vehicle_name=self.vehicle
        ).kinematics_estimated
        pos_ned = np.array(
            [kin.position.x_val, kin.position.y_val, kin.position.z_val]
        )
        vel_ned = np.array(
            [
                kin.linear_velocity.x_val,
                kin.linear_velocity.y_val,
                kin.linear_velocity.z_val,
            ]
        )
        pos_enu = _ned_to_enu(pos_ned)
        vel_enu = _ned_to_enu(vel_ned)
        ndim = self.scenario.ndim
        self._state.position = pos_enu[:ndim]
        self._state.velocity = vel_enu[:ndim]
        self._state.t += self.dt
        self._step_count += 1
        if self.lidars:
            self._state.extra["lidar_points"] = {
                name: _ned_pointcloud_to_enu(
                    client.getLidarData(name, vehicle_name=self.vehicle).point_cloud
                )
                for name in self.lidars
            }
        if self.cameras:
            requests = _sensors.build_image_requests(self.cameras)
            responses = client.simGetImages(requests, vehicle_name=self.vehicle)
            self._state.extra["camera_images"] = {
                spec["name"]: bytes(getattr(resp, "image_data_uint8", b"") or b"")
                for spec, resp in zip(self.cameras, responses)
            }
        if self.depths:
            depth_requests = _sensors.build_depth_requests(self.depths)
            depth_responses = client.simGetImages(
                depth_requests, vehicle_name=self.vehicle
            )
            depth_bag: dict[str, dict[str, Any]] = {}
            for spec, resp in zip(self.depths, depth_responses):
                floats = getattr(resp, "image_data_float", None)
                if not floats:
                    continue
                arr = np.asarray(list(floats), dtype=np.float32).reshape(
                    spec["height"], spec["width"]
                )
                depth_bag[spec["name"]] = {
                    "depth": arr,
                    "intrinsics": _sensors.intrinsics_from_fov(
                        spec["fov_deg"], spec["width"], spec["height"]
                    ),
                }
            if depth_bag:
                self._state.extra["depth_images"] = depth_bag
        ci = client.simGetCollisionInfo(vehicle_name=self.vehicle)
        collision = bool(ci.has_collided)
        if collision:
            self._state.extra["collision_object"] = str(
                getattr(ci, "object_name", "") or ""
            )
        else:
            self._state.extra.pop("collision_object", None)
        goal = (
            self.scenario.goal
            if not hasattr(self, "_goal_override") or self._goal_override is None
            else self._goal_override
        )
        goal_reached = bool(
            np.linalg.norm(self._state.position - goal[:ndim]) <= self.goal_radius
        )
        truncated = self._step_count >= self.max_steps
        return self._state.copy(), SimStepInfo(
            collision=collision, goal_reached=goal_reached, truncated=truncated
        )

    def step(self, command: np.ndarray) -> tuple[SimState, SimStepInfo]:
        self.step_command(command)
        return self.step_readback()

    # ---- Accessors -----------------------------------------------------------

    @property
    def state(self) -> SimState:
        assert self._state is not None
        return self._state.copy()

    @property
    def goal(self) -> np.ndarray:
        if getattr(self, "_goal_override", None) is not None:
            return np.asarray(self._goal_override, dtype=float)
        return np.asarray(self.scenario.goal, dtype=float)

    def set_goal(self, goal: np.ndarray) -> None:
        """Override the goal used by the goal-reached check (multi-drone)."""
        self._goal_override = np.asarray(goal, dtype=float).reshape(
            self.scenario.ndim
        )

    @property
    def obstacle_map(self) -> Any:
        return self.scenario.occupancy
