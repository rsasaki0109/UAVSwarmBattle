"""Multi-drone aerobatic / formation choreography scenario.

Each drone follows an analytical reference trajectory parameterised by
time. The §3 N=4 cross-pattern scenario has each drone with a static
goal; this one extends multi_drone_voxel with a *time-varying* goal that
the runner re-queries each replan.

Initial patterns:
- ``synchronized_loop``: all N drones share one circular loop in a
  configurable plane (xz / xy), phase-offset by 360°/N. Iconic
  "cycle of death" air-show pattern. Tests whether the planners can
  maintain phase offset under peer prediction.

Static and dynamic obstacle handling is inherited from
``MultiDroneVoxelScenario``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import SCENARIO_REGISTRY
from .multi_drone_voxel import DroneSpec3D, MultiDroneVoxelScenario
from .voxel_world import _DynamicObstacle3D, _ObstacleSpec


@SCENARIO_REGISTRY.register("multi_drone_aerobatic")
class MultiDroneAerobaticScenario(MultiDroneVoxelScenario):
    """Synchronised choreography scenario.

    Drones have *no* free start/goal pair — instead each drone's
    position at any time t is derived from a parametric trajectory and
    a per-drone phase offset. The runner queries
    :meth:`dynamic_goal_at` each replan to update the planner's target.

    The episode is terminated by max_steps (not goal-reach); evaluation
    reports trajectory tracking error rather than binary success.
    """

    def __init__(
        self,
        size: tuple[int, int, int],
        n_drones: int,
        obstacles: _ObstacleSpec,
        *,
        pattern: str = "synchronized_loop",
        center: tuple[float, float, float] = (20.0, 20.0, 7.0),
        radius: float = 4.0,
        period: float = 8.0,
        n_loops: int = 2,
        normal_axis: str = "y",
        lookahead_t: float = 0.4,
        resolution: float = 1.0,
        dynamic_obstacles: list[_DynamicObstacle3D] | None = None,
        drone_radius: float = 0.4,
    ) -> None:
        if pattern != "synchronized_loop":
            raise ValueError(f"unknown aerobatic pattern: {pattern!r}")
        if n_drones < 1:
            raise ValueError("multi_drone_aerobatic needs at least one drone")
        self.pattern = pattern
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.period = float(period)
        if self.period <= 0:
            raise ValueError(f"period must be > 0; got {self.period!r}")
        self.omega = 2.0 * np.pi / self.period
        self.n_loops = int(n_loops)
        self.normal_axis = str(normal_axis)
        if self.normal_axis not in ("y", "z"):
            raise ValueError(f"normal_axis must be 'y' or 'z'; got {normal_axis!r}")
        self.lookahead_t = float(lookahead_t)
        self.phases = [
            2.0 * np.pi * i / max(1, n_drones) for i in range(n_drones)
        ]
        # Derive per-drone start (t=0) and goal (t = n_loops * period) from
        # the trajectory. The "goal" is what the runner uses for the
        # initial planner.plan() call; subsequent replans pick up the
        # dynamic goal via dynamic_goal_at().
        drones = [
            DroneSpec3D(
                start=self.reference_position(i, 0.0),
                goal=self.reference_position(i, self.n_loops * self.period),
                radius=float(drone_radius),
                name=f"drone_{i}",
            )
            for i in range(n_drones)
        ]
        super().__init__(
            size=size,
            drones=drones,
            obstacles=obstacles,
            resolution=resolution,
            dynamic_obstacles=dynamic_obstacles,
        )

    def reference_position(self, drone_idx: int, t: float) -> np.ndarray:
        """Analytical reference position for drone *i* at time *t*."""
        phi = self.phases[drone_idx]
        cx, cy, cz = float(self.center[0]), float(self.center[1]), float(self.center[2])
        angle = self.omega * t + phi
        if self.normal_axis == "y":
            # Vertical loop in xz plane
            x = cx + self.radius * np.cos(angle)
            y = cy
            z = cz + self.radius * np.sin(angle)
        else:  # "z"
            # Horizontal loop in xy plane
            x = cx + self.radius * np.cos(angle)
            y = cy + self.radius * np.sin(angle)
            z = cz
        return np.array([x, y, z], dtype=float)

    def reference_velocity(self, drone_idx: int, t: float) -> np.ndarray:
        """Reference tangent velocity at time *t* (m/s)."""
        phi = self.phases[drone_idx]
        angle = self.omega * t + phi
        v = self.radius * self.omega
        if self.normal_axis == "y":
            return np.array([-v * np.sin(angle), 0.0, v * np.cos(angle)], dtype=float)
        return np.array([-v * np.sin(angle), v * np.cos(angle), 0.0], dtype=float)

    def dynamic_goal_at(self, drone_idx: int, t: float) -> np.ndarray:
        """Goal the planner should pursue at time *t*: a lookahead point on
        the reference trajectory ``lookahead_t`` seconds in the future."""
        return self.reference_position(drone_idx, t + self.lookahead_t)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "MultiDroneAerobaticScenario":
        size = tuple(cfg.get("size", (40, 40, 12)))
        if len(size) != 3:
            raise ValueError("multi_drone_aerobatic.size must be 3D")
        obs_cfg = dict(cfg.get("obstacles", {}))
        obstacles = _ObstacleSpec(
            type=str(obs_cfg.get("type", "none")),
            count=int(obs_cfg.get("count", 0)),
            seed=int(obs_cfg.get("seed", 0)),
            cells=obs_cfg.get("cells"),
            boxes=obs_cfg.get("boxes"),
        )
        dynamic_specs = cfg.get("dynamic_obstacles", []) or []
        dynamic = [
            _DynamicObstacle3D(
                pos0=np.asarray(d["start"], dtype=float),
                velocity=np.asarray(d["velocity"], dtype=float),
                reflect=bool(d.get("reflect", True)),
                radius=float(d.get("radius", 0.5)),
            )
            for d in dynamic_specs
        ]
        return cls(
            size=(int(size[0]), int(size[1]), int(size[2])),
            n_drones=int(cfg.get("n_drones", 4)),
            obstacles=obstacles,
            pattern=str(cfg.get("pattern", "synchronized_loop")),
            center=tuple(cfg.get("center", (20.0, 20.0, 7.0))),
            radius=float(cfg.get("radius", 4.0)),
            period=float(cfg.get("period", 8.0)),
            n_loops=int(cfg.get("n_loops", 2)),
            normal_axis=str(cfg.get("normal_axis", "y")),
            lookahead_t=float(cfg.get("lookahead_t", 0.4)),
            resolution=float(cfg.get("resolution", 1.0)),
            dynamic_obstacles=dynamic,
            drone_radius=float(cfg.get("drone_radius", 0.4)),
        )
