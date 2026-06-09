"""NavRL velocity-policy adapter (Zhefan-Xu/NavRL RA-L 2025).

Wraps the upstream ``quick-demos`` PPO checkpoint. Observations are built from
the runner's ``dynamic_obstacles`` list:

* entries **with** ``goal`` → other drones (dynamic-obstacle tensor)
* entries **without** → scene threats (LiDAR ray cast circles)

Requires a local NavRL clone + PyTorch stack::

  bash scripts/setup_navrl_adapter.sh
  pip install -e '.[navrl]'

YAML::

  planner:
    type: navrl
    navrl_root: third_party/NavRL
    max_speed: 5.0
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .base import PLANNER_REGISTRY, Plan, Planner
from . import navrl_bridge as bridge


def _split_obstacles(
    dynamic_obstacles: list[dict] | None,
) -> tuple[list[tuple[float, float, float]], list[np.ndarray], list[np.ndarray]]:
    """Return (static_circles, peer_positions, peer_velocities)."""
    circles: list[tuple[float, float, float]] = []
    peers_p: list[np.ndarray] = []
    peers_v: list[np.ndarray] = []
    for d in dynamic_obstacles or []:
        pos = np.asarray(d["position"], dtype=float)[:2]
        vel = np.asarray(d.get("velocity", (0.0, 0.0)), dtype=float)[:2]
        if "goal" in d:
            peers_p.append(pos)
            peers_v.append(vel)
        else:
            r = float(d.get("radius", 0.5))
            circles.append((float(pos[0]), float(pos[1]), r))
    return circles, peers_p, peers_v


@PLANNER_REGISTRY.register("navrl")
class NavRLPlanner(Planner):
    """Pretrained NavRL PPO policy (single-agent checkpoint, multi-robot obs)."""

    def __init__(
        self,
        *,
        navrl_root: str | Path,
        max_speed: float = 5.0,
        goal_radius: float = 1.5,
        lidar_range: float = 4.0,
        hres_deg: float = 10.0,
        device: str = "cpu",
    ) -> None:
        self.max_speed = float(max_speed)
        self.goal_radius = float(goal_radius)
        self.lidar_range = float(lidar_range)
        self.hres_deg = float(hres_deg)
        self._root = Path(navrl_root)
        self._device = device
        self._utils = bridge.load_utils(self._root)
        self._agent = bridge.load_agent(self._root, device=device)
        self._cur_vel: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "NavRLPlanner":
        root = cfg.get("navrl_root") or str(bridge.default_navrl_root())
        return cls(
            navrl_root=root,
            max_speed=float(cfg.get("max_speed", 5.0)),
            goal_radius=float(cfg.get("goal_radius", 1.5)),
            lidar_range=float(cfg.get("lidar_range", 4.0)),
            device=str(cfg.get("device", "cpu")),
        )

    def reset(self) -> None:
        self._cur_vel = None

    def set_current_state(
        self, position: np.ndarray, velocity: np.ndarray | None = None,
    ) -> None:
        if velocity is not None:
            self._cur_vel = np.asarray(velocity, dtype=float)[:2]

    def plan(
        self,
        observation: np.ndarray,
        goal: np.ndarray,
        obstacle_map: Any,
        *,
        dynamic_obstacles: list[dict] | None = None,
    ) -> Plan:
        del obstacle_map
        pos = np.asarray(observation, dtype=float)[:2]
        gl = np.asarray(goal, dtype=float)[:2]
        vel = self._cur_vel if self._cur_vel is not None else np.zeros(2)

        if float(np.linalg.norm(gl - pos)) < self.goal_radius:
            return Plan(
                waypoints=np.asarray([pos], dtype=float),
                target_velocity=np.zeros(2),
                meta={"planner": "navrl"},
            )

        circles, peer_p, peer_v = _split_obstacles(dynamic_obstacles)
        target_dir = gl - pos
        if float(np.linalg.norm(target_dir)) < 1e-6:
            target_dir = np.array([1.0, 0.0])

        u = self._utils
        import torch

        dev = self._agent.device
        robot_state = u.get_robot_state(pos, gl, vel, target_dir, device=dev)
        static_obs_input, _, _ = u.get_ray_cast(
            pos,
            circles,
            max_range=self.lidar_range,
            hres_deg=self.hres_deg,
            vfov_angles_deg=[-10.0, 0.0, 10.0, 20.0],
            start_angle_deg=float(np.degrees(np.arctan2(target_dir[1], target_dir[0]))),
            device=dev,
        )
        target_tensor = torch.tensor(
            np.append(target_dir[:2], 0.0), dtype=torch.float, device=dev,
        ).unsqueeze(0).unsqueeze(0)

        if peer_p:
            dyn_obs_input = u.get_dyn_obs_state(
                pos, vel, peer_p, peer_v, target_tensor, device=dev,
            )
        else:
            dyn_obs_input = torch.zeros((1, 1, 5, 10), dtype=torch.float, device=dev)

        v = self._agent.plan(robot_state, static_obs_input, dyn_obs_input, target_tensor)
        v = np.asarray(v, dtype=float)[:2]
        if not np.all(np.isfinite(v)):
            d = gl - pos
            n = float(np.linalg.norm(d))
            v = (d / n * self.max_speed) if n > 1e-6 else np.zeros(2)
        sp = float(np.linalg.norm(v))
        if sp > self.max_speed and sp > 1e-9:
            v = v * (self.max_speed / sp)

        wp = pos + v * 0.1
        return Plan(
            waypoints=np.asarray([wp], dtype=float),
            target_velocity=v,
            meta={"planner": "navrl", "n_peers": len(peer_p), "n_threats": len(circles)},
        )
