"""TeamHOI-style learned swarm planner (cross-attention over ally / threat tokens).

A single decentralized policy reads local observations plus a variable-length
set of peer tokens from the runner's ``dynamic_obstacles`` list:

* entries **with** a ``goal`` key → ally UAV (``ROLE_ALLY``)
* entries **without** → scene dynamic obstacle (``ROLE_THREAT``)

Weights come from a NumPy checkpoint produced by
``scripts/train_swarm_transformer_checkpoint.py`` (BC on the convention teacher
or REINFORCE). 2-D only — matches ``dummy_2d`` antipodal / threat-hub probes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..predictor import build_predictor
from .base import PLANNER_REGISTRY, Plan, Planner
from . import swarm_transformer_core as core


@PLANNER_REGISTRY.register("swarm_transformer")
class SwarmTransformerPlanner(Planner):
    """Cross-attention teammate-token policy; checkpoint-backed."""

    def __init__(
        self,
        *,
        max_speed: float = 1.0,
        neighbor_dist: float = 15.0,
        interaction_radius: float = 4.0,
        checkpoint: str | Path,
        goal_radius: float = 1.5,
        predictor: Any | None = None,
    ) -> None:
        self.max_speed = float(max_speed)
        self.neighbor_dist = float(neighbor_dist)
        self.interaction_radius = float(interaction_radius)
        self.goal_radius = float(goal_radius)
        self._predictor = predictor
        ckpt = Path(checkpoint)
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"swarm_transformer checkpoint not found: {ckpt}. "
                "Run scripts/train_swarm_transformer_checkpoint.py first."
            )
        self._params, self._stats = core.load_checkpoint(ckpt)
        self._cur_vel: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "SwarmTransformerPlanner":
        ckpt = cfg.get("checkpoint")
        if not ckpt:
            raise ValueError("planner.checkpoint is required for swarm_transformer")
        return cls(
            max_speed=float(cfg.get("max_speed", 1.0)),
            neighbor_dist=float(cfg.get("neighbor_dist", 15.0)),
            interaction_radius=float(cfg.get("interaction_radius", 4.0)),
            checkpoint=ckpt,
            goal_radius=float(cfg.get("goal_radius", 1.5)),
            predictor=build_predictor(cfg.get("predictor")),
        )

    def reset(self) -> None:
        self._cur_vel = None
        if self._predictor is not None:
            self._predictor.reset()

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
        pos = np.asarray(observation, dtype=float)
        if pos.shape[0] != 2:
            raise ValueError(
                f"SwarmTransformerPlanner is 2-D only; got {pos.shape[0]}-D observation."
            )
        gl = np.asarray(goal, dtype=float)[:2]
        vel = self._cur_vel if self._cur_vel is not None else np.zeros(2)

        if float(np.linalg.norm(gl - pos[:2])) < self.goal_radius:
            return Plan(
                waypoints=np.asarray([pos[:2]], dtype=float),
                target_velocity=np.zeros(2),
                meta={"planner": "swarm_transformer", "n_tokens": 0},
            )

        ego, peers, mask, rot = core.build_tokens(
            pos[:2], vel, gl, dynamic_obstacles,
            neighbor_dist=self.neighbor_dist,
            interaction_radius=self.interaction_radius,
            predictor=self._predictor,
        )
        a_ego = core.predict_velocity_ego(self._params, self._stats, ego, peers, mask)
        a_ego = a_ego * self.max_speed  # training teachers used VMAX=1
        v_world = core.clamp_speed(rot.T @ a_ego, self.max_speed)
        wp = pos[:2] + v_world * 0.1
        return Plan(
            waypoints=np.asarray([wp], dtype=float),
            target_velocity=v_world,
            meta={
                "planner": "swarm_transformer",
                "n_tokens": int(mask.sum()),
            },
        )
