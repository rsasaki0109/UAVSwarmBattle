"""Game-theoretic (best-response) predictor for multi-drone coordination.

The constant-velocity predictor models every dynamic obstacle as coasting in a
straight line. For *peer drones* that is wrong: a peer is itself a rational
planner steering toward its own goal, so its near-future path curves toward
that goal rather than continuing along its current heading. Treating peers as
ballistic makes the ego planner over- or under-react to crossing traffic.

This predictor performs one level of best-response reasoning: if a tracked
obstacle carries a ``goal`` key (peers do — see ``_peers_view``), it is
predicted to move from its current position straight toward that goal at a
constant speed (its current speed, or ``peer_speed`` if configured). Obstacles
without a ``goal`` (scene dynamic obstacles, intruders) fall back to
constant-velocity, so this is a safe drop-in for any scenario.

It is "one level" because it assumes peers head naively for their goals
without themselves accounting for the ego drone — a single Stackelberg step,
not a converged Nash equilibrium. That is the cheap, robust 80% of the benefit.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import PREDICTOR_REGISTRY, Predictor


@PREDICTOR_REGISTRY.register("game_theoretic")
class GameTheoreticPredictor(Predictor):
    """Predict goal-carrying peers as heading to their goal; others ballistic.

    Config keys
    -----------
    ``peer_speed`` : optional fixed cruise speed (m/s) for goal-seeking peers.
        Default ``None`` → use the peer's current observed speed (falling back
        to ballistic if it is essentially stationary, since direction is then
        undefined).
    """

    def __init__(self, peer_speed: float | None = None) -> None:
        self.peer_speed = None if peer_speed is None else float(peer_speed)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "GameTheoreticPredictor":
        ps = cfg.get("peer_speed")
        return cls(peer_speed=(float(ps) if ps is not None else None))

    def predict(
        self,
        dynamic_obstacles: list[dict],
        horizon_dts: np.ndarray,
    ) -> np.ndarray:
        if not dynamic_obstacles:
            return np.zeros((0, len(horizon_dts), 0), dtype=float)
        ndim = len(dynamic_obstacles[0]["position"])
        H = len(horizon_dts)
        out = np.empty((len(dynamic_obstacles), H, ndim), dtype=float)
        dts = np.asarray(horizon_dts, dtype=float)
        for k, d in enumerate(dynamic_obstacles):
            p0 = np.asarray(d["position"], dtype=float)[:ndim]
            v = np.asarray(d["velocity"], dtype=float)[:ndim]
            goal = d.get("goal")
            if goal is not None:
                g = np.asarray(goal, dtype=float)[:ndim]
                to_goal = g - p0
                dist = float(np.linalg.norm(to_goal))
                speed = self.peer_speed
                if speed is None:
                    speed = float(np.linalg.norm(v))
                if dist > 1e-9 and speed > 1e-9:
                    direction = to_goal / dist
                    # don't overshoot the goal: clamp travel to the remaining
                    # distance, then hold at the goal.
                    travel = np.minimum(dts * speed, dist)
                    out[k] = p0[None, :] + travel[:, None] * direction[None, :]
                    continue
            # no goal, or undefined direction/speed → constant velocity
            out[k] = p0[None, :] + dts[:, None] * v[None, :]
        return out
