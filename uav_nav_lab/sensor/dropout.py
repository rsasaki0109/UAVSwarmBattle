"""Intermittent peer / obstacle observation dropout (a packet-loss channel).

Every other tracker reports the moving threats it can see at ground truth or with
noise/delay applied to the *value*; none model losing the observation entirely.
``dropout`` is that missing channel: each replan, each dynamic obstacle (peer or
scene body) is independently *missing* from the observation with probability
``dropout_prob`` — a Bernoulli packet-loss / intermittent-occlusion model. The ego
pose is still reported at ground truth (this models only the peer-tracking link).

It exists to ask what a coordination rule actually NEEDS to communicate. A global
right-of-way (``lateral_bias``) is comms-free — each drone tilts off its OWN goal
heading and never reads a peer — so it should be untouched by dropout, while a
rule that must SEE each neighbour to decide which side to pass (``pairwise_bias``)
or a peer forecast that extrapolates the neighbour's state (a predictor) loses its
input as peers vanish. (Collision avoidance degrades for everyone, since the
avoider also stops seeing the dropped peer; the question is whether the
symmetry-breaking on top survives the loss.)
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import SENSOR_REGISTRY, SensorModel


@SENSOR_REGISTRY.register("dropout")
class DropoutSensor(SensorModel):
    """Bernoulli per-observation dropout of dynamic obstacles.

    Parameters
    ----------
    dropout_prob : float
        Probability in [0, 1] that any given dynamic obstacle is omitted from a
        single ``observe_dynamics`` call (drawn i.i.d. per obstacle per call).
    """

    def __init__(self, dropout_prob: float = 0.0) -> None:
        self.dropout_prob = float(dropout_prob)
        self._rng = np.random.default_rng()

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "DropoutSensor":
        return cls(dropout_prob=float(cfg.get("dropout_prob", 0.0)))

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def observe(self, t: float, true_position: np.ndarray) -> np.ndarray:
        # Ego pose is ground truth — this sensor models only the peer link.
        return np.asarray(true_position, dtype=float).copy()

    def observe_dynamics(
        self, t: float, true_position: np.ndarray, dynamic_obstacles: list[dict]
    ) -> list[dict]:
        if self.dropout_prob <= 0.0:
            return [dict(d) for d in dynamic_obstacles]
        keep = self._rng.random(len(dynamic_obstacles)) >= self.dropout_prob
        return [dict(d) for d, k in zip(dynamic_obstacles, keep) if k]
