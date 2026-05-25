"""Identity sensor — returns the true state. Useful as a baseline."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import SENSOR_REGISTRY, SensorModel


@SENSOR_REGISTRY.register("perfect")
class PerfectSensor(SensorModel):
    def __init__(self, *, dynamic_velocity_scale: float = 1.0) -> None:
        self.dynamic_velocity_scale = float(dynamic_velocity_scale)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "PerfectSensor":
        return cls(
            dynamic_velocity_scale=float(cfg.get("dynamic_velocity_scale", 1.0)),
        )

    def reset(self, *, seed: int | None = None) -> None:
        pass

    def observe(self, t: float, true_position: np.ndarray) -> np.ndarray:
        return np.asarray(true_position, dtype=float).copy()

    def observe_dynamics(
        self, t: float, true_position: np.ndarray, dynamic_obstacles: list[dict]
    ) -> list[dict]:
        rows: list[dict] = []
        for d in dynamic_obstacles:
            row = dict(d)
            if self.dynamic_velocity_scale != 1.0:
                velocity = np.asarray(row.get("velocity", ()), dtype=float)
                if velocity.size > 0:
                    row["velocity"] = [
                        float(v) for v in (velocity * self.dynamic_velocity_scale)
                    ]
            rows.append(row)
        return rows
