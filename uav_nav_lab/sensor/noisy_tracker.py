"""Noisy, delayed tracker for *dynamic obstacles*.

Every existing sensor reports moving threats at ground truth (``perfect`` and
``delayed`` both pass ``dynamic_obstacles`` through untouched in
``observe_dynamics``); position-level delay/noise only ever applied to the ego
pose. That makes the obstacle's current position perfectly known, so a planner
that simply avoids the *observed* obstacle never needs a forecast — and a
risk-aware planner that hedges the forecast tail (CVaR-MPPI) has nothing to win.

``noisy_tracker`` is the missing perception model: it corrupts each obstacle's
reported position with a fixed sensor delay (a ring buffer per obstacle) plus
Gaussian measurement noise, and optionally jitters the reported velocity. Now
the obstacle's *current* state is itself uncertain, the predictor genuinely
errs, and planning for the bad tail of that error becomes worth something.

The ego pose still comes from ``observe`` (left as ground truth here — compose
with a delayed ego sensor if you want both); this class models only the
obstacle-tracking channel, keyed per obstacle by list index.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping

import numpy as np

from .base import SENSOR_REGISTRY, SensorModel


@SENSOR_REGISTRY.register("noisy_tracker")
class NoisyTrackerSensor(SensorModel):
    """Per-obstacle delayed + noisy position tracker.

    Parameters
    ----------
    delay : float
        Fixed perception delay (s); reported position lags the truth by
        ``round(delay/dt)`` control steps via a per-obstacle ring buffer.
    dt : float
        Control step (s), used to size the delay buffer.
    position_noise_std : float
        Std (m) of i.i.d. Gaussian noise added to each reported position.
    velocity_noise_std : float
        Std (m/s) of Gaussian noise added to each reported velocity. Velocity
        is otherwise passed through (the scenario already supplies it); a noisy
        velocity makes a constant-velocity predictor's forecast drift.
    """

    def __init__(
        self,
        delay: float = 0.0,
        dt: float = 0.05,
        position_noise_std: float = 0.0,
        velocity_noise_std: float = 0.0,
    ) -> None:
        self.delay = float(delay)
        self.dt = float(dt)
        self.position_noise_std = float(position_noise_std)
        self.velocity_noise_std = float(velocity_noise_std)
        self._buffer_len = max(1, int(round(self.delay / self.dt)))
        # One position ring buffer per obstacle, created lazily on first sight
        # (the obstacle count is not known until the first observe_dynamics).
        self._buffers: list[deque[np.ndarray]] = []
        self._rng = np.random.default_rng()

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "NoisyTrackerSensor":
        return cls(
            delay=float(cfg.get("delay", 0.0)),
            dt=float(cfg.get("dt", 0.05)),
            position_noise_std=float(cfg.get("position_noise_std", 0.0)),
            velocity_noise_std=float(cfg.get("velocity_noise_std", 0.0)),
        )

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._buffers = []

    def observe(self, t: float, true_position: np.ndarray) -> np.ndarray:
        # Ego pose is reported at ground truth — this sensor models only the
        # obstacle-tracking channel. Compose with a `delayed` ego sensor if you
        # want a laggy ego estimate too.
        return np.asarray(true_position, dtype=float).copy()

    def observe_dynamics(
        self, t: float, true_position: np.ndarray, dynamic_obstacles: list[dict]
    ) -> list[dict]:
        # Grow the per-obstacle buffer list if the obstacle count changed (or on
        # the first call). A new buffer is pre-filled with the current position
        # so early steps report a coherent (if stale) estimate instead of None.
        while len(self._buffers) < len(dynamic_obstacles):
            self._buffers.append(deque(maxlen=self._buffer_len))
        out: list[dict] = []
        for i, d in enumerate(dynamic_obstacles):
            buf = self._buffers[i]
            pos = np.asarray(d["position"], dtype=float)
            buf.append(pos.copy())
            stale = buf[0]  # leftmost = oldest within the delay window
            rep_pos = stale.copy()
            if self.position_noise_std > 0.0:
                rep_pos = rep_pos + self._rng.normal(
                    0.0, self.position_noise_std, size=rep_pos.shape
                )
            nd = dict(d)
            nd["position"] = [float(v) for v in rep_pos]
            if self.velocity_noise_std > 0.0:
                vel = np.asarray(d["velocity"], dtype=float)
                vel = vel + self._rng.normal(
                    0.0, self.velocity_noise_std, size=vel.shape
                )
                nd["velocity"] = [float(v) for v in vel]
            out.append(nd)
        return out
