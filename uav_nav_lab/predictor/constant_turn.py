"""Constant-turn-rate (CTR) predictor.

The constant-velocity predictor models every obstacle as coasting in a straight
line, so it systematically mis-forecasts anything that *curves* — most sharply a
``pursue``/``intercept`` hunter (proportional-navigation lead) or a peer drone
arcing toward its goal. This predictor estimates each obstacle's turn rate ω
(rad/s) from the rotation of its velocity vector between successive observations
and rolls the state forward along the resulting circular arc, instead of a
straight line.

Closed-form 2D arc (constant speed ``s``, heading ``θ0``, turn rate ``ω``)::

    x(t) = x0 + (s/ω)·[sin(θ0 + ω·t) − sin θ0]
    y(t) = y0 − (s/ω)·[cos(θ0 + ω·t) − cos θ0]

As ω → 0 this reduces to the straight-line (constant-velocity) forecast, so on
non-curving traffic CTR is a no-op rather than a regression. Only the 2D turn is
modelled; for ``ndim != 2`` the predictor falls back to constant velocity.

State
-----
ω is estimated from the *change in velocity direction* between this call and the
previous one, so the predictor is stateful: it remembers the last observation of
each obstacle and associates incoming observations to it by greedy
nearest-neighbour within ``association_threshold`` (m). Newly seen obstacles have
no rotation history and forecast as constant-velocity for one step.

Config keys
-----------
``dt`` : seconds between successive ``predict`` calls — i.e. the planner's
    ``replan_period``. ω = (signed angle between last and current velocity) / dt,
    so a wrong ``dt`` scales every turn-rate estimate. Default 0.1.
``smoothing`` : EMA weight in (0, 1] on the newest ω estimate (1.0 = trust the
    latest sample fully; lower resists velocity-field noise, e.g. under the
    ``noisy_tracker`` sensor). Default 1.0.
``max_turn_rate`` : optional clamp (rad/s) on |ω| so a noisy velocity flip cannot
    fling the arc. Default ``None`` (no clamp).
``association_threshold`` : NN gate (m) for matching observations across calls.
    Default 3.0.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import PREDICTOR_REGISTRY, Predictor


@PREDICTOR_REGISTRY.register("constant_turn")
class ConstantTurnPredictor(Predictor):
    def __init__(
        self,
        dt: float = 0.1,
        smoothing: float = 1.0,
        max_turn_rate: float | None = None,
        association_threshold: float = 3.0,
    ) -> None:
        self.dt = float(dt)
        self.smoothing = float(smoothing)
        self.max_turn_rate = None if max_turn_rate is None else float(max_turn_rate)
        self.assoc_thr = float(association_threshold)
        # each track: {"pos": np.ndarray, "vel": np.ndarray, "omega": float}
        self._tracks: list[dict[str, Any]] = []

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "ConstantTurnPredictor":
        mtr = cfg.get("max_turn_rate")
        return cls(
            dt=float(cfg.get("dt", 0.1)),
            smoothing=float(cfg.get("smoothing", 1.0)),
            max_turn_rate=(float(mtr) if mtr is not None else None),
            association_threshold=float(cfg.get("association_threshold", 3.0)),
        )

    def reset(self, *, seed: int | None = None) -> None:
        self._tracks = []

    def _match(self, pos: np.ndarray, ndim: int, used: set[int]) -> int | None:
        """Greedy nearest previous track to `pos` within the gate, unused."""
        best_i: int | None = None
        best_d2 = np.inf
        for i, t in enumerate(self._tracks):
            if i in used or t["pos"].shape[0] != ndim:
                continue
            d2 = float(np.sum((t["pos"] - pos) ** 2))
            if d2 < best_d2 and d2 <= self.assoc_thr * self.assoc_thr:
                best_i, best_d2 = i, d2
        return best_i

    @staticmethod
    def _signed_turn(v_prev: np.ndarray, v_cur: np.ndarray) -> float:
        """Signed angle (rad) rotating v_prev onto v_cur; 0 if either is ~0."""
        if np.linalg.norm(v_prev) < 1e-9 or np.linalg.norm(v_cur) < 1e-9:
            return 0.0
        cross = v_prev[0] * v_cur[1] - v_prev[1] * v_cur[0]
        dot = float(np.dot(v_prev, v_cur))
        return float(np.arctan2(cross, dot))

    def _arc(self, p0: np.ndarray, v: np.ndarray, omega: float,
             dts: np.ndarray) -> np.ndarray:
        s = float(np.linalg.norm(v))
        if s < 1e-9:
            return np.repeat(p0[None, :], len(dts), axis=0)
        if abs(omega) < 1e-6:
            return p0[None, :] + dts[:, None] * v[None, :]  # straight line
        th0 = float(np.arctan2(v[1], v[0]))
        ang = th0 + omega * dts
        out = np.empty((len(dts), 2), dtype=float)
        out[:, 0] = p0[0] + (s / omega) * (np.sin(ang) - np.sin(th0))
        out[:, 1] = p0[1] - (s / omega) * (np.cos(ang) - np.cos(th0))
        return out

    def predict(
        self,
        dynamic_obstacles: list[dict],
        horizon_dts: np.ndarray,
    ) -> np.ndarray:
        if not dynamic_obstacles:
            return np.zeros((0, len(horizon_dts), 0), dtype=float)
        ndim = len(dynamic_obstacles[0]["position"])
        H = len(horizon_dts)
        dts = np.asarray(horizon_dts, dtype=float)
        out = np.empty((len(dynamic_obstacles), H, ndim), dtype=float)

        used: set[int] = set()
        next_tracks: list[dict[str, Any]] = []
        for k, d in enumerate(dynamic_obstacles):
            p0 = np.asarray(d["position"], dtype=float)[:ndim]
            v = np.asarray(d["velocity"], dtype=float)[:ndim]

            omega = 0.0
            if ndim == 2:
                j = self._match(p0, ndim, used)
                if j is not None:
                    used.add(j)
                    est = self._signed_turn(self._tracks[j]["vel"], v) / self.dt
                    prev_w = self._tracks[j]["omega"]
                    omega = self.smoothing * est + (1.0 - self.smoothing) * prev_w
                    if self.max_turn_rate is not None:
                        omega = float(np.clip(omega, -self.max_turn_rate,
                                              self.max_turn_rate))

            if ndim == 2:
                out[k] = self._arc(p0, v, omega, dts)
            else:  # turn only modelled in 2D; fall back to constant velocity
                out[k] = p0[None, :] + dts[:, None] * v[None, :]
            next_tracks.append({"pos": p0, "vel": v, "omega": omega})

        self._tracks = next_tracks
        return out
