"""Non-holonomic (unicycle / Dubins-car) 2D simulator.

A point-mass drone can strafe — it changes its velocity vector in any direction,
limited only by ``max_accel``. A real fixed-wing UAV or a wheeled robot cannot:
it can only move along its current heading and *turn* that heading at a bounded
rate. This simulator imposes exactly that constraint while reusing everything
else in :class:`~uav_nav_lab.sim.dummy.DummySim` (occupancy collisions,
disturbances, synthetic perception, the multi-drone goal override).

It consumes the *same* velocity setpoint the planners already emit, so the whole
controller stack — including the right-of-way ``lateral_bias`` convention — runs
unchanged; only the kinematics differ. The drone:

* turns its heading toward the commanded direction, rate-limited by
  ``turn_rate_max`` (rad/s); and
* drives forward along its (new) heading at a speed that tracks the command
  magnitude, accel-limited by ``max_accel``.

So a commanded sideways velocity is realised only after the drone has turned to
face it — it cannot translate sideways. As ``turn_rate_max`` → ∞ the unicycle
recovers the holonomic point-mass; small values make it strongly non-holonomic.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import SIM_REGISTRY, SimState
from .dummy import DummySim, _DummyParams


def _wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class UnicycleSim(DummySim):
    """2D non-holonomic drone: forward drive + rate-limited turn."""

    def __init__(
        self,
        params: _DummyParams,
        scenario: Any,
        *,
        turn_rate_max: float = 3.0,
        advance_scenario: bool = True,
    ) -> None:
        super().__init__(params, scenario, advance_scenario=advance_scenario)
        if self._ndim != 2:
            raise ValueError("UnicycleSim is 2D only (needs a planar heading)")
        self._turn_rate_max = float(turn_rate_max)
        self._heading = 0.0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], scenario: Any) -> "UnicycleSim":
        base = DummySim.from_config(cfg, scenario)
        sim = cls(base.p, scenario, turn_rate_max=float(cfg.get("turn_rate_max", 3.0)))
        return sim

    def reset(self, *, seed: int | None = None, initial_position=None) -> SimState:
        state = super().reset(seed=seed, initial_position=initial_position)
        # Face the goal at the start (a sensible initial heading; the drone is
        # otherwise free to turn away). Falls back to +x if start == goal.
        goal = self.goal
        d = np.asarray(goal, dtype=float)[:2] - state.position[:2]
        self._heading = float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 1e-9 else 0.0
        return state

    def _update_velocity(self, cmd: np.ndarray) -> None:
        desired_speed = float(np.linalg.norm(cmd))
        if desired_speed > 1e-6:
            desired_dir = float(np.arctan2(cmd[1], cmd[0]))
            err = _wrap_to_pi(desired_dir - self._heading)
            max_turn = self._turn_rate_max * self.dt
            self._heading += float(np.clip(err, -max_turn, max_turn))
        # forward speed tracks the command magnitude, accel-limited
        cur_speed = float(np.linalg.norm(self._state.velocity))
        max_dv = self.p.max_accel * self.dt
        new_speed = cur_speed + float(np.clip(desired_speed - cur_speed, -max_dv, max_dv))
        new_speed = max(0.0, new_speed)
        self._state.velocity = new_speed * np.array(
            [np.cos(self._heading), np.sin(self._heading)]
        )
        self._state.extra["heading"] = self._heading


SIM_REGISTRY.register("dummy_unicycle")(UnicycleSim)
