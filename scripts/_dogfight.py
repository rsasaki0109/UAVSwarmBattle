"""A 1-v-1 UAV dogfight: two unicycles pursuing each other — who wins?

A standalone adversarial-game sim (no AirSim, pure NumPy), the competitive
counterpoint to the cooperative-convention work. Each UAV is a planar unicycle
(state x, y, heading θ; forward speed v; bounded turn rate ω ∈ [−ω_max, ω_max]).
Both run the *same* pursuit law — steer toward the opponent — so any asymmetry
in the *outcome* comes only from an asymmetry in the *dynamics* (speed, turn rate).

A UAV is "on the opponent's six" when it sits inside the opponent's rear cone
within `capture_range`. A UAV WINS by holding a *solo* rear position (it is on the
opponent's six while the opponent is NOT on its) for `hold_steps` consecutive
steps. A *mutual* lock (the symmetric "circle of death", where both are on each
other's six) resets both, so a perfectly matched duel runs to a STALEMATE.

`aim`:
  "body"  — steer straight at the opponent (the natural "go for the kill" law).
  "tail"  — steer at a point `tail_dist` behind the opponent (lag pursuit).

`duel` returns a DuelResult: winner ∈ {0, 1, None}, the win step, and (optionally)
the recorded trajectories for rendering.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DuelResult:
    winner: int | None          # 0, 1, or None (stalemate)
    win_step: int | None
    steps_run: int
    traj: list = field(default_factory=list)   # (state0, state1) samples over time


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _on_six(att, dfn, capture_range, cone):
    """Is `att` (x,y,θ) inside `dfn`'s rear cone within range? cone = half-angle.

    `att` is on `dfn`'s six when it sits *behind* `dfn`: the bearing from `dfn`
    to `att` points opposite `dfn`'s heading (within `cone` of dead astern).
    """
    d = dfn[:2] - att[:2]
    dist = float(np.hypot(d[0], d[1]))
    if dist > capture_range:
        return False
    bearing_d2a = np.arctan2(att[1] - dfn[1], att[0] - dfn[0])  # defender -> attacker
    return abs(_wrap(bearing_d2a - dfn[2])) > (np.pi - cone)     # attacker astern


def _step(state, target, v, w_max, dt):
    x, y, th = state
    desired = np.arctan2(target[1] - y, target[0] - x)
    dth = np.clip(_wrap(desired - th), -w_max * dt, w_max * dt)
    th = _wrap(th + dth)
    return np.array([x + v * np.cos(th) * dt, y + v * np.sin(th) * dt, th])


def duel(
    *,
    v0: float = 4.0, v1: float = 4.0,            # forward speeds
    wmax0: float = 1.5, wmax1: float = 1.5,      # turn-rate limits (rad/s)
    aim: str = "body",                            # "body" or "tail"
    tail_dist: float = 4.0,                       # lag-pursuit offset (aim="tail")
    capture_range: float = 8.0,
    cone: float = 0.7,                            # rear-cone half-angle (rad)
    hold_steps: int = 30,                         # ~0.6 s at dt=0.02
    arena: float = 40.0,
    dt: float = 0.02,
    steps: int = 4000,
    seed: int = 0,
    record: bool = False,
) -> DuelResult:
    rng = np.random.default_rng(seed)
    r = arena * 0.35
    a = rng.uniform(0, 2 * np.pi)
    s0 = np.array([arena / 2 + r * np.cos(a), arena / 2 + r * np.sin(a),
                   rng.uniform(0, 2 * np.pi)])
    s1 = np.array([arena / 2 - r * np.cos(a), arena / 2 - r * np.sin(a),
                   rng.uniform(0, 2 * np.pi)])
    hold0 = hold1 = 0
    traj = []
    for t in range(steps):
        if aim == "tail":
            tgt0 = s1[:2] - tail_dist * np.array([np.cos(s1[2]), np.sin(s1[2])])
            tgt1 = s0[:2] - tail_dist * np.array([np.cos(s0[2]), np.sin(s0[2])])
        else:  # body
            tgt0, tgt1 = s1[:2], s0[:2]
        s0 = _step(s0, tgt0, v0, wmax0, dt)
        s1 = _step(s1, tgt1, v1, wmax1, dt)
        for s in (s0, s1):
            if not (0 < s[0] < arena and 0 < s[1] < arena):
                c = np.array([arena / 2, arena / 2])
                s[2] = np.arctan2(c[1] - s[1], c[0] - s[0])
                s[0] = float(np.clip(s[0], 0.5, arena - 0.5))
                s[1] = float(np.clip(s[1], 0.5, arena - 0.5))
        if record and t % 4 == 0:
            traj.append((s0.copy(), s1.copy()))
        on0 = _on_six(s0, s1, capture_range, cone)
        on1 = _on_six(s1, s0, capture_range, cone)
        # only a SOLO rear position counts; a mutual lock resets both → stalemate.
        hold0 = hold0 + 1 if (on0 and not on1) else 0
        hold1 = hold1 + 1 if (on1 and not on0) else 0
        if hold0 >= hold_steps:
            return DuelResult(winner=0, win_step=t, steps_run=t, traj=traj)
        if hold1 >= hold_steps:
            return DuelResult(winner=1, win_step=t, steps_run=t, traj=traj)
    return DuelResult(winner=None, win_step=None, steps_run=steps, traj=traj)
