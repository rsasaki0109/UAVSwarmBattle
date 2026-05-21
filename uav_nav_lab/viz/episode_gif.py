"""Shared GIF-rendering helpers for ``scripts/render_*_gif.py``.

The three renderers used to each carry their own ``DRONE_COLORS``,
their own ``load_drones`` shim, and their own ``trajectory_arrays``
implementation. They have drifted over time (e.g. the race renderer
added ``T_pad`` and a ``collision_step`` return so failed planners
freeze in place instead of vanishing). Consolidating those helpers
here keeps the renderers' code budget for what is actually
script-specific (axis setup, animation glue, scenario geometry).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np


DRONE_COLORS: list[str] = ["#e8443b", "#3aa54a", "#3865bf", "#d49b1c"]


def load_drones(run_dir: Path, ep: int, n_drones: int = 4) -> list[dict]:
    """Read ``episode_NNN_drone_KK.json`` for ``KK = 0 .. n_drones-1``."""
    out: list[dict] = []
    for i in range(n_drones):
        path = Path(run_dir) / f"episode_{ep:03d}_drone_{i:02d}.json"
        out.append(json.loads(path.read_text()))
    return out


def trajectory_arrays(
    drones: list[dict],
    *,
    T_pad: int | None = None,
    fit: Literal["min", "max"] = "max",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(true_pos[D,T,3], ref_pos[D,T,3], collision_step[D])``.

    ``T`` is chosen by, in order of precedence:

    * ``T_pad`` if it is ``>=`` the longest drone's step count
      (race-renderer behaviour: all panes share the longest timeline);
    * ``min(len(d.steps))`` when ``fit='min'`` (aerobatic renderer:
      every drone is the same synchronised length anyway, so the
      stricter ``min`` is the historical choice);
    * ``max(len(d.steps))`` otherwise.

    Drones shorter than ``T`` are right-padded by holding their last
    logged position — the planner has crashed or finished, so freezing
    in place is the most honest visualisation. ``collision_step[i]``
    is the first step index at which drone ``i`` reports
    ``collision=True`` (or, falling back, the index of the last step
    of a drone whose final ``outcome`` is ``"collision"``); it equals
    ``T`` when the drone never collided.
    """
    D = len(drones)
    longest = max(len(d["steps"]) for d in drones)
    if T_pad is not None and T_pad > longest:
        T = T_pad
    elif fit == "min":
        T = min(len(d["steps"]) for d in drones)
    else:
        T = longest

    true_pos = np.zeros((D, T, 3))
    ref_pos = np.zeros((D, T, 3))
    collision_step = np.full(D, T, dtype=int)
    for i, d in enumerate(drones):
        steps = d["steps"]
        last_true = None
        last_ref = None
        for k in range(T):
            if k < len(steps):
                s = steps[k]
                last_true = s["true_pos"]
                last_ref = s.get("reference_pos", s["true_pos"])
                if collision_step[i] == T and s.get("collision"):
                    collision_step[i] = k
            true_pos[i, k] = last_true
            ref_pos[i, k] = last_ref
        if collision_step[i] == T and d.get("outcome") == "collision":
            collision_step[i] = len(steps) - 1
    return true_pos, ref_pos, collision_step
