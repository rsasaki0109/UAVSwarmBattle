"""Peer-aware helpers for multi-drone episodes.

Both helpers are pure (no I/O, no mutation), so they unit-test cleanly
against `SimpleNamespace` stand-ins for the SimState objects.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _peers_view(
    states: list[Any], radii: list[float], finished: list[bool], me: int
) -> list[dict]:
    """Build a `dynamic_obstacles`-shaped list describing the *other* drones.

    A finished peer (success or collision) is reported with zero velocity —
    the simplest reasonable model of "stuck in place". This is what a
    downstream tracker would feed back to the planner anyway.
    """
    peers = []
    for j, s in enumerate(states):
        if j == me:
            continue
        v = s.velocity if not finished[j] else np.zeros_like(s.velocity)
        peers.append(
            {
                "position": [float(x) for x in s.position],
                "velocity": [float(x) for x in v],
                "radius": float(radii[j]),
            }
        )
    return peers


def _check_peer_collision(
    states: list[Any], radii: list[float], drone_radius: float
) -> list[bool]:
    """Returns a boolean per drone: True if it is currently overlapping a peer."""
    n = len(states)
    hit = [False] * n
    for i in range(n):
        for j in range(i + 1, n):
            r = drone_radius + radii[j]  # treat both with their own radii
            d2 = float(np.sum((states[i].position - states[j].position) ** 2))
            if d2 <= r * r:
                hit[i] = True
                hit[j] = True
    return hit
