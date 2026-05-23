"""Shared N-D occupancy-grid utilities for grid-based planners.

Used by ``astar``, ``mpc``, and ``mppi`` — the neighbour table, obstacle
dilation, Dijkstra cost-to-go, point-to-cell mapping, occupancy lookup,
and dynamic-obstacle masking all live here so a fix applies in one
place and the planner files stay small.
"""

from __future__ import annotations

import heapq
import itertools
from typing import Any, Mapping

import numpy as np


def build_neighbours(ndim: int) -> list[tuple[tuple[int, ...], float]]:
    """All ±1/0 offsets with weight = Euclidean length, excluding the origin."""
    out: list[tuple[tuple[int, ...], float]] = []
    for delta in itertools.product((-1, 0, 1), repeat=ndim):
        if all(d == 0 for d in delta):
            continue
        w = float(np.sqrt(sum(d * d for d in delta)))
        out.append((delta, w))
    return out


def inflate_obstacles(occ: np.ndarray, n: int) -> np.ndarray:
    """Binary dilation by `n` cells along each axis (N-D)."""
    if n <= 0:
        return occ
    out = occ.copy()
    for _ in range(n):
        shifted = np.zeros_like(out)
        for axis in range(out.ndim):
            slc_dst = [slice(None)] * out.ndim
            slc_src = [slice(None)] * out.ndim
            slc_dst[axis] = slice(1, None)
            slc_src[axis] = slice(None, -1)
            shifted[tuple(slc_dst)] |= out[tuple(slc_src)]
            slc_dst[axis] = slice(None, -1)
            slc_src[axis] = slice(1, None)
            shifted[tuple(slc_dst)] |= out[tuple(slc_src)]
        out |= shifted
    return out


def dijkstra_cost_to_go(occ: np.ndarray, goal_cell: tuple[int, ...]) -> np.ndarray:
    """Dijkstra from `goal_cell` over free cells; obstacles get +inf.

    Diagonal corner-cutting is disallowed: a diagonal step is rejected if any
    of its component axis-aligned moves passes through an obstacle.
    """
    ndim = occ.ndim
    neighbours = build_neighbours(ndim)
    dist = np.full(occ.shape, np.inf, dtype=float)
    if occ[goal_cell]:
        return dist
    dist[goal_cell] = 0.0
    heap: list[tuple[float, int, tuple[int, ...]]] = [(0.0, 0, goal_cell)]
    counter = 1
    while heap:
        d, _, cur = heapq.heappop(heap)
        if d > dist[cur]:
            continue
        for delta, w in neighbours:
            nb = tuple(cur[i] + delta[i] for i in range(ndim))
            if any(not (0 <= nb[i] < occ.shape[i]) for i in range(ndim)):
                continue
            if occ[nb]:
                continue
            nz = sum(1 for d2 in delta if d2 != 0)
            if nz > 1:
                blocked = False
                for i in range(ndim):
                    if delta[i] == 0:
                        continue
                    probe = list(cur)
                    probe[i] += delta[i]
                    if occ[tuple(probe)]:
                        blocked = True
                        break
                if blocked:
                    continue
            nd = d + w
            if nd < dist[nb]:
                dist[nb] = nd
                heapq.heappush(heap, (nd, counter, nb))
                counter += 1
    return dist


def point_to_cell(
    p: np.ndarray,
    shape: tuple[int, ...],
    resolution: float,
) -> tuple[int, ...]:
    """Map a continuous point to its grid cell (clipped to bounds).

    Duplicated as ``_cell()`` on the MPC and MPPI classes before the
    S0 cleanup — kept identical to preserve numerical behaviour.
    """
    return tuple(
        int(np.clip(p[i] / resolution, 0, shape[i] - 1)) for i in range(len(shape))
    )


def point_is_occupied(
    occ: np.ndarray,
    p: np.ndarray,
    resolution: float,
) -> bool:
    """Lookup at a continuous point. Out-of-bounds counts as obstacle."""
    ndim = occ.ndim
    coords = []
    for i in range(ndim):
        ci = int(p[i] / resolution)
        if not (0 <= ci < occ.shape[i]):
            return True
        coords.append(ci)
    return bool(occ[tuple(coords)])


def mask_dynamic_obstacle_cells(
    occ_raw: np.ndarray,
    d: Mapping[str, Any],
    resolution: float,
) -> None:
    """Zero out cells inside a dynamic obstacle's footprint (in-place).

    The heuristic ignores movers — the rollout's sphere-sphere distance
    check is what enforces dynamic-obstacle avoidance. Mask is applied
    once per episode against the raw (un-inflated) occupancy.
    """
    pos = np.asarray(d.get("position", ()), dtype=float)
    if pos.size == 0:
        return
    radius = float(d.get("radius", 0.5))
    cells = max(1, int(np.ceil(radius / resolution)))
    ndim = occ_raw.ndim
    center = point_to_cell(pos[:ndim], occ_raw.shape, resolution)
    if ndim == 2:
        for dx in range(-cells + 1, cells):
            for dy in range(-cells + 1, cells):
                cx, cy = center[0] + dx, center[1] + dy
                if 0 <= cx < occ_raw.shape[0] and 0 <= cy < occ_raw.shape[1]:
                    occ_raw[cx, cy] = False
    else:  # 3D
        for dx in range(-cells + 1, cells):
            for dy in range(-cells + 1, cells):
                for dz in range(-cells + 1, cells):
                    cx, cy, cz = center[0] + dx, center[1] + dy, center[2] + dz
                    if (
                        0 <= cx < occ_raw.shape[0]
                        and 0 <= cy < occ_raw.shape[1]
                        and 0 <= cz < occ_raw.shape[2]
                    ):
                        occ_raw[cx, cy, cz] = False


def sample_unit_directions(ndim: int, n_samples: int, base: np.ndarray) -> np.ndarray:
    """Return (n_samples, ndim) unit vectors covering all directions.

    `base` (assumed already unit-length) is included as the first sample so
    the goal direction always gets evaluated.

    - 2D: evenly-spaced angles in [-π, π) rotated by `base`.
    - 3D: `base` first, then a Fibonacci sphere of size n_samples - 1 (uniform).
    """
    base = np.asarray(base, dtype=float).reshape(ndim)
    if ndim == 2:
        out = np.empty((n_samples, 2), dtype=float)
        angles = np.linspace(-np.pi, np.pi, n_samples, endpoint=False)
        for i, ang in enumerate(angles):
            ca, sa = float(np.cos(ang)), float(np.sin(ang))
            out[i] = (ca * base[0] - sa * base[1], sa * base[0] + ca * base[1])
        return out
    if ndim == 3:
        out = np.empty((n_samples, 3), dtype=float)
        out[0] = base
        if n_samples == 1:
            return out
        phi = np.pi * (3.0 - np.sqrt(5.0))
        rest = n_samples - 1
        for i in range(rest):
            t = i / max(1, rest - 1) if rest > 1 else 0.5
            y = 1.0 - 2.0 * t
            r = float(np.sqrt(max(0.0, 1.0 - y * y)))
            theta = phi * i
            out[i + 1] = (float(np.cos(theta)) * r, y, float(np.sin(theta)) * r)
        return out
    raise NotImplementedError(f"sample_unit_directions: unsupported ndim={ndim}")
