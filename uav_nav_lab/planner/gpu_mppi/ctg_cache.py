"""Dijkstra cost-to-go cache with a tolerance window for moving goals.

Recomputing Dijkstra every replan dominates wallclock on moving-lookahead
scenarios (race / aerobatic). The cache reuses the previous CTG grid as
long as the integer goal cell has not drifted by more than `tolerance`
along any axis.
"""

from __future__ import annotations

import numpy as np

from .._grid import dijkstra_cost_to_go


class CTGCache:
    """Caches the Dijkstra cost-to-go grid keyed by goal cell."""

    def __init__(self, tolerance: int = 0) -> None:
        self.tolerance = max(0, int(tolerance))
        self._cache: np.ndarray | None = None
        self._goal_cell: tuple[int, ...] | None = None

    def reset(self) -> None:
        self._cache = None
        self._goal_cell = None

    def get(self, static_occ: np.ndarray, goal_cell: tuple[int, ...]) -> np.ndarray:
        if self._cache is None or self._needs_recompute(goal_cell):
            self._cache = dijkstra_cost_to_go(static_occ, goal_cell)
            self._goal_cell = goal_cell
        return self._cache

    def _needs_recompute(self, goal_cell: tuple[int, ...]) -> bool:
        if self._goal_cell is None:
            return True
        if self.tolerance <= 0:
            return self._goal_cell != goal_cell
        return any(
            abs(int(a) - int(b)) > self.tolerance
            for a, b in zip(goal_cell, self._goal_cell)
        )
