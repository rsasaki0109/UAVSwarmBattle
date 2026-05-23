"""Shared fixtures for planner unit tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from uav_nav_lab.planner import PLANNER_REGISTRY


@pytest.fixture
def planner_registry():
    """Planner registry imported once for tests that exercise registration."""
    return PLANNER_REGISTRY


@pytest.fixture
def empty_grid() -> Callable[[tuple[int, ...]], np.ndarray]:
    """Factory for a fresh obstacle-free occupancy grid."""

    def make(shape: tuple[int, ...] = (20, 20)) -> np.ndarray:
        return np.zeros(shape, dtype=bool)

    return make


@pytest.fixture
def empty_grid_20(empty_grid) -> np.ndarray:
    return empty_grid((20, 20))


@pytest.fixture
def empty_grid_30(empty_grid) -> np.ndarray:
    return empty_grid((30, 30))
