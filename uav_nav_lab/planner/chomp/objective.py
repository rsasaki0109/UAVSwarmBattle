"""CHOMP objective primitives shared by CHOMP-family planners."""

from __future__ import annotations

import numpy as np


def distance_field(occ: np.ndarray, resolution: float, cap: float) -> np.ndarray:
    """Per-cell Euclidean distance to the nearest occupied cell.

    Distances are in world units and capped at ``cap``. The brute-force
    vectorized implementation is fine for the framework's typical grids
    (50^2 up to 40x40x12); large 3D grids should pre-inflate to keep the
    obstacle count tractable.

    The cap keeps obstacle-free cells finite so centered-difference
    gradients elsewhere do not underflow to ``+inf - +inf = NaN``.
    Anything beyond the CHOMP ``epsilon`` already contributes zero cost,
    so any cap greater than epsilon is sound.
    """
    obstacle_idx = np.argwhere(occ)
    if obstacle_idx.shape[0] == 0:
        return np.full(occ.shape, float(cap), dtype=np.float64)
    grid = np.indices(occ.shape).reshape(occ.ndim, -1).T  # (M, ndim)
    diff = grid[:, None, :] - obstacle_idx[None, :, :]  # (M, K, ndim)
    d_cells = np.sqrt((diff * diff).sum(axis=2)).min(axis=1)
    out = np.minimum(d_cells * float(resolution), float(cap))
    return out.reshape(occ.shape)


def smoothness_hessian(n: int) -> np.ndarray:
    """A^T A for the (n-2, n) second-difference matrix A.

    This is the same Hessian for every spatial dimension because the
    CHOMP smoothness term is separable across axes.
    """
    if n < 3:
        return np.zeros((n, n), dtype=np.float64)
    a = np.zeros((n - 2, n), dtype=np.float64)
    for i in range(n - 2):
        a[i, i] = 1.0
        a[i, i + 1] = -2.0
        a[i, i + 2] = 1.0
    return a.T @ a


def obstacle_cost_and_grad(
    x: np.ndarray, dist_field: np.ndarray, epsilon: float, resolution: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-waypoint CHOMP obstacle potential and its spatial gradient.

    The gradient is estimated with centered finite differences on the
    precomputed distance field. Out-of-bounds queries clamp; the runner is
    responsible for keeping the drone inside the world.
    """
    n, ndim = x.shape
    cells = np.clip(
        np.round(x / resolution).astype(int),
        0,
        np.array(dist_field.shape, dtype=int) - 1,
    )
    d = dist_field[tuple(cells.T)]  # (n,)
    eps = float(epsilon)
    c = np.where(
        d < 0.0,
        -d + eps / 2.0,
        np.where(d <= eps, (d - eps) ** 2 / (2.0 * eps), 0.0),
    )

    grad = np.zeros_like(x)
    for k in range(ndim):
        plus = cells.copy()
        minus = cells.copy()
        plus[:, k] = np.clip(plus[:, k] + 1, 0, dist_field.shape[k] - 1)
        minus[:, k] = np.clip(minus[:, k] - 1, 0, dist_field.shape[k] - 1)
        d_plus = dist_field[tuple(plus.T)]
        d_minus = dist_field[tuple(minus.T)]
        dcdd = np.where(
            d < 0.0,
            -1.0,
            np.where(d <= eps, (d - eps) / eps, 0.0),
        )
        ddx = (d_plus - d_minus) / (2.0 * resolution)
        grad[:, k] = dcdd * ddx
    return c, grad
