"""MPPI action aggregation — softmax over rollout costs.

The single aggregation strategy used by :class:`.planner.MPPIPlanner`:
weight each sampled action by ``exp(-(cost - cost_min) / temperature)``
and return the normalised weighted average. Subtracting the min before
``exp`` keeps magnitudes finite even when the best rollout carries the
-1e6 reach-goal bonus; the shift cancels in normalisation, so the
result is unchanged.

Low temperature → collapses to argmin MPC. High temperature → approaches
the uniform mean over all sampled actions.
"""

from __future__ import annotations

import numpy as np


def softmax_aggregate(
    costs: np.ndarray,
    actions: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Softmax-weighted average action.

    Returns ``(chosen_action, weights, cost_min, best_k)`` where
    ``best_k = argmax(weights)`` — useful for picking the most-weighted
    rollout to attach to the returned ``Plan`` for visualisation /
    pure-pursuit fallback.
    """
    cost_min = float(np.min(costs))
    weights = np.exp(-(costs - cost_min) / temperature)
    weights = weights / float(np.sum(weights))
    chosen_action = (weights[:, None] * actions).sum(axis=0)
    best_k = int(np.argmax(weights))
    return chosen_action, weights, cost_min, best_k
