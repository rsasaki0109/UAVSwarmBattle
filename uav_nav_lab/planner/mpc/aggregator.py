"""MPC action aggregation — argmin over rollout costs.

The single aggregation strategy used by :class:`.planner.SamplingMPCPlanner`:
pick the action whose rollout produced the lowest cost. Compared with
:func:`uav_nav_lab.planner.mppi.aggregator.softmax_aggregate`, this is
the temperature → 0 limit (one sample takes all weight).
"""

from __future__ import annotations

import numpy as np


def argmin_aggregate(
    costs: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    """Lowest-cost action selection.

    Returns ``(chosen_action, cost_min, best_k)``. ``best_k`` picks the
    rollout to attach to the returned :class:`.base.Plan`.
    """
    best_k = int(np.argmin(costs))
    return actions[best_k], float(costs[best_k]), best_k
