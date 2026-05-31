"""CVaR (Conditional Value-at-Risk) cost aggregation for risk-aware MPPI.

Vanilla MPPI scores each candidate action with a single deterministic
rollout cost, so it optimises the *expected* outcome under the predictor's
point estimate. Under perception/prediction uncertainty that expectation can
hide a fat tail: an action that is great on average may collide in the worst
10% of futures. CVaR replaces the per-action scalar cost with the mean of its
worst-case tail, so the softmax then prefers actions that stay safe even when
the prediction is wrong.

This module is the pure-math core: given a [n_samples, n_scenarios] cost
matrix (one row per candidate action, one column per sampled future), return
a [n_samples] risk-adjusted cost. Kept separate from the planner so the tail
math is unit-testable in isolation.
"""

from __future__ import annotations

import numpy as np


def cvar_costs(cost_matrix: np.ndarray, risk_alpha: float) -> np.ndarray:
    """Return per-sample CVaR cost from a [n_samples, n_scenarios] matrix.

    ``risk_alpha`` is the worst-case tail fraction that is averaged:

      - ``risk_alpha = 1.0`` → mean over all scenarios (risk-neutral; this
        recovers vanilla MPPI's expected cost).
      - ``risk_alpha = 0.25`` → mean of the worst 25% of scenarios.
      - ``risk_alpha → 0`` → the single worst scenario (pure min-max).

    The tail size is ``m = max(1, ceil(risk_alpha * n_scenarios))`` so at
    least one scenario always contributes, and the worst scenario is always
    included for any alpha. Costs are "higher = worse", matching
    ``score_rollouts``.
    """
    c = np.asarray(cost_matrix, dtype=float)
    if c.ndim != 2:
        raise ValueError(f"cost_matrix must be 2D [n_samples, n_scenarios]; got {c.shape}")
    n_scenarios = c.shape[1]
    if n_scenarios == 0:
        raise ValueError("cost_matrix needs at least one scenario column")
    if not (0.0 < risk_alpha <= 1.0):
        raise ValueError(f"risk_alpha must be in (0, 1]; got {risk_alpha!r}")
    m = max(1, int(np.ceil(risk_alpha * n_scenarios)))
    if m >= n_scenarios:
        return c.mean(axis=1)
    # Worst m per row = m largest costs. partition is O(n) vs full sort.
    # Take the top-m columns (largest), then average them.
    part = np.partition(c, n_scenarios - m, axis=1)[:, n_scenarios - m:]
    return part.mean(axis=1)
