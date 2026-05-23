"""Per-sample rollout scoring for MPPI.

Simulates each sampled constant-velocity action for ``horizon`` steps,
accumulating goal-attraction cost (Dijkstra cost-to-go), obstacle
penalties (static occupancy + dynamic-obstacle prediction), and a
smoothness term against the previous chosen action. Returns the full
cost vector plus the rollout polylines for downstream weighting.

Same cost shape as :mod:`..mpc` so MPPI and MPC scoring stay
directly comparable — the only difference between the two planners is
the action-aggregation rule (argmin vs softmax).
"""

from __future__ import annotations

import numpy as np

from .._grid import point_is_occupied, point_to_cell


def score_rollouts(
    *,
    actions: np.ndarray,
    obs: np.ndarray,
    gl: np.ndarray,
    occ: np.ndarray,
    ctg: np.ndarray,
    unreachable_penalty: float,
    horizon: int,
    dt_plan: float,
    resolution: float,
    goal_radius: float,
    pred_traj: np.ndarray | None,
    r2_arr: np.ndarray | None,
    wind_step: np.ndarray | None,
    prev_action: np.ndarray | None,
    w_goal: float,
    w_obs: float,
    w_smooth: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Score every sampled action by simulating a horizon-step rollout.

    Parameters mirror the MPPIPlanner internals — they are passed
    explicitly so this function is pure (no self) and unit-testable.
    Returns ``(costs[n_samples], rollouts[n_samples])`` where each
    rollout is truncated at the first goal-reach or end of horizon.
    """
    n_samples = actions.shape[0]
    ndim = obs.shape[0]
    costs = np.empty(n_samples, dtype=float)
    rollouts: list[np.ndarray] = []
    gr2 = goal_radius * goal_radius
    for k, v in enumerate(actions):
        rollout = np.empty((horizon + 1, ndim), dtype=float)
        rollout[0] = obs
        collision_pen = 0
        ctg_min = np.inf
        ctg_sum_until = 0.0
        steps_until = 0
        reaches_goal = False
        for h in range(1, horizon + 1):
            step = v * dt_plan
            if wind_step is not None:
                step = step + wind_step * dt_plan
            rollout[h] = rollout[h - 1] + step
            d2 = float(np.sum((rollout[h] - gl) ** 2))
            if d2 <= gr2:
                reaches_goal = True
                break
            if point_is_occupied(occ, rollout[h], resolution):
                collision_pen += 1
            if pred_traj is not None:
                diffs = pred_traj[:, h - 1, :] - rollout[h]
                sep2 = np.sum(diffs * diffs, axis=1)
                if np.any(sep2 <= r2_arr):
                    collision_pen += 1
            cell_h = point_to_cell(rollout[h], occ.shape, resolution)
            ctg_h = float(ctg[cell_h]) if np.isfinite(ctg[cell_h]) else unreachable_penalty
            ctg_sum_until += ctg_h
            if ctg_h < ctg_min:
                ctg_min = ctg_h
            steps_until = h
        smooth_pen = 0.0
        if prev_action is not None:
            smooth_pen = float(np.linalg.norm(v - prev_action))
        if reaches_goal and collision_pen == 0:
            cost = -1e6 + w_smooth * smooth_pen
        elif reaches_goal:
            ctg_avg = ctg_sum_until / max(1, steps_until)
            cost = (
                w_goal * ctg_avg
                + w_obs * collision_pen
                + w_smooth * smooth_pen
            )
        else:
            ctg_avg = ctg_sum_until / max(1, steps_until)
            cost = (
                w_goal * (0.5 * ctg_avg + 0.5 * ctg_min)
                + w_obs * collision_pen
                + w_smooth * smooth_pen
            )
        costs[k] = cost
        rollouts.append(rollout[: steps_until + 1] if steps_until > 0 else rollout)
    return costs, rollouts
