"""GPU batched rollout, cost, and softmax weighting for MPPI.

Takes the prepared inputs (occupancy, cost-to-go, sampled actions, wind,
dynamic-obstacle predictions) and runs the per-replan batched simulation:
positions, collision, cost, softmax weights, vanilla softmax action,
argmin action. The action-aggregation decision (vanilla / argmin /
mode-aware cluster commit) lives in ``aggregator`` so the rollout module
stays a pure compute kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _to_tensor(x: np.ndarray, device: Any) -> Any:
    import torch

    return torch.as_tensor(x, dtype=torch.float32, device=device)


@dataclass
class RolloutResult:
    """All tensors the aggregator and the viz pipeline need from a rollout."""

    rollouts: Any  # torch.Tensor [S, H, D]
    actions_t: Any  # torch.Tensor [S, D]
    costs: Any  # torch.Tensor [S]
    weights: Any  # torch.Tensor [S]
    softmax_action: Any  # torch.Tensor [D]
    argmin_action: Any  # torch.Tensor [D]
    argmin_idx: int
    reaches_goal_any: Any  # torch.Tensor [S]
    first_goal_h: Any  # torch.Tensor [S]
    cost_min: Any  # torch.Tensor scalar


def run_rollout(
    *,
    obs: np.ndarray,
    gl: np.ndarray,
    actions_np: np.ndarray,
    occ: np.ndarray,
    ctg_np: np.ndarray,
    pred_traj: np.ndarray | None,
    r2_arr: np.ndarray | None,
    wind_step: np.ndarray | None,
    prev_action: np.ndarray | None,
    horizon: int,
    dt_plan: float,
    resolution: float,
    goal_radius: float,
    n_samples: int,
    w_goal: float,
    w_obs: float,
    w_smooth: float,
    temperature: float,
    device: Any,
    w_reach_time: float = 0.0,
    w_clean_ctg: float = 0.0,
    score_collision_after_goal: bool = False,
) -> RolloutResult:
    """Run the batched rollout and return all per-sample tensors.

    Mirrors the original ``GPUMPPIPlanner.plan`` rollout block bit-for-bit;
    only the surrounding orchestration is moved into the planner class.
    """
    import torch

    ndim = occ.ndim
    occ_t = _to_tensor(occ.astype(np.float32), device)
    ctg_t = _to_tensor(ctg_np, device)
    obs_t = _to_tensor(obs, device)
    gl_t = _to_tensor(gl, device)
    actions_t = _to_tensor(actions_np, device)
    gr2 = goal_radius ** 2
    max_finite = float(ctg_np[ctg_np < np.inf].max()) if np.any(ctg_np < np.inf) else 1e6
    unreachable_penalty = max_finite + 100.0

    # Rollout positions for every sample × horizon step.
    h = torch.arange(1, horizon + 1, dtype=torch.float32, device=device) * dt_plan  # [H]
    rollouts = obs_t[None, None, :] + actions_t[:, None, :] * h[None, :, None]  # [S, H, D]
    if wind_step is not None:
        ws_t = _to_tensor(wind_step, device)
        rollouts = rollouts + ws_t[None, None, :] * h[None, :, None]

    # Cell indices for occupancy + OOB.
    shape_t = torch.tensor(list(occ.shape), dtype=torch.long, device=device)
    cell_indices_raw = (rollouts / resolution).long()
    oob = (
        (cell_indices_raw < 0)
        | (cell_indices_raw >= shape_t[None, None, :])
    ).any(dim=-1)  # [S, H]
    cell_indices = cell_indices_raw.clamp(
        torch.zeros_like(shape_t), shape_t - 1
    )  # [S, H, D]

    if ndim == 2:
        occ_collision = occ_t[cell_indices[:, :, 0], cell_indices[:, :, 1]].float()
    else:
        occ_collision = occ_t[
            cell_indices[:, :, 0], cell_indices[:, :, 1], cell_indices[:, :, 2]
        ].float()
    collision_mask = occ_collision + oob.float()

    # First-goal-reach index — used to scope collision sums to pre-goal steps
    # (matches the CPU MPPI's `break` after first reach).
    dist2 = ((rollouts - gl_t[None, None, :]) ** 2).sum(dim=-1)  # [S, H]
    reaches_goal_any = (dist2 <= gr2).any(dim=1)  # [S]
    first_goal_h = torch.where(
        reaches_goal_any,
        (dist2 <= gr2).float().argmax(dim=1),
        torch.tensor(horizon, device=device),
    )
    step_idx = torch.arange(horizon, device=device)
    pre_goal_mask = (step_idx[None, :] < first_goal_h[:, None]).float()  # [S, H]

    collision_scope = (
        torch.ones_like(pre_goal_mask) if score_collision_after_goal else pre_goal_mask
    )
    collision_pen = (collision_mask * collision_scope).sum(dim=1)  # [S]
    if pred_traj is not None and r2_arr is not None:
        pred_t = _to_tensor(pred_traj, device)  # [O, H, D]
        r2_t = _to_tensor(r2_arr, device)  # [O]
        diffs = rollouts[:, None, :, :] - pred_t[None, :, :, :]  # [S, O, H, D]
        sep2 = (diffs * diffs).sum(dim=-1)  # [S, O, H]
        dyn_collision = (sep2 <= r2_t[None, :, None]).any(dim=1).float()  # [S, H]
        collision_pen = collision_pen + (dyn_collision * collision_scope).sum(dim=1)

    if ndim == 2:
        ctg_roll = ctg_t[cell_indices[:, :, 0], cell_indices[:, :, 1]]
    else:
        ctg_roll = ctg_t[cell_indices[:, :, 0], cell_indices[:, :, 1], cell_indices[:, :, 2]]
    ctg_roll = torch.where(
        torch.isfinite(ctg_roll),
        ctg_roll,
        torch.tensor(unreachable_penalty, device=device),
    )
    ctg_min = ctg_roll.min(dim=1).values  # [S]
    ctg_avg = ctg_roll.mean(dim=1)  # [S]

    smooth_pen = torch.zeros(n_samples, device=device)
    if prev_action is not None:
        prev_t = _to_tensor(prev_action, device)
        smooth_pen = torch.norm(actions_t - prev_t[None, :], dim=1)
    reach_time_pen = first_goal_h.float()

    no_coll = collision_pen == 0
    clean_reach = reaches_goal_any & no_coll
    dirty_reach = reaches_goal_any & ~no_coll
    neither = ~reaches_goal_any

    costs = torch.empty(n_samples, device=device)
    costs[clean_reach] = (
        -1e6
        + w_reach_time * reach_time_pen[clean_reach]
        + w_clean_ctg * ctg_avg[clean_reach]
        + w_smooth * smooth_pen[clean_reach]
    )
    costs[dirty_reach] = (
        w_goal * ctg_avg[dirty_reach]
        + w_obs * collision_pen[dirty_reach]
        + w_smooth * smooth_pen[dirty_reach]
    )
    costs[neither] = (
        w_goal * (0.5 * ctg_avg[neither] + 0.5 * ctg_min[neither])
        + w_obs * collision_pen[neither]
        + w_smooth * smooth_pen[neither]
    )

    cost_min = costs.min()
    weights = torch.exp(-(costs - cost_min) / temperature)
    weights = weights / weights.sum()
    softmax_action = (weights[:, None] * actions_t).sum(dim=0)
    argmin_idx = int(costs.argmin().item())
    argmin_action = actions_t[argmin_idx]

    return RolloutResult(
        rollouts=rollouts,
        actions_t=actions_t,
        costs=costs,
        weights=weights,
        softmax_action=softmax_action,
        argmin_action=argmin_action,
        argmin_idx=argmin_idx,
        reaches_goal_any=reaches_goal_any,
        first_goal_h=first_goal_h,
        cost_min=cost_min,
    )
