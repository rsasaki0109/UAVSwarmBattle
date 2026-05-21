"""Action aggregation strategies for GPU MPPI.

Three strategies, dispatched in order of priority:

1. **mode-aware sampling** (Smart MPPI v4 / v5) — split rollouts into L/R
   clusters by their lateral principal-component sign and emit the
   lower-cost cluster's softmax-weighted action. Targets bidirectional
   escape cancellation. Optional cost-asymmetry and lateral-cancellation
   gates control when the commit fires.
2. **argmin fallback** (Smart MPPI v1) — when the vanilla softmax mean's
   lateral component is much smaller than the argmin rollout's lateral
   component, the rollout cloud is bimodal and the softmax mean cancels
   the two sides toward zero. Commit to argmin for ``commit_steps`` replans.
3. **vanilla softmax** — MPPI default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AggregatorResult:
    chosen_action: Any  # torch.Tensor [D]
    best_k: int
    fallback_triggered: bool
    mode_aware_triggered: bool
    mode_aware_cluster_sign: int


class ActionAggregator:
    """Stateful dispatcher that picks one of three action strategies per replan."""

    def __init__(
        self,
        *,
        fallback_to_argmin: bool,
        fallback_lateral_threshold: float,
        fallback_lateral_ratio: float,
        fallback_commit_steps: int,
        mode_aware_sampling: bool,
        mode_aware_min_size: int,
        mode_aware_cost_ratio: float,
        mode_aware_lateral_threshold: float,
        mode_aware_lateral_ratio: float,
        temperature: float,
    ) -> None:
        self.fallback_to_argmin = bool(fallback_to_argmin)
        self.fallback_lateral_threshold = float(fallback_lateral_threshold)
        self.fallback_lateral_ratio = float(fallback_lateral_ratio)
        self.fallback_commit_steps = max(1, int(fallback_commit_steps))
        self.mode_aware_sampling = bool(mode_aware_sampling)
        self.mode_aware_min_size = max(1, int(mode_aware_min_size))
        self.mode_aware_cost_ratio = max(1.0, float(mode_aware_cost_ratio))
        self.mode_aware_lateral_threshold = max(0.0, float(mode_aware_lateral_threshold))
        self.mode_aware_lateral_ratio = max(0.0, float(mode_aware_lateral_ratio))
        self.temperature = float(temperature)
        self._fallback_commit_remaining = 0

    def reset(self) -> None:
        self._fallback_commit_remaining = 0

    def select(
        self,
        *,
        actions_t: Any,
        costs: Any,
        weights: Any,
        softmax_action: Any,
        argmin_action: Any,
        argmin_idx: int,
        base: np.ndarray,
        device: Any,
    ) -> AggregatorResult:
        import torch

        mode_aware_action = None
        mode_aware_best_k: int | None = None
        mode_aware_cluster_sign = 0
        mode_aware_triggered = False

        if self.mode_aware_sampling:
            base_t = torch.as_tensor(base, dtype=torch.float32, device=device)
            actions_along = (actions_t * base_t[None, :]).sum(dim=1)
            lat_components = actions_t - actions_along[:, None] * base_t[None, :]
            try:
                _, _, V = torch.linalg.svd(lat_components, full_matrices=False)
                pc = V[0]
                proj = lat_components @ pc  # [S]
            except Exception:
                proj = None
            if proj is not None:
                pos_mask = proj > 0
                neg_mask = ~pos_mask
                n_pos = int(pos_mask.sum().item())
                n_neg = int(neg_mask.sum().item())
                if (
                    n_pos >= self.mode_aware_min_size
                    and n_neg >= self.mode_aware_min_size
                ):
                    pos_action, pos_cost, pos_idx = self._cluster(
                        pos_mask, costs, actions_t
                    )
                    neg_action, neg_cost, neg_idx = self._cluster(
                        neg_mask, costs, actions_t
                    )
                    lo, hi = sorted((pos_cost, neg_cost))
                    cost_ok = (
                        self.mode_aware_cost_ratio <= 1.0
                        or (hi >= self.mode_aware_cost_ratio * max(lo, 1e-6))
                    )
                    lateral_ok = self._lateral_cancellation_ok(
                        softmax_action, argmin_action, base_t
                    )
                    if cost_ok and lateral_ok:
                        if pos_cost <= neg_cost:
                            mode_aware_action = pos_action
                            mode_aware_best_k = pos_idx
                            mode_aware_cluster_sign = 1
                        else:
                            mode_aware_action = neg_action
                            mode_aware_best_k = neg_idx
                            mode_aware_cluster_sign = -1
                        mode_aware_triggered = True

        fallback_triggered = False
        if self.fallback_to_argmin:
            if self._fallback_commit_remaining > 0:
                fallback_triggered = True
                self._fallback_commit_remaining -= 1
            else:
                base_t = torch.as_tensor(base, dtype=torch.float32, device=device)
                if self._lateral_cancellation_detected(
                    softmax_action,
                    argmin_action,
                    base_t,
                    threshold=self.fallback_lateral_threshold,
                    ratio=self.fallback_lateral_ratio,
                ):
                    fallback_triggered = True
                    self._fallback_commit_remaining = self.fallback_commit_steps - 1

        if mode_aware_triggered and mode_aware_action is not None:
            chosen = mode_aware_action
            best_k = mode_aware_best_k or 0
        elif fallback_triggered:
            chosen = argmin_action
            best_k = argmin_idx
        else:
            chosen = softmax_action
            best_k = int(weights.argmax().item())

        return AggregatorResult(
            chosen_action=chosen,
            best_k=best_k,
            fallback_triggered=fallback_triggered,
            mode_aware_triggered=mode_aware_triggered,
            mode_aware_cluster_sign=mode_aware_cluster_sign,
        )

    def _cluster(self, mask: Any, costs: Any, actions_t: Any) -> tuple[Any, float, int]:
        import torch

        c = costs[mask]
        acts = actions_t[mask]
        cmin = c.min()
        w = torch.exp(-(c - cmin) / self.temperature)
        w = w / w.sum()
        action = (w[:, None] * acts).sum(dim=0)
        avg_cost = (w * c).sum()
        idx_local = int(c.argmin().item())
        global_idx = int(torch.nonzero(mask, as_tuple=False)[idx_local].item())
        return action, float(avg_cost.item()), global_idx

    def _lateral_cancellation_ok(
        self, softmax_action: Any, argmin_action: Any, base_t: Any
    ) -> bool:
        """Optional gate for mode-aware sampling: only commit when the
        vanilla softmax mean shows the lateral-cancellation signature."""
        if self.mode_aware_lateral_threshold <= 0.0:
            return True
        return self._lateral_cancellation_detected(
            softmax_action,
            argmin_action,
            base_t,
            threshold=self.mode_aware_lateral_threshold,
            ratio=self.mode_aware_lateral_ratio,
        )

    def _lateral_cancellation_detected(
        self,
        softmax_action: Any,
        argmin_action: Any,
        base_t: Any,
        *,
        threshold: float,
        ratio: float,
    ) -> bool:
        import torch

        softmax_along = (softmax_action * base_t).sum()
        argmin_along = (argmin_action * base_t).sum()
        softmax_lat = softmax_action - softmax_along * base_t
        argmin_lat = argmin_action - argmin_along * base_t
        softmax_lat_mag = float(torch.norm(softmax_lat).item())
        argmin_lat_mag = float(torch.norm(argmin_lat).item())
        return (
            argmin_lat_mag > threshold
            and softmax_lat_mag < ratio * argmin_lat_mag
        )
