"""MPPI with single-episode warmup-driven softmax temperature selection.

Episode 0 runs at `warmup_temperature` (default 1.0 — vanilla MPPI) and
accumulates two per-replan diagnostics from the parent MPPI's
`_last_actions`, `_last_weights`, `_last_chosen_action`, `_last_goal_dir`:

- top-2 weighted-rollout angular disagreement (degrees)
- chosen-action vs goal-direction angle (degrees)

At the start of episode 1, the per-replan means feed the N+P rule
(empirically calibrated in `scripts/n_rule_summary.py` across v1, wave,
4-way, peer, chokepoint):

    if mean_top2 > appl_cut:
        # rollouts disagree wildly — cost landscape is chaos, all
        # aggregators are equivalent. Stay at warmup_temperature
        # rather than pretending we have signal.
        temperature = warmup_temperature
    elif mean_chosen_vs_goal < choice_cut_low:
        # vanilla MPPI's chosen action already hugs the goal — the
        # prior is correct, the best move is to average many
        # successful rollouts toward it.
        temperature = uniform_temperature
    else:
        # vanilla MPPI deviates from the prior — there IS a specific
        # evasion direction worth committing to. Argmin finds it.
        temperature = argmin_temperature

Episodes >=1 use the selected temperature. The warmup buffers stay
populated so callers (and the meta dict on returned Plans) can read
back which temperature was picked and why.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import PLANNER_REGISTRY, Plan
from .mppi import MPPIPlanner


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    c = float(np.dot(a, b) / (na * nb))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def _top2_angle(actions: np.ndarray, weights: np.ndarray) -> float:
    if actions.shape[0] < 2:
        return float("nan")
    idx = np.argsort(-weights)
    return _angle_between(actions[idx[0]], actions[idx[1]])


@PLANNER_REGISTRY.register("warmup_select_mppi")
class WarmupSelectMPPIPlanner(MPPIPlanner):
    def __init__(
        self,
        *,
        warmup_temperature: float = 1.0,
        uniform_temperature: float = 10.0,
        argmin_temperature: float = 0.1,
        appl_cut: float = 50.0,
        choice_cut: float = 12.5,
        **mppi_kwargs: Any,
    ) -> None:
        # Force the parent to start at warmup_temperature regardless of
        # what the caller passed via `temperature`. Episode 0 is the
        # warmup pass; the runtime-selected temperature kicks in at the
        # second reset().
        mppi_kwargs["temperature"] = float(warmup_temperature)
        super().__init__(**mppi_kwargs)
        self._warmup_temperature = float(warmup_temperature)
        self._uniform_temperature = float(uniform_temperature)
        self._argmin_temperature = float(argmin_temperature)
        self._appl_cut = float(appl_cut)
        self._choice_cut = float(choice_cut)
        # Mutable state, reset on each call to reset().
        self._episode_idx: int = -1  # -1 means "no reset() called yet"
        self._warm_top2: list[float] = []
        self._warm_cvg: list[float] = []
        self._selected_reason: str = "pending"

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "WarmupSelectMPPIPlanner":
        return cls(
            warmup_temperature=float(cfg.get("warmup_temperature", 1.0)),
            uniform_temperature=float(cfg.get("uniform_temperature", 10.0)),
            argmin_temperature=float(cfg.get("argmin_temperature", 0.1)),
            appl_cut=float(cfg.get("appl_cut", 50.0)),
            choice_cut=float(cfg.get("choice_cut", 12.5)),
            max_speed=float(cfg.get("max_speed", 10.0)),
            horizon=int(cfg.get("horizon", 60)),
            dt_plan=float(cfg.get("dt_plan", 0.05)),
            n_samples=int(cfg.get("n_samples", 32)),
            resolution=float(cfg.get("resolution", 1.0)),
            inflate=int(cfg.get("inflate", 1)),
            goal_radius=float(cfg.get("goal_radius", 1.5)),
            safety_margin=float(cfg.get("safety_margin", 0.4)),
            use_prediction=bool(cfg.get("use_prediction", True)),
            wind=tuple(cfg.get("wind", ())),
            w_goal=float(cfg.get("w_goal", 1.0)),
            w_obs=float(cfg.get("w_obs", 100.0)),
            w_smooth=float(cfg.get("w_smooth", 0.05)),
            predictor=_build_predictor_from_cfg(cfg.get("predictor")),
        )

    def reset(self) -> None:
        super().reset()
        self._episode_idx += 1
        if self._episode_idx == 0:
            self.temperature = self._warmup_temperature
            self._warm_top2.clear()
            self._warm_cvg.clear()
            self._selected_reason = "warmup_pass"
        elif self._episode_idx == 1:
            self.temperature, self._selected_reason = self._select_from_warmup()

    def _select_from_warmup(self) -> tuple[float, str]:
        if not self._warm_top2 or not self._warm_cvg:
            return self._warmup_temperature, "no_warmup_samples_fallback"
        mean_top2 = float(np.nanmean(self._warm_top2))
        mean_cvg = float(np.nanmean(self._warm_cvg))
        if mean_top2 > self._appl_cut:
            return (
                self._warmup_temperature,
                f"chaotic_top2={mean_top2:.1f}>appl_cut={self._appl_cut}",
            )
        if mean_cvg < self._choice_cut:
            return (
                self._uniform_temperature,
                f"prior_aligned_cvg={mean_cvg:.1f}<choice_cut={self._choice_cut}",
            )
        return (
            self._argmin_temperature,
            f"prior_misses_cvg={mean_cvg:.1f}>=choice_cut={self._choice_cut}",
        )

    def plan(
        self,
        observation: np.ndarray,
        goal: np.ndarray,
        obstacle_map: Any,
        *,
        dynamic_obstacles: list[dict] | None = None,
    ) -> Plan:
        plan = super().plan(
            observation, goal, obstacle_map, dynamic_obstacles=dynamic_obstacles
        )
        # Only accumulate during warmup. Subsequent episodes still
        # populate `_last_*` but we ignore them — the selection is
        # frozen.
        if (
            self._episode_idx == 0
            and self._last_actions is not None
            and self._last_weights is not None
            and self._last_chosen_action is not None
            and self._last_goal_dir is not None
        ):
            self._warm_top2.append(_top2_angle(self._last_actions, self._last_weights))
            self._warm_cvg.append(
                _angle_between(self._last_chosen_action, self._last_goal_dir)
            )
        plan.meta = dict(plan.meta or {})
        plan.meta["warmup_select"] = {
            "episode_idx": self._episode_idx,
            "temperature": self.temperature,
            "selected_reason": self._selected_reason,
            "n_warmup_samples": len(self._warm_top2),
        }
        return plan


def _build_predictor_from_cfg(cfg: Mapping[str, Any] | None):
    from ..predictor import build_predictor

    return build_predictor(cfg)
