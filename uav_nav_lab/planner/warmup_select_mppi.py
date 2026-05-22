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

Multi-drone runs default to `share_warmup=True`: planner instances
built from the same config pool their warmup samples into a single
shared session (keyed by `share_warmup_key`, default "_default") and
the N+P rule fires once against the pooled means, so every drone in
the run adopts the same temperature for ep 1+. This matches how the
N+P rule was calibrated (per-cell pooled means) and prevents the
per-drone selection drift observed on v1 in the first cut, where two
drones whose individual cvg signals straddled the choice_cut picked
opposite temperatures. Opt out with `share_warmup: false` to recover
per-drone behavior.
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


class _WarmupSession:
    """Pooled warmup state shared across planner instances in a run.

    Members write into `top2` / `cvg` during their ep 0 plan() calls.
    The first member to hit ep 1 in reset() picks a temperature from
    the pooled means and stores it on the session; subsequent members
    read it back instead of recomputing from their own buffer."""

    __slots__ = ("top2", "cvg", "selected_temperature", "selected_reason")

    def __init__(self) -> None:
        self.top2: list[float] = []
        self.cvg: list[float] = []
        self.selected_temperature: float | None = None
        self.selected_reason: str | None = None

    def reset_for_ep0(self) -> None:
        self.top2.clear()
        self.cvg.clear()
        self.selected_temperature = None
        self.selected_reason = None


# Module-level registry of active warmup sessions, keyed by
# share_warmup_key. Cleared on each member's ep 0 reset (idempotent — all
# members clear to the same empty state before any plan() runs).
_SHARED_SESSIONS: dict[str, _WarmupSession] = {}


def _get_session(key: str | None) -> _WarmupSession | None:
    if key is None:
        return None
    return _SHARED_SESSIONS.setdefault(key, _WarmupSession())


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
        share_warmup: bool = True,
        share_warmup_key: str = "_default",
        **mppi_kwargs: Any,
    ) -> None:
        mppi_kwargs["temperature"] = float(warmup_temperature)
        super().__init__(**mppi_kwargs)
        self._warmup_temperature = float(warmup_temperature)
        self._uniform_temperature = float(uniform_temperature)
        self._argmin_temperature = float(argmin_temperature)
        self._appl_cut = float(appl_cut)
        self._choice_cut = float(choice_cut)
        self._share_warmup_key: str | None = (
            str(share_warmup_key) if share_warmup else None
        )
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
            share_warmup=bool(cfg.get("share_warmup", True)),
            share_warmup_key=str(cfg.get("share_warmup_key", "_default")),
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
        session = _get_session(self._share_warmup_key)
        if self._episode_idx == 0:
            self.temperature = self._warmup_temperature
            self._warm_top2.clear()
            self._warm_cvg.clear()
            if session is not None:
                # Idempotent — all members of the session clear to the
                # same empty state before any plan() runs in ep 0.
                session.reset_for_ep0()
            self._selected_reason = "warmup_pass"
        elif self._episode_idx == 1:
            self.temperature, self._selected_reason = self._select_from_warmup(session)

    def _select_from_warmup(
        self, session: _WarmupSession | None
    ) -> tuple[float, str]:
        # If another member of the session already picked, adopt it.
        if session is not None and session.selected_temperature is not None:
            return session.selected_temperature, session.selected_reason or "shared"
        # Otherwise compute from pooled (or local-only) buffers.
        if session is not None:
            top2 = session.top2
            cvg = session.cvg
        else:
            top2 = self._warm_top2
            cvg = self._warm_cvg
        if not top2 or not cvg:
            decision = (self._warmup_temperature, "no_warmup_samples_fallback")
        else:
            mean_top2 = float(np.nanmean(top2))
            mean_cvg = float(np.nanmean(cvg))
            tag = "pooled" if session is not None else "local"
            if mean_top2 > self._appl_cut:
                decision = (
                    self._warmup_temperature,
                    f"chaotic_{tag}_top2={mean_top2:.1f}>appl_cut={self._appl_cut}",
                )
            elif mean_cvg < self._choice_cut:
                decision = (
                    self._uniform_temperature,
                    f"prior_aligned_{tag}_cvg={mean_cvg:.1f}<choice_cut={self._choice_cut}",
                )
            else:
                decision = (
                    self._argmin_temperature,
                    f"prior_misses_{tag}_cvg={mean_cvg:.1f}>=choice_cut={self._choice_cut}",
                )
        if session is not None:
            session.selected_temperature = decision[0]
            session.selected_reason = decision[1]
        return decision

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
        if (
            self._episode_idx == 0
            and self._last_actions is not None
            and self._last_weights is not None
            and self._last_chosen_action is not None
            and self._last_goal_dir is not None
        ):
            top2 = _top2_angle(self._last_actions, self._last_weights)
            cvg = _angle_between(self._last_chosen_action, self._last_goal_dir)
            self._warm_top2.append(top2)
            self._warm_cvg.append(cvg)
            session = _get_session(self._share_warmup_key)
            if session is not None:
                session.top2.append(top2)
                session.cvg.append(cvg)
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
