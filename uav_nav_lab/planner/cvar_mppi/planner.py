"""Risk-aware MPPI via CVaR over sampled prediction futures.

Same rollout/cost model and softmax update as :class:`MPPIPlanner` — the only
change is *what scalar cost each candidate action is scored with*:

  MPPI      : one deterministic rollout cost (expected case under the
              predictor's point estimate).
  CVaR-MPPI : score each action against ``n_scenarios`` perturbed copies of
              the predicted dynamic-obstacle trajectories (noise grows with
              look-ahead time), then take the CVaR — the mean of the worst
              ``risk_alpha`` tail — as the action's cost.

So an action that threads a gap only if every obstacle moves exactly as
predicted is penalised by its bad tail and the softmax steers away from it.
With no dynamic obstacles, or ``risk_alpha = 1.0``, every scenario is
identical / fully averaged and this reduces to vanilla MPPI — making the two
directly comparable (any difference is the risk term, not the cost model).

The cost model is reused verbatim from :mod:`uav_nav_lab.planner._rollout`,
so head-to-head ablations against ``mppi`` isolate the CVaR contribution.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .._grid import (
    dijkstra_cost_to_go,
    inflate_obstacles,
    mask_dynamic_obstacle_cells,
    point_to_cell,
    sample_unit_directions,
)
from .._rollout import score_rollouts
from ..base import PLANNER_REGISTRY, Plan
from ..mppi.aggregator import softmax_aggregate
from ..mppi.planner import MPPIPlanner
from .cvar import cvar_costs


@PLANNER_REGISTRY.register("cvar_mppi")
class CVaRMPPIPlanner(MPPIPlanner):
    """MPPI that minimises the CVaR of cost over sampled prediction futures.

    Adds three knobs on top of :class:`MPPIPlanner`:

    - ``n_scenarios``  : how many perturbed futures to score each action
      against (1 → deterministic, == vanilla MPPI).
    - ``risk_alpha``   : worst-case tail fraction averaged into the cost
      (1.0 → risk-neutral mean; smaller → more conservative). See
      :func:`cvar_costs`.
    - ``pred_noise_std``: per-second std (m/√s) of the Gaussian perturbation
      added to predicted obstacle positions; the perturbation at look-ahead
      time t scales with √t, so uncertainty grows the further out we predict.
    """

    def __init__(
        self,
        *,
        n_scenarios: int = 16,
        risk_alpha: float = 0.2,
        pred_noise_std: float = 0.5,
        base_seed: int = 0,
        **mppi_kwargs: Any,
    ) -> None:
        super().__init__(**mppi_kwargs)
        self.n_scenarios = int(n_scenarios)
        if self.n_scenarios < 1:
            raise ValueError(f"n_scenarios must be >= 1; got {n_scenarios}")
        if not (0.0 < risk_alpha <= 1.0):
            raise ValueError(f"risk_alpha must be in (0, 1]; got {risk_alpha!r}")
        self.risk_alpha = float(risk_alpha)
        self.pred_noise_std = float(pred_noise_std)
        self.base_seed = int(base_seed)
        self._episode = -1
        self._rng = np.random.default_rng(self.base_seed)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "CVaRMPPIPlanner":
        # Reuse the parent's parsing for the shared MPPI knobs, then layer the
        # CVaR-specific ones on top.
        base = MPPIPlanner.from_config(cfg)
        mppi_kwargs = dict(
            max_speed=base.max_speed,
            horizon=base.horizon,
            dt_plan=base.dt_plan,
            n_samples=base.n_samples,
            resolution=base.resolution,
            inflate=base.inflate,
            goal_radius=base.goal_radius,
            safety_margin=base.safety_margin,
            use_prediction=base.use_prediction,
            wind=tuple(base._wind) if base._wind is not None else (),
            w_goal=base.w_goal,
            w_obs=base.w_obs,
            w_smooth=base.w_smooth,
            temperature=base.temperature,
            predictor=base._predictor,
        )
        return cls(
            n_scenarios=int(cfg.get("n_scenarios", 16)),
            risk_alpha=float(cfg.get("risk_alpha", 0.2)),
            pred_noise_std=float(cfg.get("pred_noise_std", 0.5)),
            base_seed=int(cfg.get("base_seed", 0)),
            **mppi_kwargs,
        )

    def reset(self) -> None:
        super().reset()
        # Standalone determinism (e.g. unit tests that call reset() directly,
        # no runner): decorrelate scenario noise across successive resets while
        # staying reproducible. The runner additionally calls seed_episode()
        # right after this to key the RNG on the real episode seed.
        self._episode += 1
        self._rng = np.random.default_rng(self.base_seed + self._episode)

    # Offset added to the episode seed so the perturbation RNG does not share a
    # stream with the sim/sensor/predictor, which are seeded from the same
    # episode seed (sensor at `seed`, predictor at `seed + 7777`). default_rng(N)
    # yields an identical bit stream regardless of consumer, so an un-offset key
    # would correlate the CVaR scenario noise with a stochastic sensor's
    # position noise. Distinct from the predictor's 7777 for the same reason.
    _SEED_OFFSET = 90001

    def seed_episode(self, seed: int) -> None:
        # Key the scenario-perturbation RNG on the actual episode seed (like
        # the predictor's reseed), so re-running a single episode in isolation
        # reproduces the same perturbed futures as when it ran inside a batch.
        self._rng = np.random.default_rng(self.base_seed + int(seed) + self._SEED_OFFSET)

    def _perturbed_predictions(
        self, pred_traj: np.ndarray, horizon_dts: np.ndarray
    ) -> np.ndarray:
        """Return [n_scenarios, n_obs, horizon, ndim] perturbed predictions.

        Noise std at look-ahead time t is ``pred_noise_std * sqrt(t)`` (a
        Brownian-like growth of uncertainty), so near-term predictions are
        trusted more than far-term ones. Scenario 0 is always the unperturbed
        prediction so the risk-neutral nominal is represented.
        """
        n_obs, horizon, ndim = pred_traj.shape
        out = np.broadcast_to(
            pred_traj, (self.n_scenarios, n_obs, horizon, ndim)
        ).copy()
        if self.pred_noise_std <= 0.0 or self.n_scenarios == 1:
            return out
        std_t = self.pred_noise_std * np.sqrt(np.maximum(horizon_dts, 0.0))  # [horizon]
        noise = self._rng.normal(
            0.0, 1.0, size=(self.n_scenarios - 1, n_obs, horizon, ndim)
        )
        noise = noise * std_t[None, None, :, None]
        out[1:] += noise
        return out

    def plan(
        self,
        observation: np.ndarray,
        goal: np.ndarray,
        obstacle_map: Any,
        *,
        dynamic_obstacles: list[dict] | None = None,
    ) -> Plan:
        occ_raw = np.asarray(obstacle_map, dtype=bool)
        ndim = occ_raw.ndim
        obs = np.asarray(observation, dtype=float)[:ndim]
        gl = np.asarray(goal, dtype=float)[:ndim]

        to_goal = gl - obs
        dist_goal = float(np.linalg.norm(to_goal))
        if dist_goal < 1e-6:
            return Plan(waypoints=np.asarray([gl], dtype=float),
                        meta={"planner": "cvar_mppi"})

        # Predicted dynamic-obstacle trajectories (shared with vanilla MPPI).
        if self.use_prediction and dynamic_obstacles:
            horizon_dts = np.arange(1, self.horizon + 1, dtype=float) * self.dt_plan
            pred_traj = self._predictor.predict(dynamic_obstacles, horizon_dts)[:, :, :ndim]
            r2_arr = np.array(
                [(float(d.get("radius", 0.5)) + self.safety_margin) ** 2
                 for d in dynamic_obstacles],
                dtype=float,
            )
            scenarios = self._perturbed_predictions(pred_traj, horizon_dts)
        else:
            pred_traj = None
            r2_arr = None
            scenarios = None

        occ = inflate_obstacles(occ_raw, self.inflate)
        if occ[point_to_cell(obs, occ.shape, self.resolution)] or \
                occ[point_to_cell(gl, occ.shape, self.resolution)]:
            occ = occ_raw

        # Static cost-to-go cache (identical to MPPIPlanner).
        if self._static_occ_inflated is None or self._static_occ_inflated.shape != occ.shape:
            static_raw = occ_raw.copy()
            if dynamic_obstacles:
                for d in dynamic_obstacles:
                    mask_dynamic_obstacle_cells(static_raw, d, self.resolution)
            self._static_occ_inflated = inflate_obstacles(static_raw, self.inflate)
            self._ctg_cache = None
            self._ctg_cache_goal = None

        goal_cell = point_to_cell(gl, self._static_occ_inflated.shape, self.resolution)
        if self._ctg_cache is None or self._ctg_cache_goal != goal_cell:
            self._ctg_cache = dijkstra_cost_to_go(self._static_occ_inflated, goal_cell)
            self._ctg_cache_goal = goal_cell
        ctg = self._ctg_cache
        max_finite = float(np.max(ctg[np.isfinite(ctg)])) if np.any(np.isfinite(ctg)) else 1e6
        unreachable_penalty = max_finite + 100.0

        base = to_goal / dist_goal
        directions = sample_unit_directions(ndim, self.n_samples, base)
        actions = directions * self.max_speed
        if self._wind is not None and self._wind.size > 0:
            wind_step = np.zeros(ndim)
            n = min(self._wind.size, ndim)
            wind_step[:n] = self._wind[:n]
        else:
            wind_step = None

        def _score(pred: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
            return score_rollouts(
                actions=actions, obs=obs, gl=gl, occ=occ, ctg=ctg,
                unreachable_penalty=unreachable_penalty,
                horizon=self.horizon, dt_plan=self.dt_plan,
                resolution=self.resolution, goal_radius=self.goal_radius,
                pred_traj=pred, r2_arr=r2_arr, wind_step=wind_step,
                prev_action=self._prev_action,
                w_goal=self.w_goal, w_obs=self.w_obs, w_smooth=self.w_smooth,
            )

        if scenarios is None:
            # No prediction → deterministic, identical to vanilla MPPI.
            costs, rollouts = _score(None)
            cvar = costs
            n_eff_scenarios = 1
        else:
            # Score each candidate action against every perturbed future.
            cost_cols = []
            rollouts = None
            for k in range(scenarios.shape[0]):
                c_k, roll_k = _score(scenarios[k])
                cost_cols.append(c_k)
                if rollouts is None:
                    rollouts = roll_k  # rollouts are scenario-independent
            cost_matrix = np.stack(cost_cols, axis=1)  # [n_samples, n_scenarios]
            cvar = cvar_costs(cost_matrix, self.risk_alpha)
            n_eff_scenarios = scenarios.shape[0]

        # Softmax over the risk-adjusted costs (same update as MPPI).
        chosen_action, weights, cost_min, best_k = softmax_aggregate(
            cvar, actions, self.temperature,
        )
        self._last_costs = cvar.copy()
        self._last_weights = weights.copy()
        self._last_chosen_action = chosen_action.copy()
        self._last_actions = actions.copy()
        gd_norm = float(np.linalg.norm(to_goal))
        self._last_goal_dir = (to_goal / gd_norm) if gd_norm > 1e-9 else np.zeros_like(to_goal)

        speed = float(np.linalg.norm(chosen_action))
        if speed > self.max_speed:
            chosen_action = chosen_action * (self.max_speed / speed)
        best_rollout = rollouts[best_k]
        self._prev_action = chosen_action

        return Plan(
            waypoints=best_rollout[1:],
            target_velocity=chosen_action,
            meta={
                "planner": "cvar_mppi",
                "cost_min": cost_min,
                "risk_alpha": self.risk_alpha,
                "n_scenarios": n_eff_scenarios,
                "weight_max": float(weights.max()),
                "weight_entropy": float(-np.sum(weights * np.log(weights + 1e-12))),
            },
        )
