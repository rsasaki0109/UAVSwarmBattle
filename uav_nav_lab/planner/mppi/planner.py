"""MPPI planner orchestration.

Same per-step pipeline as :class:`uav_nav_lab.planner.mpc.SamplingMPCPlanner`
(Dijkstra cost-to-go heuristic, n direction samples covering 2D circle /
3D Fibonacci sphere, horizon-step rollouts scored by goal + obstacle +
smoothness terms), with one substitution at the end:

  argmin → softmax-weighted average

The per-sample cost compute lives in :mod:`.rollout`; the softmax
selection lives in :mod:`.aggregator`. This file only orchestrates:
setup (occupancy / cost-to-go cache / direction sampling), delegation
to the two helpers, and the final :class:`.base.Plan` packaging.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...predictor import Predictor, build_predictor
from .._grid import (
    dijkstra_cost_to_go,
    inflate_obstacles,
    mask_dynamic_obstacle_cells,
    point_to_cell,
    sample_unit_directions,
)
from ..base import PLANNER_REGISTRY, Plan, Planner
from .aggregator import softmax_aggregate
from .rollout import score_rollouts


@PLANNER_REGISTRY.register("mppi")
class MPPIPlanner(Planner):
    def __init__(
        self,
        max_speed: float = 10.0,
        horizon: int = 60,
        dt_plan: float = 0.05,
        n_samples: int = 32,
        resolution: float = 1.0,
        inflate: int = 1,
        goal_radius: float = 1.5,
        safety_margin: float = 0.4,
        use_prediction: bool = True,
        wind: tuple[float, ...] = (),
        w_goal: float = 1.0,
        w_obs: float = 100.0,
        w_smooth: float = 0.05,
        # MPPI-specific. Lower → behaves like argmin MPC; higher → behaves
        # like uniform action average (a degenerate "average of all
        # directions" → speed collapse). Default 1.0 was identified as
        # the sweet spot on the predictive scenario by the temperature
        # sweep in `examples/exp_compare_mppi.yaml` — see that YAML's
        # header for the full table.
        temperature: float = 1.0,
        predictor: Predictor | None = None,
    ) -> None:
        self.max_speed = float(max_speed)
        self.horizon = int(horizon)
        self.dt_plan = float(dt_plan)
        self.n_samples = int(n_samples)
        self.resolution = float(resolution)
        self.inflate = int(inflate)
        self.goal_radius = float(goal_radius)
        self.safety_margin = float(safety_margin)
        self.use_prediction = bool(use_prediction)
        self._wind = np.asarray(wind, dtype=float) if wind else None
        self.w_goal = float(w_goal)
        self.w_obs = float(w_obs)
        self.w_smooth = float(w_smooth)
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0; got {temperature!r}")
        self.temperature = float(temperature)
        self._predictor: Predictor = (
            predictor if predictor is not None else build_predictor(None)
        )
        self._prev_action: np.ndarray | None = None
        # Same per-episode caches as MPC. Cleared by reset().
        self._static_occ_inflated: np.ndarray | None = None
        self._ctg_cache: np.ndarray | None = None
        self._ctg_cache_goal: tuple[int, ...] | None = None
        # Last-replan internals — exposed for mechanism analysis. Not
        # written to disk by the runner; standalone scripts can opt in
        # by reading these after each call to plan().
        self._last_costs: np.ndarray | None = None
        self._last_weights: np.ndarray | None = None
        self._last_chosen_action: np.ndarray | None = None
        self._last_actions: np.ndarray | None = None  # [n_samples, ndim]
        self._last_goal_dir: np.ndarray | None = None  # unit vector toward goal

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "MPPIPlanner":
        return cls(
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
            temperature=float(cfg.get("temperature", 1.0)),
            predictor=build_predictor(cfg.get("predictor")),
        )

    def reset(self) -> None:
        self._prev_action = None
        self._predictor.reset()
        self._static_occ_inflated = None
        self._ctg_cache = None
        self._ctg_cache_goal = None

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
        if self.use_prediction and dynamic_obstacles:
            horizon_dts = np.arange(1, self.horizon + 1, dtype=float) * self.dt_plan
            pred_traj = self._predictor.predict(
                dynamic_obstacles, horizon_dts
            )[:, :, :ndim]
            r2_arr = np.array(
                [(float(d.get("radius", 0.5)) + self.safety_margin) ** 2 for d in dynamic_obstacles],
                dtype=float,
            )
        else:
            pred_traj = None
            r2_arr = None
        occ = inflate_obstacles(occ_raw, self.inflate)
        obs = np.asarray(observation, dtype=float)[:ndim]
        gl = np.asarray(goal, dtype=float)[:ndim]

        to_goal = gl - obs
        dist_goal = float(np.linalg.norm(to_goal))
        if dist_goal < 1e-6:
            return Plan(waypoints=np.asarray([gl], dtype=float), meta={"planner": "mppi"})

        if occ[point_to_cell(obs, occ.shape, self.resolution)] or \
                occ[point_to_cell(gl, occ.shape, self.resolution)]:
            occ = occ_raw

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

        # Per-sample rollout scoring (no early-out — softmax needs all costs).
        costs, rollouts = score_rollouts(
            actions=actions, obs=obs, gl=gl, occ=occ, ctg=ctg,
            unreachable_penalty=unreachable_penalty,
            horizon=self.horizon, dt_plan=self.dt_plan,
            resolution=self.resolution, goal_radius=self.goal_radius,
            pred_traj=pred_traj, r2_arr=r2_arr, wind_step=wind_step,
            prev_action=self._prev_action,
            w_goal=self.w_goal, w_obs=self.w_obs, w_smooth=self.w_smooth,
        )

        # Softmax aggregation.
        chosen_action, weights, cost_min, best_k = softmax_aggregate(
            costs, actions, self.temperature,
        )
        self._last_costs = costs.copy()
        self._last_weights = weights.copy()
        self._last_chosen_action = chosen_action.copy()
        self._last_actions = actions.copy()
        gd_norm = float(np.linalg.norm(to_goal))
        self._last_goal_dir = (to_goal / gd_norm) if gd_norm > 1e-9 else np.zeros_like(to_goal)
        # Cap the weighted action to max_speed (the average can drop below
        # max_speed when weights are spread; that's expected). The
        # rollout we attach to the Plan is the highest-weighted sample's
        # — for visualisation / pure-pursuit fallback only.
        speed = float(np.linalg.norm(chosen_action))
        if speed > self.max_speed:
            chosen_action = chosen_action * (self.max_speed / speed)
        best_rollout = rollouts[best_k]
        self._prev_action = chosen_action

        return Plan(
            waypoints=best_rollout[1:],
            target_velocity=chosen_action,
            meta={
                "planner": "mppi",
                "cost_min": cost_min,
                "weight_max": float(weights.max()),
                "weight_entropy": float(-np.sum(weights * np.log(weights + 1e-12))),
            },
        )
