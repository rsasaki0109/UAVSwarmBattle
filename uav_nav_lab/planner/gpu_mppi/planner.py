"""GPU MPPI planner — orchestrates predict / cache / direction / rollout / aggregate.

The heavy GPU rollout lives in :mod:`.rollout`, the multi-strategy action
selection in :mod:`.aggregator`, and the Dijkstra cost-to-go cache in
:mod:`.ctg_cache`. This class wires those pieces together and converts the
result into a :class:`~uav_nav_lab.planner.base.Plan`.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

try:
    import torch
except ImportError as e:  # pragma: no cover - exercised in subprocess smoke test
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: ImportError | None = e
else:
    _TORCH_IMPORT_ERROR = None

from ...predictor import Predictor, build_predictor
from .._grid import inflate_obstacles, sample_unit_directions
from ..base import PLANNER_REGISTRY, Plan, Planner
from .aggregator import ActionAggregator
from .ctg_cache import CTGCache
from .rollout import run_rollout


def _require_torch() -> Any:
    if torch is None:
        raise SystemExit(
            "gpu_mppi requires PyTorch. Install it with `pip install -e '.[gpu]'` "
            "or choose a non-GPU planner such as `mpc` / `mppi`."
        ) from _TORCH_IMPORT_ERROR
    return torch


@PLANNER_REGISTRY.register("gpu_mppi")
class GPUMPPIPlanner(Planner):
    def __init__(
        self,
        max_speed: float = 10.0,
        horizon: int = 60,
        dt_plan: float = 0.05,
        n_samples: int = 128,
        resolution: float = 1.0,
        inflate: int = 1,
        goal_radius: float = 1.5,
        safety_margin: float = 0.4,
        use_prediction: bool = True,
        wind: tuple[float, ...] = (),
        w_goal: float = 1.0,
        w_obs: float = 100.0,
        w_smooth: float = 0.05,
        temperature: float = 1.0,
        fallback_to_argmin: bool = False,
        fallback_lateral_threshold: float = 0.5,
        fallback_lateral_ratio: float = 0.5,
        fallback_commit_steps: int = 1,
        asymmetric_bias: float = 0.0,
        mode_aware_sampling: bool = False,
        mode_aware_min_size: int = 8,
        mode_aware_cost_ratio: float = 1.0,
        mode_aware_lateral_threshold: float = 0.0,
        mode_aware_lateral_ratio: float = 0.5,
        ctg_cache_tolerance: int = 0,
        viz_rollouts: int = 24,
        predictor: Predictor | None = None,
        device: str = "cuda",
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
        # Mirror aggregator knobs onto self for from_config roundtrip + viz/meta.
        self.fallback_to_argmin = bool(fallback_to_argmin)
        self.fallback_lateral_threshold = float(fallback_lateral_threshold)
        self.fallback_lateral_ratio = float(fallback_lateral_ratio)
        self.fallback_commit_steps = max(1, int(fallback_commit_steps))
        self.asymmetric_bias = float(asymmetric_bias)
        self.mode_aware_sampling = bool(mode_aware_sampling)
        self.mode_aware_min_size = max(1, int(mode_aware_min_size))
        self.mode_aware_cost_ratio = max(1.0, float(mode_aware_cost_ratio))
        self.mode_aware_lateral_threshold = max(0.0, float(mode_aware_lateral_threshold))
        self.mode_aware_lateral_ratio = max(0.0, float(mode_aware_lateral_ratio))
        self.ctg_cache_tolerance = max(0, int(ctg_cache_tolerance))
        self.viz_rollouts = int(viz_rollouts)
        self._predictor: Predictor = (
            predictor if predictor is not None else build_predictor(None)
        )
        self._prev_action: np.ndarray | None = None
        self._bias_vec: np.ndarray | None = None
        self._static_occ_inflated: np.ndarray | None = None
        self._ctg_cache = CTGCache(tolerance=self.ctg_cache_tolerance)
        self._aggregator = ActionAggregator(
            fallback_to_argmin=self.fallback_to_argmin,
            fallback_lateral_threshold=self.fallback_lateral_threshold,
            fallback_lateral_ratio=self.fallback_lateral_ratio,
            fallback_commit_steps=self.fallback_commit_steps,
            mode_aware_sampling=self.mode_aware_sampling,
            mode_aware_min_size=self.mode_aware_min_size,
            mode_aware_cost_ratio=self.mode_aware_cost_ratio,
            mode_aware_lateral_threshold=self.mode_aware_lateral_threshold,
            mode_aware_lateral_ratio=self.mode_aware_lateral_ratio,
            temperature=self.temperature,
        )
        torch_mod = _require_torch()
        self._device = torch_mod.device(device if torch_mod.cuda.is_available() else "cpu")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "GPUMPPIPlanner":
        return cls(
            max_speed=float(cfg.get("max_speed", 10.0)),
            horizon=int(cfg.get("horizon", 60)),
            dt_plan=float(cfg.get("dt_plan", 0.05)),
            n_samples=int(cfg.get("n_samples", 128)),
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
            fallback_to_argmin=bool(cfg.get("fallback_to_argmin", False)),
            fallback_lateral_threshold=float(cfg.get("fallback_lateral_threshold", 0.5)),
            fallback_lateral_ratio=float(cfg.get("fallback_lateral_ratio", 0.5)),
            fallback_commit_steps=int(cfg.get("fallback_commit_steps", 1)),
            asymmetric_bias=float(cfg.get("asymmetric_bias", 0.0)),
            mode_aware_sampling=bool(cfg.get("mode_aware_sampling", False)),
            mode_aware_min_size=int(cfg.get("mode_aware_min_size", 8)),
            mode_aware_cost_ratio=float(cfg.get("mode_aware_cost_ratio", 1.0)),
            mode_aware_lateral_threshold=float(cfg.get("mode_aware_lateral_threshold", 0.0)),
            mode_aware_lateral_ratio=float(cfg.get("mode_aware_lateral_ratio", 0.5)),
            ctg_cache_tolerance=int(cfg.get("ctg_cache_tolerance", 0)),
            viz_rollouts=int(cfg.get("viz_rollouts", 24)),
            predictor=build_predictor(cfg.get("predictor")),
            device=str(cfg.get("device", "cuda")),
        )

    def reset(self) -> None:
        self._prev_action = None
        self._predictor.reset()
        self._static_occ_inflated = None
        self._ctg_cache.reset()
        self._bias_vec = None
        self._aggregator.reset()

    def _cell(self, p: np.ndarray, shape: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            int(np.clip(p[i] / self.resolution, 0, shape[i] - 1)) for i in range(len(shape))
        )

    def _mask_dynamic_cells(self, occ_raw: np.ndarray, d: Mapping[str, Any]) -> None:
        """Zero out cells overlapping a dynamic obstacle so the static cache
        sees only static-only occupancy."""
        pos = np.asarray(d.get("position", ()), dtype=float)
        if pos.size == 0:
            return
        radius = float(d.get("radius", 0.5))
        cells = max(1, int(np.ceil(radius / self.resolution)))
        ndim = occ_raw.ndim
        center = self._cell(pos[:ndim], occ_raw.shape)
        if ndim == 2:
            for dx in range(-cells + 1, cells):
                for dy in range(-cells + 1, cells):
                    cx, cy = center[0] + dx, center[1] + dy
                    if 0 <= cx < occ_raw.shape[0] and 0 <= cy < occ_raw.shape[1]:
                        occ_raw[cx, cy] = False
        else:
            for dx in range(-cells + 1, cells):
                for dy in range(-cells + 1, cells):
                    for dz in range(-cells + 1, cells):
                        cx, cy, cz = center[0] + dx, center[1] + dy, center[2] + dz
                        if (
                            0 <= cx < occ_raw.shape[0]
                            and 0 <= cy < occ_raw.shape[1]
                            and 0 <= cz < occ_raw.shape[2]
                        ):
                            occ_raw[cx, cy, cz] = False

    def _apply_asymmetric_bias(self, base: np.ndarray, obs: np.ndarray, ndim: int) -> np.ndarray:
        """Rotate `base` toward a deterministic per-drone perpendicular axis
        to break L/R rollout symmetry. The bias vector is seeded once from
        the drone's first observation so a multi-drone fleet gets
        non-identical preferred avoidance directions."""
        if self._bias_vec is None:
            seed_int = int(abs(hash(tuple(np.round(obs, 1).tolist()))) & 0xFFFFFFFF)
            rng = np.random.default_rng(seed_int)
            v = rng.standard_normal(ndim)
            vn = float(np.linalg.norm(v))
            self._bias_vec = (v / vn).astype(float) if vn > 1e-9 else np.zeros(ndim, dtype=float)
        bias_perp = self._bias_vec - float(self._bias_vec @ base) * base
        bn = float(np.linalg.norm(bias_perp))
        if bn <= 1e-9:
            return base
        bias_perp = bias_perp / bn
        biased = base + self.asymmetric_bias * bias_perp
        return biased / float(np.linalg.norm(biased))

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
            pred_traj = self._predictor.predict(dynamic_obstacles, horizon_dts)[:, :, :ndim]
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
            return Plan(waypoints=np.asarray([gl], dtype=float), meta={"planner": "gpu_mppi"})

        if occ[self._cell(obs, occ.shape)] or occ[self._cell(gl, occ.shape)]:
            occ = occ_raw

        # Static-only cache so dynamic obstacles don't bias the Dijkstra grid.
        if self._static_occ_inflated is None or self._static_occ_inflated.shape != occ.shape:
            static_raw = occ_raw.copy()
            if dynamic_obstacles:
                for d in dynamic_obstacles:
                    self._mask_dynamic_cells(static_raw, d)
            self._static_occ_inflated = inflate_obstacles(static_raw, self.inflate)
            self._ctg_cache.reset()

        goal_cell = self._cell(gl, self._static_occ_inflated.shape)
        ctg_np = self._ctg_cache.get(self._static_occ_inflated, goal_cell)

        base = to_goal / dist_goal
        if self.asymmetric_bias > 0.0:
            base = self._apply_asymmetric_bias(base, obs, ndim)
        directions = sample_unit_directions(ndim, self.n_samples, base)
        actions_np = directions * self.max_speed

        wind_step = None
        if self._wind is not None and self._wind.size > 0:
            wind_step = np.zeros(ndim)
            nw = min(self._wind.size, ndim)
            wind_step[:nw] = self._wind[:nw]

        rr = run_rollout(
            obs=obs,
            gl=gl,
            actions_np=actions_np,
            occ=occ,
            ctg_np=ctg_np,
            pred_traj=pred_traj,
            r2_arr=r2_arr,
            wind_step=wind_step,
            prev_action=self._prev_action,
            horizon=self.horizon,
            dt_plan=self.dt_plan,
            resolution=self.resolution,
            goal_radius=self.goal_radius,
            n_samples=self.n_samples,
            w_goal=self.w_goal,
            w_obs=self.w_obs,
            w_smooth=self.w_smooth,
            temperature=self.temperature,
            device=self._device,
        )

        agg = self._aggregator.select(
            actions_t=rr.actions_t,
            costs=rr.costs,
            weights=rr.weights,
            softmax_action=rr.softmax_action,
            argmin_action=rr.argmin_action,
            argmin_idx=rr.argmin_idx,
            base=base,
            device=self._device,
        )

        chosen_action_t = agg.chosen_action
        torch_mod = _require_torch()
        speed = torch_mod.norm(chosen_action_t)
        if speed > self.max_speed:
            chosen_action_t = chosen_action_t * (self.max_speed / speed)

        best_k = agg.best_k
        best_rollout = rr.rollouts[best_k].cpu().numpy()
        best_full = np.concatenate([obs.reshape(1, -1), best_rollout], axis=0)
        if rr.reaches_goal_any[best_k]:
            best_full = best_full[: rr.first_goal_h[best_k].item() + 2]

        k_vis = min(self.viz_rollouts, self.n_samples)
        if k_vis > 0:
            stride = max(1, self.n_samples // k_vis)
            rollouts_vis_t = rr.rollouts[::stride][:k_vis]
            obs_t = torch_mod.as_tensor(obs, dtype=torch_mod.float32, device=self._device)
            obs_prefix = obs_t[None, None, :].expand(rollouts_vis_t.shape[0], 1, ndim)
            rollouts_vis = torch_mod.cat([obs_prefix, rollouts_vis_t], dim=1).cpu().numpy()
            best_vis_idx = min(best_k // stride, rollouts_vis.shape[0] - 1)
            rollouts_meta = np.round(rollouts_vis, 3).tolist()
        else:
            rollouts_meta = None
            best_vis_idx = 0

        chosen_action = chosen_action_t.cpu().numpy()
        self._prev_action = chosen_action

        return Plan(
            waypoints=best_full[1:],
            target_velocity=chosen_action,
            meta={
                "planner": "gpu_mppi",
                "cost_min": float(rr.cost_min.item()),
                "weight_max": float(rr.weights.max().item()),
                "weight_entropy": float(
                    (-rr.weights * torch_mod.log(rr.weights + 1e-12)).sum().item()
                ),
                "n_samples": self.n_samples,
                "device": str(self._device),
                "rollouts": rollouts_meta,
                "best_rollout_idx": int(best_vis_idx),
                "fallback_to_argmin": bool(agg.fallback_triggered),
                "mode_aware_triggered": bool(agg.mode_aware_triggered),
                "mode_aware_cluster_sign": int(agg.mode_aware_cluster_sign),
            },
        )
