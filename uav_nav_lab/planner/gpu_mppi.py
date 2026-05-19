"""GPU MPPI — batched PyTorch rollout with softmax-weighted action selection.

Same interface as `MPPIPlanner` but the per-sample rollout loop is replaced
by a single batched tensor operation on GPU.  For n_samples=128/256 this
brings plan_dt from O(1000 ms) to O(10 ms), unlocking the rightward shift
of the compute Pareto curve.
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

from ..predictor import Predictor, build_predictor
from ._grid import dijkstra_cost_to_go, inflate_obstacles, sample_unit_directions
from .base import PLANNER_REGISTRY, Plan, Planner


def _require_torch() -> Any:
    if torch is None:
        raise SystemExit(
            "gpu_mppi requires PyTorch. Install it with `pip install -e '.[gpu]'` "
            "or choose a non-GPU planner such as `mpc` / `mppi`."
        ) from _TORCH_IMPORT_ERROR
    return torch


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


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
        self.fallback_to_argmin = bool(fallback_to_argmin)
        self.fallback_lateral_threshold = float(fallback_lateral_threshold)
        self.fallback_lateral_ratio = float(fallback_lateral_ratio)
        # Once the bidirectional-cancellation detector fires, hold the
        # argmin-fallback choice for `fallback_commit_steps` consecutive
        # replans before re-evaluating. K=1 = per-replan (Smart v1
        # behaviour); K=5 ≈ 1 s at replan_period=0.2 s (Smart v3 default).
        self.fallback_commit_steps = max(1, int(fallback_commit_steps))
        self._fallback_commit_remaining = 0
        # Asymmetric perturbation: a per-episode, perpendicular-to-goal
        # bias added to the rollout base direction. The goal is to break
        # the L/R symmetry that produces softmax cancellation under
        # symmetric dynamic obstacles (§3 Table 2 mechanism). The bias
        # vector is derived deterministically from the drone's initial
        # observation, so each drone gets a different (but consistent)
        # preferred avoidance direction.
        self.asymmetric_bias = float(asymmetric_bias)
        self._bias_vec: np.ndarray | None = None
        # Mode-aware sampling (Smart MPPI v4): instead of taking the softmax
        # mean over ALL rollouts (which cancels left/right escapes under a
        # bidirectional cancellation regime, §3 Table 2), project lateral
        # action components onto their first principal direction, split
        # rollouts into L/R clusters by sign, and emit the softmax-weighted
        # action of the lower-cost cluster only. This commits to one side
        # while keeping MPPI's smoothing within that side.
        self.mode_aware_sampling = bool(mode_aware_sampling)
        self.mode_aware_min_size = max(1, int(mode_aware_min_size))
        # `mode_aware_cost_ratio` gates the mode-aware cluster commit on
        # cluster-cost asymmetry. Default 1.0 = always commit when the
        # L/R cluster sizes are above `mode_aware_min_size` (Smart v4
        # behaviour). Setting > 1.0 turns v4 into a *switcher*: commit
        # only when one cluster's softmax cost is at least this factor
        # higher than the other's, i.e. when bimodality is driven by a
        # *real* cost asymmetry (dynamic-obstacle hit penalty on one
        # side, the §3 mode 2 cancellation signature) — and fall through
        # to vanilla softmax otherwise (which keeps mode 1 static-peer
        # clustering and mode 4 aerobatic smoothing intact).
        self.mode_aware_cost_ratio = max(1.0, float(mode_aware_cost_ratio))
        # `ctg_cache_tolerance`: integer cells of slack for the Dijkstra
        # cost-to-go cache. Default 0 → recompute every time the integer
        # goal cell changes (per-replan, exact). For scenarios with a
        # *moving* lookahead goal (e.g. `multi_drone_aerobatic` /
        # `multi_drone_race`) the goal cell can change every replan and
        # Dijkstra dominates wallclock; setting tolerance to 2–3 cells
        # caches across small lookahead drifts and gives a ~5–10x runner
        # speedup at a negligible cost-to-go staleness (~few cells of
        # approximation in a 40-cell-wide world).
        self.ctg_cache_tolerance = max(0, int(ctg_cache_tolerance))
        self.viz_rollouts = int(viz_rollouts)
        self._predictor: Predictor = (
            predictor if predictor is not None else build_predictor(None)
        )
        self._prev_action: np.ndarray | None = None
        self._static_occ_inflated: np.ndarray | None = None
        self._ctg_cache: np.ndarray | None = None
        self._ctg_cache_goal: tuple[int, ...] | None = None
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
            ctg_cache_tolerance=int(cfg.get("ctg_cache_tolerance", 0)),
            viz_rollouts=int(cfg.get("viz_rollouts", 24)),
            predictor=build_predictor(cfg.get("predictor")),
            device=str(cfg.get("device", "cuda")),
        )

    def reset(self) -> None:
        self._prev_action = None
        self._predictor.reset()
        self._static_occ_inflated = None
        self._ctg_cache = None
        self._ctg_cache_goal = None
        self._bias_vec = None  # lazily set on first plan() call
        self._fallback_commit_remaining = 0

    def _cell(self, p: np.ndarray, shape: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            int(np.clip(p[i] / self.resolution, 0, shape[i] - 1)) for i in range(len(shape))
        )

    def _mask_dynamic_cells(self, occ_raw: np.ndarray, d: Mapping[str, Any]) -> None:
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

        # Dijkstra cost-to-go — computed on CPU, cached per episode (minor cost)
        if self._static_occ_inflated is None or self._static_occ_inflated.shape != occ.shape:
            static_raw = occ_raw.copy()
            if dynamic_obstacles:
                for d in dynamic_obstacles:
                    self._mask_dynamic_cells(static_raw, d)
            self._static_occ_inflated = inflate_obstacles(static_raw, self.inflate)
            self._ctg_cache = None
            self._ctg_cache_goal = None

        goal_cell = self._cell(gl, self._static_occ_inflated.shape)
        if self._ctg_cache is None or self._ctg_cache_goal is None:
            need_recompute = True
        elif self.ctg_cache_tolerance <= 0:
            need_recompute = self._ctg_cache_goal != goal_cell
        else:
            # Cache hit when the goal has drifted by at most
            # `ctg_cache_tolerance` cells along *every* axis.
            need_recompute = any(
                abs(int(a) - int(b)) > self.ctg_cache_tolerance
                for a, b in zip(goal_cell, self._ctg_cache_goal)
            )
        if need_recompute:
            self._ctg_cache = dijkstra_cost_to_go(self._static_occ_inflated, goal_cell)
            self._ctg_cache_goal = goal_cell
        ctg_np = self._ctg_cache

        base = to_goal / dist_goal
        # Asymmetric perturbation: rotate `base` by a small angle toward a
        # deterministic per-drone perpendicular axis to break L/R rollout
        # symmetry. The bias direction is seeded once from the drone's
        # initial observation so each drone in a multi-drone fleet gets
        # a different preferred avoidance side.
        if self.asymmetric_bias > 0.0:
            if self._bias_vec is None:
                seed_int = int(
                    abs(hash(tuple(np.round(obs, 1).tolist()))) & 0xFFFFFFFF
                )
                rng = np.random.default_rng(seed_int)
                v = rng.standard_normal(ndim)
                vn = float(np.linalg.norm(v))
                if vn > 1e-9:
                    self._bias_vec = (v / vn).astype(float)
                else:
                    self._bias_vec = np.zeros(ndim, dtype=float)
            # Project bias_vec perpendicular to current `base`
            bias_perp = self._bias_vec - float(self._bias_vec @ base) * base
            bn = float(np.linalg.norm(bias_perp))
            if bn > 1e-9:
                bias_perp = bias_perp / bn
                biased_base = base + self.asymmetric_bias * bias_perp
                biased_base = biased_base / float(np.linalg.norm(biased_base))
                base = biased_base
        directions = sample_unit_directions(ndim, self.n_samples, base)
        actions_np = directions * self.max_speed

        wind_step = None
        if self._wind is not None and self._wind.size > 0:
            wind_step = np.zeros(ndim)
            nw = min(self._wind.size, ndim)
            wind_step[:nw] = self._wind[:nw]

        # --- GPU batched rollout ---
        device = self._device
        occ_t = _to_tensor(occ.astype(np.float32), device)
        ctg_t = _to_tensor(ctg_np, device)
        obs_t = _to_tensor(obs, device)
        gl_t = _to_tensor(gl, device)
        actions_t = _to_tensor(actions_np, device)
        gr2 = self.goal_radius ** 2
        max_finite = float(ctg_np[ctg_np < np.inf].max()) if np.any(ctg_np < np.inf) else 1e6
        unreachable_penalty = max_finite + 100.0

        # Rollout: obs + v * dt * h  for all samples and horizon steps
        # Shape: [n_samples, horizon, ndim]
        dt = self.dt_plan
        h = torch.arange(1, self.horizon + 1, dtype=torch.float32, device=device) * dt  # [H]
        rollouts = obs_t[None, None, :] + actions_t[:, None, :] * h[None, :, None]  # [S, H, D]
        if wind_step is not None:
            ws_t = _to_tensor(wind_step, device)
            rollouts = rollouts + ws_t[None, None, :] * h[None, :, None]

        # Cell indices for occupancy check
        shape_t = torch.tensor(list(occ.shape), dtype=torch.long, device=device)
        cell_indices_float = rollouts / self.resolution  # [S, H, D]
        cell_indices_raw = cell_indices_float.long()
        # Out-of-bounds detection: OOB counts as obstacle
        oob = (
            (cell_indices_raw < 0)
            | (cell_indices_raw >= shape_t[None, None, :])
        ).any(dim=-1)  # [S, H]
        cell_indices = cell_indices_raw.clamp(
            torch.zeros_like(shape_t), shape_t - 1
        )  # [S, H, D]

        # Collision: gather occupancy values at cell indices + OOB
        if ndim == 2:
            occ_collision = occ_t[cell_indices[:, :, 0], cell_indices[:, :, 1]].float()  # [S, H]
        else:
            occ_collision = occ_t[cell_indices[:, :, 0], cell_indices[:, :, 1], cell_indices[:, :, 2]].float()
        collision_mask = occ_collision + oob.float()  # OOB = collision

        # CTG: gather cost-to-go values at each rollout position (needed
        # below; pre-computed here so we can also use the goal-reach mask
        # to scope collision sums to pre-goal steps — matching CPU MPPI's
        # `break` after the first goal reach so rollouts aren't penalised
        # for incidental obstacles past the goal).
        dist2 = ((rollouts - gl_t[None, None, :]) ** 2).sum(dim=-1)  # [S, H]
        reaches_goal_any = (dist2 <= gr2).any(dim=1)  # [S]
        first_goal_h = torch.where(
            reaches_goal_any,
            (dist2 <= gr2).float().argmax(dim=1),
            torch.tensor(self.horizon, device=device),
        )
        step_idx = torch.arange(self.horizon, device=device)  # [H]
        pre_goal_mask = (step_idx[None, :] < first_goal_h[:, None]).float()  # [S, H]

        collision_pen = (collision_mask * pre_goal_mask).sum(dim=1)  # [S]
        if pred_traj is not None and r2_arr is not None:
            pred_t = _to_tensor(pred_traj, device)  # [O, H, D]
            r2_t = _to_tensor(r2_arr, device)  # [O]
            diffs = rollouts[:, None, :, :] - pred_t[None, :, :, :]  # [S, O, H, D]
            sep2 = (diffs * diffs).sum(dim=-1)  # [S, O, H]
            dyn_collision = (sep2 <= r2_t[None, :, None]).any(dim=1).float()  # [S, H]
            collision_pen = collision_pen + (dyn_collision * pre_goal_mask).sum(dim=1)

        # CTG: gather cost-to-go values at each rollout position
        if ndim == 2:
            ctg_roll = ctg_t[cell_indices[:, :, 0], cell_indices[:, :, 1]]  # [S, H]
        else:
            ctg_roll = ctg_t[cell_indices[:, :, 0], cell_indices[:, :, 1], cell_indices[:, :, 2]]

        ctg_roll = torch.where(torch.isfinite(ctg_roll), ctg_roll, torch.tensor(unreachable_penalty, device=device))
        ctg_min = ctg_roll.min(dim=1).values  # [S]
        ctg_avg = ctg_roll.mean(dim=1)  # [S]
        # `dist2` / `reaches_goal_any` / `first_goal_h` already computed above
        # to build the pre-goal collision mask.

        # Cost computation (matches CPU mppi logic)
        smooth_pen = torch.zeros(self.n_samples, device=device)
        if self._prev_action is not None:
            prev_t = _to_tensor(self._prev_action, device)
            smooth_pen = torch.norm(actions_t - prev_t[None, :], dim=1)

        no_coll = collision_pen == 0
        clean_reach = reaches_goal_any & no_coll
        dirty_reach = reaches_goal_any & ~no_coll
        neither = ~reaches_goal_any

        costs = torch.empty(self.n_samples, device=device)
        costs[clean_reach] = -1e6 + self.w_smooth * smooth_pen[clean_reach]
        costs[dirty_reach] = (
            self.w_goal * ctg_avg[dirty_reach]
            + self.w_obs * collision_pen[dirty_reach]
            + self.w_smooth * smooth_pen[dirty_reach]
        )
        costs[neither] = (
            self.w_goal * (0.5 * ctg_avg[neither] + 0.5 * ctg_min[neither])
            + self.w_obs * collision_pen[neither]
            + self.w_smooth * smooth_pen[neither]
        )

        # MPPI softmax-weighted average
        cost_min = costs.min()
        weights = torch.exp(-(costs - cost_min) / self.temperature)
        weights = weights / weights.sum()
        softmax_action_t = (weights[:, None] * actions_t).sum(dim=0)
        argmin_idx = int(costs.argmin().item())
        argmin_action_t = actions_t[argmin_idx]

        # Mode-aware sampling (Smart MPPI v4): cluster rollouts by lateral
        # principal-component sign and pick the lower-cost cluster's
        # softmax-weighted action. This targets the §3 dynamic-obstacle
        # cancellation mode: when the rollout cloud is bimodal (escape via
        # left vs right), the global softmax averages the two modes toward
        # zero. Splitting first preserves MPPI smoothing while breaking the
        # cancellation.
        mode_aware_triggered = False
        mode_aware_action_t = None
        mode_aware_best_k: int | None = None
        mode_aware_cluster_sign = 0
        if self.mode_aware_sampling:
            base_t_ma = _to_tensor(base, device)
            actions_along_ma = (actions_t * base_t_ma[None, :]).sum(dim=1)
            lat_components = actions_t - actions_along_ma[:, None] * base_t_ma[None, :]
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
                if n_pos >= self.mode_aware_min_size and n_neg >= self.mode_aware_min_size:
                    def _cluster(mask: torch.Tensor) -> tuple[torch.Tensor, float, int]:
                        c = costs[mask]
                        acts = actions_t[mask]
                        cmin = c.min()
                        w = torch.exp(-(c - cmin) / self.temperature)
                        w = w / w.sum()
                        action = (w[:, None] * acts).sum(dim=0)
                        avg_cost = (w * c).sum()
                        idx_local = int(c.argmin().item())
                        global_idx = int(
                            torch.nonzero(mask, as_tuple=False)[idx_local].item()
                        )
                        return action, float(avg_cost.item()), global_idx

                    pos_action, pos_cost, pos_idx = _cluster(pos_mask)
                    neg_action, neg_cost, neg_idx = _cluster(neg_mask)
                    lo, hi = sorted((pos_cost, neg_cost))
                    # When cost_ratio gate > 1, require one cluster to be
                    # meaningfully worse before committing (mode-aware
                    # switcher behaviour). At the default 1.0 the gate
                    # is always satisfied (Smart v4 unconditional commit).
                    ratio_ok = (
                        self.mode_aware_cost_ratio <= 1.0
                        or (hi >= self.mode_aware_cost_ratio * max(lo, 1e-6))
                    )
                    if ratio_ok:
                        if pos_cost <= neg_cost:
                            mode_aware_action_t = pos_action
                            mode_aware_best_k = pos_idx
                            mode_aware_cluster_sign = 1
                        else:
                            mode_aware_action_t = neg_action
                            mode_aware_best_k = neg_idx
                            mode_aware_cluster_sign = -1
                        mode_aware_triggered = True

        # Hybrid argmin-fallback: detect bidirectional cancellation by
        # checking whether the softmax averaged action's lateral component
        # (perpendicular to the goal direction) is much smaller than the
        # argmin rollout's lateral component. When that holds, the rollout
        # cloud has bimodal left/right escapes with similar costs and the
        # softmax mean averages them toward zero — exactly the §3
        # dynamic-obstacle cancellation mechanism. Falling back to argmin
        # commits to one side instead.
        fallback_triggered = False
        if self.fallback_to_argmin:
            # If the previous replan committed to argmin, hold the
            # commit for `fallback_commit_steps - 1` further replans.
            if self._fallback_commit_remaining > 0:
                fallback_triggered = True
                self._fallback_commit_remaining -= 1
            else:
                base_t = _to_tensor(base, device)
                softmax_along = (softmax_action_t * base_t).sum()
                argmin_along = (argmin_action_t * base_t).sum()
                softmax_lat = softmax_action_t - softmax_along * base_t
                argmin_lat = argmin_action_t - argmin_along * base_t
                softmax_lat_mag = float(torch.norm(softmax_lat).item())
                argmin_lat_mag = float(torch.norm(argmin_lat).item())
                if (
                    argmin_lat_mag > self.fallback_lateral_threshold
                    and softmax_lat_mag
                    < self.fallback_lateral_ratio * argmin_lat_mag
                ):
                    fallback_triggered = True
                    # Hold for the next `K - 1` replans
                    self._fallback_commit_remaining = self.fallback_commit_steps - 1
        if mode_aware_triggered and mode_aware_action_t is not None:
            chosen_action_t = mode_aware_action_t
        elif fallback_triggered:
            chosen_action_t = argmin_action_t
        else:
            chosen_action_t = softmax_action_t

        speed = torch.norm(chosen_action_t)
        if speed > self.max_speed:
            chosen_action_t = chosen_action_t * (self.max_speed / speed)
        if mode_aware_triggered and mode_aware_best_k is not None:
            best_k = mode_aware_best_k
        elif fallback_triggered:
            best_k = argmin_idx
        else:
            best_k = int(weights.argmax().item())

        # Build best rollout for visualisation (on CPU)
        best_rollout = rollouts[best_k].cpu().numpy()
        best_full = np.concatenate([obs.reshape(1, -1), best_rollout], axis=0)
        if reaches_goal_any[best_k]:
            best_full = best_full[: first_goal_h[best_k].item() + 2]

        # Subsampled rollouts for anim overlay: K uniformly-spaced samples
        # prepended with `obs` so each polyline starts at the drone. Stored
        # in `meta` for the recorder; cost is ~K * (H+1) * D floats per
        # replan, dominated by JSON overhead.
        k_vis = min(self.viz_rollouts, self.n_samples)
        if k_vis > 0:
            stride = max(1, self.n_samples // k_vis)
            rollouts_vis_t = rollouts[::stride][:k_vis]
            obs_prefix = obs_t[None, None, :].expand(rollouts_vis_t.shape[0], 1, ndim)
            rollouts_vis = torch.cat([obs_prefix, rollouts_vis_t], dim=1).cpu().numpy()
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
                "cost_min": float(cost_min.item()),
                "weight_max": float(weights.max().item()),
                "weight_entropy": float((-weights * torch.log(weights + 1e-12)).sum().item()),
                "n_samples": self.n_samples,
                "device": str(device),
                "rollouts": rollouts_meta,
                "best_rollout_idx": int(best_vis_idx),
                "fallback_to_argmin": bool(fallback_triggered),
                "mode_aware_triggered": bool(mode_aware_triggered),
                "mode_aware_cluster_sign": int(mode_aware_cluster_sign),
            },
        )
