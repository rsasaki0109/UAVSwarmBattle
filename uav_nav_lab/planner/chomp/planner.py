"""CHOMP -- gradient-based trajectory smoothing planner.

CHOMP (Covariant Hamiltonian Optimization for Motion Planning, Zucker
et al. 2013) optimizes a fixed-length sequence of waypoints between
start and goal under

    U(x) = w_smooth * ||A x||^2 / 2  +  w_obs * sum_i c(x_i)

where ``A`` is the second-difference matrix and ``c`` is the standard
CHOMP obstacle potential from :mod:`.objective`.

Updates use the M^-1-preconditioned step from the original paper, where
M is the interior block of the smoothness Hessian K = A^T A. This keeps
the optimizer well-behaved at non-trivial trajectory lengths. Endpoints
stay clamped by optimizing only the interior block and folding endpoint
contributions into the gradient as a constant offset.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .._grid import inflate_obstacles
from ..base import PLANNER_REGISTRY, Plan, Planner
from ..rrt import RRTPlanner
from .objective import distance_field, obstacle_cost_and_grad, smoothness_hessian


def _resample_polyline(wps: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to exactly ``n`` points along its arc length.

    RRT/RRT* return variable-length, unevenly spaced waypoint sequences.
    CHOMP needs a fixed ``n`` with roughly uniform spacing so the
    smoothness Hessian's per-waypoint scale is consistent.
    """
    wps = np.asarray(wps, dtype=float)
    if wps.shape[0] <= 1 or n <= 1:
        return np.repeat(wps[:1], max(n, 1), axis=0)
    seg = np.linalg.norm(np.diff(wps, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-12:
        return np.repeat(wps[:1], n, axis=0)
    targets = np.linspace(0.0, cum[-1], n)
    out = np.empty((n, wps.shape[1]), dtype=float)
    for k in range(wps.shape[1]):
        out[:, k] = np.interp(targets, cum, wps[:, k])
    return out


@PLANNER_REGISTRY.register("chomp")
class ChompPlanner(Planner):
    def __init__(
        self,
        max_speed: float = 10.0,
        replan_period: float = 0.5,
        n_waypoints: int = 30,
        n_iters: int = 100,
        learning_rate: float = 0.05,
        max_step_norm: float = 1.0,
        w_smooth: float = 1.0,
        w_obs: float = 5.0,
        epsilon: float = 2.0,
        resolution: float = 1.0,
        inflate: int = 0,
        goal_tolerance: float = 1.5,
        init: str = "straight",
        rrt_max_samples: int = 1000,
        rrt_step_size: float = 2.0,
        rrt_goal_tolerance: float = 1.5,
        rrt_goal_bias: float = 0.1,
        rrt_seed: int = 0,
    ) -> None:
        self.max_speed = float(max_speed)
        self.replan_period = float(replan_period)
        self.n_waypoints = int(n_waypoints)
        self.n_iters = int(n_iters)
        self.learning_rate = float(learning_rate)
        self.max_step_norm = float(max_step_norm)
        self.w_smooth = float(w_smooth)
        self.w_obs = float(w_obs)
        self.epsilon = float(epsilon)
        self.resolution = float(resolution)
        self.inflate = int(inflate)
        self.goal_tolerance = float(goal_tolerance)
        if init not in ("straight", "rrt"):
            raise ValueError(f"init must be 'straight' or 'rrt'; got {init!r}")
        self.init = init
        self._rrt: RRTPlanner | None = None
        if init == "rrt":
            self._rrt = RRTPlanner(
                max_speed=self.max_speed,
                max_samples=int(rrt_max_samples),
                step_size=float(rrt_step_size),
                goal_tolerance=float(rrt_goal_tolerance),
                goal_bias=float(rrt_goal_bias),
                resolution=self.resolution,
                inflate=self.inflate,
                seed=int(rrt_seed),
            )

        self._K: np.ndarray | None = None
        self._K_int_inv: np.ndarray | None = None
        self._K_endpts: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "ChompPlanner":
        return cls(
            max_speed=float(cfg.get("max_speed", 10.0)),
            replan_period=float(cfg.get("replan_period", 0.5)),
            n_waypoints=int(cfg.get("n_waypoints", 30)),
            n_iters=int(cfg.get("n_iters", 100)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            max_step_norm=float(cfg.get("max_step_norm", 1.0)),
            w_smooth=float(cfg.get("w_smooth", 1.0)),
            w_obs=float(cfg.get("w_obs", 5.0)),
            epsilon=float(cfg.get("epsilon", 2.0)),
            resolution=float(cfg.get("resolution", 1.0)),
            inflate=int(cfg.get("inflate", 0)),
            goal_tolerance=float(cfg.get("goal_tolerance", 1.5)),
            init=str(cfg.get("init", "straight")),
            rrt_max_samples=int(cfg.get("rrt_max_samples", 1000)),
            rrt_step_size=float(cfg.get("rrt_step_size", 2.0)),
            rrt_goal_tolerance=float(cfg.get("rrt_goal_tolerance", 1.5)),
            rrt_goal_bias=float(cfg.get("rrt_goal_bias", 0.1)),
            rrt_seed=int(cfg.get("rrt_seed", 0)),
        )

    def reset(self) -> None:
        if self._rrt is not None:
            self._rrt.reset()

    def plan(
        self,
        observation: np.ndarray,
        goal: np.ndarray,
        obstacle_map: Any,
        *,
        dynamic_obstacles: list[dict] | None = None,  # noqa: ARG002
    ) -> Plan:
        occ_raw = np.asarray(obstacle_map, dtype=bool)
        ndim = occ_raw.ndim
        occ = inflate_obstacles(occ_raw, self.inflate)
        start = np.asarray(observation, dtype=float)[:ndim]
        gl = np.asarray(goal, dtype=float)[:ndim]

        n = max(4, self.n_waypoints)
        if self._K is None or self._K.shape[0] != n:
            self._K = smoothness_hessian(n)
            k_int = self._K[1:-1, 1:-1]
            self._K_int_inv = np.linalg.inv(k_int + 1e-6 * np.eye(n - 2))
            self._K_endpts = self._K[1:-1][:, [0, -1]]

        init_used = self.init
        if self._rrt is not None:
            rrt_plan = self._rrt.plan(start, gl, occ_raw)
            if (
                rrt_plan.meta.get("status") == "ok"
                and rrt_plan.waypoints.shape[0] >= 2
            ):
                x = _resample_polyline(rrt_plan.waypoints, n)
            else:
                init_used = "rrt_fallback_straight"
                ts = np.linspace(0.0, 1.0, n)
                x = start[None, :] + (gl - start)[None, :] * ts[:, None]
        else:
            ts = np.linspace(0.0, 1.0, n)
            x = start[None, :] + (gl - start)[None, :] * ts[:, None]

        dist = distance_field(occ, self.resolution, cap=2.0 * self.epsilon)
        endpts = np.stack([x[0], x[-1]])
        k_int = self._K[1:-1, 1:-1]
        for _ in range(self.n_iters):
            _c, grad_obs = obstacle_cost_and_grad(
                x, dist, self.epsilon, self.resolution
            )
            grad_smooth_int = k_int @ x[1:-1] + self._K_endpts @ endpts
            grad_int = self.w_smooth * grad_smooth_int + self.w_obs * grad_obs[1:-1]
            step = self.learning_rate * (self._K_int_inv @ grad_int)
            step_norms = np.linalg.norm(step, axis=1, keepdims=True)
            scale = np.minimum(
                1.0,
                self.max_step_norm / np.maximum(step_norms, 1e-12),
            )
            x[1:-1] = x[1:-1] - step * scale

        cells = np.clip(
            np.round(x / self.resolution).astype(int),
            0,
            np.array(occ_raw.shape, dtype=int) - 1,
        )
        in_obstacle = bool(occ_raw[tuple(cells.T)].any())
        status = "local_minimum" if in_obstacle else "ok"

        return Plan(
            waypoints=x.astype(float),
            meta={
                "planner": "chomp",
                "status": status,
                "n_waypoints": n,
                "n_iters": self.n_iters,
                "init": init_used,
            },
        )
