"""HRVO — Hybrid Reciprocal Velocity Obstacles (Snape, van den Berg, Guy, Manocha,
IEEE T-RO 2011; the method dates to 2009).

The HISTORICAL fix for RVO's oscillation that PREDATES ORCA, and the reason it
matters here: it removes the reciprocal dance WITHOUT abandoning the
velocity-obstacle sampling framework for ORCA's half-plane LP. It is the
constructive test of the lineage's corrected mechanism claim (see
docs/findings.md): if ORCA's cure is a *structural* commitment to one side of the
obstacle — not continuity, and not the LP specifically — then HRVO, which adds
exactly that commitment to a sampled VO/RVO, should also remove the oscillation.

Construction. For neighbour B, the velocity obstacle is a cone (apex at v_B,
half-angle asin(R/d) about the line A->B). RVO shifts the apex to (v_A+v_B)/2 so
each agent takes half the avoidance. The ambiguity that makes RVO oscillate is
*which side* to pass: the penalty-minimum can flip between the two legs step to
step. HRVO resolves it by committing to the side the agent is already favouring:
it keeps the RVO leg on the chosen side but swaps in the VO (non-reciprocal) leg
on the other side, moving the apex to the intersection of those two legs. The
enlarged "wrong-side" region makes switching sides costly, so the agent commits.

Clean-room implementation of the published apex construction, evaluated in the
same sampled-velocity framework as VO/RVO (so the three are directly comparable).
2-D, agent-agent only; uses set_current_state for the ego velocity.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .base import PLANNER_REGISTRY, Plan, Planner
from .rvo import _norm

_EPS = 1e-9


def _rot(v: np.ndarray, ang: float) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _ray_intersect(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray):
    """Intersection of line (p1 + t d1) and (p2 + s d2); None if near-parallel."""
    det = d1[0] * (-d2[1]) - d1[1] * (-d2[0])
    if abs(det) < 1e-9:
        return None
    rhs = p2 - p1
    t = (rhs[0] * (-d2[1]) - rhs[1] * (-d2[0])) / det
    return p1 + t * d1


@PLANNER_REGISTRY.register("hrvo")
class HRVOPlanner(Planner):
    """Hybrid Reciprocal Velocity Obstacles (Snape et al. 2011); 2-D, sampled."""

    def __init__(
        self,
        max_speed: float = 5.0,
        radius: float = 0.4,
        safety_margin: float = 0.1,
        time_horizon: float = 2.0,
        neighbor_dist: float = 15.0,
        goal_radius: float = 1.5,
        n_speeds: int = 4,
        n_angles: int = 24,
        w_collision: float = 2.0,
        time_step: float = 0.05,
    ) -> None:
        self.max_speed = float(max_speed)
        self.radius = float(radius)
        self.safety_margin = float(safety_margin)
        self.time_horizon = float(time_horizon)
        self.neighbor_dist = float(neighbor_dist)
        self.goal_radius = float(goal_radius)
        self.n_speeds = int(n_speeds)
        self.n_angles = int(n_angles)
        self.w_collision = float(w_collision)
        self.time_step = float(time_step)
        self._cur_vel: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "HRVOPlanner":
        return cls(
            max_speed=float(cfg.get("max_speed", 5.0)),
            radius=float(cfg.get("radius", 0.4)),
            safety_margin=float(cfg.get("safety_margin", 0.1)),
            time_horizon=float(cfg.get("time_horizon", 2.0)),
            neighbor_dist=float(cfg.get("neighbor_dist", 15.0)),
            goal_radius=float(cfg.get("goal_radius", 1.5)),
            n_speeds=int(cfg.get("n_speeds", 4)),
            n_angles=int(cfg.get("n_angles", 24)),
            w_collision=float(cfg.get("w_collision", 2.0)),
            time_step=float(cfg.get("time_step", cfg.get("dt", 0.05))),
        )

    def reset(self) -> None:
        self._cur_vel = None

    def set_current_state(self, position, velocity=None) -> None:
        if velocity is not None:
            self._cur_vel = np.asarray(velocity, dtype=float)[:2]

    def _candidates(self, v_pref):
        cands = [np.zeros(2), np.array(v_pref, dtype=float)]
        for si in range(1, self.n_speeds + 1):
            sp = self.max_speed * si / self.n_speeds
            for ai in range(self.n_angles):
                a = 2.0 * math.pi * ai / self.n_angles
                cands.append(np.array([sp * math.cos(a), sp * math.sin(a)]))
        return cands

    def plan(
        self,
        observation: np.ndarray,
        goal: np.ndarray,
        obstacle_map: Any,
        *,
        dynamic_obstacles: list[dict] | None = None,
    ) -> Plan:
        pos = np.asarray(observation, dtype=float)
        if pos.shape[0] != 2:
            raise ValueError(f"HRVOPlanner is 2-D only; got {pos.shape[0]}-D.")
        gl = np.asarray(goal, dtype=float)[:2]
        v_cur = self._cur_vel if self._cur_vel is not None else np.zeros(2)
        to_goal = gl - pos
        dist = _norm(to_goal)
        if dist < self.goal_radius:
            return Plan(waypoints=np.asarray([pos], dtype=float),
                        target_velocity=np.zeros(2), meta={"planner": "hrvo"})
        v_pref = to_goal / dist * self.max_speed

        # Precompute per-neighbour HRVO cones (apex, axis u, half-angle theta).
        cones = []
        nb2 = self.neighbor_dist * self.neighbor_dist
        for dd in (dynamic_obstacles or []):
            p_o = np.asarray(dd["position"], dtype=float)[:2]
            rel_p = p_o - pos
            d2 = float(rel_p @ rel_p)
            if d2 > nb2:
                continue
            v_o = np.asarray(dd.get("velocity", (0.0, 0.0)), dtype=float)[:2]
            R = self.radius + float(dd.get("radius", 0.5)) + self.safety_margin
            d = math.sqrt(d2)
            u = rel_p / max(d, _EPS)
            theta = math.asin(min(R / max(d, R), 1.0))
            leg_l = _rot(u, theta)    # left leg direction
            leg_r = _rot(u, -theta)   # right leg direction
            c_rvo = 0.5 * (v_cur + v_o)
            c_vo = v_o
            # which side is the agent already favouring? sign of cross(u, v_rel)
            v_rel = v_cur - v_o
            side = u[0] * v_rel[1] - u[1] * v_rel[0]  # >0: left, <=0: right
            if side > 0.0:  # pass on the left: RVO left leg + VO right leg
                apex = _ray_intersect(c_rvo, leg_l, c_vo, leg_r)
            else:           # pass on the right: RVO right leg + VO left leg
                apex = _ray_intersect(c_rvo, leg_r, c_vo, leg_l)
            if apex is None:
                apex = c_rvo
            cones.append((apex, u, theta, d, R))

        best, best_pen = v_pref, math.inf
        cos_h = [math.cos(t) for (_, _, t, _, _) in cones]
        for v in self._candidates(v_pref):
            if _norm(v) > self.max_speed + _EPS:
                continue
            min_tau = math.inf
            for (apex, u, theta, d, R), ch in zip(cones, cos_h):
                w = v - apex
                wn = _norm(w)
                if wn < _EPS:
                    min_tau = 0.0
                    continue
                along = float(w @ u)
                if along <= 0.0:
                    continue  # heading away from the obstacle apex -> outside cone
                if along >= wn * ch - _EPS:  # angle(w,u) <= theta -> inside HRVO cone
                    # time proxy: clearance / closing speed (monotone, scale-robust)
                    tau = max(d - R, 0.05) / max(along, 0.05)
                    if tau < min_tau:
                        min_tau = tau
            if math.isinf(min_tau):
                penalty = _norm(v - v_pref)                       # feasible
            else:
                penalty = self.w_collision / max(min_tau, 1e-3) + _norm(v - v_pref)
            if penalty < best_pen:
                best_pen, best = penalty, v

        return Plan(waypoints=np.asarray([pos + best * self.time_step], dtype=float),
                    target_velocity=best, meta={"planner": "hrvo"})
