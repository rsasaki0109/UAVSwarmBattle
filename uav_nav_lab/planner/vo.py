"""VO — Velocity Obstacles (Fiorini & Shiller, IJRR 1998).

The FOUNDATIONAL velocity-space avoider and the direct ancestor of both RVO
(2008) and ORCA (2011). A velocity obstacle VO_{A|B} is the set of A's
velocities that would, at some future time, put A on a collision course with B
assuming B holds its current velocity v_B. A chooses a velocity outside the union
of its neighbours' VOs, as close as possible to its preferred velocity.

VO's defining assumption — and its weakness — is that it is NON-RECIPROCAL: each
agent plans as if every neighbour will keep its current velocity, i.e. as if it
alone is responsible for avoiding the collision. When two VO agents meet, BOTH
fully avoid the same encounter, so they over-react; on the next step each sees the
other has moved and over-reacts again — the oscillation that RVO's reciprocal
half-velocity construction (each takes half the avoidance) was introduced to
damp, and that ORCA's half-plane LP then removed almost entirely.

This planner is the 1998 baseline of that lineage. It shares RVO's sampled
selection machinery and time-to-collision helper, differing in exactly one place:
the candidate velocity v is tested directly against the neighbour (rel velocity
v - v_o), with no reciprocal split. 2-D, agent-agent only.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .base import PLANNER_REGISTRY, Plan, Planner
from .rvo import _norm, _ttc

_EPS = 1e-9


@PLANNER_REGISTRY.register("vo")
class VOPlanner(Planner):
    """Velocity Obstacles (Fiorini & Shiller 1998); 2-D, non-reciprocal, sampled."""

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

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "VOPlanner":
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
        pass

    def set_current_state(self, position, velocity=None) -> None:  # noqa: D401 - VO is memoryless
        # VO does not need the ego velocity (non-reciprocal); accept the hook for
        # runner uniformity.
        return None

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
            raise ValueError(f"VOPlanner is 2-D only; got {pos.shape[0]}-D.")
        gl = np.asarray(goal, dtype=float)[:2]
        to_goal = gl - pos
        dist = _norm(to_goal)
        if dist < self.goal_radius:
            return Plan(waypoints=np.asarray([pos], dtype=float),
                        target_velocity=np.zeros(2), meta={"planner": "vo"})
        v_pref = to_goal / dist * self.max_speed

        nbrs = []
        nb2 = self.neighbor_dist * self.neighbor_dist
        for d in (dynamic_obstacles or []):
            p_o = np.asarray(d["position"], dtype=float)[:2]
            rel_p = p_o - pos
            if float(rel_p @ rel_p) > nb2:
                continue
            v_o = np.asarray(d.get("velocity", (0.0, 0.0)), dtype=float)[:2]
            R = self.radius + float(d.get("radius", 0.5)) + self.safety_margin
            nbrs.append((rel_p, v_o, R))

        best, best_pen = v_pref, math.inf
        for v in self._candidates(v_pref):
            if _norm(v) > self.max_speed + _EPS:
                continue
            min_ttc = math.inf
            for rel_p, v_o, R in nbrs:
                # non-reciprocal: test the candidate against B holding v_o
                rel_v = v - v_o
                tau = _ttc(rel_p, -rel_v, R, self.time_horizon)
                if tau < min_ttc:
                    min_ttc = tau
            if math.isinf(min_ttc):
                penalty = _norm(v - v_pref)              # collision-free
            else:
                penalty = self.w_collision / max(min_ttc, 1e-3) + _norm(v - v_pref)
            if penalty < best_pen:
                best_pen, best = penalty, v

        return Plan(waypoints=np.asarray([pos + best * self.time_step], dtype=float),
                    target_velocity=best, meta={"planner": "vo"})
