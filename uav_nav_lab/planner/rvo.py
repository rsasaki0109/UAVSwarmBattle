"""RVO — Reciprocal Velocity Obstacles (van den Berg, Lin, Manocha, ICRA 2008).

The DIRECT PRECURSOR to ORCA (2011). Both are reciprocal velocity-space avoiders,
but they differ in how they pick the new velocity:

  * RVO (this file) SAMPLES candidate velocities and scores each by a penalty
    that trades distance-to-preferred against time-to-collision, choosing the
    minimum. The reciprocal velocity obstacle of B for A is the VO translated so
    its apex sits at (v_A + v_B)/2 — i.e. a candidate v' is forbidden iff the
    *reciprocal* effective velocity 2v' - v_A - v_B leads to collision.
  * ORCA (planner.type: orca) instead derives, per neighbour, a single linear
    half-plane of permitted velocities and solves a tiny LP.

RVO's famous weakness is OSCILLATION: with a discrete, penalty-ranked velocity
choice, two reciprocating agents can flip between mirror-image candidate
velocities step after step (the "reciprocal dance"), so their tracks jitter.
ORCA's half-plane construction was introduced precisely to remove that
oscillation by making the choice a smooth convex projection. This planner exists
to reproduce and quantify that classic RVO→ORCA improvement.

Clean-room implementation from the published RVO algorithm (sampled-velocity
selection with the reciprocal half-velocity penalty). 2-D, agent-agent only;
uses the runner's set_current_state hook for the ego velocity.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .base import PLANNER_REGISTRY, Plan, Planner

_EPS = 1e-9


def _norm(v: np.ndarray) -> float:
    return float(np.hypot(v[0], v[1]))


def _ttc(rel_p: np.ndarray, rel_v: np.ndarray, R: float, horizon: float) -> float:
    """Time to collision of a point at rel_p moving at rel_v with the disc of
    radius R at the origin; inf if no collision within `horizon`."""
    # |rel_p + t rel_v| = R  ->  a t^2 + b t + c = 0
    a = float(rel_v @ rel_v)
    if a < _EPS:
        return math.inf
    b = 2.0 * float(rel_p @ rel_v)
    c = float(rel_p @ rel_p) - R * R
    if c < 0.0:
        return 0.0  # already overlapping
    disc = b * b - 4.0 * a * c
    if disc <= 0.0:
        return math.inf
    t = (-b - math.sqrt(disc)) / (2.0 * a)
    if t < 0.0 or t > horizon:
        return math.inf
    return t


@PLANNER_REGISTRY.register("rvo")
class RVOPlanner(Planner):
    """Reciprocal Velocity Obstacles (van den Berg 2008); 2-D, sampled selection."""

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
    def from_config(cls, cfg: Mapping[str, Any]) -> "RVOPlanner":
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
            raise ValueError(f"RVOPlanner is 2-D only; got {pos.shape[0]}-D.")
        gl = np.asarray(goal, dtype=float)[:2]
        v_cur = self._cur_vel if self._cur_vel is not None else np.zeros(2)
        to_goal = gl - pos
        dist = _norm(to_goal)
        if dist < self.goal_radius:
            return Plan(waypoints=np.asarray([pos], dtype=float),
                        target_velocity=np.zeros(2), meta={"planner": "rvo"})
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
                # reciprocal: A takes half -> test effective rel velocity
                rel_v = (2.0 * v - v_cur) - v_o
                # relative position closes as -rel_v (B - A frame): use rel_p, rel_v
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
                    target_velocity=best, meta={"planner": "rvo"})
