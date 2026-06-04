"""APF — Artificial Potential Field navigation (Khatib, 1986).

The oldest and most ubiquitous reactive navigation method, and the one paradigm
in this repo's reactive line-up that is NOT reciprocal: each agent independently
descends a potential field — an attractive well at the goal plus a repulsive
barrier around every peer — with no model of what the peer will do. ORCA / CBF /
BVC all assume a *cooperating* peer that takes its share; APF assumes nothing,
which makes it the natural test of whether the swarm findings (right-of-way
convention, heterogeneity) hold outside the reciprocal family.

Field (dimension-agnostic):
  F_att = k_att * unit(goal - p)                                  (constant pull)
  F_rep = sum_j  k_rep * (1/d_j - 1/d0) / d_j^2 * unit(p - p_j)   for d_j < d0
  v     = clip(F_att + F_rep, max_speed)

Its signature failure is the *local minimum*: where attraction and repulsion
cancel (classically, a symmetric obstacle directly between agent and goal), the
field has a stationary point. On the antipodal swarm the symmetric hub is exactly
such a point. This controller steers at constant cruise speed along the gradient
(`v = max_speed * unit(F)`, matching the other reactive baselines), so at the hub
it plows through the stationary point and *collides* rather than stalling; a
variable-speed APF (`v ∝ F`) would instead halt there. Either reading, the
symmetric hub defeats stock APF, and the in-plane right-of-way convention breaks
the symmetry that creates the stationary point.

Scope: 2-D and 3-D, agent-agent only (static occupancy ignored), matching the
other reactive baselines. The right-of-way conventions (`lateral_bias` /
`pairwise_bias`) tilt the in-plane (xy) heading to break the symmetry, as for the
other planners.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import PLANNER_REGISTRY, Plan, Planner

_EPS = 1e-9


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


@PLANNER_REGISTRY.register("apf")
class APFPlanner(Planner):
    """Artificial Potential Field controller; 2-D/3-D, agent-agent only.

    Config keys
    -----------
    ``max_speed``      : cruise / cap speed (m/s).
    ``radius``         : ego collision radius (m).
    ``k_att``          : attractive gain (pull toward goal).
    ``k_rep``          : repulsive gain (push from peers).
    ``influence_dist`` : d0 — peers farther than this exert no repulsion (m).
    ``time_step``      : step for waypoint extrapolation; runner uses velocity.
    ``goal_radius``    : within this the agent stops (arrived).
    ``lateral_bias``   : GLOBAL right-of-way; tilt the goal heading right (xy).
    ``pairwise_bias``  : PAIRWISE right-of-way; tilt toward pass-on-the-right of
                         each nearby peer (xy), exp(-d/radius) weighted.
    ``pairwise_radius``: neighbour-conflict falloff length (m) for pairwise_bias.
    """

    def __init__(
        self,
        max_speed: float = 5.0,
        radius: float = 0.4,
        k_att: float = 1.0,
        k_rep: float = 6.0,
        influence_dist: float = 4.0,
        time_step: float = 0.05,
        goal_radius: float = 1.5,
        lateral_bias: float = 0.0,
        pairwise_bias: float = 0.0,
        pairwise_radius: float = 5.0,
    ) -> None:
        self.max_speed = float(max_speed)
        self.radius = float(radius)
        self.k_att = float(k_att)
        self.k_rep = float(k_rep)
        self.influence_dist = float(influence_dist)
        self.time_step = float(time_step)
        self.goal_radius = float(goal_radius)
        self.lateral_bias = float(lateral_bias)
        self.pairwise_bias = float(pairwise_bias)
        self.pairwise_radius = max(1e-6, float(pairwise_radius))

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "APFPlanner":
        return cls(
            max_speed=float(cfg.get("max_speed", 5.0)),
            radius=float(cfg.get("radius", 0.4)),
            k_att=float(cfg.get("k_att", 1.0)),
            k_rep=float(cfg.get("k_rep", 6.0)),
            influence_dist=float(cfg.get("influence_dist", 4.0)),
            time_step=float(cfg.get("time_step", cfg.get("dt", 0.05))),
            goal_radius=float(cfg.get("goal_radius", 1.5)),
            lateral_bias=float(cfg.get("lateral_bias", 0.0)),
            pairwise_bias=float(cfg.get("pairwise_bias", 0.0)),
            pairwise_radius=float(cfg.get("pairwise_radius", 5.0)),
        )

    def plan(
        self,
        observation: np.ndarray,
        goal: np.ndarray,
        obstacle_map: Any,
        *,
        dynamic_obstacles: list[dict] | None = None,
    ) -> Plan:
        pos = np.asarray(observation, dtype=float)
        ndim = pos.shape[0]
        if ndim not in (2, 3):
            raise ValueError(f"APFPlanner supports 2-D/3-D; got {ndim}-D observation.")
        gl = np.asarray(goal, dtype=float)[:ndim]
        to_goal = gl - pos
        dist = _norm(to_goal)
        if dist < self.goal_radius:
            return Plan(waypoints=np.asarray([pos], dtype=float),
                        target_velocity=np.zeros(ndim), meta={"planner": "apf"})

        gdir = to_goal / dist
        # In-plane right-of-way (xy only; symmetry-break, see findings.md).
        if self.lateral_bias > 0.0:
            g2 = gdir[:2]
            n2 = _norm(g2)
            if n2 > _EPS:
                right = np.array([g2[1], -g2[0]]) / n2
                gdir = gdir.copy()
                gdir[:2] = gdir[:2] + self.lateral_bias * right
                gn = _norm(gdir)
                if gn > _EPS:
                    gdir = gdir / gn
        if self.pairwise_bias > 0.0 and dynamic_obstacles:
            tilt = np.zeros(2)
            for d in dynamic_obstacles:
                rel = np.asarray(d["position"], dtype=float)[:2] - pos[:2]
                dn = _norm(rel)
                if dn < _EPS:
                    continue
                nr = rel / dn
                tilt = tilt + np.exp(-dn / self.pairwise_radius) * np.array([nr[1], -nr[0]])
            gdir = gdir.copy()
            gdir[:2] = gdir[:2] + self.pairwise_bias * tilt
            gn = _norm(gdir)
            if gn > _EPS:
                gdir = gdir / gn

        force = self.k_att * gdir
        d0 = self.influence_dist
        for ob in (dynamic_obstacles or []):
            p_other = np.asarray(ob["position"], dtype=float)[:ndim]
            away = pos - p_other
            d = _norm(away)
            r_safe = self.radius + float(ob.get("radius", 0.5))
            surf = max(d - r_safe, 1e-3)  # distance to the peer's surface
            if surf >= d0 or d < _EPS:
                continue
            mag = self.k_rep * (1.0 / surf - 1.0 / d0) / (surf * surf)
            force = force + mag * (away / d)

        # Steer at cruise speed along the field gradient (comparable to the other
        # reactive baselines). At a stationary point of the field (attraction and
        # repulsion cancel — the classic local minimum) the force vanishes and the
        # agent stalls; near one it oscillates and never converges (timeout).
        sp = _norm(force)
        vel = force / sp * self.max_speed if sp > _EPS else np.zeros(ndim)
        wp = pos + vel * self.time_step
        return Plan(waypoints=np.asarray([wp], dtype=float), target_velocity=vel,
                    meta={"planner": "apf"})
