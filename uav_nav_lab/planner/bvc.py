"""BVC — Buffered Voronoi Cell collision avoidance (Zhou et al., 2017).

A clean-room 2-D implementation of the position-space reciprocal avoider from

    D. Zhou, Z. Wang, S. Bandyopadhyay, M. Schwager,
    "Fast, On-line Collision Avoidance for Dynamic Vehicles Using Buffered
    Voronoi Cells", IEEE RA-L 2017.

Where ORCA (`planner.type: orca`) reasons in VELOCITY space — each pairwise
encounter becomes a half-plane of permitted velocities — BVC reasons in
POSITION space: each agent restricts its next position to its Voronoi cell,
shrunk inward by the combined safety radius (the "buffer"). Buffered Voronoi
cells are disjoint by construction, so an agent that stays inside its own cell
**cannot collide with a peer, ever** — BVC has a hard, geometric safety
guarantee that ORCA's reciprocal split only approximates.

The price is the mirror-image failure mode. ORCA fails the symmetric antipodal
swap by COLLISION (the reciprocal dance funnels everyone onto the hub). BVC
cannot collide, so it fails the same swap by DEADLOCK / TIMEOUT: at the hub each
agent's buffered cell is a sliver whose only goal-ward face is cut off by the
opposing agents' bisectors, so the closest-to-goal point in the cell is the
agent's own position — it stops and never arrives. Two reciprocal schools, two
opposite failure signatures on the same benchmark.

Each replan: build the buffered Voronoi half-planes from the observed peer
positions, project the (optionally right-of-way-tilted) goal onto their
intersection with Dykstra's alternating projection, and command the velocity
that heads to that point (capped at max_speed).

Scope / caveats
---------------
* **2-D only** (Voronoi bisectors in the plane); raises on ndim != 2.
* **Agent-agent only** — static occupancy is ignored (open-arena swarm
  scenarios, obstacles: none), matching the ORCA baseline.
* Position-space: needs peer *positions* (not velocities), so it ignores the
  reported peer velocity — the stock method is purely geometric.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import PLANNER_REGISTRY, Plan, Planner

_EPS = 1e-9


def _norm(v: np.ndarray) -> float:
    return float(np.hypot(v[0], v[1]))


def _project_intersection(target, lines, pos, max_step, iters):
    """Closest point to ``target`` in the intersection of the half-planes
    ``a·x <= b`` (``lines`` = list of (a, b)) and the step disk |x-pos|<=max_step,
    via Dykstra's alternating projection (converges to the true projection onto
    the convex intersection, unlike naive cyclic projection)."""
    x = np.array(target, dtype=float)
    corr = [np.zeros(2) for _ in lines]
    step_corr = np.zeros(2)
    for _ in range(iters):
        for k, (a, b) in enumerate(lines):
            y = x + corr[k]
            viol = float(a @ y) - b
            a2 = float(a @ a)
            x_new = y - (viol / a2) * a if (viol > 0.0 and a2 > _EPS) else y
            corr[k] = y - x_new
            x = x_new
        # step-disk projection (keeps the move within one control step)
        y = x + step_corr
        d = y - pos
        dn = _norm(d)
        x_new = pos + d / dn * max_step if dn > max_step else y
        step_corr = y - x_new
        x = x_new
    return x


@PLANNER_REGISTRY.register("bvc")
class BVCPlanner(Planner):
    """Buffered Voronoi Cell controller; 2-D, agent-agent only.

    Config keys
    -----------
    ``max_speed``      : cruise / cap speed (m/s).
    ``radius``         : ego collision radius (m).
    ``safety_margin``  : extra buffer added to each pairwise combined radius (m).
    ``neighbor_dist``  : ignore peers farther than this (m).
    ``time_step``      : control step used to convert the projected target point
                         into a velocity command (m/s = step / time_step).
    ``goal_radius``    : within this, the agent stops (arrived).
    ``proj_iters``     : Dykstra projection iterations (accuracy vs cost).
    ``lateral_bias``   : GLOBAL right-of-way (default 0). Tilts the projected
                         goal to the ego's right (clockwise perp of goal heading)
                         by this fraction of the goal distance before projection,
                         breaking the symmetric hub stall into a clockwise pass.
                         Fires unconditionally (even with no peer).
    ``pairwise_bias``  : PAIRWISE right-of-way (default 0). Tilts the projected
                         goal toward the sum over nearby peers of "pass this peer
                         on the right", weighted exp(-dist/pairwise_radius); with
                         no peer the tilt vanishes. The neighbour-conditional
                         analogue of ``lateral_bias`` (cf. the MPC / ORCA ports).
    ``pairwise_radius``: neighbour-conflict falloff length (m) for pairwise_bias.
    """

    def __init__(
        self,
        max_speed: float = 5.0,
        radius: float = 0.4,
        safety_margin: float = 0.1,
        neighbor_dist: float = 15.0,
        time_step: float = 0.1,
        goal_radius: float = 1.5,
        proj_iters: int = 20,
        lateral_bias: float = 0.0,
        pairwise_bias: float = 0.0,
        pairwise_radius: float = 5.0,
    ) -> None:
        self.max_speed = float(max_speed)
        self.radius = float(radius)
        self.safety_margin = float(safety_margin)
        self.neighbor_dist = float(neighbor_dist)
        self.time_step = float(time_step)
        self.goal_radius = float(goal_radius)
        self.proj_iters = int(proj_iters)
        self.lateral_bias = float(lateral_bias)
        self.pairwise_bias = float(pairwise_bias)
        self.pairwise_radius = max(1e-6, float(pairwise_radius))

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "BVCPlanner":
        return cls(
            max_speed=float(cfg.get("max_speed", 5.0)),
            radius=float(cfg.get("radius", 0.4)),
            safety_margin=float(cfg.get("safety_margin", 0.1)),
            neighbor_dist=float(cfg.get("neighbor_dist", 15.0)),
            time_step=float(cfg.get("time_step", cfg.get("dt", 0.1))),
            goal_radius=float(cfg.get("goal_radius", 1.5)),
            proj_iters=int(cfg.get("proj_iters", 20)),
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
        if pos.shape[0] != 2:
            raise ValueError(
                f"BVCPlanner is 2-D only; got {pos.shape[0]}-D observation. "
                "Use a sampling planner (mpc/mppi) for 3-D swarms."
            )
        gl = np.asarray(goal, dtype=float)[:2]
        to_goal = gl - pos
        dist = _norm(to_goal)
        if dist < self.goal_radius:
            return Plan(waypoints=np.asarray([pos], dtype=float),
                        target_velocity=np.zeros(2), meta={"planner": "bvc"})

        # Right-of-way: tilt the goal-seeking DIRECTION before we pick a target
        # point to project. Same conventions as the MPC / ORCA ports.
        gdir = to_goal / dist
        if self.lateral_bias > 0.0:
            right = np.array([gdir[1], -gdir[0]])
            gdir = gdir + self.lateral_bias * right
            gdir = gdir / _norm(gdir)
        if self.pairwise_bias > 0.0 and dynamic_obstacles:
            tilt = np.zeros(2)
            for d in dynamic_obstacles:
                rel = np.asarray(d["position"], dtype=float)[:2] - pos
                dn = _norm(rel)
                if dn < _EPS:
                    continue
                nrel = rel / dn
                tilt = tilt + np.exp(-dn / self.pairwise_radius) * np.array([nrel[1], -nrel[0]])
            gdir = gdir + self.pairwise_bias * tilt
            gn = _norm(gdir)
            if gn > _EPS:
                gdir = gdir / gn

        # Target the (possibly tilted) goal point; projection clamps the move.
        target = pos + gdir * dist

        # Buffered Voronoi half-planes a·x <= b from the peers.
        lines = []
        for d in (dynamic_obstacles or []):
            p_other = np.asarray(d["position"], dtype=float)[:2]
            a = p_other - pos
            an = _norm(a)
            if an < _EPS or an > self.neighbor_dist:
                continue
            r_buf = self.radius + float(d.get("radius", 0.5)) + self.safety_margin
            mid = 0.5 * (pos + p_other)
            b = float(a @ mid) - r_buf * an
            lines.append((a, b))

        max_step = self.max_speed * self.time_step
        nxt = _project_intersection(target, lines, pos, max_step, self.proj_iters)

        # Empty / infeasible buffered cell: when enough peers crowd the hub the
        # intersection of buffered half-planes is empty (the agent is already
        # inside someone's buffer band), and the projection cannot satisfy every
        # constraint. Moving to that infeasible point would breach a buffer, so
        # the safety-preserving action is to STAY PUT — which is exactly the BVC
        # deadlock (collision-free timeout) on the symmetric swap. We hold
        # position unless the move stays feasible for every buffered half-plane.
        feasible = all(float(a @ nxt) <= b + 1e-6 for a, b in lines)
        if not feasible:
            return Plan(waypoints=np.asarray([pos], dtype=float),
                        target_velocity=np.zeros(2),
                        meta={"planner": "bvc", "n_lines": len(lines), "held": True})

        vel = (nxt - pos) / self.time_step
        sp = _norm(vel)
        if sp > self.max_speed:
            vel = vel / sp * self.max_speed
        return Plan(waypoints=np.asarray([nxt], dtype=float),
                    target_velocity=vel,
                    meta={"planner": "bvc", "n_lines": len(lines)})
