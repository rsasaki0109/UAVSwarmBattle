"""Merry-Go-Round — explicit shared-roundabout deadlock avoidance (Zhou et al.,
"Merry-Go-Round: Safe Control of Decentralized Multi-Robot Systems with Deadlock
Prevention", 2025; arXiv:2503.05848).

Where the lab's `lateral_bias` / `pairwise_bias` conventions break a symmetric
convergence *implicitly* — a small cost nudge that lets the base planner discover
a roundabout — Merry-Go-Round does it *explicitly*: all robots in conflict agree
on a shared circular reference path around a common centre and traverse it in the
same direction until they can peel off to goal. Spread on one ring, all turning
the same way at the same rate, they keep their angular spacing and cannot collide,
which is why the method scales to very dense fleets — at the price of riding a
half-circumference arc instead of the straight diameter.

This is a clean-room steering realisation for the antipodal swap, where the shared
centre is known by symmetry (`center`, default the arena middle). Each drone:
  1. ORBIT: steer tangentially (counter-clockwise) around the centre while a
     radial term holds it on the ring of radius `ring_radius`, until its bearing
     has advanced (CCW) to within `exit_angle` of the goal's bearing;
  2. EXIT: once aligned (or already close to goal), steer straight to the goal.

Scope: 2-D, agent-agent (the ring *is* the coordination — no peer state is read,
exactly like the global convention, so it is sensing-independent). Single shared
centre; for general scenes the centre would be negotiated per conflict cluster
(not modelled here).
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .base import PLANNER_REGISTRY, Plan, Planner

_EPS = 1e-9


@PLANNER_REGISTRY.register("roundabout")
class RoundaboutPlanner(Planner):
    """Explicit shared-roundabout (Merry-Go-Round) controller; 2-D."""

    def __init__(
        self,
        max_speed: float = 5.0,
        center: tuple[float, float] = (25.0, 25.0),
        ring_radius: float = 20.0,
        k_radial: float = 1.0,
        exit_angle: float = 0.35,
        time_step: float = 0.05,
        goal_radius: float = 1.5,
    ) -> None:
        self.max_speed = float(max_speed)
        self.center = np.asarray(center, dtype=float)
        self.ring_radius = float(ring_radius)
        self.k_radial = float(k_radial)
        self.exit_angle = float(exit_angle)
        self.time_step = float(time_step)
        self.goal_radius = float(goal_radius)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "RoundaboutPlanner":
        c = cfg.get("center", (25.0, 25.0))
        return cls(
            max_speed=float(cfg.get("max_speed", 5.0)),
            center=(float(c[0]), float(c[1])),
            ring_radius=float(cfg.get("ring_radius", 20.0)),
            k_radial=float(cfg.get("k_radial", 1.0)),
            exit_angle=float(cfg.get("exit_angle", 0.35)),
            time_step=float(cfg.get("time_step", cfg.get("dt", 0.05))),
            goal_radius=float(cfg.get("goal_radius", 1.5)),
        )

    def plan(
        self,
        observation: np.ndarray,
        goal: np.ndarray,
        obstacle_map: Any,
        *,
        dynamic_obstacles: list[dict] | None = None,
    ) -> Plan:
        pos = np.asarray(observation, dtype=float)[:2]
        gl = np.asarray(goal, dtype=float)[:2]
        to_goal = gl - pos
        dist = float(np.linalg.norm(to_goal))
        if dist < self.goal_radius:
            return Plan(waypoints=np.asarray([pos], dtype=float),
                        target_velocity=np.zeros(2), meta={"planner": "roundabout"})

        rel = pos - self.center
        r = float(np.linalg.norm(rel))
        ang_cur = math.atan2(rel[1], rel[0])
        g_rel = gl - self.center
        ang_goal = math.atan2(g_rel[1], g_rel[0])
        # CCW angular distance still to travel to reach the goal's bearing.
        ccw_to_go = (ang_goal - ang_cur) % (2.0 * math.pi)

        # Exit when nearly aligned with the goal bearing, or already inside the
        # ring near the goal (so a goal off the ring is still reached).
        if ccw_to_go < self.exit_angle or ccw_to_go > 2.0 * math.pi - _EPS or dist < self.goal_radius * 2.5:
            vel = to_goal / dist * self.max_speed
        else:
            # ORBIT: tangential (CCW) + radial pull onto the ring.
            if r < _EPS:
                tang = to_goal / dist
            else:
                u = rel / r
                tang = np.array([-u[1], u[0]])              # CCW tangent
                radial = u * (self.ring_radius - r)         # toward the ring
                steer = tang + self.k_radial * radial / max(self.ring_radius, 1.0)
                n = float(np.linalg.norm(steer))
                tang = steer / n if n > _EPS else tang
            vel = tang * self.max_speed

        wp = pos + vel * self.time_step
        return Plan(waypoints=np.asarray([wp], dtype=float), target_velocity=vel,
                    meta={"planner": "roundabout"})
