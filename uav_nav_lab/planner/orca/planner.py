"""ORCA (Optimal Reciprocal Collision Avoidance) — the canonical reactive
multi-agent baseline.

This is a clean-room 2D implementation of the ORCA velocity controller from

    J. van den Berg, S. J. Guy, M. Lin, D. Manocha,
    "Reciprocal n-Body Collision Avoidance", ISRR 2009 / Robotics Research 2011.

The reference open-source implementation is the C++ RVO2 library
(https://github.com/snape/RVO2, Apache-2.0) and its pure-Python port
chengji253/RVO2-python (MIT). The half-plane construction and the
randomized-incremental 2-D linear program below (``_linear_program1/2/3``)
follow that algorithm; the code itself is written from the published
algorithm to fit this repo's planner registry, not copied.

Why this planner exists in a repo full of sampling planners
-----------------------------------------------------------
Every other multi-drone planner here (mpc / mppi / chomp) is a *predict-then-
optimize* controller: it forecasts peers (constant_velocity / game_theoretic)
and scores sampled trajectories. ORCA is the *reciprocal* school's answer to
the same problem and the standard literature baseline for swarm collision
avoidance: each agent assumes every neighbour shares the avoidance effort
50/50, turns each pairwise encounter into a velocity-space half-plane, and
picks the velocity closest to its goal-seeking preferred velocity that lies in
the intersection of all half-planes (a tiny 2-D linear program). No forecast,
no sampling, no communication.

It is famously prone to symmetric deadlock on the antipodal swap — which is
exactly the benchmark this repo's right-of-way / predictor findings live on, so
ORCA gives those findings a published baseline to be measured against.

Scope / caveats
---------------
* **2-D only.** The 3-D ORCA (RVO2-3D) uses a different LP on the velocity
  sphere; this planner raises if asked for ndim != 2.
* **Agent-agent only.** Static occupancy is ignored (no static ORCA lines).
  Appropriate for the open-arena swarm scenarios (obstacles: none); it is *not*
  a clutter planner. Scene dynamic obstacles are treated as neighbours with
  their reported radius and (reciprocity-off) full responsibility.
* Needs the runner's ``set_current_state`` hook for the ego velocity (the LP
  centres the reciprocal split on the *current* velocity). The multi-drone
  runner calls it every replan; falls back to the last commanded velocity.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..base import PLANNER_REGISTRY, Plan, Planner

_EPS = 1e-10


def _det(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _norm(v: np.ndarray) -> float:
    return float(np.hypot(v[0], v[1]))


class _Line:
    __slots__ = ("point", "direction")

    def __init__(self, point: np.ndarray, direction: np.ndarray) -> None:
        self.point = point
        self.direction = direction


def _linear_program1(
    lines: list[_Line], line_no: int, radius: float,
    opt_velocity: np.ndarray, direction_opt: bool, result: np.ndarray,
) -> tuple[bool, np.ndarray]:
    """Optimise along the single constraint line ``line_no`` subject to all
    earlier lines and the max-speed circle. Returns (feasible, result)."""
    line = lines[line_no]
    dot = float(line.point @ line.direction)
    discriminant = dot * dot + radius * radius - float(line.point @ line.point)
    if discriminant < 0.0:
        # max-speed circle does not intersect this line
        return False, result
    sqrt_disc = np.sqrt(discriminant)
    t_left = -dot - sqrt_disc
    t_right = -dot + sqrt_disc
    for i in range(line_no):
        denom = _det(line.direction, lines[i].direction)
        numer = _det(lines[i].direction, line.point - lines[i].point)
        if abs(denom) <= _EPS:
            # lines nearly parallel
            if numer < 0.0:
                return False, result
            continue
        t = numer / denom
        if denom >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return False, result
    if direction_opt:
        if float(opt_velocity @ line.direction) > 0.0:
            result = line.point + t_right * line.direction
        else:
            result = line.point + t_left * line.direction
    else:
        t = float(line.direction @ (opt_velocity - line.point))
        if t < t_left:
            result = line.point + t_left * line.direction
        elif t > t_right:
            result = line.point + t_right * line.direction
        else:
            result = line.point + t * line.direction
    return True, result


def _linear_program2(
    lines: list[_Line], radius: float, opt_velocity: np.ndarray,
    direction_opt: bool,
) -> tuple[int, np.ndarray]:
    """Find the velocity in the intersection of all half-planes closest to
    ``opt_velocity`` (or, if direction_opt, farthest in that direction).
    Returns (n_satisfied, result); n_satisfied < len(lines) signals the first
    infeasible line."""
    if direction_opt:
        result = opt_velocity * radius
    elif float(opt_velocity @ opt_velocity) > radius * radius:
        result = opt_velocity / _norm(opt_velocity) * radius
    else:
        result = opt_velocity.copy()
    for i in range(len(lines)):
        if _det(lines[i].direction, lines[i].point - result) > 0.0:
            temp = result.copy()
            ok, result = _linear_program1(
                lines, i, radius, opt_velocity, direction_opt, result
            )
            if not ok:
                return i, temp
    return len(lines), result


def _linear_program3(
    lines: list[_Line], num_obst_lines: int, begin_line: int,
    radius: float, result: np.ndarray,
) -> np.ndarray:
    """Densest-deadlock fallback: when LP2 is infeasible, relax toward the
    least-violated velocity (push all reciprocal lines out together)."""
    distance = 0.0
    for i in range(begin_line, len(lines)):
        if _det(lines[i].direction, lines[i].point - result) > distance:
            proj: list[_Line] = list(lines[:num_obst_lines])
            for j in range(num_obst_lines, i):
                determinant = _det(lines[i].direction, lines[j].direction)
                if abs(determinant) <= _EPS:
                    if float(lines[i].direction @ lines[j].direction) > 0.0:
                        continue  # same direction
                    point = 0.5 * (lines[i].point + lines[j].point)
                else:
                    point = lines[i].point + (
                        _det(lines[j].direction, lines[i].point - lines[j].point)
                        / determinant
                    ) * lines[i].direction
                dirv = lines[j].direction - lines[i].direction
                proj.append(_Line(point, dirv / _norm(dirv)))
            temp = result.copy()
            opt_dir = np.array([-lines[i].direction[1], lines[i].direction[0]])
            cnt, result = _linear_program2(proj, radius, opt_dir, True)
            if cnt < len(proj):
                result = temp
            distance = _det(lines[i].direction, lines[i].point - result)
    return result


@PLANNER_REGISTRY.register("orca")
class ORCAPlanner(Planner):
    """Reciprocal velocity-obstacle (ORCA) controller; 2-D, agent-agent only.

    Config keys
    -----------
    ``max_speed``      : cruise / cap speed (m/s).
    ``radius``         : ego collision radius (m); default 0.4 to match the
                         sim drone radius.
    ``time_horizon``   : seconds of lookahead for the reciprocal half-planes
                         (larger → earlier, more conservative avoidance).
    ``time_step``      : step used only for the already-overlapping recovery
                         case; smaller → firmer separation push.
    ``neighbor_dist``  : ignore neighbours farther than this (m).
    ``safety_margin``  : added to each pairwise combined radius (m).
    ``goal_radius``    : within this, preferred velocity is zero (arrive/stop).
    ``lateral_bias``   : right-of-way convention strength (default 0 = stock
                         ORCA). When > 0, the preferred velocity is tilted to
                         the ego drone's right by this fraction before the LP,
                         so a symmetric head-on encounter resolves into a
                         consistent clockwise pass instead of a mirror-swerve
                         deadlock. This is the ORCA port of the MPC
                         ``planner.lateral_bias`` right-of-way knob — included
                         to test whether that convention generalises beyond the
                         sampling planner to the canonical reciprocal one. It is
                         a GLOBAL rule: the tilt fires whenever the agent is
                         moving toward goal, even with no neighbour around, which
                         is why too large a value over-rotates into an orbit
                         (timeout) — see ``pairwise_bias``.
    ``pairwise_bias``  : PAIRWISE right-of-way strength (default 0 = off). The
                         ORCA port of the MPC ``planner.pairwise_bias`` knob.
                         Instead of tilting toward a fixed global "right of goal"
                         heading, it tilts the preferred velocity toward the sum
                         over nearby neighbours of "pass this neighbour on the
                         right" (clockwise perpendicular of the bearing to it),
                         each weighted ``exp(-dist / pairwise_radius)``. With no
                         neighbour in conflict (or neighbours that cancel by
                         symmetry) the tilt vanishes, so unlike ``lateral_bias``
                         it does not over-rotate a lone agent — testing whether
                         the pairwise rule removes the global rule's
                         over-rotation timeout cliff on ORCA as it removes its
                         no-deadlock harm on the MPC.
    ``pairwise_radius``: neighbour-conflict falloff length (m) for pairwise_bias.
    """

    def __init__(
        self,
        max_speed: float = 5.0,
        radius: float = 0.4,
        time_horizon: float = 2.0,
        time_step: float = 0.25,
        neighbor_dist: float = 15.0,
        safety_margin: float = 0.0,
        goal_radius: float = 1.5,
        lateral_bias: float = 0.0,
        pairwise_bias: float = 0.0,
        pairwise_radius: float = 5.0,
    ) -> None:
        self.max_speed = float(max_speed)
        self.radius = float(radius)
        self.time_horizon = float(time_horizon)
        self.time_step = float(time_step)
        self.neighbor_dist = float(neighbor_dist)
        self.safety_margin = float(safety_margin)
        self.goal_radius = float(goal_radius)
        self.lateral_bias = float(lateral_bias)
        self.pairwise_bias = float(pairwise_bias)
        self.pairwise_radius = max(1e-6, float(pairwise_radius))
        self._cur_vel: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "ORCAPlanner":
        return cls(
            max_speed=float(cfg.get("max_speed", 5.0)),
            radius=float(cfg.get("radius", 0.4)),
            time_horizon=float(cfg.get("time_horizon", 2.0)),
            time_step=float(cfg.get("time_step", 0.25)),
            neighbor_dist=float(cfg.get("neighbor_dist", 15.0)),
            safety_margin=float(cfg.get("safety_margin", 0.0)),
            goal_radius=float(cfg.get("goal_radius", 1.5)),
            lateral_bias=float(cfg.get("lateral_bias", 0.0)),
            pairwise_bias=float(cfg.get("pairwise_bias", 0.0)),
            pairwise_radius=float(cfg.get("pairwise_radius", 5.0)),
        )

    def reset(self) -> None:
        self._cur_vel = None

    def set_current_state(
        self, position: np.ndarray, velocity: np.ndarray | None = None
    ) -> None:
        if velocity is not None:
            self._cur_vel = np.asarray(velocity, dtype=float)[:2]

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
                f"ORCAPlanner is 2-D only; got {pos.shape[0]}-D observation. "
                "Use a sampling planner (mpc/mppi) for 3-D swarms."
            )
        gl = np.asarray(goal, dtype=float)[:2]
        v_cur = self._cur_vel if self._cur_vel is not None else np.zeros(2)

        # Preferred velocity: straight to goal at cruise speed; stop in goal disc.
        to_goal = gl - pos
        dist = _norm(to_goal)
        if dist < self.goal_radius:
            v_pref = np.zeros(2)
        else:
            v_pref = to_goal / dist * self.max_speed
            if self.lateral_bias > 0.0:
                # Right-of-way: tilt the preferred velocity to the ego's right
                # (clockwise perpendicular = (y, -x) of the goal direction),
                # then renormalise to cruise speed. Turns a symmetric head-on
                # into a consistent clockwise pass.
                gdir = to_goal / dist
                right = np.array([gdir[1], -gdir[0]])
                v_pref = v_pref + self.lateral_bias * self.max_speed * right
                v_pref = v_pref / _norm(v_pref) * self.max_speed
            if self.pairwise_bias > 0.0 and dynamic_obstacles:
                # PAIRWISE right-of-way: tilt the preferred velocity toward the
                # sum of "pass each nearby neighbour on the right" (clockwise
                # perpendicular of the bearing to it), weighted exp(-d/radius).
                # Conditional on real neighbours, so a lone agent (or one whose
                # neighbours cancel by symmetry) is not tilted and cannot
                # over-rotate into an orbit — the ORCA port of the MPC pairwise
                # knob.
                tilt = np.zeros(2)
                for d in dynamic_obstacles:
                    rel = np.asarray(d["position"], dtype=float)[:2] - pos
                    dn = _norm(rel)
                    if dn < _EPS:
                        continue
                    nrel = rel / dn
                    cw = np.array([nrel[1], -nrel[0]])  # right of the neighbour
                    tilt = tilt + np.exp(-dn / self.pairwise_radius) * cw
                v_pref = v_pref + self.pairwise_bias * self.max_speed * tilt
                vp = _norm(v_pref)
                if vp > _EPS:
                    v_pref = v_pref / vp * self.max_speed

        lines: list[_Line] = []
        inv_th = 1.0 / self.time_horizon
        inv_ts = 1.0 / self.time_step
        nb2 = self.neighbor_dist * self.neighbor_dist
        for d in dynamic_obstacles or []:
            p_other = np.asarray(d["position"], dtype=float)[:2]
            v_other = np.asarray(d.get("velocity", (0.0, 0.0)), dtype=float)[:2]
            rel_pos = p_other - pos
            dist_sq = float(rel_pos @ rel_pos)
            if dist_sq > nb2:
                continue
            rel_vel = v_cur - v_other
            comb_r = self.radius + float(d.get("radius", 0.5)) + self.safety_margin
            comb_r_sq = comb_r * comb_r

            if dist_sq > comb_r_sq:
                # No collision yet: half-plane from the truncated VO cone.
                w = rel_vel - inv_th * rel_pos
                w_len_sq = float(w @ w)
                dot1 = float(w @ rel_pos)
                if dot1 < 0.0 and dot1 * dot1 > comb_r_sq * w_len_sq:
                    # project on the cut-off circle (front)
                    w_len = np.sqrt(w_len_sq)
                    unit_w = w / w_len if w_len > _EPS else np.zeros(2)
                    direction = np.array([unit_w[1], -unit_w[0]])
                    u = (comb_r * inv_th - w_len) * unit_w
                else:
                    # project on the nearer leg of the cone
                    leg = np.sqrt(max(dist_sq - comb_r_sq, 0.0))
                    if _det(rel_pos, w) > 0.0:
                        direction = np.array([
                            rel_pos[0] * leg - rel_pos[1] * comb_r,
                            rel_pos[0] * comb_r + rel_pos[1] * leg,
                        ]) / dist_sq
                    else:
                        direction = -np.array([
                            rel_pos[0] * leg + rel_pos[1] * comb_r,
                            -rel_pos[0] * comb_r + rel_pos[1] * leg,
                        ]) / dist_sq
                    dot2 = float(rel_vel @ direction)
                    u = dot2 * direction - rel_vel
            else:
                # Already overlapping: use the control step to push apart.
                w = rel_vel - inv_ts * rel_pos
                w_len = _norm(w)
                unit_w = w / w_len if w_len > _EPS else np.zeros(2)
                direction = np.array([unit_w[1], -unit_w[0]])
                u = (comb_r * inv_ts - w_len) * unit_w

            lines.append(_Line(v_cur + 0.5 * u, direction))

        count, new_vel = _linear_program2(lines, self.max_speed, v_pref, False)
        if count < len(lines):
            new_vel = _linear_program3(lines, 0, count, self.max_speed, new_vel)

        # Short straight extrapolation as waypoints (viz / pure-pursuit fallback);
        # the runner consumes target_velocity directly.
        wp = pos + new_vel * self.time_step
        return Plan(
            waypoints=np.asarray([wp], dtype=float),
            target_velocity=new_vel,
            meta={"planner": "orca", "n_lines": len(lines)},
        )
