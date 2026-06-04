"""CBF — Control-Barrier-Function reciprocal collision avoidance.

A clean-room 2-D single-integrator CBF-QP safety filter, the reactive school
behind the recent decentralized-deadlock papers (e.g. LivePoint, arXiv:2503.13098;
"Deadlock-free … via Discrete-Time CBFs", arXiv:2308.10966). It is the third
reactive family in this repo, alongside the velocity-obstacle ORCA
(`planner.type: orca`) and the position-space BVC (`planner.type: bvc`).

Mechanism. For each peer j define the safety function
    h_ij = |p_i - p_j|^2 - (r_i + r_j + margin)^2   (>= 0 means safe).
The discrete CBF condition ``ḣ + alpha·h >= 0`` with single-integrator dynamics
(``ḣ = 2 (p_i - p_j)·(v_i - v_j)``) is linear in the control velocity v_i:
    (p_j - p_i)·v_i  <=  (p_j - p_i)·v_j + (alpha/2)·h_ij,
i.e. one velocity half-plane per peer, exactly like ORCA — but derived from
barrier theory, not a truncated velocity-obstacle cone. The controller then
solves the safety-filter QP
    min |v_i - v_nom|^2   s.t. all CBF half-planes and |v_i| <= max_speed,
where v_nom drives to the goal. ``reciprocal=True`` (default) splits the burden
50/50 with the peer (each half-plane keeps half its slack), matching the
reciprocal school; ``reciprocal=False`` is the conservative full-responsibility
filter.

Like ORCA and BVC, a plain CBF filter has a hard safety property but no liveness
guarantee: on the symmetric antipodal swap the barrier constraints brake every
agent at the hub and it DEADLOCKS (timeout). The same right-of-way conventions
proven planner-agnostic for MPC (#68/#84), ORCA (#85) and BVC are ported here
(``lateral_bias`` / ``pairwise_bias``) to test whether they generalise to a
fourth controller.

Scope: 2-D, agent-agent only (static occupancy ignored); raises on ndim != 2.
The half-plane QP reuses the standard randomized-incremental 2-D linear program
(van den Berg 2011 / RVO2), which solves exactly this min-distance-to-nominal
problem under linear velocity constraints.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import PLANNER_REGISTRY, Plan, Planner

_EPS = 1e-10


def _det(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _norm(v: np.ndarray) -> float:
    return float(np.hypot(v[0], v[1]))


# --- 2-D velocity linear program (lines are (point, direction); feasible side
# is to the LEFT of direction) — same algorithm as the ORCA baseline. ----------
def _lp1(lines, i, radius, opt, dir_opt, result):
    pt, dr = lines[i]
    dot = float(pt @ dr)
    disc = dot * dot + radius * radius - float(pt @ pt)
    if disc < 0.0:
        return False, result
    sq = np.sqrt(disc)
    t_left, t_right = -dot - sq, -dot + sq
    for k in range(i):
        pk, dk = lines[k]
        denom = _det(dr, dk)
        numer = _det(dk, pt - pk)
        if abs(denom) <= _EPS:
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
    if dir_opt:
        result = pt + (t_right if float(opt @ dr) > 0.0 else t_left) * dr
    else:
        t = float(dr @ (opt - pt))
        result = pt + (t_left if t < t_left else t_right if t > t_right else t) * dr
    return True, result


def _lp2(lines, radius, opt, dir_opt):
    if dir_opt:
        result = opt * radius
    elif float(opt @ opt) > radius * radius:
        result = opt / _norm(opt) * radius
    else:
        result = opt.copy()
    for i in range(len(lines)):
        if _det(lines[i][1], lines[i][0] - result) > 0.0:
            tmp = result.copy()
            ok, result = _lp1(lines, i, radius, opt, dir_opt, result)
            if not ok:
                return i, tmp
    return len(lines), result


def _lp3(lines, begin, radius, result):
    distance = 0.0
    for i in range(begin, len(lines)):
        if _det(lines[i][1], lines[i][0] - result) > distance:
            proj = []
            for j in range(i):
                det = _det(lines[i][1], lines[j][1])
                if abs(det) <= _EPS:
                    if float(lines[i][1] @ lines[j][1]) > 0.0:
                        continue
                    point = 0.5 * (lines[i][0] + lines[j][0])
                else:
                    point = lines[i][0] + (
                        _det(lines[j][1], lines[i][0] - lines[j][0]) / det
                    ) * lines[i][1]
                dirv = lines[j][1] - lines[i][1]
                proj.append((point, dirv / _norm(dirv)))
            tmp = result.copy()
            opt_dir = np.array([-lines[i][1][1], lines[i][1][0]])
            cnt, result = _lp2(proj, radius, opt_dir, True)
            if cnt < len(proj):
                result = tmp
            distance = _det(lines[i][1], lines[i][0] - result)
    return result


@PLANNER_REGISTRY.register("cbf")
class CBFPlanner(Planner):
    """Control-Barrier-Function QP safety filter; 2-D, agent-agent only.

    Config keys
    -----------
    ``max_speed``      : cruise / cap speed (m/s).
    ``radius``         : ego collision radius (m).
    ``safety_margin``  : extra margin added to each pairwise safe distance (m).
    ``alpha``          : CBF class-K gain (1/s). Larger = act later / more
                         aggressively; smaller = brake earlier / more cautious.
    ``neighbor_dist``  : ignore peers farther than this (m).
    ``time_step``      : step for waypoint extrapolation; runner uses the
                         velocity directly.
    ``goal_radius``    : within this the nominal velocity is zero (arrive/stop).
    ``reciprocal``     : split the avoidance 50/50 with the peer (default True).
    ``lateral_bias``   : GLOBAL right-of-way; tilt the nominal velocity right of
                         the goal heading (fraction of cruise), unconditionally.
    ``pairwise_bias``  : PAIRWISE right-of-way; tilt the nominal velocity toward
                         "pass each nearby peer on the right", exp(-d/radius)
                         weighted — vanishes with no peer (cf. MPC/ORCA/BVC ports).
    ``pairwise_radius``: neighbour-conflict falloff length (m) for pairwise_bias.
    """

    def __init__(
        self,
        max_speed: float = 5.0,
        radius: float = 0.4,
        safety_margin: float = 0.1,
        alpha: float = 2.0,
        neighbor_dist: float = 15.0,
        time_step: float = 0.1,
        goal_radius: float = 1.5,
        reciprocal: bool = True,
        lateral_bias: float = 0.0,
        pairwise_bias: float = 0.0,
        pairwise_radius: float = 5.0,
    ) -> None:
        self.max_speed = float(max_speed)
        self.radius = float(radius)
        self.safety_margin = float(safety_margin)
        self.alpha = float(alpha)
        self.neighbor_dist = float(neighbor_dist)
        self.time_step = float(time_step)
        self.goal_radius = float(goal_radius)
        self.reciprocal = bool(reciprocal)
        self.lateral_bias = float(lateral_bias)
        self.pairwise_bias = float(pairwise_bias)
        self.pairwise_radius = max(1e-6, float(pairwise_radius))
        self._cur_vel: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "CBFPlanner":
        return cls(
            max_speed=float(cfg.get("max_speed", 5.0)),
            radius=float(cfg.get("radius", 0.4)),
            safety_margin=float(cfg.get("safety_margin", 0.1)),
            alpha=float(cfg.get("alpha", 2.0)),
            neighbor_dist=float(cfg.get("neighbor_dist", 15.0)),
            time_step=float(cfg.get("time_step", cfg.get("dt", 0.1))),
            goal_radius=float(cfg.get("goal_radius", 1.5)),
            reciprocal=bool(cfg.get("reciprocal", True)),
            lateral_bias=float(cfg.get("lateral_bias", 0.0)),
            pairwise_bias=float(cfg.get("pairwise_bias", 0.0)),
            pairwise_radius=float(cfg.get("pairwise_radius", 5.0)),
        )

    def reset(self) -> None:
        self._cur_vel = None

    def set_current_state(self, position, velocity=None) -> None:
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
                f"CBFPlanner is 2-D only; got {pos.shape[0]}-D observation. "
                "Use a sampling planner (mpc/mppi) for 3-D swarms."
            )
        gl = np.asarray(goal, dtype=float)[:2]
        to_goal = gl - pos
        dist = _norm(to_goal)
        if dist < self.goal_radius:
            return Plan(waypoints=np.asarray([pos], dtype=float),
                        target_velocity=np.zeros(2), meta={"planner": "cbf"})

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
                nr = rel / dn
                tilt = tilt + np.exp(-dn / self.pairwise_radius) * np.array([nr[1], -nr[0]])
            gdir = gdir + self.pairwise_bias * tilt
            gn = _norm(gdir)
            if gn > _EPS:
                gdir = gdir / gn
        v_nom = gdir * self.max_speed

        v_cur = self._cur_vel if self._cur_vel is not None else np.zeros(2)
        share = 0.5 if self.reciprocal else 1.0

        # CBF half-plane per peer: (p_j - p_i)·v <= (p_j-p_i)·v_j + (alpha/2)·h.
        # Encoded as an ORCA-style line (point, direction): the boundary is the
        # set { v : a·v = b }; feasible region a·v <= b is to the LEFT of
        # direction = (-a)/|a| rotated... we store direction so that the LP's
        # "left of direction" test selects a·v <= b. With direction d s.t.
        # det(d, p - v) > 0 means infeasible, choose point on the boundary and
        # direction = unit(a) rotated -90deg.
        lines = []
        for ob in (dynamic_obstacles or []):
            p_other = np.asarray(ob["position"], dtype=float)[:2]
            v_other = np.asarray(ob.get("velocity", (0.0, 0.0)), dtype=float)[:2]
            a = p_other - pos
            an = _norm(a)
            if an < _EPS or an > self.neighbor_dist:
                continue
            r_safe = self.radius + float(ob.get("radius", 0.5)) + self.safety_margin
            h = an * an - r_safe * r_safe
            # a·v <= b
            b = share * (float(a @ v_other) + 0.5 * self.alpha * h)
            n = a / an                      # unit normal (a = n * an)
            boundary_pt = n * (b / an)      # a point with a·pt = b
            # LP feasibility is det(direction, point - v) <= 0; with this
            # direction that reduces to a·v <= b (verified in tests).
            direction = np.array([-n[1], n[0]])
            lines.append((boundary_pt, direction))

        cnt, v = _lp2(lines, self.max_speed, v_nom, False)
        if cnt < len(lines):
            v = _lp3(lines, cnt, self.max_speed, v)

        wp = pos + v * self.time_step
        return Plan(waypoints=np.asarray([wp], dtype=float), target_velocity=v,
                    meta={"planner": "cbf", "n_lines": len(lines)})
