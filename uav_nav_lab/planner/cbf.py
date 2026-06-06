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

Scope: 2-D and 3-D, agent-agent only (static occupancy ignored). The CBF
half-space algebra is identical in any dimension; in 2-D the min-distance-to-
nominal QP is solved with the standard randomized-incremental linear program
(van den Berg 2011 / RVO2), in 3-D with a dimension-agnostic Dykstra projection
onto the half-spaces and the speed ball. The right-of-way conventions
(``lateral_bias`` / ``pairwise_bias``) are an in-plane rule applied to the
horizontal (xy) components in both 2-D and 3-D: the 3-D vertical escape that
dissolves the deadlock for a *sampling* planner is unused by this reactive filter
(its goal-seeking nominal stays in-plane — see findings.md "the 3-D dissolution
is a planner property"), so the in-plane convention is what restores liveness to
a reactive controller in 3-D as well as 2-D.
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


def _solve_qp_nd(v_nom, cons, vmax, iters=40):
    """Closest velocity to ``v_nom`` in the intersection of the half-spaces
    ``a·v <= b`` (``cons`` = list of (a, b)) and the speed ball ``|v| <= vmax``,
    by Dykstra's alternating projection. Dimension-agnostic — the CBF-QP solver
    used for ndim != 2 (the 2-D path keeps the exact RVO2 linear program above so
    the merged 2-D behaviour is unchanged)."""
    x = np.array(v_nom, dtype=float)
    corr = [np.zeros_like(x) for _ in cons]
    bcorr = np.zeros_like(x)
    for _ in range(iters):
        for k, (a, b) in enumerate(cons):
            y = x + corr[k]
            a2 = float(a @ a)
            viol = float(a @ y) - b
            x_new = y - (viol / a2) * a if (viol > 0.0 and a2 > _EPS) else y
            corr[k] = y - x_new
            x = x_new
        y = x + bcorr
        n = float(np.linalg.norm(y))
        x_new = y * (vmax / n) if n > vmax else y
        bcorr = y - x_new
        x = x_new
    return x


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
        priority_yield: bool = False,
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
        # Priority deconfliction: a decentralized total order from each peer's
        # (observable, fixed) GOAL position, lexicographic. The ego avoids only
        # HIGHER-priority peers (goal lexicographically greater) and ignores
        # lower-priority ones, assuming they yield. A fixed goal-key order is
        # consistent (unlike a current-position order, which flips when agents
        # cross), so it breaks the symmetric deadlock by a social hierarchy
        # rather than a geometric convention or a shared roundabout. 0 = off.
        self.priority_yield = bool(priority_yield)
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
            priority_yield=bool(cfg.get("priority_yield", False)),
        )

    def reset(self) -> None:
        self._cur_vel = None

    def set_current_state(self, position, velocity=None) -> None:
        if velocity is not None:
            self._cur_vel = np.asarray(velocity, dtype=float)

    def _nominal_velocity(self, pos, gl, dynamic_obstacles):
        """Nominal (pre-filter) velocity the CBF QP is projected onto: straight
        to goal at cruise speed, optionally tilted by the in-plane right-of-way
        convention. Factored out so subclasses can inject a different reference
        (e.g. the Merry-Go-Round roundabout tangent) and reuse the barrier QP.

        Right-of-way conventions are an IN-PLANE rule (a clockwise pass in the
        horizontal plane): in 3-D they tilt only the (xy) components, leaving the
        vertical axis alone (the 3-D vertical escape is a sampling-planner
        property, unused by this reactive filter).
        """
        to_goal = gl - pos
        gdir = to_goal / float(np.linalg.norm(to_goal))
        if self.lateral_bias > 0.0:
            g2 = gdir[:2]
            n2 = _norm(g2)
            if n2 > _EPS:
                right = np.array([g2[1], -g2[0]]) / n2
                gdir = gdir.copy()
                gdir[:2] = gdir[:2] + self.lateral_bias * right
                gn = float(np.linalg.norm(gdir))
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
            gn = float(np.linalg.norm(gdir))
            if gn > _EPS:
                gdir = gdir / gn
        return gdir * self.max_speed

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
            raise ValueError(f"CBFPlanner supports 2-D/3-D; got {ndim}-D observation.")
        gl = np.asarray(goal, dtype=float)[:ndim]
        to_goal = gl - pos
        dist = float(np.linalg.norm(to_goal))
        if dist < self.goal_radius:
            return Plan(waypoints=np.asarray([pos], dtype=float),
                        target_velocity=np.zeros(ndim),
                        meta={"planner": getattr(self, "_planner_name", "cbf")})

        v_nom = self._nominal_velocity(pos, gl, dynamic_obstacles)

        share = 0.5 if self.reciprocal else 1.0

        # One CBF half-space per peer: (p_j - p_i)·v <= (p_j-p_i)·v_j + (alpha/2)·h,
        # h = |p_j-p_i|^2 - R^2. Identical algebra in 2-D and 3-D — only the
        # vector dimension changes.
        ego_key = (round(float(gl[0]), 3), round(float(gl[1]), 3)) if ndim >= 2 else (0.0, 0.0)
        cons = []  # (a, b): a·v <= b
        for ob in (dynamic_obstacles or []):
            p_other = np.asarray(ob["position"], dtype=float)[:ndim]
            v_other = np.asarray(ob.get("velocity", np.zeros(ndim)), dtype=float)[:ndim]
            # Priority deconfliction: avoid only peers of higher priority (goal
            # lexicographically greater); assume lower-priority peers yield. A
            # peer with no reported goal is always avoided (cannot rank it).
            if self.priority_yield and "goal" in ob:
                g = np.asarray(ob["goal"], dtype=float)
                peer_key = (round(float(g[0]), 3), round(float(g[1]), 3))
                if peer_key <= ego_key:
                    continue
            a = p_other - pos
            an = float(np.linalg.norm(a))
            if an < _EPS or an > self.neighbor_dist:
                continue
            r_safe = self.radius + float(ob.get("radius", 0.5)) + self.safety_margin
            h = an * an - r_safe * r_safe
            b = share * (float(a @ v_other) + 0.5 * self.alpha * h)
            cons.append((a, b))

        if ndim == 2:
            # Keep the exact RVO2 2-D linear program (merged behaviour).
            lines = []
            for a, b in cons:
                an = _norm(a)
                n = a / an
                lines.append((n * (b / an), np.array([-n[1], n[0]])))
            cnt, v = _lp2(lines, self.max_speed, v_nom, False)
            if cnt < len(lines):
                v = _lp3(lines, cnt, self.max_speed, v)
        else:
            v = _solve_qp_nd(v_nom, cons, self.max_speed)

        wp = pos + v * self.time_step
        return Plan(waypoints=np.asarray([wp], dtype=float), target_velocity=v,
                    meta={"planner": getattr(self, "_planner_name", "cbf"),
                          "n_lines": len(cons), "mode": getattr(self, "_mode", None)})
