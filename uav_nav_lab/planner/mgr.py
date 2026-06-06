"""Merry-Go-Round — *decentralized, triggered* roundabout deadlock prevention
(Zhou et al., "Merry-Go-Round: Safe Control of Decentralized Multi-Robot Systems
with Deadlock Prevention", 2025; arXiv:2503.05848).

The lab already has a `roundabout` planner, but it is the *simplified* form the
paper's own framing punts on: an always-on ring around a **fixed centre handed in
by symmetry** (the arena middle). This planner implements the parts that actually
make Merry-Go-Round a decentralized method, none of which the `roundabout` planner
models:

  1. **Triggered, not always-on.** The roundabout engages only when the ego
     detects a deadlock locally — it has been braked to a near-stop (low
     along-goal speed) with a peer close ahead (the paper's barrier-distance
     condition). With no deadlock the controller is the plain base avoider, so it
     cannot harm the unstructured traffic where the always-on conventions are a
     net liability.
  2. **Centre negotiated locally, not handed in.** On trigger the ego sets the
     ring centre to the centroid of its own conflict cluster (itself + nearby
     peers) — computed from sensing, with no global knowledge of the symmetric
     hub. Whether independent agents *agree* on a common ring this way (and so
     reproduce the clean fixed-centre result) is the open question this tests.
  3. **Capacity-tiered radius.** The ring radius grows with the cluster size so
     the circumference always holds the crowd (the paper's overflow-to-larger-
     radius rule), instead of a single fixed `ring_radius`.
  4. **Peel-off with a clear exit sector.** The ego leaves the ring when its
     bearing (CCW) reaches the goal's *and* the forward cone toward the goal is
     clear of peers; otherwise it rides one more arc.

It is layered on the CBF barrier-QP as the base avoider: the roundabout only
supplies the *nominal* velocity, and the CBF half-planes keep it collision-safe.
The CBF base is what makes the trigger reliable — on the antipodal hub a plain
CBF brakes every agent to a safe stop (a *timeout* deadlock, not a collision),
giving the detector a clean stall to fire on. Everyone orbits counter-clockwise;
that shared turn direction is the convention the roundabout carries — the novelty
is that it is *triggered and local*, not that it is convention-free.

Scope: 2-D agent-agent (the roundabout is an in-plane manoeuvre; a 3-D goal's
vertical component passes through to the base nominal).
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .base import PLANNER_REGISTRY
from .cbf import CBFPlanner, _norm, _EPS


@PLANNER_REGISTRY.register("mgr")
class MerryGoRoundPlanner(CBFPlanner):
    """Decentralized triggered Merry-Go-Round on a CBF base avoider; 2-D."""

    def __init__(
        self,
        *,
        # --- deadlock detection (when to engage the roundabout) ---
        detect_dist: float = 5.0,
        stall_frac: float = 0.5,
        trigger_persist: int = 4,
        # --- local ring construction ---
        cluster_dist: float = 15.0,
        ring_min: float = 4.0,
        ring_gap: float = 1.6,
        k_radial: float = 1.0,
        # --- peel-off / exit ---
        exit_angle: float = 0.35,
        exit_sector: float = 0.5,
        exit_range: float = 4.0,
        **cbf_kwargs: Any,
    ) -> None:
        super().__init__(**cbf_kwargs)
        self.detect_dist = float(detect_dist)
        self.stall_frac = float(stall_frac)
        self.trigger_persist = int(trigger_persist)
        self.cluster_dist = float(cluster_dist)
        self.ring_min = float(ring_min)
        self.ring_gap = float(ring_gap)
        self.k_radial = float(k_radial)
        self.exit_angle = float(exit_angle)
        self.exit_sector = float(exit_sector)
        self.exit_range = float(exit_range)
        self._planner_name = "mgr"
        self._mode = "free"
        self._center: np.ndarray | None = None
        self._cluster_n = 1
        self._stall_count = 0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "MerryGoRoundPlanner":
        return cls(
            detect_dist=float(cfg.get("detect_dist", 5.0)),
            stall_frac=float(cfg.get("stall_frac", 0.5)),
            trigger_persist=int(cfg.get("trigger_persist", 4)),
            cluster_dist=float(cfg.get("cluster_dist", 15.0)),
            ring_min=float(cfg.get("ring_min", 4.0)),
            ring_gap=float(cfg.get("ring_gap", 1.6)),
            k_radial=float(cfg.get("k_radial", 1.0)),
            exit_angle=float(cfg.get("exit_angle", 0.35)),
            exit_sector=float(cfg.get("exit_sector", 0.5)),
            exit_range=float(cfg.get("exit_range", 4.0)),
            max_speed=float(cfg.get("max_speed", 5.0)),
            radius=float(cfg.get("radius", 0.4)),
            safety_margin=float(cfg.get("safety_margin", 0.1)),
            alpha=float(cfg.get("alpha", 2.0)),
            neighbor_dist=float(cfg.get("neighbor_dist", 15.0)),
            time_step=float(cfg.get("time_step", cfg.get("dt", 0.1))),
            goal_radius=float(cfg.get("goal_radius", 1.5)),
            reciprocal=bool(cfg.get("reciprocal", True)),
        )

    def reset(self) -> None:
        super().reset()
        self._mode = "free"
        self._center = None
        self._cluster_n = 1
        self._stall_count = 0

    # ------------------------------------------------------------------ helpers
    def _deadlock(self, pos, gdir, v_cur, peers) -> bool:
        """Local deadlock proxy: the ego has been braked to a near-stop AND a
        peer is close ahead (the paper's barrier-distance condition). On the
        antipodal hub the CBF base brakes everyone to exactly this state."""
        if float(v_cur @ gdir) >= self.stall_frac * self.max_speed:
            return False
        for pj, _vj, _rj in peers:
            rel = pj - pos
            if _norm(rel) <= self.detect_dist and float(rel @ gdir) > 0.0:
                return True
        return False

    def _ring_radius(self) -> float:
        """Capacity-tiered radius: circumference must hold the cluster."""
        cap = self._cluster_n * self.ring_gap / (2.0 * math.pi)
        return max(self.ring_min, cap)

    def _orbit_velocity(self, pos, gl) -> np.ndarray:
        rel = pos - self._center
        r = _norm(rel)
        if r < _EPS:
            to_goal = gl - pos
            return to_goal / max(_norm(to_goal), _EPS) * self.max_speed
        u = rel / r
        ring = self._ring_radius()
        tang = np.array([-u[1], u[0]])              # counter-clockwise tangent
        radial = u * (ring - r)                     # pull onto the ring
        steer = tang + self.k_radial * radial / max(ring, 1.0)
        s = _norm(steer)
        v = steer / s if s > _EPS else tang
        return v * self.max_speed

    def _can_exit(self, pos, gl, peers) -> bool:
        if self._center is None:
            return True
        rel = pos - self._center
        g_rel = gl - self._center
        ccw_to_go = (math.atan2(g_rel[1], g_rel[0]) - math.atan2(rel[1], rel[0])) % (2.0 * math.pi)
        if not (ccw_to_go < self.exit_angle or ccw_to_go > 2.0 * math.pi - _EPS):
            return False
        to_goal = gl - pos
        dist = _norm(to_goal)
        if dist < _EPS:
            return True
        gdir = to_goal / dist
        cos_sec = math.cos(self.exit_sector)
        for pj, _vj, _rj in peers:
            relp = pj - pos
            dp = _norm(relp)
            if dp <= self.exit_range and float(relp @ gdir) / max(dp, _EPS) > cos_sec:
                return False  # a peer is blocking the exit cone
        return True

    # --------------------------------------------------------------- nominal vel
    def _nominal_velocity(self, pos, gl, dynamic_obstacles):
        base = super()._nominal_velocity(pos, gl, dynamic_obstacles)  # straight-to-goal
        pos2, gl2 = np.asarray(pos, float)[:2], np.asarray(gl, float)[:2]
        to_goal = gl2 - pos2
        dist = _norm(to_goal)
        if dist < _EPS:
            return base
        gdir = to_goal / dist
        v_cur = self._cur_vel[:2] if self._cur_vel is not None else np.zeros(2)

        peers = []
        for d in dynamic_obstacles or []:
            pj = np.asarray(d["position"], dtype=float)[:2]
            vj = np.asarray(d.get("velocity", (0.0, 0.0)), dtype=float)[:2]
            peers.append((pj, vj, float(d.get("radius", 0.5))))

        if self._mode == "free":
            # Debounce: engage only after the deadlock has PERSISTED for several
            # replans. A genuine symmetric-hub deadlock is permanent (the stall
            # never clears on its own); a transient stall in dense unstructured
            # traffic clears as the crossing resolves, so it must not trip the
            # roundabout (the paper's predicted-trajectory horizon, discretised).
            if self._deadlock(pos2, gdir, v_cur, peers):
                self._stall_count += 1
            else:
                self._stall_count = 0
            if self._stall_count >= self.trigger_persist:
                pts = [pos2] + [pj for pj, _, _ in peers if _norm(pj - pos2) < self.cluster_dist]
                self._center = np.mean(np.asarray(pts), axis=0)
                self._cluster_n = len(pts)
                self._mode = "orbit"
                self._stall_count = 0

        if self._mode == "orbit":
            if self._can_exit(pos2, gl2, peers):
                self._mode = "free"
            else:
                out = np.array(base, dtype=float)
                out[:2] = self._orbit_velocity(pos2, gl2)
                return out

        return base
