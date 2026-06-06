"""Self-contained cooperative aerial transport sim (2D top-down).

N drones rigidly carry a beam-shaped payload (a virtual rigid structure) from a
start to a goal through a wall with a single doorway gap. The team is modelled as
a virtual structure: the beam centre `c` and orientation `theta` are the controlled
DOF, and drone i sits at a fixed body-frame anchor `s_i` along the beam, so the
formation stays rigid by construction (cooperative carrying, not free swarm).

This module is the TeamHOI-style probe (CVPR 2026, cooperative human-object
interaction with any team size): TeamHOI's headline is a single decentralized
policy that is "team-size- and shape-agnostic". Here we interrogate that claim on
the canonical hard case for a RIGID carried payload -- threading an aperture --
where the geometry, not the policy, may decide success.

Two arms, paired by seed:
  fixed     beam orientation held at its initial (perpendicular-to-travel) pose
  adaptive  beam reorients to align with the travel direction near the doorway,
            shrinking its cross-aperture footprint so it can slip through

The only difference between arms is whether the beam is allowed to rotate; the
translational controller (goal attraction + doorway y-centring) is identical, so
any success gap is attributable to reorientation alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TransportResult:
    success: bool
    reason: str                       # "goal" | "collision" | "timeout"
    n_steps: int
    # per-step replay for rendering (only populated when record=True)
    centres: list = field(default_factory=list)      # (T,2)
    thetas: list = field(default_factory=list)        # (T,)
    drones: list = field(default_factory=list)        # (T,N,2)
    collide_step: int = -1


def _anchors(n: int, bar_len: float) -> np.ndarray:
    """Body-frame x-positions of the N drones evenly spread along the beam."""
    if n == 1:
        return np.zeros((1,))
    return np.linspace(-bar_len / 2.0, bar_len / 2.0, n)


def _drone_world(c: np.ndarray, theta: float, anchors: np.ndarray) -> np.ndarray:
    """World positions of all drones for beam pose (c, theta)."""
    d = np.stack([np.cos(theta) * anchors, np.sin(theta) * anchors], axis=1)
    return c[None, :] + d


def _wall_hit(pts: np.ndarray, wall_x: float, wall_t: float,
              gap_y: float, gap_w: float, margin: float) -> bool:
    """True if any point (with `margin` radius) intersects the wall outside its gap."""
    in_x = np.abs(pts[:, 0] - wall_x) <= (wall_t / 2.0 + margin)
    in_gap = np.abs(pts[:, 1] - gap_y) <= (gap_w / 2.0 - margin)
    return bool(np.any(in_x & ~in_gap))


def simulate(
    *,
    n: int = 4,
    bar_len: float = 10.0,
    gap_w: float = 6.0,
    seed: int = 0,
    adaptive: bool = True,
    wall_x: float = 25.0,
    wall_t: float = 1.5,
    gap_y: float = 25.0,
    start_x: float = 8.0,
    goal_x: float = 42.0,
    centre_y: float = 25.0,
    world: tuple[float, float] = (50.0, 50.0),
    drone_r: float = 0.5,
    beam_t: float = 0.4,
    speed: float = 4.0,
    dt: float = 0.05,
    max_steps: int = 1400,
    k_rot: float = 2.5,
    approach: float = 14.0,
    jitter: float = 1.2,
    record: bool = False,
    beam_samples: int = 25,
) -> TransportResult:
    """Run one cooperative-transport episode; return outcome (+ replay if record).

    The beam starts perpendicular to the travel direction (theta=pi/2, spanning y).
    `adaptive` lets it rotate toward travel (theta->0, spanning x) as it nears the
    wall, shrinking the y-footprint that must fit the gap. `fixed` keeps theta=pi/2,
    so it can only pass when gap_w exceeds the full bar length.
    """
    rng = np.random.default_rng(seed)
    # seed-paired perturbations (identical for both arms at a given seed)
    gy = gap_y + float(rng.normal(0.0, jitter))
    cy = centre_y + float(rng.normal(0.0, jitter))
    theta0 = math.pi / 2.0

    anchors = _anchors(n, bar_len)
    c = np.array([start_x, cy], dtype=float)
    theta = theta0
    goal = np.array([goal_x, gy], dtype=float)

    centres: list = []
    thetas: list = []
    drones: list = []

    margin_d = drone_r
    margin_b = beam_t / 2.0

    for step in range(max_steps):
        if record:
            centres.append(c.copy())
            thetas.append(theta)
            drones.append(_drone_world(c, theta, anchors))

        # --- translational control (identical for both arms) ---
        to_goal = goal - c
        dist_goal = float(np.linalg.norm(to_goal))
        if dist_goal < 1e-6:
            v = np.zeros(2)
        else:
            v = to_goal / dist_goal * speed
        # doorway y-centring: as the beam nears the wall, pull its centre toward
        # the gap centre so it actually aims at the opening (both arms equally).
        dx_wall = abs(c[0] - wall_x)
        if dx_wall < approach:
            w = 1.0 - dx_wall / approach
            v[1] += 3.0 * w * (gy - c[1])

        c = c + v * dt
        c[0] = min(c[0], world[0])

        # --- orientation control (the ONLY arm difference) ---
        if adaptive:
            # align beam long-axis with travel direction near the doorway
            if dx_wall < approach:
                theta_t = math.atan2(v[1], v[0])  # travel heading
                err = math.atan2(math.sin(theta_t - theta), math.cos(theta_t - theta))
                theta += k_rot * err * dt
        # fixed: theta stays at theta0

        # --- collision test (drones + sampled beam body) ---
        dpos = _drone_world(c, theta, anchors)
        bs = np.linspace(-bar_len / 2.0, bar_len / 2.0, beam_samples)
        beam_pts = c[None, :] + np.stack(
            [np.cos(theta) * bs, np.sin(theta) * bs], axis=1
        )
        hit = (
            _wall_hit(dpos, wall_x, wall_t, gy, gap_w, margin_d)
            or _wall_hit(beam_pts, wall_x, wall_t, gy, gap_w, margin_b)
        )
        if hit:
            if record:
                centres.append(c.copy())
                thetas.append(theta)
                drones.append(dpos)
            return TransportResult(
                success=False, reason="collision", n_steps=step,
                centres=centres, thetas=thetas, drones=drones,
                collide_step=len(centres) - 1,
            )

        # --- goal test (beam centre cleared the wall and reached goal x) ---
        if c[0] >= goal_x - 0.5:
            if record:
                centres.append(c.copy())
                thetas.append(theta)
                drones.append(dpos)
            return TransportResult(
                success=True, reason="goal", n_steps=step,
                centres=centres, thetas=thetas, drones=drones,
            )

    return TransportResult(
        success=False, reason="timeout", n_steps=max_steps,
        centres=centres, thetas=thetas, drones=drones,
    )


# --------------------------------------------------------------------------- #
# L-shaped corner: the "ladder around a corner" (piano-mover) ceiling.
#
# A straight doorway lets reorientation make carrying team-size-agnostic (above).
# An L-corner does NOT: rounding a right-angle junction of two corridors of width
# a and b, a rigid segment can be maneuvered through iff its length is at most the
# classical bound L_max = (a^(2/3) + b^(2/3))^(3/2) -- a HARD geometric ceiling no
# reorientation can beat. With beam length growing in the team size, this caps the
# maximum team that can round the corner, however the formation reshapes.
#
# Free space (L): horizontal corridor {0<=y<=w, x<=W} UNION vertical corridor
# {W-w<=x<=W, y>=0}; concave inner corner at I=(W-w, w). The optimal rounding
# motion keeps the beam through I; at orientation theta its span between the two
# outer walls is a/sin(theta) + b/cos(theta), minimised at the critical angle.
# --------------------------------------------------------------------------- #

def corner_Lmax(a: float, b: float) -> float:
    """Longest rigid segment that can round a right-angle a×b corridor corner."""
    return (a ** (2.0 / 3.0) + b ** (2.0 / 3.0)) ** 1.5


def _in_L_free(pts: np.ndarray, w: float, W: float, margin: float) -> bool:
    """All points inside the L free space (with `margin` clearance from walls)?"""
    horiz = (pts[:, 1] >= margin) & (pts[:, 1] <= w - margin) & (pts[:, 0] <= W - margin)
    vert = (pts[:, 0] >= (W - w) + margin) & (pts[:, 0] <= W - margin) & (pts[:, 1] >= margin)
    return bool(np.all(horiz | vert))


def simulate_corner(
    *,
    n: int = 4,
    bar_len: float = 10.0,
    corridor_w: float = 4.0,
    seed: int = 0,
    jitter: float = 0.4,
    W: float = 30.0,
    drone_r: float = 0.5,
    beam_t: float = 0.4,
    n_theta: int = 120,
    record: bool = False,
    beam_samples: int = 31,
) -> TransportResult:
    """Round the beam through an L-corner via the optimal (tangent-to-inner-corner)
    motion. Succeeds iff the beam fits the tightest configuration -- i.e. its length
    is within the ladder-around-corner bound for the (seed-jittered) corridor widths.

    This is the corner counterpart to `simulate(adaptive=True)`: the formation is
    free to reorient continuously, but the corner geometry imposes a length ceiling
    the straight doorway does not.
    """
    rng = np.random.default_rng(seed)
    a = corridor_w + float(rng.normal(0.0, jitter))
    b = corridor_w + float(rng.normal(0.0, jitter))
    w = min(a, b)  # render with the binding width; bound uses (a, b)
    I = np.array([W - w, w])  # inner (concave) corner

    # The rigorous feasibility for a rigid segment rounding a right-angle corner is
    # the classical ladder bound: it can be maneuvered through iff its length (here
    # inflated by the drone footprint) does not exceed L_max = (a^2/3 + b^2/3)^3/2.
    # No reorientation beats this -- it is the geometric ceiling the doorway lacks.
    L_eff = bar_len + 2.0 * drone_r
    ok = L_eff <= corner_Lmax(a, b)

    centres: list = []
    thetas: list = []
    drones: list = []
    if record:
        anchors = _anchors(n, bar_len)
        # offset the beam centre off the concave corner into free space so the
        # rendered pivot grazes (not crosses) the inner corner
        off = (beam_t / 2.0 + drone_r) * np.array([1.0, -1.0]) / math.sqrt(2.0)
        # the jam (if any) happens at the critical ~45deg configuration
        crit = math.degrees(math.atan2(b ** (1.0 / 3.0), a ** (1.0 / 3.0)))
        for k in range(n_theta):
            theta = math.radians(5.0 + 80.0 * k / (n_theta - 1))
            dir_ = np.array([math.cos(theta), math.sin(theta)])
            c = I + off
            centres.append(c.copy())
            thetas.append(theta)
            drones.append(c[None, :] + anchors[:, None] * dir_[None, :])
            if (not ok) and math.degrees(theta) >= crit:
                break  # freeze at the jam

    return TransportResult(
        success=ok, reason="goal" if ok else "collision",
        n_steps=len(centres) if record else n_theta,
        centres=centres, thetas=thetas, drones=drones,
        collide_step=(len(centres) - 1) if (not ok and record) else -1,
    )


if __name__ == "__main__":
    # quick smoke: fixed should fail a sub-bar gap, adaptive should pass.
    for ad in (False, True):
        ok = sum(
            simulate(n=4, bar_len=10.0, gap_w=4.0, seed=s, adaptive=ad).success
            for s in range(20)
        )
        print(f"adaptive={ad!s:5}  gap=4.0 bar=10.0  success={ok}/20")
    # corner: short beam rounds, long beam jams (L_max = 2.83 * width)
    print(f"L_max(w=4) = {corner_Lmax(4.0, 4.0):.2f}")
    for bl in (8.0, 11.0, 13.0):
        ok = sum(simulate_corner(n=4, bar_len=bl, corridor_w=4.0, seed=s).success
                 for s in range(20))
        print(f"corner bar_len={bl:5} (w=4)  success={ok}/20")
