"""Two Olfati-Saber flocks crossing head-on — and a right-of-way convention.

This reuses the α-lattice dynamics of `_flocking.py` (the same σ-norm gradient +
velocity consensus from Olfati-Saber 2006) but drives TWO cohesive flocks at each
other: group 0 starts on the left migrating right, group 1 starts on the right
migrating left, their moving γ-goals passing through a shared centre. The α-lattice
potential is *universal* — it repels any agent closer than the desired spacing `d`,
regardless of which flock it belongs to — so when the two flocks meet they cannot
interpenetrate: they form a mutual wall and JAM at the centre while their goals run
on ahead. This is the cohesive-flocking analogue of the antipodal swap deadlock that
runs through this repo's convention work (MPC, ORCA, HRVO).

The fix tested here is the same `lateral_bias` right-of-way convention: every agent
adds a constant veer to the RIGHT of its own goal heading. It is decentralized and
comms-free (each agent uses only its own goal direction), and it turns the symmetric
head-on jam into a pass-on-the-right slip-through.

`simulate_crossing` returns enough to score three outcomes:
  passed     — both flock centroids reached the far side (the jam is broken)
  on_lane    — neither centroid drifted more than `lane_tol` off the x-axis
               (too strong a bias slips them past but flings them off their lane)
  cohesive   — each flock is still a single connected component at the end
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# reuse the exact Olfati-Saber math helpers from the single-flock sim
_SPEC = importlib.util.spec_from_file_location(
    "_flocking", str(Path(__file__).resolve().parent / "_flocking.py"))
_F = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("_flocking", _F)
_SPEC.loader.exec_module(_F)
_sigma_norm = _F._sigma_norm
_sigma_norm_scalar = _F._sigma_norm_scalar
_sigma1 = _F._sigma1
_bump = _F._bump
_phi = _F._phi
_components = _F._components
_EPS = _F.EPS


@dataclass
class CrossResult:
    passed: bool                      # both flock centroids reached the far side within the budget
    on_lane: bool                     # neither centroid drifted > lane_tol off axis
    cohesive: bool                    # each flock still one connected component
    max_lateral: float                # largest |centroid_y| over the run
    min_pair: float                   # smallest inter-agent distance over the run (collision proxy)
    pass_step: int                    # first step both centroids cleared pass_x (-1 if never)
    traj: list = field(default_factory=list)
    grp: np.ndarray = None

    @property
    def on_time(self) -> bool:        # cleared the crossing within the time budget AND kept its lane
        return self.passed and self.on_lane


def simulate_crossing(
    *,
    n: int = 24,
    d: float = 7.0,
    ratio: float = 1.2,
    c1a: float = 3.0,
    c2a: float = 8.0,
    c1g: float = 1.0,
    c2g: float = 0.6,
    bias: float = 0.0,            # right-of-way: constant veer right of each agent's goal heading
    sep: float = 40.0,            # initial half-separation of the two flocks along x
    spread: float = 10.0,         # initial spread of each flock
    gv: float = 5.0,              # goal speed (group 0 +x, group 1 -x)
    pass_x: float = 30.0,         # centroid must cross beyond this to count as "passed"
    lane_tol: float = 30.0,       # centroid must stay within this of the x-axis to be "on lane"
    dt: float = 0.02,
    steps: int = 1400,
    vmax: float = 12.0,
    seed: int = 0,
    record: bool = False,
) -> CrossResult:
    rng = np.random.default_rng(seed)
    r = ratio * d
    r_a = _sigma_norm_scalar(r)
    d_a = _sigma_norm_scalar(d)
    half = n // 2
    qA = rng.uniform(-spread, spread, size=(half, 2)) + np.array([-sep, 0.0])
    qB = rng.uniform(-spread, spread, size=(n - half, 2)) + np.array([sep, 0.0])
    q = np.vstack([qA, qB])
    p = np.zeros((n, 2))
    grp = np.array([0] * half + [1] * (n - half))
    gA = np.array([-sep, 0.0]); gB = np.array([sep, 0.0])
    vA = np.array([gv, 0.0]); vB = np.array([-gv, 0.0])
    eye = np.eye(n, dtype=bool)

    traj = []
    max_lat = 0.0
    min_pair = np.inf
    pass_step = -1
    for t in range(steps):
        diff = q[None, :, :] - q[:, None, :]
        z2 = (diff * diff).sum(-1)
        z_sig = _sigma_norm(z2)
        n_ij = diff / np.sqrt(1.0 + _EPS * z2)[..., None]
        a_ij = _bump(z_sig / r_a)
        a_ij[eye] = 0.0
        phi_a = a_ij * _phi(z_sig - d_a)
        grad = (phi_a[..., None] * n_ij).sum(axis=1)
        consensus = (a_ij[..., None] * (p[None, :, :] - p[:, None, :])).sum(axis=1)
        u = c1a * grad + c2a * consensus

        q_g = np.where(grp[:, None] == 0, gA, gB)
        p_g = np.where(grp[:, None] == 0, vA, vB)
        u += -c1g * _sigma1(q - q_g) - c2g * (p - p_g)

        if bias != 0.0:
            g_dir = p_g / np.linalg.norm(p_g, axis=1, keepdims=True)
            perp = np.stack([g_dir[:, 1], -g_dir[:, 0]], axis=1)   # to the RIGHT of heading
            u += bias * perp

        gA = gA + vA * dt; gB = gB + vB * dt
        p = p + u * dt
        sp = np.sqrt((p * p).sum(-1, keepdims=True))
        over = (sp > vmax).ravel()
        if over.any():
            p[over] = p[over] / sp[over] * vmax
        q = q + p * dt

        cA_t = q[grp == 0].mean(0); cB_t = q[grp == 1].mean(0)
        max_lat = max(max_lat, abs(cA_t[1]), abs(cB_t[1]))
        # collision proxy = closest approach between agents of DIFFERENT flocks
        # (intra-flock spacing is the lattice; only inter-flock contact is a "collision")
        inter = np.sqrt(z2)[np.ix_(grp == 0, grp == 1)]
        min_pair = min(min_pair, float(inter.min()))
        if pass_step < 0 and cA_t[0] > pass_x and cB_t[0] < -pass_x:
            pass_step = t
        if record and t % 5 == 0:
            traj.append(q.copy())

    cA = q[grp == 0].mean(0); cB = q[grp == 1].mean(0)
    passed = bool(cA[0] > pass_x and cB[0] < -pass_x)
    on_lane = bool(max_lat <= lane_tol)
    kA, _ = _components(q[grp == 0], r)
    kB, _ = _components(q[grp == 1], r)
    cohesive = bool(kA == 1 and kB == 1)
    return CrossResult(passed=passed, on_lane=on_lane, cohesive=cohesive,
                       max_lateral=max_lat, min_pair=min_pair, pass_step=pass_step,
                       traj=traj, grp=grp)
