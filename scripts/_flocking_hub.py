"""K Olfati-Saber flocks converging on a shared hub — a multi-way roundabout.

Generalises the [two-flock crossing](_flocking_crossing.py) to K flocks placed
evenly on a circle, each migrating through the centre to the antipodal point. For
K=2 this is the head-on crossing; for K>2 it is a K-way intersection. Without a
convention the flocks **jam** at the hub (the universal α-repulsion makes a mutual
wall), exactly as in the N-drone antipodal hub of the convention thread. The
right-of-way veer (each agent biases right of its goal heading) turns the fan-in
into a **roundabout**.

The question this isolates is what *crowding the hub* costs. For point-agent fleets
the cost of a denser hub is collisions (the convention's density cliff). For
cohesive flocks the roundabout self-spaces over a wide range, so the FIRST casualty
is not safety but the **flocks' own cohesion**: the roundabout shears each group as
it threads the hub. Only at the densest hubs (very high fan-in) does the safety
margin finally fail too — a collision cliff, pushed far out past where cohesion erodes.

`simulate_hub` returns:
  n_passed    — how many of the K flocks cleared the hub (all_passed = n_passed==K)
  cohesion    — mean over flocks of the largest-connected-fraction at the end
  min_inter   — closest approach between agents of *different* flocks (collision proxy)
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

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
class HubResult:
    n_passed: int                     # flocks that cleared the hub
    all_passed: bool                  # n_passed == K
    cohesion: float                   # mean largest-flock fraction at the end
    min_inter: float                  # closest inter-flock approach (collision proxy)
    K: int = 0
    traj: list = field(default_factory=list)
    grp: np.ndarray = None


def simulate_hub(
    *,
    K: int = 4,
    per_flock: int = 10,
    d: float = 7.0,
    ratio: float = 1.2,
    c1a: float = 3.0,
    c2a: float = 8.0,
    c1g: float = 1.0,
    c2g: float = 0.6,
    bias: float = 0.0,
    adopt: np.ndarray = None,     # per-agent bool mask of who applies the bias (None = all)
    sep: float = 40.0,            # radius of the ring the flocks start on
    spread: float = 8.0,
    gv: float = 5.0,
    pass_frac: float = 0.6,       # a flock "passed" once it has travelled pass_frac*sep past the hub
    coh_radius: float = 1.6,      # cohesion graph radius = coh_radius * r
    dt: float = 0.02,
    steps: int = 1600,
    vmax: float = 12.0,
    seed: int = 0,
    record: bool = False,
) -> HubResult:
    rng = np.random.default_rng(seed)
    r = ratio * d
    r_a = _sigma_norm_scalar(r)
    d_a = _sigma_norm_scalar(d)
    n = K * per_flock
    q = np.zeros((n, 2))
    grp = np.zeros(n, dtype=int)
    u_dir = np.zeros((K, 2))
    start = np.zeros((K, 2))
    for k in range(K):
        ang = 2.0 * np.pi * k / K
        pos = sep * np.array([np.cos(ang), np.sin(ang)])
        idx = slice(k * per_flock, (k + 1) * per_flock)
        q[idx] = rng.uniform(-spread, spread, size=(per_flock, 2)) + pos
        grp[idx] = k
        u_dir[k] = -pos / np.linalg.norm(pos)          # head through the centre
        start[k] = pos
    p = np.zeros((n, 2))
    g = start.copy()
    v = gv * u_dir
    eye = np.eye(n, dtype=bool)

    traj = []
    min_inter = np.inf
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

        q_g = g[grp]; p_g = v[grp]
        u += -c1g * _sigma1(q - q_g) - c2g * (p - p_g)

        if bias != 0.0:
            g_dir = p_g / np.linalg.norm(p_g, axis=1, keepdims=True)
            perp = np.stack([g_dir[:, 1], -g_dir[:, 0]], axis=1)
            b_vec = bias * perp
            if adopt is not None:
                b_vec = b_vec * adopt[:, None]
            u += b_vec

        g = g + v * dt
        p = p + u * dt
        sp = np.sqrt((p * p).sum(-1, keepdims=True))
        over = (sp > vmax).ravel()
        if over.any():
            p[over] = p[over] / sp[over] * vmax
        q = q + p * dt

        dmat = np.sqrt(z2)
        diff_flock = grp[:, None] != grp[None, :]
        if diff_flock.any():
            min_inter = min(min_inter, float(dmat[diff_flock].min()))
        if record and t % 5 == 0:
            traj.append(q.copy())

    n_passed = 0
    fracs = []
    for k in range(K):
        qk = q[grp == k]
        if (qk.mean(0) @ u_dir[k]) > pass_frac * sep:
            n_passed += 1
        fracs.append(_components(qk, coh_radius * r)[1])
    return HubResult(n_passed=n_passed, all_passed=(n_passed == K),
                     cohesion=float(np.mean(fracs)), min_inter=min_inter,
                     K=K, traj=traj, grp=grp)
