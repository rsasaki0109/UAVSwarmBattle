"""Self-contained Olfati-Saber flocking sim (2D, double-integrator agents).

Reproduces the two flocking algorithms of

    R. Olfati-Saber, "Flocking for Multi-Agent Dynamic Systems: Algorithms and
    Theory", IEEE TAC 51(3), 2006.

Each agent is a double integrator ``q̇ = p, ṗ = u``.  The control law has up to
three parts (all smoothed with the σ-norm so the gradient is well-defined):

  gradient   Σ_j φ_α(‖q_j-q_i‖_σ) · n_ij      collective potential
             (repels closer than the desired distance ``d``, attracts beyond it
             up to the interaction range ``r``; ZERO past ``r`` — finite support)
  consensus  Σ_j a_ij(q) · (p_j - p_i)         velocity alignment
  navigation -c1γ·σ1(q_i-q_γ) - c2γ·(p_i-p_γ)  pull toward a shared objective γ

ALGORITHM 1 (free flocking) = gradient + consensus only.  Olfati-Saber proves
this FRAGMENTS for generic initial states: the group splits into several flocks.

ALGORITHM 2 = Algorithm 1 + the navigational feedback term (a γ-agent / shared
goal), which the paper proves keeps the whole group a single cohesive flock.

The ``grad_gain`` knob multiplies the gradient (cohesion+separation) term so a
caller can ask the counterintuitive question: can MORE cohesion substitute for
the structural navigational term?  (It can't: the gradient has finite support —
an agent past range ``r`` of every neighbour feels zero force at any gain.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 0.1          # σ-norm parameter
H_BUMP = 0.2       # bump-function transition point
A_PHI = 5.0        # action-function shape (a = b ⇒ symmetric attract/repel, c = 0)
B_PHI = 5.0


def _sigma_norm(z2: np.ndarray) -> np.ndarray:
    """σ-norm given squared Euclidean norms ``z2``: (√(1+ε·z²)−1)/ε."""
    return (np.sqrt(1.0 + EPS * z2) - 1.0) / EPS


def _sigma_norm_scalar(x: float) -> float:
    return (np.sqrt(1.0 + EPS * x * x) - 1.0) / EPS


def _sigma1(z: np.ndarray) -> np.ndarray:
    return z / np.sqrt(1.0 + z * z)


def _bump(z: np.ndarray) -> np.ndarray:
    """ρ_h: 1 on [0,h), cosine taper on [h,1], 0 past 1."""
    out = np.zeros_like(z)
    out[z < H_BUMP] = 1.0
    mid = (z >= H_BUMP) & (z <= 1.0)
    out[mid] = 0.5 * (1.0 + np.cos(np.pi * (z[mid] - H_BUMP) / (1.0 - H_BUMP)))
    return out


def _phi(z: np.ndarray) -> np.ndarray:
    c = abs(A_PHI - B_PHI) / np.sqrt(4.0 * A_PHI * B_PHI)  # = 0 when a == b
    return 0.5 * ((A_PHI + B_PHI) * _sigma1(z + c) + (A_PHI - B_PHI))


@dataclass
class FlockResult:
    n_components: int                 # connected components of the final r-graph
    largest_frac: float               # fraction of agents in the biggest cluster
    connected: bool                   # n_components == 1
    reached: bool                     # group centroid within goal_tol of γ (algo 2)
    # per-step replay for rendering (only when record=True)
    traj: list = field(default_factory=list)          # (T, n, 2)


def _components(q: np.ndarray, radius: float) -> tuple[int, float]:
    """Connected components of the graph with an edge when ‖q_i-q_j‖ < radius."""
    n = len(q)
    diff = q[:, None, :] - q[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))
    adj = (dist < radius) & ~np.eye(n, dtype=bool)
    seen = np.zeros(n, dtype=bool)
    sizes = []
    for s in range(n):
        if seen[s]:
            continue
        stack, size = [s], 0
        seen[s] = True
        while stack:
            u = stack.pop()
            size += 1
            for v in np.nonzero(adj[u] & ~seen)[0]:
                seen[v] = True
                stack.append(v)
        sizes.append(size)
    return len(sizes), max(sizes) / n


def simulate(
    *,
    algorithm: int = 1,
    n: int = 20,
    d: float = 7.0,
    ratio: float = 1.2,          # interaction range r = ratio * d
    grad_gain: float = 1.0,      # the cohesion+separation knob
    c1a: float = 3.0,            # base gradient gain
    c2a: float = 2.0,            # consensus gain
    c1g: float = 0.6,            # navigational position gain (algo 2)
    c2g: float = 0.4,            # navigational velocity gain (algo 2)
    spread: float = 26.0,        # initial position spread (half-width of the box)
    v_spread: float = 0.0,       # initial velocity spread
    goal: tuple[float, float] = (0.0, 0.0),
    goal_vel: tuple[float, float] = (0.0, 0.0),
    goal_moves: bool = False,     # advance the γ-agent at goal_vel (a migrating flock)
    obstacles: tuple = (),        # disk β-agents: each (cx, cy, R) -> repulsion
    c_obs: float = 20.0,          # β-agent (obstacle) repulsion gain
    obs_infl: float = 7.0,        # influence margin beyond each disk's radius
    dt: float = 0.02,
    steps: int = 1500,
    vmax: float = 12.0,
    seed: int = 0,
    record: bool = False,
) -> FlockResult:
    rng = np.random.default_rng(seed)
    r = ratio * d
    r_a = _sigma_norm_scalar(r)
    d_a = _sigma_norm_scalar(d)
    q_g = np.asarray(goal, dtype=float)
    p_g = np.asarray(goal_vel, dtype=float)

    q = rng.uniform(-spread, spread, size=(n, 2))
    p = rng.uniform(-v_spread, v_spread, size=(n, 2)) if v_spread > 0 else np.zeros((n, 2))
    obs = np.asarray(obstacles, dtype=float).reshape(-1, 3) if len(obstacles) else None

    traj = []
    eye = np.eye(n, dtype=bool)
    for t in range(steps):
        diff = q[None, :, :] - q[:, None, :]          # diff[i,j] = q_j - q_i
        z2 = (diff * diff).sum(-1)                     # squared distance
        z_sig = _sigma_norm(z2)                        # σ-norm distance
        n_ij = diff / np.sqrt(1.0 + EPS * z2)[..., None]

        a_ij = _bump(z_sig / r_a)
        a_ij[eye] = 0.0
        phi_a = a_ij * _phi(z_sig - d_a)              # φ_α (already gated by bump)

        grad = (phi_a[..., None] * n_ij).sum(axis=1)
        consensus = (a_ij[..., None] * (p[None, :, :] - p[:, None, :])).sum(axis=1)

        u = grad_gain * c1a * grad + c2a * consensus
        if algorithm == 2:
            u += -c1g * _sigma1(q - q_g) - c2g * (p - p_g)
        if obs is not None:
            for cx, cy, R in obs:                         # β-agent disk repulsion (APF form)
                vec = q - np.array([cx, cy])
                dist = np.sqrt((vec * vec).sum(-1))
                gap = np.clip(dist - R, 0.05, None)       # surface gap
                hit = gap < obs_infl
                if hit.any():
                    mag = c_obs * (1.0 / gap - 1.0 / obs_infl) / (gap * gap)
                    u[hit] += (mag[hit, None] * vec[hit] / dist[hit, None])

        if goal_moves:
            q_g = q_g + p_g * dt
        p = p + u * dt
        sp = np.sqrt((p * p).sum(-1, keepdims=True))
        over = (sp > vmax).ravel()
        if over.any():
            p[over] = p[over] / sp[over] * vmax
        q = q + p * dt
        if record and t % 5 == 0:
            traj.append(q.copy())

    ncomp, frac = _components(q, r)
    reached = bool(np.linalg.norm(q.mean(0) - q_g) < d)
    return FlockResult(
        n_components=ncomp,
        largest_frac=frac,
        connected=(ncomp == 1),
        reached=reached,
        traj=traj,
    )
