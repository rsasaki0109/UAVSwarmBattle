"""2D hub crossing with K moving threat tokens (non-agent obstacles).

Threats are point obstacles with velocity — encoded as ROLE_THREAT tokens in the
transformer policy but invisible to the ally-only deep-set baseline.

Teachers repel from both peers and threats; the convention teacher also applies the
right-of-way bias from scripts/_swarm_policy.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import _swarm_policy as sp
import _swarm_transformer as st

THREAT_R = 0.55
THREAT_VMAX = 1.0
WORLD = 14.0


@dataclass
class ThreatRoll:
    success: bool
    reason: str
    threat_hits: int
    peer_hits: int


def _threat_list(pos: np.ndarray, vel: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(pos[k], vel[k]) for k in range(len(pos))]


def make_threats(k: int, rng, *, speed: float = 0.9, lane_y: float = 0.0):
    """K threats drifting horizontally through the hub."""
    pos = np.zeros((k, 2))
    vel = np.zeros((k, 2))
    for i in range(k):
        pos[i, 0] = rng.uniform(-4.0, 4.0)
        pos[i, 1] = lane_y + rng.normal(0.0, 0.8)
        vel[i, 0] = speed * float(rng.choice([-1.0, 1.0]))
        vel[i, 1] = rng.normal(0.0, 0.15)
    return pos, vel


def hub_scene(n: int, k: int, rng, *, ring: float = 6.5, jitter: float = 0.25):
    start, goal = sp.antipodal(n, rng, ring=ring, jitter=jitter)
    tpos, tvel = make_threats(k, rng)
    return start, goal, tpos, tvel


def advance_threats(pos: np.ndarray, vel: np.ndarray, dt: float = sp.DT):
    pos = pos + vel * dt
    for i in range(len(pos)):
        for ax in (0, 1):
            if pos[i, ax] < -WORLD:
                pos[i, ax] = -WORLD
                vel[i, ax] = abs(vel[i, ax])
            elif pos[i, ax] > WORLD:
                pos[i, ax] = WORLD
                vel[i, ax] = -abs(vel[i, ax])
    return pos, vel


def _threat_repulsion(p_i: np.ndarray, threats: list[tuple[np.ndarray, np.ndarray]], w=1.4):
    acc = np.zeros(2)
    for tp, _tv in threats:
        d = p_i - tp
        dist = np.linalg.norm(d)
        if dist < sp.PEER_R and dist > 1e-6:
            acc += (d / dist) * (sp.PEER_R - dist) / sp.PEER_R * w
    return acc


def teacher_plain(p, vel, goals, i, threats):
    to_goal = goals[i] - p[i]
    dg = np.linalg.norm(to_goal)
    v = (to_goal / dg) * sp.VMAX if dg > 1e-6 else np.zeros(2)
    v = v + sp._peer_repulsion(p, vel, i) + _threat_repulsion(p[i], threats)
    return sp._clamp(v)


def teacher_conv(p, vel, goals, i, threats, bias=0.9):
    to_goal = goals[i] - p[i]
    dg = np.linalg.norm(to_goal)
    v = (to_goal / dg) * sp.VMAX if dg > 1e-6 else np.zeros(2)
    v = v + sp._peer_repulsion(p, vel, i) + _threat_repulsion(p[i], threats)
    h = to_goal / dg if dg > 1e-6 else np.array([1.0, 0.0])
    right = np.array([h[1], -h[0]])
    near = 0.0
    for j in range(len(p)):
        if j == i:
            continue
        d = p[j] - p[i]
        dist = np.linalg.norm(d)
        if dist < sp.PEER_R and dist > 1e-6 and np.dot(d, h) > 0:
            near = max(near, (sp.PEER_R - dist) / sp.PEER_R)
    v = v + right * bias * near
    return sp._clamp(v)


def _peer_collision(p, done):
    n = len(p)
    for i in range(n):
        for j in range(i + 1, n):
            if not (done[i] and done[j]) and np.linalg.norm(p[i] - p[j]) < 2 * sp.RADIUS:
                return True
    return False


def _threat_collision(p, done, tpos):
    for i in range(len(p)):
        if done[i]:
            continue
        for k in range(len(tpos)):
            if np.linalg.norm(p[i] - tpos[k]) < sp.RADIUS + THREAT_R:
                return True
    return False


def rollout(start, goal, tpos0, tvel0, controller, *, max_steps=350):
    """controller(p, vel, goals, i, threats) -> velocity."""
    p = start.copy()
    vel = np.zeros_like(p)
    tpos = tpos0.copy()
    tvel = tvel0.copy()
    n = len(p)
    done = np.zeros(n, dtype=bool)
    threat_hits = 0
    peer_hits = 0
    for _ in range(max_steps):
        threats = _threat_list(tpos, tvel)
        newv = np.zeros_like(p)
        for i in range(n):
            if done[i]:
                continue
            newv[i] = controller(p, vel, goal, i, threats)
        p = p + newv * sp.DT
        vel = newv
        tpos, tvel = advance_threats(tpos, tvel)
        for i in range(n):
            if np.linalg.norm(goal[i] - p[i]) < sp.GOAL_TOL:
                done[i] = True
        if _threat_collision(p, done, tpos):
            return ThreatRoll(False, "threat", threat_hits + 1, peer_hits)
        if _peer_collision(p, done):
            return ThreatRoll(False, "peer", threat_hits, peer_hits + 1)
        if done.all():
            return ThreatRoll(True, "goal", threat_hits, peer_hits)
    return ThreatRoll(False, "timeout", threat_hits, peer_hits)


def _ds_feat(p, vel, goals, i, reflect=False):
    ego, pe, R = sp._ego_frame(p, vel, goals, i, reflect)
    pad = np.zeros((sp.MAX_PEERS, 5))
    m = np.zeros(sp.MAX_PEERS)
    k = min(len(pe), sp.MAX_PEERS)
    if k:
        pad[:k] = pe[:k]
        m[:k] = 1.0
    return ego, pad, m, R


def make_ds_dataset(teacher, *, n_list, k_list, n_scenes, seed0):
    """Ally-only 5D features; teacher sees threats during rollouts."""
    egos, peerl, masks, acts = [], [], [], []
    sc = 0
    s = seed0
    while sc < n_scenes:
        rng = np.random.default_rng(s)
        s += 1
        n = int(rng.choice(n_list))
        k = int(rng.choice(k_list))
        st0, gl, tpos, tvel = hub_scene(n, k, rng)
        p = st0.copy()
        vel = np.zeros_like(p)
        done = np.zeros(n, bool)
        for _ in range(220):
            threats = _threat_list(tpos, tvel)
            newv = np.zeros_like(p)
            for i in range(n):
                if done[i]:
                    continue
                a = teacher(p, vel, gl, i, threats)
                ego, pad, m, R = _ds_feat(p, vel, gl, i)
                egos.append(ego)
                peerl.append(pad)
                masks.append(m)
                acts.append(R @ a)
                newv[i] = a
            p = p + newv * sp.DT
            vel = newv
            tpos, tvel = advance_threats(tpos, tvel)
            for i in range(n):
                if np.linalg.norm(gl[i] - p[i]) < sp.GOAL_TOL:
                    done[i] = True
            if done.all():
                break
        sc += 1
    return np.array(egos), np.array(peerl), np.array(masks), np.array(acts)


def make_xf_dataset(teacher, *, n_list, k_list, n_scenes, seed0):
    """Ally + threat 6D tokens."""
    egos, peerl, masks, acts = [], [], [], []
    sc = 0
    s = seed0
    while sc < n_scenes:
        rng = np.random.default_rng(s)
        s += 1
        n = int(rng.choice(n_list))
        k = int(rng.choice(k_list))
        st0, gl, tpos, tvel = hub_scene(n, k, rng)
        p = st0.copy()
        vel = np.zeros_like(p)
        done = np.zeros(n, bool)
        for _ in range(220):
            threats = _threat_list(tpos, tvel)
            newv = np.zeros_like(p)
            for i in range(n):
                if done[i]:
                    continue
                a = teacher(p, vel, gl, i, threats)
                ego, pad, m, R = st.featurize(p, vel, gl, i, threats=threats)
                egos.append(ego)
                peerl.append(pad)
                masks.append(m)
                acts.append(R @ a)
                newv[i] = a
            p = p + newv * sp.DT
            vel = newv
            tpos, tvel = advance_threats(tpos, tvel)
            for i in range(n):
                if np.linalg.norm(gl[i] - p[i]) < sp.GOAL_TOL:
                    done[i] = True
            if done.all():
                break
        sc += 1
    return np.array(egos), np.array(peerl), np.array(masks), np.array(acts)


def make_ds_controller(P, stats):
    em, es, pm, ps = stats["em"], stats["es"], stats["pm"], stats["ps"]

    def ctrl(p, vel, goals, i, threats):
        del threats
        ego, pad, m, R = _ds_feat(p, vel, goals, i)
        e = ((ego - em) / es)[None, :]
        pn = np.where(m[None, :, None] > 0, (pad[None] - pm) / ps, 0.0)
        a_ego = sp.forward(P, e, pn, m[None, :])[0]
        return sp._clamp(R.T @ a_ego)

    return ctrl


def make_xf_controller(P, stats):
    em, es, pm, ps = stats["em"], stats["es"], stats["pm"], stats["ps"]

    def ctrl(p, vel, goals, i, threats):
        ego, pad, m, R = st.featurize(p, vel, goals, i, threats=threats)
        e = ((ego - em) / es)[None, :]
        pn = np.where(m[None, :, None] > 0, (pad[None] - pm) / ps, 0.0)
        a_ego = st.forward(P, e, pn, m[None, :])[0]
        return sp._clamp(R.T @ a_ego)

    return ctrl
