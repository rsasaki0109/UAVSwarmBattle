"""Self-contained 2D multi-agent sim + teacher controllers + a NumPy teammate-token
(deep-set) policy with behavioral cloning, for the neta-A probe:

  Does a TeamHOI-style permutation-equivariant teammate-token policy reimport the
  antipodal deadlock — or does it just transport whatever symmetry-breaking its
  teacher had?

The lab's recurring result: on the symmetric antipodal hub, a symmetric reactive
avoider deadlocks (every agent mirror-swerves into the same point); the cure is an
explicit right-of-way CONVENTION that breaks the left/right symmetry. TeamHOI claims
a single decentralized teammate-token policy scales to any team size. This module
distills two teachers into the SAME deep-set architecture and asks which property —
the architecture or the teacher's convention — controls the deadlock.

Pure NumPy (the lab is numpy-only): the policy is a deep set — per-peer encoder,
permutation-invariant mean-pool over teammate tokens, ego encoder, readout — the
essence of teammate-token attention. Ego-goal frame (a rotation, so chirality /
"right" is preserved and CAN be represented).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from dataclasses import dataclass

import numpy as np

VMAX = 1.0
DT = 0.1
RADIUS = 0.5
PEER_R = 4.0          # interaction radius
GOAL_TOL = 0.6


# --------------------------------------------------------------------------- #
# Scenario generation
# --------------------------------------------------------------------------- #
def antipodal(n: int, rng, ring: float = 8.0, jitter: float = 0.4):
    """N agents on a ring, each crossing to its antipode through the centre."""
    ang = 2.0 * np.pi * np.arange(n) / n
    start = ring * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    goal = -start.copy()
    start = start + rng.normal(0, jitter, start.shape)
    return start, goal


def random_scene(n: int, rng, box: float = 8.0):
    """Random starts and goals (the bread-and-butter avoidance training data)."""
    start = rng.uniform(-box, box, (n, 2))
    goal = rng.uniform(-box, box, (n, 2))
    return start, goal


# --------------------------------------------------------------------------- #
# Teacher controllers (closed-form). plain = symmetric (deadlocks on antipodal);
# conv = plain + right-of-way convention (breaks the symmetry).
# --------------------------------------------------------------------------- #
def _peer_repulsion(p, vel, i, w=1.2):
    acc = np.zeros(2)
    for j in range(len(p)):
        if j == i:
            continue
        d = p[i] - p[j]
        dist = np.linalg.norm(d)
        if dist < PEER_R and dist > 1e-6:
            acc += (d / dist) * (PEER_R - dist) / PEER_R * w
    return acc


def teacher_plain(p, vel, goals, i):
    to_goal = goals[i] - p[i]
    dg = np.linalg.norm(to_goal)
    v = (to_goal / dg) * VMAX if dg > 1e-6 else np.zeros(2)
    v = v + _peer_repulsion(p, vel, i)
    return _clamp(v)


def teacher_conv(p, vel, goals, i, bias=0.9):
    """plain + a right-of-way bias: when a peer is near & ahead, steer to the RIGHT
    of the current goal heading (a global handedness that breaks the symmetry)."""
    to_goal = goals[i] - p[i]
    dg = np.linalg.norm(to_goal)
    v = (to_goal / dg) * VMAX if dg > 1e-6 else np.zeros(2)
    v = v + _peer_repulsion(p, vel, i)
    # right-of-way: rotate the goal heading clockwise (to the agent's right),
    # weighted by how close the nearest conflicting peer is
    h = to_goal / dg if dg > 1e-6 else np.array([1.0, 0.0])
    right = np.array([h[1], -h[0]])  # clockwise normal = "right" of heading
    near = 0.0
    for j in range(len(p)):
        if j == i:
            continue
        d = p[j] - p[i]
        dist = np.linalg.norm(d)
        if dist < PEER_R and dist > 1e-6 and np.dot(d, h) > 0:  # ahead
            near = max(near, (PEER_R - dist) / PEER_R)
    v = v + right * bias * near
    return _clamp(v)


def _clamp(v):
    s = np.linalg.norm(v)
    return v / s * VMAX if s > VMAX else v


# --------------------------------------------------------------------------- #
# Sim rollout
# --------------------------------------------------------------------------- #
@dataclass
class Roll:
    success: bool
    reason: str
    traj: list  # (T, N, 2) if recorded


def rollout(start, goal, controller, *, max_steps=300, record=False):
    """controller(p, vel, goals, i) -> velocity for agent i. Returns success."""
    p = start.copy()
    vel = np.zeros_like(p)
    n = len(p)
    done = np.zeros(n, dtype=bool)
    traj = []
    for _ in range(max_steps):
        if record:
            traj.append(p.copy())
        newv = np.zeros_like(p)
        for i in range(n):
            if done[i]:
                continue
            newv[i] = controller(p, vel, goal, i)
        p = p + newv * DT
        vel = newv
        for i in range(n):
            if np.linalg.norm(goal[i] - p[i]) < GOAL_TOL:
                done[i] = True
        # collision
        for i in range(n):
            for j in range(i + 1, n):
                if not (done[i] and done[j]) and np.linalg.norm(p[i] - p[j]) < 2 * RADIUS:
                    return Roll(False, "collision", traj)
        if done.all():
            return Roll(True, "goal", traj)
    return Roll(False, "timeout", traj)


# --------------------------------------------------------------------------- #
# Ego-goal-frame featurization (a rotation -> rotation-equivariant but chirality
# preserved: "right" is a fixed direction in this frame, so a convention CAN be
# represented). The policy is a deep set over teammate tokens.
# --------------------------------------------------------------------------- #
MAX_PEERS = 8


def _ego_frame(p, vel, goal, i):
    to_goal = goal[i] - p[i]
    dg = np.linalg.norm(to_goal)
    h = to_goal / dg if dg > 1e-6 else np.array([1.0, 0.0])
    R = np.array([[h[0], h[1]], [-h[1], h[0]]])  # world -> ego (goal along +x)
    ego = np.array([min(dg, 10.0) / 10.0, *(R @ vel[i])])  # [dist, vx, vy]
    peers = []
    for j in range(len(p)):
        if j == i:
            continue
        d = p[j] - p[i]
        dist = np.linalg.norm(d)
        if dist < PEER_R and dist > 1e-6:
            de = R @ d
            ve = R @ (vel[j] - vel[i])
            peers.append([de[0], de[1], dist / PEER_R, ve[0], ve[1]])
    return ego, np.array(peers, dtype=float).reshape(-1, 5), R


def make_dataset(teacher, *, n_list, n_scenes, seed0, antipodal_frac=0.0):
    """Roll the teacher on random (and optionally antipodal) scenes, recording
    (ego, peers, teacher-action-in-ego-frame) at every step."""
    egos, peerl, masks, acts = [], [], [], []
    sc = 0
    s = seed0
    while sc < n_scenes:
        rng = np.random.default_rng(s); s += 1
        n = int(rng.choice(n_list))
        if rng.random() < antipodal_frac:
            st, gl = antipodal(n, rng)
        else:
            st, gl = random_scene(n, rng)
        p = st.copy(); vel = np.zeros_like(p); done = np.zeros(n, bool)
        for _ in range(200):
            newv = np.zeros_like(p)
            for i in range(n):
                if done[i]:
                    continue
                a = teacher(p, vel, gl, i)
                ego, pe, R = _ego_frame(p, vel, gl, i)
                pad = np.zeros((MAX_PEERS, 5)); m = np.zeros(MAX_PEERS)
                k = min(len(pe), MAX_PEERS)
                if k:
                    pad[:k] = pe[:k]; m[:k] = 1.0
                egos.append(ego); peerl.append(pad); masks.append(m)
                acts.append(R @ a)  # teacher action in ego frame
                newv[i] = a
            p = p + newv * DT; vel = newv
            for i in range(n):
                if np.linalg.norm(gl[i] - p[i]) < GOAL_TOL:
                    done[i] = True
            if done.all():
                break
        sc += 1
    return (np.array(egos), np.array(peerl), np.array(masks), np.array(acts))


# --------------------------------------------------------------------------- #
# Deep-set MLP (NumPy, manual backprop + Adam). phi: per-peer encoder; mean-pool
# over teammate tokens (permutation invariant); ego encoder; readout.
# --------------------------------------------------------------------------- #
def init_model(h=32, seed=0):
    rng = np.random.default_rng(seed)
    def W(a, b):
        return rng.normal(0, np.sqrt(2.0 / a), (a, b))
    return {
        "phi1": W(5, h), "phi1b": np.zeros(h),
        "phi2": W(h, h), "phi2b": np.zeros(h),
        "ego1": W(3, h), "ego1b": np.zeros(h),
        "out1": W(2 * h, h), "out1b": np.zeros(h),
        "out2": W(h, 2), "out2b": np.zeros(2),
    }


def forward(P, ego, peers, mask, cache=None):
    B, M, _ = peers.shape
    flat = peers.reshape(B * M, 5)
    z1 = flat @ P["phi1"] + P["phi1b"]; a1 = np.tanh(z1)
    z2 = a1 @ P["phi2"] + P["phi2b"]; a2 = np.tanh(z2)
    a2 = a2.reshape(B, M, -1)
    msum = mask.sum(1, keepdims=True); msum = np.maximum(msum, 1.0)
    pooled = (a2 * mask[:, :, None]).sum(1) / msum          # (B,h)
    ze = ego @ P["ego1"] + P["ego1b"]; ae = np.tanh(ze)     # (B,h)
    cat = np.concatenate([ae, pooled], axis=1)              # (B,2h)
    zo = cat @ P["out1"] + P["out1b"]; ao = np.tanh(zo)
    out = ao @ P["out2"] + P["out2b"]                       # (B,2)
    if cache is not None:
        cache.update(dict(flat=flat, a1=a1, z1=z1, a2r=a2, z2=z2, mask=mask,
                          msum=msum, pooled=pooled, ae=ae, ze=ze, ego=ego,
                          cat=cat, ao=ao, zo=zo, B=B, M=M))
    return out


def backward(P, cache, dout):
    g = {}
    ao = cache["ao"]; cat = cache["cat"]
    g["out2"] = ao.T @ dout; g["out2b"] = dout.sum(0)
    dao = dout @ P["out2"].T
    dzo = dao * (1 - ao ** 2)
    g["out1"] = cat.T @ dzo; g["out1b"] = dzo.sum(0)
    dcat = dzo @ P["out1"].T
    h = P["ego1"].shape[1]
    dae = dcat[:, :h]; dpooled = dcat[:, h:]
    # ego branch
    dze = dae * (1 - cache["ae"] ** 2)
    g["ego1"] = cache["ego"].T @ dze; g["ego1b"] = dze.sum(0)
    # peer branch (pool -> per-peer)
    B, M = cache["B"], cache["M"]
    da2 = (dpooled / cache["msum"])[:, None, :] * cache["mask"][:, :, None]  # (B,M,h)
    da2 = da2.reshape(B * M, -1)
    dz2 = da2 * (1 - cache["a2r"].reshape(B * M, -1) ** 2)
    g["phi2"] = cache["a1"].T @ dz2; g["phi2b"] = dz2.sum(0)
    da1 = dz2 @ P["phi2"].T
    dz1 = da1 * (1 - cache["a1"] ** 2)
    g["phi1"] = cache["flat"].T @ dz1; g["phi1b"] = dz1.sum(0)
    return g


def train_bc(data, *, h=32, epochs=300, batch=256, lr=3e-3, seed=0, verbose=False):
    egos, peers, masks, acts = data
    N = len(egos)
    # standardize inputs
    em, es = egos.mean(0), egos.std(0) + 1e-6
    pm, ps = peers.reshape(-1, 5).mean(0), peers.reshape(-1, 5).std(0) + 1e-6
    egos = (egos - em) / es
    peers = np.where(masks[:, :, None] > 0, (peers - pm) / ps, 0.0)
    P = init_model(h, seed)
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(v) for k, v in P.items()}
    rng = np.random.default_rng(seed); t = 0
    for ep in range(epochs):
        idx = rng.permutation(N)
        for b in range(0, N, batch):
            bi = idx[b:b + batch]
            cache = {}
            pred = forward(P, egos[bi], peers[bi], masks[bi], cache)
            err = pred - acts[bi]
            dout = 2.0 * err / len(bi)
            g = backward(P, cache, dout)
            t += 1
            for k in P:
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                mh = m[k] / (1 - 0.9 ** t); vh = v[k] / (1 - 0.999 ** t)
                P[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        if verbose and ep % 50 == 0:
            pr = forward(P, egos, peers, masks)
            print(f"  ep{ep:4} mse={np.mean((pr - acts) ** 2):.4f}")
    stats = dict(em=em, es=es, pm=pm, ps=ps)
    return P, stats


def make_student_controller(P, stats):
    """Wrap the trained deep set as a controller(p, vel, goals, i)."""
    em, es, pm, ps = stats["em"], stats["es"], stats["pm"], stats["ps"]

    def ctrl(p, vel, goals, i):
        ego, pe, R = _ego_frame(p, vel, goals, i)
        pad = np.zeros((1, MAX_PEERS, 5)); mask = np.zeros((1, MAX_PEERS))
        k = min(len(pe), MAX_PEERS)
        if k:
            pad[0, :k] = pe[:k]; mask[0, :k] = 1.0
        e = ((ego - em) / es)[None, :]
        pn = np.where(mask[:, :, None] > 0, (pad - pm) / ps, 0.0)
        a_ego = forward(P, e, pn, mask)[0]
        return _clamp(R.T @ a_ego)  # back to world frame

    return ctrl


if __name__ == "__main__":
    print("sanity: teacher on antipodal (plain should deadlock, conv should clear)")
    for name, ctrl in (("plain", teacher_plain), ("conv", teacher_conv)):
        for n in (4, 6, 8):
            ok = 0
            for s in range(30):
                rng = np.random.default_rng(s)
                st, gl = antipodal(n, rng)
                ok += rollout(st, gl, ctrl).success
            print(f"  teacher_{name:5} N={n}  {ok}/30")
    print("\nBC distill (train on RANDOM scenes only, test antipodal):", flush=True)
    for name, teacher in (("plain", teacher_plain), ("conv", teacher_conv)):
        data = make_dataset(teacher, n_list=[3, 4, 5, 6], n_scenes=150, seed0=0)
        print(f"  [{name}] dataset {len(data[0])} samples; training...", flush=True)
        P, stats = train_bc(data, epochs=150, verbose=False)
        pr = forward(P, *(np.asarray(x) for x in (
            (data[0] - stats["em"]) / stats["es"],
            np.where(data[2][:, :, None] > 0, (data[1] - stats["pm"]) / stats["ps"], 0.0),
            data[2])))
        mse = np.mean((pr - data[3]) ** 2)
        sc = make_student_controller(P, stats)
        line = []
        for n in (4, 6, 8):
            ok = sum(rollout(*antipodal(n, np.random.default_rng(s)), sc).success
                     for s in range(30))
            line.append(f"N={n}:{ok}/30")
        print(f"  student<-{name:5} (bc_mse={mse:.4f})  " + "  ".join(line))
