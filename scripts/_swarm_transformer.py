"""TeamHOI-style teammate-token policy: ego cross-attention over variable peers.

Extends scripts/_swarm_policy.py (mean-pool deep set) with:
  - per-peer encoder -> token
  - ego query x peer keys/values (single-head cross-attention)
  - optional role channel (ally vs threat) on each token

Same NumPy-only manual backprop + BC/REINFORCE surface as the deep set so
swarm_bc_symmetry_phase.py and _swarm_rl.py comparisons stay apples-to-apples.

Peer geometry features match _swarm_policy._ego_frame; role is appended before
encoding (0 = ally UAV, 1 = threat / dynamic obstacle token).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

import _swarm_policy as sp
from uav_nav_lab.planner import swarm_transformer_core as core

MAX_PEERS = core.MAX_PEERS
PEER_DIM = core.PEER_DIM
ROLE_ALLY = core.ROLE_ALLY
ROLE_THREAT = core.ROLE_THREAT
VMAX = sp.VMAX
DT = sp.DT
init_model = core.init_model
forward = core.forward
save_checkpoint = core.save_checkpoint
load_checkpoint = core.load_checkpoint


def _pad_peers(pe: np.ndarray, *, role: float = ROLE_ALLY) -> tuple[np.ndarray, np.ndarray]:
    """(MAX_PEERS, PEER_DIM) pad + mask from raw (k, 5) ego-frame peer rows."""
    pad = np.zeros((MAX_PEERS, PEER_DIM))
    mask = np.zeros(MAX_PEERS)
    k = min(len(pe), MAX_PEERS)
    if k:
        pad[:k, :5] = pe[:k]
        pad[:k, 5] = role
        mask[:k] = 1.0
    return pad, mask


def featurize(p, vel, goals, i, *, reflect_canonical=False, threats=None):
    """Ego-goal frame features with ally peers + optional threat tokens."""
    ego, pe, R = sp._ego_frame(p, vel, goals, i, reflect_canonical)
    pads = []
    masks = []
    ap, am = _pad_peers(pe, role=ROLE_ALLY)
    pads.append(ap)
    masks.append(am)
    if threats is not None:
        for tp, tv in threats:
            d = tp - p[i]
            dist = np.linalg.norm(d)
            if dist < sp.PEER_R and dist > 1e-6:
                de = R @ d
                ve = R @ (tv - vel[i])
                row = np.array([de[0], de[1], dist / sp.PEER_R, ve[0], ve[1], ROLE_THREAT])
                tp_pad = np.zeros((MAX_PEERS, PEER_DIM))
                tm = np.zeros(MAX_PEERS)
                tp_pad[0] = row
                tm[0] = 1.0
                pads.append(tp_pad)
                masks.append(tm)
    if len(pads) == 1:
        return ego, pads[0], masks[0], R
    combined = np.zeros((MAX_PEERS, PEER_DIM))
    combined_mask = np.zeros(MAX_PEERS)
    idx = 0
    for pad, mask in zip(pads, masks):
        k = int(mask.sum())
        if idx + k > MAX_PEERS:
            k = MAX_PEERS - idx
        if k <= 0:
            continue
        slots = np.where(mask > 0)[0][:k]
        combined[idx:idx + k] = pad[slots[:k]]
        combined_mask[idx:idx + k] = 1.0
        idx += k
    return ego, combined, combined_mask, R


def backward(P, cache, dout):
    g = {}
    ao = cache["ao"]
    cat = cache["cat"]
    g["out2"] = ao.T @ dout
    g["out2b"] = dout.sum(0)
    dao = dout @ P["out2"].T
    dzo = dao * (1 - ao ** 2)
    g["out1"] = cat.T @ dzo
    g["out1b"] = dzo.sum(0)
    dcat = dzo @ P["out1"].T
    h = P["ego1"].shape[1]
    d_ego_emb = dcat[:, :h]
    d_context = dcat[:, h:]

    attn = cache["attn"]
    V = cache["V"]
    mask = cache["mask"]
    B, M = cache["B"], cache["M"]

    dV = attn[:, :, None] * d_context[:, None, :]
    d_attn = np.einsum("bh,bmh->bm", d_context, V)
    d_scores = attn * (
        d_attn - (d_attn * attn).sum(axis=1, keepdims=True)
    )
    d_scores = np.where(mask > 0, d_scores, 0.0)

    Q = cache["Q"]
    K = cache["K"]
    scale = cache["scale"]
    dQ = np.einsum("bm,bmh->bh", d_scores, K) / scale
    dK = d_scores[:, :, None] * Q[:, None, :] / scale

    g["Wq"] = cache["ego_emb"].T @ dQ
    g["Wk"] = cache["peer_emb"].reshape(B * M, h).T @ dK.reshape(B * M, h)
    g["Wv"] = cache["peer_emb"].reshape(B * M, h).T @ dV.reshape(B * M, h)

    d_peer_emb = dK @ P["Wk"].T + dV @ P["Wv"].T
    d_ego_emb = d_ego_emb + dQ @ P["Wq"].T

    dze = d_ego_emb * (1 - cache["ego_emb"] ** 2)
    g["ego1"] = cache["ego"].T @ dze
    g["ego1b"] = dze.sum(0)

    d_peer = d_peer_emb * (1 - cache["peer_emb"] ** 2)
    dz2 = d_peer.reshape(B * M, h)
    g["phi2"] = cache["a1"].T @ dz2
    g["phi2b"] = dz2.sum(0)
    da1 = dz2 @ P["phi2"].T
    dz1 = da1 * (1 - cache["a1"] ** 2)
    g["phi1"] = cache["flat"].T @ dz1
    g["phi1b"] = dz1.sum(0)
    return g


def make_dataset(teacher, *, n_list, n_scenes, seed0, antipodal_frac=0.0,
                 reflect_canonical=False):
    """BC dataset mirroring _swarm_policy.make_dataset with 6D peer tokens."""
    egos, peerl, masks, acts = [], [], [], []
    sc = 0
    s = seed0
    while sc < n_scenes:
        rng = np.random.default_rng(s)
        s += 1
        n = int(rng.choice(n_list))
        if rng.random() < antipodal_frac:
            st, gl = sp.antipodal(n, rng)
        else:
            st, gl = sp.random_scene(n, rng)
        p = st.copy()
        vel = np.zeros_like(p)
        done = np.zeros(n, bool)
        for _ in range(200):
            newv = np.zeros_like(p)
            for i in range(n):
                if done[i]:
                    continue
                a = teacher(p, vel, gl, i)
                ego, pad, m, R = featurize(
                    p, vel, gl, i, reflect_canonical=reflect_canonical,
                )
                egos.append(ego)
                peerl.append(pad)
                masks.append(m)
                acts.append(R @ a)
                newv[i] = a
            p = p + newv * DT
            vel = newv
            for i in range(n):
                if np.linalg.norm(gl[i] - p[i]) < sp.GOAL_TOL:
                    done[i] = True
            if done.all():
                break
        sc += 1
    return (
        np.array(egos),
        np.array(peerl),
        np.array(masks),
        np.array(acts),
    )


def train_bc(data, *, h=32, epochs=300, batch=256, lr=3e-3, seed=0, verbose=False):
    egos, peers, masks, acts = data
    n = len(egos)
    em, es = egos.mean(0), egos.std(0) + 1e-6
    flat = peers.reshape(-1, PEER_DIM)
    pm, ps = flat.mean(0), flat.std(0) + 1e-6
    egos = (egos - em) / es
    peers = np.where(masks[:, :, None] > 0, (peers - pm) / ps, 0.0)
    P = init_model(h, seed)
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(v) for k, v in P.items()}
    rng = np.random.default_rng(seed)
    t = 0
    for ep in range(epochs):
        idx = rng.permutation(n)
        for b in range(0, n, batch):
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
                mh = m[k] / (1 - 0.9 ** t)
                vh = v[k] / (1 - 0.999 ** t)
                P[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        if verbose and ep % 50 == 0:
            pr = forward(P, egos, peers, masks)
            print(f"  ep{ep:4} mse={np.mean((pr - acts) ** 2):.4f}")
    stats = dict(em=em, es=es, pm=pm, ps=ps)
    return P, stats


def make_student_controller(P, stats, reflect_canonical=False, threats=None):
    """Wrap trained transformer as controller(p, vel, goals, i)."""
    em, es, pm, ps = stats["em"], stats["es"], stats["pm"], stats["ps"]

    def ctrl(p, vel, goals, i):
        ego, pad, m, R = featurize(
            p, vel, goals, i,
            reflect_canonical=reflect_canonical,
            threats=threats(p, vel) if threats is not None else None,
        )
        e = ((ego - em) / es)[None, :]
        pn = np.where(m[None, :, None] > 0, (pad[None] - pm) / ps, 0.0)
        mk = m[None, :]
        a_ego = forward(P, e, pn, mk)[0]
        return sp._clamp(R.T @ a_ego)

    return ctrl


def attention_mass(P, stats, ego, pad, mask):
    """Per-peer softmax mass after training (diagnostic)."""
    em, es, pm, ps = stats["em"], stats["es"], stats["pm"], stats["ps"]
    e = ((ego - em) / es)[None, :]
    pn = np.where(mask[None, :, None] > 0, (pad[None] - pm) / ps, 0.0)
    mk = mask[None, :]
    cache = {}
    forward(P, e, pn, mk, cache)
    return cache["attn"][0]


if __name__ == "__main__":
    print("transformer BC distill (train random, test antipodal):", flush=True)
    for name, teacher in (("plain", sp.teacher_plain), ("conv", sp.teacher_conv)):
        data = make_dataset(teacher, n_list=[3, 4, 5, 6], n_scenes=120, seed0=0)
        print(f"  [{name}] {len(data[0])} samples; training...", flush=True)
        P, stats = train_bc(data, epochs=120, verbose=False)
        pr = forward(
            P,
            (data[0] - stats["em"]) / stats["es"],
            np.where(
                data[2][:, :, None] > 0,
                (data[1] - stats["pm"]) / stats["ps"],
                0.0,
            ),
            data[2],
        )
        mse = float(np.mean((pr - data[3]) ** 2))
        sc = make_student_controller(P, stats)
        line = []
        for n in (4, 6, 8):
            ok = sum(
                sp.rollout(*sp.antipodal(n, np.random.default_rng(s)), sc).success
                for s in range(20)
            )
            line.append(f"N={n}:{ok}/20")
        print(f"  xf<-{name:5} (bc_mse={mse:.4f})  " + "  ".join(line))
