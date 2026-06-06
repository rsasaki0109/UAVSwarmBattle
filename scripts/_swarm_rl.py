"""REINFORCE on the SAME NumPy teammate-token deep set used for behavioral cloning
(neta A, follow-up to swarm_bc_symmetry_phase.py / swarm_bc_chirality_phase.py).

BC showed the deep set carries a right-of-way convention iff (its teacher
demonstrates it) AND (its representation can represent handedness). The open
question those leave: does a policy *discover* the convention on its own, from a
REWARD with no built-in handedness — i.e. is the symmetry-breaking learnable, or
does symmetric optimization fall into the symmetric (deadlocking) solution?

We train the identical architecture (sp.init_model / forward / backward) by
policy-gradient instead of supervised regression. The policy is Gaussian: mean =
deep_set(features) in the ego-goal frame, fixed exploration std. The gradient
reuses sp.backward with dout = -advantage * (a - mu) / sigma^2 (the REINFORCE
gradient of a fixed-variance Gaussian). The reward is reflection-SYMMETRIC —
progress to goal, a collision penalty, a goal bonus — so any handedness the policy
ends up with is *self-generated*, not supplied. The only seed of asymmetry is the
random init + exploration noise (spontaneous symmetry breaking).

`reflect_canonical` toggles the same chirality-free representation as the BC study,
so we can ask whether RL discovery, like BC transfer, needs a chirality-capable
representation.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

import _swarm_policy as sp


def _feat(p, vel, goal, i, reflect):
    ego, pe, R = sp._ego_frame(p, vel, goal, i, reflect)
    pad = np.zeros((sp.MAX_PEERS, 5))
    m = np.zeros(sp.MAX_PEERS)
    k = min(len(pe), sp.MAX_PEERS)
    if k:
        pad[:k] = pe[:k]
        m[:k] = 1.0
    return ego, pad, m, R


def _collect_stats(reflect, n=3000, seed=0):
    """Input standardization from straight-to-goal motion on random scenes."""
    rng = np.random.default_rng(seed)
    egos, peers, masks = [], [], []
    while len(egos) < n:
        nn = int(rng.integers(2, 7))
        st, gl = sp.random_scene(nn, rng)
        p = st.copy(); vel = np.zeros_like(p)
        for _ in range(30):
            for i in range(nn):
                e, pd, mk, _R = _feat(p, vel, gl, i, reflect)
                egos.append(e); peers.append(pd); masks.append(mk)
            for i in range(nn):
                d = gl[i] - p[i]
                p[i] = p[i] + d / (np.linalg.norm(d) + 1e-9) * sp.VMAX * sp.DT
    egos = np.array(egos); peers = np.array(peers); masks = np.array(masks)
    return dict(em=egos.mean(0), es=egos.std(0) + 1e-6,
                pm=peers.reshape(-1, 5).mean(0), ps=peers.reshape(-1, 5).std(0) + 1e-6)


def _mu(P, stats, ego, pad, mask):
    e = (ego - stats["em"]) / stats["es"]
    pn = np.where(mask[:, :, None] > 0, (pad - stats["pm"]) / stats["ps"], 0.0)
    return sp.forward(P, e, pn, mask)


def _episode(P, stats, start, goal, reflect, sigma, rng, max_steps=120):
    """One stochastic rollout. Returns per-agent (records, return)."""
    nn = len(start)
    p = start.copy(); vel = np.zeros_like(p)
    done = np.zeros(nn, bool)
    rec = {i: [] for i in range(nn)}
    rew = {i: 0.0 for i in range(nn)}
    prevd = {i: float(np.linalg.norm(goal[i] - p[i])) for i in range(nn)}
    for _ in range(max_steps):
        idxs, egos, pads, masks, Rs = [], [], [], [], []
        for i in range(nn):
            if done[i]:
                continue
            e, pd, mk, R = _feat(p, vel, goal, i, reflect)
            idxs.append(i); egos.append(e); pads.append(pd); masks.append(mk); Rs.append(R)
        if not idxs:
            break
        mu = _mu(P, stats, np.array(egos), np.array(pads), np.array(masks))
        a_ego = mu + rng.normal(0, sigma, mu.shape)
        newv = np.zeros_like(p)
        for r, i in enumerate(idxs):
            newv[i] = sp._clamp(Rs[r].T @ a_ego[r])
            rec[i].append((egos[r], pads[r], masks[r], a_ego[r]))
        p = p + newv * sp.DT; vel = newv
        for i in range(nn):
            if done[i]:
                continue
            d = float(np.linalg.norm(goal[i] - p[i]))
            rew[i] += prevd[i] - d                 # progress (reflection-symmetric)
            prevd[i] = d
            if d < sp.GOAL_TOL:
                done[i] = True; rew[i] += 2.0       # goal bonus
        col = False
        for i in range(nn):
            for j in range(i + 1, nn):
                if not (done[i] and done[j]) and np.linalg.norm(p[i] - p[j]) < 2 * sp.RADIUS:
                    rew[i] -= 3.0; rew[j] -= 3.0; col = True
        if col or done.all():
            break
    return rec, rew


def train_reinforce(reflect=False, *, iters=400, episodes=32, h=32, lr=3e-3,
                    sigma=0.3, seed=0, n_list=(2, 3, 4), antipodal_frac=0.5,
                    verbose=False):
    rng = np.random.default_rng(seed)
    stats = _collect_stats(reflect, seed=seed)
    P = sp.init_model(h, seed)
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(v) for k, v in P.items()}
    baseline = 0.0; t = 0
    for it in range(iters):
        E, Pd, M, A, Ret = [], [], [], [], []
        ep_rets = []
        for _ in range(episodes):
            nn = int(rng.choice(n_list))
            if rng.random() < antipodal_frac:
                st, gl = sp.antipodal(nn, rng)
            else:
                st, gl = sp.random_scene(nn, rng)
            rec, rew = _episode(P, stats, st, gl, reflect, sigma, rng)
            for i in range(nn):
                ep_rets.append(rew[i])
                for (e, pd, mk, a) in rec[i]:
                    E.append(e); Pd.append(pd); M.append(mk); A.append(a); Ret.append(rew[i])
        if not E:
            continue
        E = np.array(E); Pd = np.array(Pd); M = np.array(M); A = np.array(A); Ret = np.array(Ret)
        baseline = 0.9 * baseline + 0.1 * float(Ret.mean())
        adv = Ret - baseline
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        mu = _mu(P, stats, E, Pd, M)
        dout = (-(adv[:, None]) * (A - mu) / (sigma ** 2)) / len(E)
        cache = {}
        sp.forward(P, (E - stats["em"]) / stats["es"],
                   np.where(M[:, :, None] > 0, (Pd - stats["pm"]) / stats["ps"], 0.0), M, cache)
        g = sp.backward(P, cache, dout)
        t += 1
        for k in P:
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
            mh = m[k] / (1 - 0.9 ** t); vh = v[k] / (1 - 0.999 ** t)
            P[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        if verbose and it % 40 == 0:
            print(f"  it{it:4d}  mean_return={np.mean(ep_rets):+.2f}", flush=True)
    return P, stats


def policy_controller(P, stats, reflect=False):
    """Deterministic (mean-action) controller for evaluation; reuses the BC wrapper."""
    return sp.make_student_controller(P, stats, reflect_canonical=reflect)


def handedness(P, stats, reflect, n=6, seeds=20):
    """Mean signed ego-frame lateral action when a peer is ahead — sign = which side
    the policy passes on, |mean| / consistency = how broken the L/R symmetry is."""
    vals = []
    for s in range(seeds):
        st, gl = sp.antipodal(n, np.random.default_rng(s))
        p = st.copy(); vel = np.zeros_like(p)
        for _ in range(60):
            newv = np.zeros_like(p)
            for i in range(len(p)):
                e, pd, mk, R = _feat(p, vel, gl, i, reflect)
                if mk.sum() > 0:
                    mu = _mu(P, stats, e[None], pd[None], mk[None])[0]
                    vals.append(mu[1])  # ego-frame lateral (−y = right)
                newv[i] = sp._clamp(R.T @ _mu(P, stats, e[None], pd[None], mk[None])[0])
            p = p + newv * sp.DT; vel = newv
    vals = np.array(vals) if vals else np.array([0.0])
    return float(vals.mean()), float(np.abs(vals).mean())


if __name__ == "__main__":
    for reflect in (False, True):
        P, stats = train_reinforce(reflect=reflect, iters=200, episodes=24, sigma=0.3,
                                   seed=0, verbose=True)
        ctrl = policy_controller(P, stats, reflect)
        line = []
        for n in (4, 6, 8):
            ok = sum(sp.rollout(*sp.antipodal(n, np.random.default_rng(s)), ctrl).success
                     for s in range(30))
            line.append(f"N={n}:{ok}/30")
        mh, ma = handedness(P, stats, reflect)
        tag = "reflect" if reflect else "standard"
        print(f"[{tag}] " + "  ".join(line) + f"   handedness mean={mh:+.3f} |lat|={ma:.3f}")
