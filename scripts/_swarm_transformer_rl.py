"""REINFORCE for the TeamHOI-style transformer policy (scripts/_swarm_transformer.py).

Mirrors scripts/_swarm_rl.py but uses cross-attention + optional threat tokens.
Supports mixed team sizes and threat counts during training (TeamHOI-style
curriculum over N and K).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

import _swarm_policy as sp
import _swarm_threat as thr
import _swarm_transformer as st


def _feat_plain(p, vel, goal, i, reflect):
    ego, pad, m, R = st.featurize(p, vel, goal, i, reflect_canonical=reflect)
    return ego, pad, m, R


def _feat_threat(p, vel, goal, i, reflect, threats):
    ego, pad, m, R = st.featurize(
        p, vel, goal, i, reflect_canonical=reflect, threats=threats,
    )
    return ego, pad, m, R


def _collect_stats(reflect, n=3000, seed=0, with_threats=False):
    rng = np.random.default_rng(seed)
    egos, peers, masks = [], [], []
    while len(egos) < n:
        nn = int(rng.integers(2, 7))
        if with_threats:
            st0, gl, tpos, tvel = thr.hub_scene(nn, int(rng.integers(1, 4)), rng)
            p = st0.copy()
            vel = np.zeros_like(p)
            threats = thr._threat_list(tpos, tvel)
        else:
            st0, gl = sp.random_scene(nn, rng)
            p = st0.copy()
            vel = np.zeros_like(p)
            threats = None
        for _ in range(30):
            for i in range(nn):
                if with_threats:
                    e, pd, mk, _ = _feat_threat(p, vel, gl, i, reflect, threats)
                else:
                    e, pd, mk, _ = _feat_plain(p, vel, gl, i, reflect)
                egos.append(e)
                peers.append(pd)
                masks.append(mk)
            for i in range(nn):
                d = gl[i] - p[i]
                p[i] = p[i] + d / (np.linalg.norm(d) + 1e-9) * sp.VMAX * sp.DT
            if with_threats:
                tpos, tvel = thr.advance_threats(tpos, tvel)
                threats = thr._threat_list(tpos, tvel)
    egos = np.array(egos)
    peers = np.array(peers)
    masks = np.array(masks)
    flat = peers.reshape(-1, st.PEER_DIM)
    return dict(
        em=egos.mean(0), es=egos.std(0) + 1e-6,
        pm=flat.mean(0), ps=flat.std(0) + 1e-6,
    )


def _mu(P, stats, ego, pad, mask):
    e = (ego - stats["em"]) / stats["es"]
    pn = np.where(mask[:, :, None] > 0, (pad - stats["pm"]) / stats["ps"], 0.0)
    return st.forward(P, e, pn, mask)


def _episode_antipodal(P, stats, start, goal, reflect, sigma, rng, max_steps=120):
    nn = len(start)
    p = start.copy()
    vel = np.zeros_like(p)
    done = np.zeros(nn, bool)
    rec = {i: [] for i in range(nn)}
    rew = {i: 0.0 for i in range(nn)}
    prevd = {i: float(np.linalg.norm(goal[i] - p[i])) for i in range(nn)}
    for _ in range(max_steps):
        idxs, egos, pads, masks, rs = [], [], [], [], []
        for i in range(nn):
            if done[i]:
                continue
            e, pd, mk, R = _feat_plain(p, vel, goal, i, reflect)
            idxs.append(i)
            egos.append(e)
            pads.append(pd)
            masks.append(mk)
            rs.append(R)
        if not idxs:
            break
        mu = _mu(P, stats, np.array(egos), np.array(pads), np.array(masks))
        a_ego = mu + rng.normal(0, sigma, mu.shape)
        newv = np.zeros_like(p)
        for r, i in enumerate(idxs):
            newv[i] = sp._clamp(rs[r].T @ a_ego[r])
            rec[i].append((egos[r], pads[r], masks[r], a_ego[r]))
        p = p + newv * sp.DT
        vel = newv
        col = _step_rewards(p, done, goal, rew, prevd, nn)
        if col or done.all():
            break
    return rec, rew


def _episode_threat(P, stats, start, goal, tpos0, tvel0, reflect, sigma, rng,
                    max_steps=180):
    nn = len(start)
    p = start.copy()
    vel = np.zeros_like(p)
    tpos = tpos0.copy()
    tvel = tvel0.copy()
    done = np.zeros(nn, bool)
    rec = {i: [] for i in range(nn)}
    rew = {i: 0.0 for i in range(nn)}
    prevd = {i: float(np.linalg.norm(goal[i] - p[i])) for i in range(nn)}
    for _ in range(max_steps):
        threats = thr._threat_list(tpos, tvel)
        idxs, egos, pads, masks, rs = [], [], [], [], []
        for i in range(nn):
            if done[i]:
                continue
            e, pd, mk, R = _feat_threat(p, vel, goal, i, reflect, threats)
            idxs.append(i)
            egos.append(e)
            pads.append(pd)
            masks.append(mk)
            rs.append(R)
        if not idxs:
            break
        mu = _mu(P, stats, np.array(egos), np.array(pads), np.array(masks))
        a_ego = mu + rng.normal(0, sigma, mu.shape)
        newv = np.zeros_like(p)
        for r, i in enumerate(idxs):
            newv[i] = sp._clamp(rs[r].T @ a_ego[r])
            rec[i].append((egos[r], pads[r], masks[r], a_ego[r]))
        p = p + newv * sp.DT
        vel = newv
        tpos, tvel = thr.advance_threats(tpos, tvel)
        col = _step_rewards(p, done, goal, rew, prevd, nn)
        if thr._threat_collision(p, done, tpos):
            for i in range(nn):
                if not done[i]:
                    rew[i] -= 3.0
            break
        if col or done.all():
            break
    return rec, rew


def _step_rewards(p, done, goal, rew, prevd, nn):
    col = False
    for i in range(nn):
        if done[i]:
            continue
        d = float(np.linalg.norm(goal[i] - p[i]))
        rew[i] += prevd[i] - d
        prevd[i] = d
        if d < sp.GOAL_TOL:
            done[i] = True
            rew[i] += 2.0
    for i in range(nn):
        for j in range(i + 1, nn):
            if not (done[i] and done[j]) and np.linalg.norm(p[i] - p[j]) < 2 * sp.RADIUS:
                rew[i] -= 3.0
                rew[j] -= 3.0
                col = True
    return col


def _apply_grad(P, stats, batch, baseline_state, lr, sigma, m, v, t):
    E, Pd, M, A, Ret = batch
    E = np.array(E)
    Pd = np.array(Pd)
    M = np.array(M)
    A = np.array(A)
    Ret = np.array(Ret)
    baseline = 0.9 * baseline_state + 0.1 * float(Ret.mean())
    adv = Ret - baseline
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    mu = _mu(P, stats, E, Pd, M)
    dout = (-(adv[:, None]) * (A - mu) / (sigma ** 2)) / len(E)
    cache = {}
    st.forward(
        P,
        (E - stats["em"]) / stats["es"],
        np.where(M[:, :, None] > 0, (Pd - stats["pm"]) / stats["ps"], 0.0),
        M,
        cache,
    )
    g = st.backward(P, cache, dout)
    t += 1
    for k in P:
        m[k] = 0.9 * m[k] + 0.1 * g[k]
        v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
        mh = m[k] / (1 - 0.9 ** t)
        vh = v[k] / (1 - 0.999 ** t)
        P[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
    return baseline, t


def train_reinforce(
    reflect=False,
    *,
    iters=400,
    episodes=32,
    h=32,
    lr=3e-3,
    sigma=0.3,
    seed=0,
    n_list=(2, 3, 4),
    k_list=(1, 2, 3),
    antipodal_frac=0.5,
    threat_frac=0.5,
    verbose=False,
):
    rng = np.random.default_rng(seed)
    stats = _collect_stats(reflect, seed=seed, with_threats=threat_frac > 0)
    P = st.init_model(h, seed)
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(v) for k, v in P.items()}
    baseline = 0.0
    t = 0
    for it in range(iters):
        E, Pd, Ma, A, Ret = [], [], [], [], []
        ep_rets = []
        for _ in range(episodes):
            nn = int(rng.choice(n_list))
            use_threat = rng.random() < threat_frac
            if use_threat:
                k = int(rng.choice(k_list))
                st0, gl, tpos, tvel = thr.hub_scene(nn, k, rng)
                rec, rew = _episode_threat(
                    P, stats, st0, gl, tpos, tvel, reflect, sigma, rng,
                )
            elif rng.random() < antipodal_frac:
                st0, gl = sp.antipodal(nn, rng)
                rec, rew = _episode_antipodal(P, stats, st0, gl, reflect, sigma, rng)
            else:
                st0, gl = sp.random_scene(nn, rng)
                rec, rew = _episode_antipodal(P, stats, st0, gl, reflect, sigma, rng)
            for i in range(nn):
                ep_rets.append(rew[i])
                for (e, pd, mk, a) in rec[i]:
                    E.append(e)
                    Pd.append(pd)
                    Ma.append(mk)
                    A.append(a)
                    Ret.append(rew[i])
        if not E:
            continue
        baseline, t = _apply_grad(
            P, stats, (E, Pd, Ma, A, Ret), baseline, lr, sigma, m, v, t,
        )
        if verbose and it % 40 == 0:
            print(f"  it{it:4d}  mean_return={np.mean(ep_rets):+.2f}", flush=True)
    return P, stats


def policy_controller(P, stats, reflect=False):
    return st.make_student_controller(P, stats, reflect_canonical=reflect)


def threat_controller(P, stats, reflect=False):
    return thr.make_xf_controller(P, stats)


def handedness(P, stats, reflect, n=6, seeds=20):
    vals = []
    for s in range(seeds):
        st0, gl = sp.antipodal(n, np.random.default_rng(s))
        p = st0.copy()
        vel = np.zeros_like(p)
        for _ in range(60):
            newv = np.zeros_like(p)
            for i in range(len(p)):
                e, pd, mk, R = _feat_plain(p, vel, gl, i, reflect)
                mu = _mu(P, stats, e[None], pd[None], mk[None])[0]
                if mk.sum() > 0:
                    vals.append(mu[1])
                newv[i] = sp._clamp(R.T @ mu)
            p = p + newv * sp.DT
            vel = newv
    vals = np.array(vals) if vals else np.array([0.0])
    return float(vals.mean()), float(np.abs(vals).mean())


if __name__ == "__main__":
    P, stats = train_reinforce(iters=200, episodes=24, verbose=True, seed=0)
    ctrl = policy_controller(P, stats)
    for n in (4, 6, 8):
        ok = sum(
            sp.rollout(*sp.antipodal(n, np.random.default_rng(s)), ctrl).success
            for s in range(20)
        )
        print(f"antipodal N={n}: {ok}/20")
    tctrl = threat_controller(P, stats)
    ok_t = sum(
        thr.rollout(*thr.hub_scene(4, 2, np.random.default_rng(s)), tctrl).success
        for s in range(20)
    )
    print(f"hub+2 threats N=4: {ok_t}/20")
