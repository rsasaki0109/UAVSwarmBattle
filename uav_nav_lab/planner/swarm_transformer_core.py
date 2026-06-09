"""NumPy teammate-token transformer core (TeamHOI-style cross-attention).

Shared by ``uav_nav_lab.planner.swarm_transformer`` (inference at replan time)
and ``scripts/_swarm_transformer.py`` (BC / REINFORCE training).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

MAX_PEERS = 8
PRED_HORIZONS = (1.0, 2.0)
PEER_DIM = 6 + 2 * len(PRED_HORIZONS)
ROLE_ALLY = 0.0
ROLE_THREAT = 1.0


def init_model(h: int = 32, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def w(a: int, b: int) -> np.ndarray:
        return rng.normal(0, np.sqrt(2.0 / a), (a, b))

    return {
        "phi1": w(PEER_DIM, h), "phi1b": np.zeros(h),
        "phi2": w(h, h), "phi2b": np.zeros(h),
        "ego1": w(3, h), "ego1b": np.zeros(h),
        "Wq": w(h, h), "Wk": w(h, h), "Wv": w(h, h),
        "out1": w(2 * h, h), "out1b": np.zeros(h),
        "out2": w(h, 2), "out2b": np.zeros(2),
    }


def _softmax(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask > 0, scores, -1e9)
    e = np.exp(masked - masked.max(axis=1, keepdims=True))
    e = e * mask
    denom = np.maximum(e.sum(axis=1, keepdims=True), 1e-9)
    return e / denom


def forward(
    params: Mapping[str, np.ndarray],
    ego: np.ndarray,
    peers: np.ndarray,
    mask: np.ndarray,
    cache: dict[str, Any] | None = None,
) -> np.ndarray:
    """ego (B,3), peers (B,M,6), mask (B,M) -> action (B,2) in ego-goal frame."""
    b, m, _ = peers.shape
    flat = peers.reshape(b * m, PEER_DIM)
    z1 = flat @ params["phi1"] + params["phi1b"]
    a1 = np.tanh(z1)
    z2 = a1 @ params["phi2"] + params["phi2b"]
    peer_emb = np.tanh(z2).reshape(b, m, -1)

    ze = ego @ params["ego1"] + params["ego1b"]
    ego_emb = np.tanh(ze)

    h = ego_emb.shape[1]
    scale = np.sqrt(float(h))
    q = ego_emb @ params["Wq"]
    k = peer_emb @ params["Wk"]
    v = peer_emb @ params["Wv"]
    scores = np.einsum("bh,bmh->bm", q, k) / scale
    attn = _softmax(scores, mask)
    context = np.einsum("bm,bmh->bh", attn, v)

    cat = np.concatenate([ego_emb, context], axis=1)
    zo = cat @ params["out1"] + params["out1b"]
    ao = np.tanh(zo)
    out = ao @ params["out2"] + params["out2b"]

    if cache is not None:
        cache.update(
            flat=flat, a1=a1, z1=z1, peer_emb=peer_emb, z2=z2,
            ze=ze, ego_emb=ego_emb, ae=ego_emb, ego=ego,
            Q=q, K=k, V=v, scores=scores, attn=attn, context=context,
            cat=cat, ao=ao, zo=zo, mask=mask, B=b, M=m, scale=scale,
        )
    return out


def backward(
    params: Mapping[str, np.ndarray],
    cache: dict[str, Any],
    dout: np.ndarray,
) -> dict[str, np.ndarray]:
    g: dict[str, np.ndarray] = {}
    ao = cache["ao"]
    cat = cache["cat"]
    g["out2"] = ao.T @ dout
    g["out2b"] = dout.sum(0)
    dao = dout @ params["out2"].T
    dzo = dao * (1 - ao ** 2)
    g["out1"] = cat.T @ dzo
    g["out1b"] = dzo.sum(0)
    dcat = dzo @ params["out1"].T
    h = params["ego1"].shape[1]
    d_ego_emb = dcat[:, :h]
    d_context = dcat[:, h:]

    attn = cache["attn"]
    v = cache["V"]
    mask = cache["mask"]
    b, m = cache["B"], cache["M"]

    dv = attn[:, :, None] * d_context[:, None, :]
    d_attn = np.einsum("bh,bmh->bm", d_context, v)
    d_scores = attn * (d_attn - (d_attn * attn).sum(axis=1, keepdims=True))
    d_scores = np.where(mask > 0, d_scores, 0.0)

    q = cache["Q"]
    k = cache["K"]
    scale = cache["scale"]
    dq = np.einsum("bm,bmh->bh", d_scores, k) / scale
    dk = d_scores[:, :, None] * q[:, None, :] / scale

    g["Wq"] = cache["ego_emb"].T @ dq
    g["Wk"] = cache["peer_emb"].reshape(b * m, h).T @ dk.reshape(b * m, h)
    g["Wv"] = cache["peer_emb"].reshape(b * m, h).T @ dv.reshape(b * m, h)

    d_peer_emb = dk @ params["Wk"].T + dv @ params["Wv"].T
    d_ego_emb = d_ego_emb + dq @ params["Wq"].T

    dze = d_ego_emb * (1 - cache["ego_emb"] ** 2)
    g["ego1"] = cache["ego"].T @ dze
    g["ego1b"] = dze.sum(0)

    d_peer = d_peer_emb * (1 - cache["peer_emb"] ** 2)
    dz2 = d_peer.reshape(b * m, h)
    g["phi2"] = cache["a1"].T @ dz2
    g["phi2b"] = dz2.sum(0)
    da1 = dz2 @ params["phi2"].T
    dz1 = da1 * (1 - cache["a1"] ** 2)
    g["phi1"] = cache["flat"].T @ dz1
    g["phi1b"] = dz1.sum(0)
    return g


def _ego_rotation(pos: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    to_goal = goal - pos
    dist = float(np.linalg.norm(to_goal))
    heading = to_goal / dist if dist > 1e-6 else np.array([1.0, 0.0])
    rot = np.array([[heading[0], heading[1]], [-heading[1], heading[0]]])
    return rot, heading


def _forecast_positions(
    obstacles: list[dict],
    horizons: tuple[float, ...],
    predictor: Any | None,
) -> np.ndarray:
    """World-frame positions at each horizon; shape (n_obs, n_horizons, 2)."""
    if not obstacles:
        return np.zeros((0, len(horizons), 2))
    h = np.asarray(horizons, dtype=float)
    if predictor is not None:
        traj = predictor.predict(obstacles, h)
        return np.asarray(traj[:, :, :2], dtype=float)
    out = np.empty((len(obstacles), len(horizons), 2))
    for k, d in enumerate(obstacles):
        p0 = np.asarray(d["position"], dtype=float)[:2]
        v = np.asarray(d.get("velocity", (0.0, 0.0)), dtype=float)[:2]
        out[k] = p0[None, :] + h[:, None] * v[None, :]
    return out


def build_tokens(
    pos: np.ndarray,
    vel: np.ndarray,
    goal: np.ndarray,
    dynamic_obstacles: list[dict] | None,
    *,
    neighbor_dist: float,
    interaction_radius: float,
    max_peers: int = MAX_PEERS,
    predictor: Any | None = None,
    pred_horizons: tuple[float, ...] = PRED_HORIZONS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack runner ``dynamic_obstacles`` into ego / peer tokens.

    Entries with a ``goal`` key are ally UAVs; others are threats (scene dynamics).
    Each token carries relative pose/velocity plus predicted relative positions at
    ``pred_horizons`` (ego-goal frame), matching the MPC planner's lookahead.
    """
    pos2 = np.asarray(pos, dtype=float)[:2]
    vel2 = np.asarray(vel, dtype=float)[:2]
    goal2 = np.asarray(goal, dtype=float)[:2]
    rot, _ = _ego_rotation(pos2, goal2)
    dist_goal = float(np.linalg.norm(goal2 - pos2))
    ego = np.array([min(dist_goal, 10.0) / 10.0, *(rot @ vel2)], dtype=float)

    visible: list[dict] = []
    nb2 = neighbor_dist * neighbor_dist
    for d in dynamic_obstacles or []:
        other = np.asarray(d["position"], dtype=float)[:2]
        rel = other - pos2
        dist_sq = float(rel @ rel)
        if dist_sq > nb2 or dist_sq < 1e-12:
            continue
        visible.append(d)

    forecasts = _forecast_positions(visible, pred_horizons, predictor)
    rows: list[np.ndarray] = []
    for k, d in enumerate(visible):
        other = np.asarray(d["position"], dtype=float)[:2]
        rel = other - pos2
        dist = float(np.linalg.norm(rel))
        ovel = np.asarray(d.get("velocity", (0.0, 0.0)), dtype=float)[:2]
        de = rot @ rel
        ve = rot @ (ovel - vel2)
        role = ROLE_ALLY if "goal" in d else ROLE_THREAT
        row = [
            de[0], de[1], dist / interaction_radius, ve[0], ve[1], role,
        ]
        for j in range(len(pred_horizons)):
            pred_rel = rot @ (forecasts[k, j] - pos2)
            row.extend([pred_rel[0], pred_rel[1]])
        rows.append(np.array(row, dtype=float))

    pad = np.zeros((max_peers, PEER_DIM))
    mask = np.zeros(max_peers)
    n = min(len(rows), max_peers)
    if n:
        pad[:n] = np.stack(rows[:n])
        mask[:n] = 1.0
    return ego, pad, mask, rot


def normalize_batch(
    ego: np.ndarray,
    peers: np.ndarray,
    mask: np.ndarray,
    stats: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    em, es, pm, ps = stats["em"], stats["es"], stats["pm"], stats["ps"]
    ego_n = (ego - em) / es
    peers_n = np.where(mask[:, :, None] > 0, (peers - pm) / ps, 0.0)
    return ego_n, peers_n


def predict_velocity_ego(
    params: Mapping[str, np.ndarray],
    stats: Mapping[str, np.ndarray],
    ego: np.ndarray,
    peers: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    ego_b = ego[None, :] if ego.ndim == 1 else ego
    peers_b = peers[None, :, :] if peers.ndim == 2 else peers
    mask_b = mask[None, :] if mask.ndim == 1 else mask
    ego_n, peers_n = normalize_batch(ego_b, peers_b, mask_b, stats)
    return forward(params, ego_n, peers_n, mask_b)[0]


def clamp_speed(v: np.ndarray, max_speed: float) -> np.ndarray:
    s = float(np.linalg.norm(v))
    if s > max_speed and s > 1e-9:
        return v / s * max_speed
    return v


def save_checkpoint(path: str | Path, params: Mapping[str, np.ndarray], stats: Mapping[str, np.ndarray]) -> None:
    payload = {f"p_{k}": v for k, v in params.items()}
    payload.update({f"s_{k}": v for k, v in stats.items()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def load_checkpoint(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    data = np.load(path)
    params = {k[2:]: data[k] for k in data.files if k.startswith("p_")}
    stats = {k[2:]: data[k] for k in data.files if k.startswith("s_")}
    if not params or not stats:
        raise ValueError(f"checkpoint missing p_/s_ keys: {path}")
    return params, stats
