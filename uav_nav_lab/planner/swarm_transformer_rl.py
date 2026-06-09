"""Framework-scale REINFORCE for swarm_transformer (obstacle / antipodal YAMLs)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import ExperimentConfig
from ..predictor import build_predictor
from ..runner.multi.peers import _check_peer_collision, _peers_view
from ..scenario import SCENARIO_REGISTRY
from . import swarm_transformer_core as core
from .swarm_transformer_bc import _DroneState, _step_velocity


@dataclass
class _RlConfig:
    dt: float
    max_steps: int
    max_accel: float
    goal_radius: float
    drone_radius: float
    max_speed: float
    neighbor_dist: float
    interaction_radius: float
    joint_bonus: float = 10.0
    collision_penalty: float = 8.0


def _mu_batch(
    params: Mapping[str, np.ndarray],
    stats: Mapping[str, np.ndarray],
    ego: np.ndarray,
    peers: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    ego_n, peers_n = core.normalize_batch(ego, peers, mask, stats)
    return core.forward(params, ego_n, peers_n, mask)


def _episode(
    scenario: Any,
    cfg: _RlConfig,
    params: Mapping[str, np.ndarray],
    stats: Mapping[str, np.ndarray],
    predictor: Any | None,
    *,
    seed: int,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[dict[int, list], dict[int, float]]:
    n = scenario.n_drones
    radii = [float(d.radius) for d in scenario.drones]
    goals = [np.asarray(d.goal, dtype=float)[:2] for d in scenario.drones]

    scenario.reseed(seed)
    if predictor is not None:
        predictor.reset(seed=seed)
    starts = scenario.episode_drone_starts(seed)
    states = [_DroneState(s[:2].copy(), np.zeros(2)) for s in starts]
    done = [False] * n
    rec: dict[int, list] = {i: [] for i in range(n)}
    rew = {i: 0.0 for i in range(n)}
    prevd = {i: float(np.linalg.norm(goals[i] - states[i].position)) for i in range(n)}

    for _ in range(cfg.max_steps):
        if all(done):
            break

        peer_states = [
            type("S", (), {"position": st.position, "velocity": st.velocity})()
            for st in states
        ]
        scenario.set_targets([st.position for st in states])
        scene_dyn = list(scenario.dynamic_obstacles)

        idxs, egos, pads, masks, rots = [], [], [], [], []
        for i in range(n):
            if done[i]:
                continue
            peer_dyn = _peers_view(peer_states, radii, done, me=i, goals=goals)
            dyn = scene_dyn + peer_dyn
            ego, pad, mask, rot = core.build_tokens(
                states[i].position,
                states[i].velocity,
                goals[i],
                dyn,
                neighbor_dist=cfg.neighbor_dist,
                interaction_radius=cfg.interaction_radius,
                predictor=predictor,
            )
            idxs.append(i)
            egos.append(ego)
            pads.append(pad)
            masks.append(mask)
            rots.append(rot)

        if not idxs:
            break

        ego_b = np.stack(egos)
        pad_b = np.stack(pads)
        mask_b = np.stack(masks)
        mu = _mu_batch(params, stats, ego_b, pad_b, mask_b)
        a_ego = mu + rng.normal(0.0, sigma, mu.shape)

        cmds: list[np.ndarray | None] = [None] * n
        for r, i in enumerate(idxs):
            v_world = core.clamp_speed(rots[r].T @ (a_ego[r] * cfg.max_speed), cfg.max_speed)
            rec[i].append((egos[r], pads[r], masks[r], a_ego[r]))
            cmds[i] = v_world

        scenario.advance(cfg.dt)
        col = False
        for i in range(n):
            if done[i] or cmds[i] is None:
                continue
            states[i].position, states[i].velocity = _step_velocity(
                states[i].position,
                states[i].velocity,
                cmds[i],
                dt=cfg.dt,
                max_accel=cfg.max_accel,
            )
            d = float(np.linalg.norm(goals[i] - states[i].position))
            rew[i] += prevd[i] - d
            prevd[i] = d
            if scenario.is_collision(states[i].position, radii[i]):
                rew[i] -= cfg.collision_penalty
                done[i] = True
                col = True
            elif d <= cfg.goal_radius:
                rew[i] += 5.0
                done[i] = True

        hits = _check_peer_collision(peer_states, radii, cfg.drone_radius)
        for i, hit in enumerate(hits):
            if hit and not done[i]:
                rew[i] -= cfg.collision_penalty
                done[i] = True
                col = True

        if col and all(done):
            break

    if cfg.joint_bonus > 0 and all(
        float(np.linalg.norm(goals[i] - states[i].position)) <= cfg.goal_radius
        for i in range(n)
    ):
        for i in range(n):
            rew[i] += cfg.joint_bonus

    return rec, rew


def _apply_grad(
    params: dict[str, np.ndarray],
    stats: Mapping[str, np.ndarray],
    batch: tuple[list, list, list, list, list],
    baseline: float,
    lr: float,
    sigma: float,
    m: dict[str, np.ndarray],
    v: dict[str, np.ndarray],
    t: int,
) -> tuple[float, int]:
    e, pd, ma, a, ret = batch
    if not e:
        return baseline, t
    ego = np.asarray(e)
    peers = np.asarray(pd)
    mask = np.asarray(ma)
    acts = np.asarray(a)
    returns = np.asarray(ret)
    baseline = 0.9 * baseline + 0.1 * float(returns.mean())
    adv = returns - baseline
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    mu = _mu_batch(params, stats, ego, peers, mask)
    dout = (-(adv[:, None]) * (acts - mu) / (sigma ** 2)) / len(ego)
    cache: dict[str, Any] = {}
    ego_n, peers_n = core.normalize_batch(ego, peers, mask, stats)
    core.forward(params, ego_n, peers_n, mask, cache)
    g = core.backward(params, cache, dout)
    t += 1
    for k in params:
        m[k] = 0.9 * m[k] + 0.1 * g[k]
        v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
        mh = m[k] / (1 - 0.9 ** t)
        vh = v[k] / (1 - 0.999 ** t)
        params[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
    return baseline, t


def train_from_config(
    cfg: ExperimentConfig | Mapping[str, Any],
    *,
    init_checkpoint: str | None = None,
    predictor_cfg: Mapping[str, Any] | None = None,
    iters: int = 80,
    episodes: int = 6,
    lr: float = 1e-3,
    sigma: float = 0.12,
    seed: int = 0,
    neighbor_dist: float = 15.0,
    interaction_radius: float = 4.0,
    joint_bonus: float = 10.0,
    collision_penalty: float = 8.0,
    episode_seeds: Sequence[int] | None = None,
    verbose: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if not isinstance(cfg, ExperimentConfig):
        cfg = ExperimentConfig.from_dict(dict(cfg))

    scenario = SCENARIO_REGISTRY.get(cfg.scenario["type"]).from_config(cfg.scenario)
    sim = cfg.simulator
    rl_cfg = _RlConfig(
        dt=float(sim.get("dt", 0.05)),
        max_steps=int(sim.get("max_steps", 1000)),
        max_accel=float(sim.get("max_accel", 6.0)),
        goal_radius=float(sim.get("goal_radius", 1.5)),
        drone_radius=float(sim.get("drone_radius", 0.4)),
        max_speed=float(cfg.planner.get("max_speed", 5.0)),
        neighbor_dist=neighbor_dist,
        interaction_radius=interaction_radius,
        joint_bonus=joint_bonus,
        collision_penalty=collision_penalty,
    )
    predictor = build_predictor(predictor_cfg) if predictor_cfg else None

    if init_checkpoint:
        params, stats = core.load_checkpoint(init_checkpoint)
        params = {k: v.copy() for k, v in params.items()}
    else:
        params = core.init_model(h=32, seed=seed)
        stats = {
            "em": np.zeros(3),
            "es": np.ones(3),
            "pm": np.zeros(core.PEER_DIM),
            "ps": np.ones(core.PEER_DIM),
        }

    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(val) for k, val in params.items()}
    rng = np.random.default_rng(seed)
    baseline = 0.0
    t = 0

    for it in range(iters):
        batch_e: list = []
        batch_p: list = []
        batch_m: list = []
        batch_a: list = []
        batch_r: list = []
        ep_rets: list[float] = []

        for ep in range(episodes):
            idx = it * episodes + ep
            if episode_seeds:
                ep_seed = int(episode_seeds[idx % len(episode_seeds)])
            else:
                ep_seed = seed + idx
            rec, rew = _episode(
                scenario, rl_cfg, params, stats, predictor,
                seed=ep_seed, sigma=sigma, rng=rng,
            )
            for i, r in rew.items():
                ep_rets.append(r)
                for row in rec[i]:
                    batch_e.append(row[0])
                    batch_p.append(row[1])
                    batch_m.append(row[2])
                    batch_a.append(row[3])
                    batch_r.append(r)

        baseline, t = _apply_grad(
            params, stats,
            (batch_e, batch_p, batch_m, batch_a, batch_r),
            baseline, lr, sigma, m, v, t,
        )
        if verbose and it % 10 == 0:
            print(f"  it{it:4d}  mean_return={np.mean(ep_rets):+.2f}", flush=True)

    return params, stats
