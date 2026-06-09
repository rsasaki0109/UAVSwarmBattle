"""Framework-scale BC dataset collection for swarm_transformer.

Rolls out a teacher planner (default: ORCA + lateral_bias) on the same
``multi_drone_grid`` geometry and ``dummy_2d`` physics as the YAML experiments,
then packages ego / teammate tokens for transformer training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..config import ExperimentConfig
from ..predictor import build_predictor
from ..runner.multi.peers import _check_peer_collision, _peers_view
from ..scenario import SCENARIO_REGISTRY
from . import PLANNER_REGISTRY
from . import swarm_transformer_core as core


@dataclass
class _DroneState:
    position: np.ndarray
    velocity: np.ndarray


def _step_velocity(
    pos: np.ndarray,
    vel: np.ndarray,
    cmd: np.ndarray,
    *,
    dt: float,
    max_accel: float,
) -> tuple[np.ndarray, np.ndarray]:
    dv = np.asarray(cmd, dtype=float)[:2] - vel
    max_dv = max_accel * dt
    n = float(np.linalg.norm(dv))
    if n > max_dv:
        dv *= max_dv / n
    new_vel = vel + dv
    return pos + new_vel * dt, new_vel


def _build_teachers(n: int, teacher_cfg: Mapping[str, Any]) -> list[Any]:
    """One teacher instance per drone (stateful planners must not be shared)."""
    tcfg = dict(teacher_cfg)
    ptype = str(tcfg.pop("type", "orca"))
    cls = PLANNER_REGISTRY.get(ptype)
    return [cls.from_config(dict(tcfg)) for _ in range(n)]


def _default_teacher_cfg(max_speed: float = 5.0, lateral_bias: float = 0.2) -> dict[str, Any]:
    return {
        "type": "orca",
        "max_speed": max_speed,
        "radius": 0.4,
        "time_horizon": 2.0,
        "time_step": 0.25,
        "neighbor_dist": 15.0,
        "safety_margin": 0.1,
        "goal_radius": 1.5,
        "lateral_bias": lateral_bias,
    }


def collect_from_config(
    cfg: ExperimentConfig | Mapping[str, Any],
    *,
    teacher_cfg: Mapping[str, Any] | None = None,
    predictor_cfg: Mapping[str, Any] | None = None,
    n_episodes: int = 200,
    seed0: int = 0,
    neighbor_dist: float = 15.0,
    interaction_radius: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect BC tuples from a multi-drone YAML config."""
    if not isinstance(cfg, ExperimentConfig):
        cfg = ExperimentConfig.from_dict(dict(cfg))

    scenario = SCENARIO_REGISTRY.get(cfg.scenario["type"]).from_config(cfg.scenario)
    sim_cfg = cfg.simulator
    dt = float(sim_cfg.get("dt", 0.05))
    max_steps = int(sim_cfg.get("max_steps", 1000))
    max_accel = float(sim_cfg.get("max_accel", 6.0))
    goal_radius = float(sim_cfg.get("goal_radius", 1.5))
    drone_radius = float(sim_cfg.get("drone_radius", 0.4))

    tcfg = dict(teacher_cfg or _default_teacher_cfg())
    max_speed = float(tcfg.get("max_speed", 5.0))
    token_predictor = build_predictor(predictor_cfg) if predictor_cfg else None

    n = scenario.n_drones
    teachers = _build_teachers(n, tcfg)
    radii = [float(d.radius) for d in scenario.drones]
    goals = [np.asarray(d.goal, dtype=float)[:2] for d in scenario.drones]

    egos: list[np.ndarray] = []
    peerl: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    acts: list[np.ndarray] = []

    for ep in range(n_episodes):
        seed = seed0 + ep
        scenario.reseed(seed)
        starts = scenario.episode_drone_starts(seed)
        states = [_DroneState(s[:2].copy(), np.zeros(2)) for s in starts]
        done = [False] * n
        for tch in teachers:
            tch.reset()
            if hasattr(tch, "seed_episode"):
                tch.seed_episode(seed)
        if token_predictor is not None:
            token_predictor.reset(seed=seed)

        for _ in range(max_steps):
            if all(done):
                break

            peer_states = [
                type("S", (), {"position": st.position, "velocity": st.velocity})()
                for st in states
            ]
            scenario.set_targets([st.position for st in states])
            scene_dyn = list(scenario.dynamic_obstacles)

            cmds: list[np.ndarray | None] = [None] * n
            for i in range(n):
                if done[i]:
                    continue
                peer_dyn = _peers_view(
                    peer_states, radii, done, me=i, goals=goals,
                )
                dyn = scene_dyn + peer_dyn
                teachers[i].set_current_state(states[i].position, states[i].velocity)
                plan = teachers[i].plan(
                    states[i].position,
                    goals[i],
                    scenario.occupancy,
                    dynamic_obstacles=dyn,
                )
                v_world = np.asarray(plan.target_velocity, dtype=float)[:2]
                ego, pad, mask, rot = core.build_tokens(
                    states[i].position,
                    states[i].velocity,
                    goals[i],
                    dyn,
                    neighbor_dist=neighbor_dist,
                    interaction_radius=interaction_radius,
                    predictor=token_predictor,
                )
                v_ego = rot @ v_world
                egos.append(ego)
                peerl.append(pad)
                masks.append(mask)
                acts.append(v_ego / max_speed)
                cmds[i] = v_world

            scenario.advance(dt)
            for i in range(n):
                if done[i] or cmds[i] is None:
                    continue
                states[i].position, states[i].velocity = _step_velocity(
                    states[i].position,
                    states[i].velocity,
                    cmds[i],
                    dt=dt,
                    max_accel=max_accel,
                )
                if scenario.is_collision(states[i].position, radii[i]):
                    done[i] = True
                elif float(np.linalg.norm(goals[i] - states[i].position)) <= goal_radius:
                    done[i] = True

            hits = _check_peer_collision(peer_states, radii, drone_radius)
            for i, hit in enumerate(hits):
                if hit:
                    done[i] = True

    return (
        np.asarray(egos),
        np.asarray(peerl),
        np.asarray(masks),
        np.asarray(acts),
    )


def collect_from_yaml(
    path: str,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return collect_from_config(ExperimentConfig.from_yaml(path), **kwargs)
