"""Smoke tests for the swarm_transformer planner."""

from __future__ import annotations

import numpy as np

from uav_nav_lab.planner import PLANNER_REGISTRY
from uav_nav_lab.planner import swarm_transformer_core as core


def _write_dummy_checkpoint(path):
    params = core.init_model(h=16, seed=0)
    stats = {
        "em": np.zeros(3),
        "es": np.ones(3),
        "pm": np.zeros(core.PEER_DIM),
        "ps": np.ones(core.PEER_DIM),
    }
    core.save_checkpoint(path, params, stats)


def test_registry_and_plan(tmp_path):
    ckpt = tmp_path / "xf.npz"
    _write_dummy_checkpoint(ckpt)
    p = PLANNER_REGISTRY.get("swarm_transformer").from_config({
        "checkpoint": str(ckpt),
        "max_speed": 1.0,
        "neighbor_dist": 15.0,
    })
    assert "swarm_transformer" in PLANNER_REGISTRY.names()
    p.set_current_state(np.array([0.0, 0.0]), np.array([0.1, 0.0]))
    peers = [{
        "position": [3.0, 0.5],
        "velocity": [-0.2, 0.0],
        "radius": 0.4,
        "goal": [10.0, 0.0],
    }]
    plan = p.plan(
        np.array([0.0, 0.0]),
        np.array([10.0, 0.0]),
        None,
        dynamic_obstacles=peers,
    )
    assert not plan.is_empty
    assert plan.target_velocity is not None
    assert plan.target_velocity.shape == (2,)
    assert plan.meta["n_tokens"] == 1


def test_role_split_threat_token(tmp_path):
    ckpt = tmp_path / "xf.npz"
    _write_dummy_checkpoint(ckpt)
    ego, peers, mask, _ = core.build_tokens(
        np.array([0.0, 0.0]),
        np.zeros(2),
        np.array([10.0, 0.0]),
        [
            {"position": [2.0, 0.0], "velocity": [0.0, 0.0], "radius": 0.5, "goal": [1.0, 1.0]},
            {"position": [1.0, 1.0], "velocity": [0.5, 0.0], "radius": 0.5},
        ],
        neighbor_dist=20.0,
        interaction_radius=4.0,
    )
    assert int(mask.sum()) == 2
    assert peers.shape[1] == core.PEER_DIM
    assert peers[0, 5] == core.ROLE_ALLY
    assert peers[1, 5] == core.ROLE_THREAT
    assert peers[0, 6] != 0.0 or peers[0, 7] != 0.0
    assert ego.shape == (3,)
