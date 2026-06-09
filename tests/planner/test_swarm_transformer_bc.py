"""Smoke test for framework-scale BC collection."""

import numpy as np

from uav_nav_lab.planner.swarm_transformer_bc import collect_from_yaml


def test_collect_antipodal_orca_layout():
    egos, peers, masks, acts = collect_from_yaml(
        "examples/exp_multi_drone_antipodal_orca.yaml",
        n_episodes=2,
        seed0=7,
    )
    assert len(egos) > 100
    assert peers.shape[1:] == (8, 10)
    assert masks.shape[1] == 8
    assert acts.shape[1] == 2
    assert float(np.abs(acts).max()) <= 1.05
