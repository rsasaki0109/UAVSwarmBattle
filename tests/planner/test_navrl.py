"""Smoke tests for NavRL planner adapter (skipped without upstream clone)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
NAVRL_ROOT = ROOT / "third_party" / "NavRL"
CKPT = NAVRL_ROOT / "quick-demos" / "ckpts" / "navrl_checkpoint.pt"


pytestmark = pytest.mark.skipif(
    not CKPT.is_file(),
    reason="NavRL not installed (run scripts/setup_navrl_adapter.sh)",
)


def test_navrl_registry_plan():
    torch = pytest.importorskip("torch")
    del torch  # noqa: F841 — ensure import works

    from uav_nav_lab.planner import PLANNER_REGISTRY

    p = PLANNER_REGISTRY.get("navrl").from_config(
        {"navrl_root": str(NAVRL_ROOT), "max_speed": 5.0, "device": "cpu"},
    )
    p.reset()
    p.set_current_state(np.array([10.0, 10.0]), np.zeros(2))
    plan = p.plan(
        np.array([10.0, 10.0]),
        np.array([40.0, 10.0]),
        None,
        dynamic_obstacles=[
            {"position": [20.0, 10.0], "velocity": [0.0, 0.0], "radius": 0.4, "goal": [30.0, 10.0]},
        ],
    )
    assert plan.target_velocity is not None
    assert plan.target_velocity.shape == (2,)
    assert plan.meta.get("planner") == "navrl"
