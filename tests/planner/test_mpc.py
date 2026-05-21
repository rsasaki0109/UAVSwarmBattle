"""MPC planner unit tests."""

from __future__ import annotations

import json  # noqa: F401
import subprocess  # noqa: F401
import sys  # noqa: F401
from pathlib import Path  # noqa: F401

import numpy as np  # noqa: F401
import pytest  # noqa: F401

from uav_nav_lab.cli import build_parser, main  # noqa: F401
from uav_nav_lab.config import ExperimentConfig  # noqa: F401
from uav_nav_lab.eval import evaluate_run  # noqa: F401
from uav_nav_lab.planner import PLANNER_REGISTRY  # noqa: F401
from uav_nav_lab.runner import expand_sweep, run_experiment  # noqa: F401

from tests._helpers import EXAMPLES, _basic_cfg, _require_mplot3d  # noqa: F401


def test_3d_mpc_runs(tmp_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_3d_mpc.yaml")
    cfg.num_episodes = 1
    cfg.simulator["max_steps"] = 400
    cfg.scenario["obstacles"]["count"] = 30  # keep it loose; MPC only needs to plan, not succeed
    cfg.planner["n_samples"] = 16
    run_dir = run_experiment(cfg, tmp_path / "3d_mpc")
    summary = evaluate_run(run_dir)
    assert summary["n_episodes"] == 1


def test_mpc_runs(tmp_path: Path) -> None:
    cfg = _basic_cfg()
    cfg.planner = {"type": "mpc", "max_speed": 5.0, "replan_period": 0.5, "horizon": 30}
    cfg.num_episodes = 1
    cfg.simulator["max_steps"] = 600
    run_dir = run_experiment(cfg, tmp_path / "mpc")
    summary = evaluate_run(run_dir)
    assert summary["n_episodes"] == 1


def test_mpc_prediction_changes_action() -> None:
    """A rollout that reaches the goal but passes through a *predicted*
    obstacle position should be rejected when use_prediction=True. Without
    prediction the planner picks the goal-direction action even though it
    walks through the obstacle's future location."""
    from uav_nav_lab.planner.mpc import SamplingMPCPlanner

    occ = np.zeros((30, 30), dtype=bool)
    goal = np.array([20.0, 5.0])
    pos = np.array([2.0, 5.0])
    # Cross-path threat: at h=20 (1.0s @ dt_plan=0.05) it sits on the
    # straight-line drone path between (10, 5) and (12, 5).
    dyn = [{"position": [11.0, 12.0], "velocity": [0.0, -7.0], "radius": 1.5}]

    args = dict(max_speed=10.0, horizon=40, dt_plan=0.05, n_samples=32, inflate=0,
                safety_margin=0.5)
    pw = SamplingMPCPlanner(use_prediction=True, **args)
    pw.reset()
    aw = pw.plan(pos, goal, occ, dynamic_obstacles=dyn).target_velocity

    pwo = SamplingMPCPlanner(use_prediction=False, **args)
    pwo.reset()
    awo = pwo.plan(pos, goal, occ, dynamic_obstacles=dyn).target_velocity

    # The "without prediction" planner has no reason to deviate from the
    # straight goal direction (1, 0) * max_speed.
    assert np.allclose(awo, [10.0, 0.0], atol=1e-3)
    # The "with prediction" planner must steer off-axis.
    assert not np.allclose(aw, awo, atol=0.1)


def test_mpc_uses_configured_predictor(tmp_path: Path) -> None:
    """MPC must accept a `planner.predictor` block and pass it through."""
    from uav_nav_lab.predictor import constant_velocity as cv
    from uav_nav_lab.predictor import noisy as ny

    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_predictor_noise.yaml")
    cfg.num_episodes = 1
    cfg.simulator["max_steps"] = 200
    cfg.planner["predictor"] = {"type": "noisy_velocity", "velocity_noise_std": 0.5}
    planner_cls = PLANNER_REGISTRY.get(cfg.planner["type"])
    p = planner_cls.from_config(cfg.planner)
    assert isinstance(p._predictor, ny.NoisyVelocityPredictor)

    cfg.planner["predictor"] = {"type": "constant_velocity"}
    p = planner_cls.from_config(cfg.planner)
    assert isinstance(p._predictor, cv.ConstantVelocityPredictor)
