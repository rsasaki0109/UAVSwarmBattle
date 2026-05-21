"""Multi-drone runner + joint metrics + viz/anim."""

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


def test_multi_drone_runs_and_logs(tmp_path: Path) -> None:
    """Two drones, head-on; per-drone episode logs land alongside each other."""
    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_multi_drone.yaml")
    cfg.num_episodes = 2
    cfg.simulator["max_steps"] = 600
    run_dir = run_experiment(cfg, tmp_path / "multi")
    drone_logs = sorted(run_dir.glob("episode_*_drone_*.json"))
    # 2 episodes × 2 drones
    assert len(drone_logs) == 4
    # parent eval treats each drone-episode as its own row
    summary = evaluate_run(run_dir)
    assert summary["n_episodes"] == 4

    # at least one drone log should have a sane outcome string
    import json as _json
    log = _json.loads(drone_logs[0].read_text())
    assert log["outcome"] in {"success", "collision", "timeout"}
    assert "drone_id" in log["meta"]


def test_multi_drone_scenario_validates_drones() -> None:
    from uav_nav_lab.scenario import SCENARIO_REGISTRY

    cls = SCENARIO_REGISTRY.get("multi_drone_grid")
    with pytest.raises(ValueError):
        # missing `drones` block must be rejected
        cls.from_config({"size": [10, 10], "obstacles": {"type": "none"}})


def test_multi_drone_joint_metrics_in_summary(tmp_path: Path) -> None:
    """Joint episode summaries are picked up and aggregated separately from
    the per-drone-episode rows."""
    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_multi_drone.yaml")
    cfg.num_episodes = 2
    cfg.simulator["max_steps"] = 600
    run_dir = run_experiment(cfg, tmp_path / "multi_joint")
    # joint files exist alongside per-drone trajectories
    assert sorted(p.name for p in run_dir.glob("episode_*_joint.json")) == [
        "episode_000_joint.json",
        "episode_001_joint.json",
    ]
    summary = evaluate_run(run_dir)
    # per-drone-episode rows: 2 episodes × 2 drones
    assert summary["n_episodes"] == 4
    # joint rows: 2 episodes
    assert summary["joint_n_episodes"] == 2
    assert summary["joint_n_drones"] == 2
    assert "joint_success_rate" in summary
    assert "joint_collision_ci95" in summary


def test_summary_includes_planner_dt_compute_metrics(tmp_path: Path) -> None:
    """The recorder logs `planner_dt_ms` per replan; eval must aggregate
    that into mean / p95 / max compute cost so compute-budget studies
    do not need a second pass over the raw episode logs."""
    cfg = _basic_cfg()
    cfg.num_episodes = 2
    cfg.simulator["max_steps"] = 200
    run_dir = run_experiment(cfg, tmp_path / "compute")
    summary = evaluate_run(run_dir)
    for key in ("planner_dt_ms_mean", "planner_dt_ms_p95", "planner_dt_ms_max"):
        assert key in summary, f"summary missing {key}"
        assert summary[key]["mean"] >= 0.0
        # consistency across statistics: max ≥ p95 ≥ mean
    assert summary["planner_dt_ms_max"]["mean"] >= summary["planner_dt_ms_p95"]["mean"]
    assert summary["planner_dt_ms_p95"]["mean"] >= summary["planner_dt_ms_mean"]["mean"]


def test_multi_drone_viz_groups_drones_per_episode(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from uav_nav_lab.viz import viz_run

    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_multi_drone.yaml")
    cfg.num_episodes = 2
    cfg.simulator["max_steps"] = 400
    run_dir = run_experiment(cfg, tmp_path / "multi_viz")
    saved = viz_run(run_dir)
    # one PNG per episode (not per drone)
    assert len(saved) == 2
    for p in saved:
        assert p.exists() and p.stat().st_size > 0


def test_multi_drone_anim_groups_drones_per_episode(tmp_path: Path) -> None:
    """`uav-nav anim` on a multi-drone run dispatches to the multi-drone
    animator: one GIF per episode (not per drone), all N drone trajectories
    rendered together with a per-drone palette colour. Mirrors the
    `viz_run` test for parity."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    from uav_nav_lab.anim import viz_anim

    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_multi_drone.yaml")
    cfg.num_episodes = 1
    cfg.simulator["max_steps"] = 100   # very short — keep test fast
    run_dir = run_experiment(cfg, tmp_path / "multi_anim")
    saved = viz_anim(run_dir, fps=10)
    # one GIF per episode (not per drone)
    assert len(saved) == 1
    p = saved[0]
    assert p.suffix == ".gif"
    assert p.stat().st_size > 1000  # non-empty animation


def test_multi_drone_voxel_anim_groups_drones_per_episode(tmp_path: Path) -> None:
    """3D multi-drone anim path: one GIF per episode (not per drone), and
    the new 3D animator is dispatched for ndim==3 multi scenarios."""
    _require_mplot3d()
    pytest.importorskip("PIL")
    from uav_nav_lab.anim import viz_anim
    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_multi_drone_3d_2.yaml")
    cfg.num_episodes = 1
    cfg.simulator["max_steps"] = 100
    run_dir = run_experiment(cfg, tmp_path / "multi_3d_anim")
    saved = viz_anim(run_dir, fps=10)
    assert len(saved) == 1
    p = saved[0]
    assert p.suffix == ".gif"
    assert p.stat().st_size > 1000


def test_multi_drone_voxel_runs_and_logs(tmp_path: Path) -> None:
    """End-to-end smoke: one episode of the 3D 2-drone YAML produces
    per-drone JSON logs and a joint-summary file, mirroring the 2D path."""
    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_multi_drone_3d_2.yaml")
    cfg.num_episodes = 1
    cfg.simulator["max_steps"] = 300
    run_dir = run_experiment(cfg, tmp_path / "multi_3d")
    drone_logs = sorted(run_dir.glob("episode_*_drone_*.json"))
    assert len(drone_logs) == 2  # 1 episode × 2 drones
    assert (run_dir / "episode_000_joint.json").exists()
