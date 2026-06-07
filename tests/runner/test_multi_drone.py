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


# ---------------------------------------------------------------------------
# Characterization tests for the helpers extracted by the multi.py split.
# Lock down current behaviour so the refactor preserves the contract.
# ---------------------------------------------------------------------------


def test_peers_view_filters_self_and_zeros_finished_velocity() -> None:
    """`_peers_view` returns the *other* drones' true poses, with finished
    peers reported at zero velocity (the simplest reasonable 'stuck' model)."""
    from types import SimpleNamespace
    from uav_nav_lab.runner.multi import _peers_view

    states = [
        SimpleNamespace(position=np.array([0.0, 0.0]), velocity=np.array([1.0, 2.0])),
        SimpleNamespace(position=np.array([5.0, 5.0]), velocity=np.array([3.0, 4.0])),
        SimpleNamespace(position=np.array([9.0, 9.0]), velocity=np.array([7.0, 8.0])),
    ]
    radii = [0.4, 0.5, 0.6]
    finished = [False, True, False]

    peers = _peers_view(states, radii, finished, me=0)

    # me=0 filtered out → 2 peers remain.
    assert len(peers) == 2
    # peer at index 1 is finished → velocity zeroed.
    assert peers[0]["position"] == [5.0, 5.0]
    assert peers[0]["velocity"] == [0.0, 0.0]
    assert peers[0]["radius"] == 0.5
    # peer at index 2 is alive → velocity passed through.
    assert peers[1]["position"] == [9.0, 9.0]
    assert peers[1]["velocity"] == [7.0, 8.0]
    assert peers[1]["radius"] == 0.6


def test_check_peer_collision_pairwise_inflation() -> None:
    """Two drones within (drone_radius + peer_radius) overlap → both flagged."""
    from types import SimpleNamespace
    from uav_nav_lab.runner.multi import _check_peer_collision

    # Drones 0 and 1 are at distance 0.6 with combined radii 0.7 → collide.
    # Drones 0 and 2 are at distance 5 with combined radii 0.7 → safe.
    states = [
        SimpleNamespace(position=np.array([0.0, 0.0])),
        SimpleNamespace(position=np.array([0.6, 0.0])),
        SimpleNamespace(position=np.array([5.0, 0.0])),
    ]
    radii = [0.35, 0.35, 0.35]
    hit = _check_peer_collision(states, radii, drone_radius=0.35)
    assert hit == [True, True, False]


def test_build_multi_rejects_vehicles_count_mismatch() -> None:
    """`simulator.vehicles` must match `scenario.n_drones` if provided."""
    from uav_nav_lab.runner.multi import _build_multi

    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_multi_drone.yaml")
    cfg.simulator["vehicles"] = ["Drone1"]  # scenario has 2 drones
    with pytest.raises(ValueError, match="simulator.vehicles"):
        _build_multi(cfg)


def test_build_multi_assigns_per_drone_components() -> None:
    """`_build_multi` returns (scenario, sims, planners, sensors) with
    one entry per drone, and only sim 0 advances the shared scenario."""
    from uav_nav_lab.runner.multi import _build_multi

    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_multi_drone.yaml")
    scenario, sims, planners, sensors = _build_multi(cfg)
    n = scenario.n_drones
    assert len(sims) == n
    assert len(planners) == n
    assert len(sensors) == n
    # Only sim 0 advances the scenario; the rest are passive.
    advance_flags = [getattr(s, "_advance_scenario", True) for s in sims]
    assert advance_flags[0] is True
    assert all(f is False for f in advance_flags[1:])


def test_build_multi_per_drone_planner_override() -> None:
    """`planner.per_drone` shallow-merges a partial override per drone, so a
    heterogeneous fleet can mix knobs and predictors; `{}` leaves the base."""
    from uav_nav_lab.runner.multi import _build_multi

    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_multi_drone.yaml")  # 2 drones
    base_speed = float(cfg.planner["max_speed"])
    cfg.planner["per_drone"] = [
        {"max_speed": 3.0, "predictor": {"type": "game_theoretic"}},
        {},  # drone 1 keeps the shared config unchanged
    ]
    _, _, planners, _ = _build_multi(cfg)

    # Drone 0 took the override; drone 1 kept the shared config.
    assert planners[0].max_speed == 3.0
    assert planners[1].max_speed == base_speed
    # Mixed predictors: the two drones forecast peers differently.
    assert type(planners[0]._predictor) is not type(planners[1]._predictor)
    # The override must not mutate the shared planner config in place.
    assert "max_speed" in cfg.planner and cfg.planner["max_speed"] == base_speed


def test_build_multi_rejects_per_drone_count_mismatch() -> None:
    """`planner.per_drone` must match `scenario.n_drones` if provided."""
    from uav_nav_lab.runner.multi import _build_multi

    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_multi_drone.yaml")  # 2 drones
    cfg.planner["per_drone"] = [{}]  # only 1 entry
    with pytest.raises(ValueError, match="planner.per_drone"):
        _build_multi(cfg)


def test_build_multi_per_drone_simulator_override() -> None:
    """`simulator.per_drone` shallow-merges a partial sim override per drone, so a
    fleet can fly mixed dynamics (e.g. heterogeneous `max_accel`); `{}` leaves
    the base. Mirrors `planner.per_drone` but for the sim backend."""
    from uav_nav_lab.runner.multi import _build_multi

    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_multi_drone.yaml")  # 2 drones
    base_accel = float(cfg.simulator.get("max_accel", 50.0))
    cfg.simulator["per_drone"] = [
        {"max_accel": 2.0},
        {},  # drone 1 keeps the shared sim config unchanged
    ]
    _, sims, _, _ = _build_multi(cfg)

    # Drone 0 took the sluggish override; drone 1 kept the shared accel limit.
    assert sims[0].p.max_accel == 2.0
    assert sims[1].p.max_accel == base_accel
    # The override must not mutate the shared simulator config in place.
    assert "per_drone" in cfg.simulator
    assert cfg.simulator.get("max_accel", 50.0) == base_accel


def test_build_multi_rejects_per_drone_simulator_count_mismatch() -> None:
    """`simulator.per_drone` must match `scenario.n_drones` if provided."""
    from uav_nav_lab.runner.multi import _build_multi

    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_multi_drone.yaml")  # 2 drones
    cfg.simulator["per_drone"] = [{}]  # only 1 entry
    with pytest.raises(ValueError, match="simulator.per_drone"):
        _build_multi(cfg)
