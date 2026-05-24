"""Characterization tests for GPUMPPIPlanner.

Locks down observable plan() behaviour before the 603-line module is
split into a subpackage. Tests run on CPU when CUDA is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from uav_nav_lab.planner.gpu_mppi import GPUMPPIPlanner
from uav_nav_lab.planner.gpu_mppi import ctg_cache as ctg_cache_mod


def _free_grid(shape: tuple[int, ...] = (20, 20)) -> np.ndarray:
    return np.zeros(shape, dtype=bool)


def _basic_planner(**overrides) -> GPUMPPIPlanner:
    cfg = {
        "max_speed": 5.0,
        "horizon": 10,
        "dt_plan": 0.1,
        "n_samples": 64,
        "resolution": 1.0,
        "inflate": 0,
        "goal_radius": 1.0,
        "device": "cpu",
    }
    cfg.update(overrides)
    return GPUMPPIPlanner.from_config(cfg)


def test_gpu_mppi_plan_returns_required_meta_keys() -> None:
    planner = _basic_planner()
    plan = planner.plan(np.array([5.0, 5.0]), np.array([15.0, 15.0]), _free_grid())

    assert plan.waypoints.ndim == 2
    assert plan.waypoints.shape[1] == 2
    assert plan.target_velocity is not None
    assert plan.target_velocity.shape == (2,)
    for key in (
        "planner",
        "cost_min",
        "weight_max",
        "weight_entropy",
        "n_samples",
        "device",
        "rollouts",
        "best_rollout_idx",
        "fallback_to_argmin",
        "mode_aware_triggered",
        "mode_aware_cluster_sign",
    ):
        assert key in plan.meta, f"missing meta key: {key}"
    assert plan.meta["planner"] == "gpu_mppi"
    assert plan.meta["n_samples"] == 64
    # Default config: neither dispatch flag should fire.
    assert plan.meta["fallback_to_argmin"] is False
    assert plan.meta["mode_aware_triggered"] is False
    assert plan.meta["mode_aware_cluster_sign"] == 0


def test_gpu_mppi_from_config_roundtrip_preserves_knobs() -> None:
    cfg = {
        "max_speed": 7.5,
        "horizon": 30,
        "dt_plan": 0.04,
        "n_samples": 256,
        "resolution": 0.5,
        "inflate": 2,
        "goal_radius": 0.8,
        "safety_margin": 0.3,
        "temperature": 2.5,
        "fallback_to_argmin": True,
        "fallback_lateral_threshold": 0.7,
        "fallback_lateral_ratio": 0.4,
        "fallback_commit_steps": 4,
        "asymmetric_bias": 0.2,
        "mode_aware_sampling": True,
        "mode_aware_min_size": 5,
        "mode_aware_cost_ratio": 1.5,
        "mode_aware_lateral_threshold": 0.6,
        "mode_aware_lateral_ratio": 0.7,
        "ctg_cache_tolerance": 3,
        "viz_rollouts": 16,
        "log_action_provenance": True,
        "w_goal": 0.5,
        "w_obs": 50.0,
        "w_smooth": 0.1,
        "device": "cpu",
    }
    p = GPUMPPIPlanner.from_config(cfg)
    for key, expected in cfg.items():
        if key == "device":
            continue
        actual = getattr(p, key)
        assert actual == expected, f"{key}: expected {expected!r}, got {actual!r}"


def test_gpu_mppi_action_norm_does_not_exceed_max_speed() -> None:
    planner = _basic_planner(max_speed=3.0)
    plan = planner.plan(np.array([5.0, 5.0]), np.array([15.0, 15.0]), _free_grid())
    speed = float(np.linalg.norm(plan.target_velocity))
    assert speed <= 3.0 + 1e-6


def test_gpu_mppi_action_provenance_meta_is_opt_in() -> None:
    obs = np.array([5.0, 5.0])
    goal = np.array([15.0, 15.0])

    default_plan = _basic_planner().plan(obs, goal, _free_grid())
    assert "action_provenance" not in default_plan.meta

    plan = _basic_planner(log_action_provenance=True).plan(obs, goal, _free_grid())
    provenance = plan.meta["action_provenance"]
    assert provenance["action_source"] == "softmax"
    assert provenance["chosen_action"] == pytest.approx(plan.target_velocity.tolist())
    assert len(provenance["softmax_action"]) == 2
    assert len(provenance["argmax_weight_action"]) == 2
    assert len(provenance["argmin_action"]) == 2
    assert provenance["top_weighted_actions"]
    assert set(provenance["weight_mass_by_action_y_sign"]) == {
        "positive",
        "negative",
        "near_zero",
    }


def test_gpu_mppi_ctg_cache_reused_when_goal_cell_unchanged(monkeypatch) -> None:
    """Dijkstra runs once when consecutive replans share an integer goal cell."""
    calls = {"n": 0}
    real = ctg_cache_mod.dijkstra_cost_to_go

    def spy(occ, goal_cell):
        calls["n"] += 1
        return real(occ, goal_cell)

    monkeypatch.setattr(ctg_cache_mod, "dijkstra_cost_to_go", spy)

    planner = _basic_planner()
    planner.reset()
    obs = np.array([5.0, 5.0])
    goal = np.array([15.0, 15.0])
    for _ in range(3):
        planner.plan(obs, goal, _free_grid())
    assert calls["n"] == 1


def test_gpu_mppi_ctg_cache_respects_tolerance_window(monkeypatch) -> None:
    calls = {"n": 0}
    real = ctg_cache_mod.dijkstra_cost_to_go

    def spy(occ, goal_cell):
        calls["n"] += 1
        return real(occ, goal_cell)

    monkeypatch.setattr(ctg_cache_mod, "dijkstra_cost_to_go", spy)

    planner = _basic_planner(ctg_cache_tolerance=2)
    planner.reset()
    obs = np.array([5.0, 5.0])
    planner.plan(obs, np.array([15.0, 15.0]), _free_grid())  # recompute
    planner.plan(obs, np.array([16.0, 16.0]), _free_grid())  # 1-cell drift, cached
    planner.plan(obs, np.array([18.0, 18.0]), _free_grid())  # 3-cell drift, recompute
    assert calls["n"] == 2


def test_gpu_mppi_asymmetric_bias_is_deterministic_per_observation() -> None:
    obs = np.array([5.0, 5.0])
    goal = np.array([15.0, 15.0])

    planner = _basic_planner(asymmetric_bias=0.3)
    planner.reset()
    planner.plan(obs, goal, _free_grid())
    bias_first = planner._bias_vec.copy()

    planner.reset()
    planner.plan(obs, goal, _free_grid())
    bias_second = planner._bias_vec.copy()

    assert np.allclose(bias_first, bias_second)
    assert float(np.linalg.norm(bias_first)) == pytest.approx(1.0, abs=1e-6)


def test_gpu_mppi_mode_aware_sampling_meta_flag_fires_when_enabled() -> None:
    """Default mode-aware thresholds (cost_ratio=1.0, lateral_threshold=0.0)
    trigger as soon as L/R clusters meet the min-size guard."""
    planner = _basic_planner(
        mode_aware_sampling=True,
        mode_aware_min_size=4,
        n_samples=128,
    )
    plan = planner.plan(np.array([5.0, 5.0]), np.array([15.0, 15.0]), _free_grid())
    assert plan.meta["mode_aware_triggered"] is True
    assert plan.meta["mode_aware_cluster_sign"] in (-1, 1)


def test_gpu_mppi_short_circuits_when_at_goal() -> None:
    """Observation ≈ goal → single-waypoint plan with no rollouts."""
    planner = _basic_planner()
    obs = np.array([15.0, 15.0])
    plan = planner.plan(obs, obs.copy(), _free_grid())
    assert plan.waypoints.shape == (1, 2)
    assert plan.meta["planner"] == "gpu_mppi"
