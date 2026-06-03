"""Construct the per-drone (scenario, sims, planners, sensors) tuple.

Wires the YAML config to one sim / planner / sensor instance per drone,
binds optional ``simulator.vehicles[i]`` names (airsim only), and marks
sim 0 as the scenario-clock master in multi-drone runs.
"""

from __future__ import annotations

from typing import Any

from ...config import ExperimentConfig
from ...planner import PLANNER_REGISTRY, Planner
from ...scenario import SCENARIO_REGISTRY
from ...sensor import SENSOR_REGISTRY
from ...sim import SIM_REGISTRY


def _build_multi(
    cfg: ExperimentConfig,
) -> tuple[Any, list[Any], list[Planner], list[Any]]:
    scenario_cls = SCENARIO_REGISTRY.get(cfg.scenario.get("type", "multi_drone_grid"))
    scenario = scenario_cls.from_config(cfg.scenario)
    n = scenario.n_drones

    sim_cls = SIM_REGISTRY.get(cfg.simulator.get("type", "dummy_2d"))

    # Heterogeneous fleets: `planner.per_drone` is an optional list of partial
    # planner overrides applied by drone index — each entry is shallow-merged
    # over the shared `planner` config (`{}` = use the shared config unchanged).
    # This lets a multi-drone run mix predictors, planner types, or any planner
    # knob per drone (e.g. half the swarm on `game_theoretic`, half on
    # `constant_velocity`). When absent, every drone shares one planner config.
    base_planner = {k: v for k, v in cfg.planner.items() if k != "per_drone"}
    per_drone_planner = cfg.planner.get("per_drone") or []
    if per_drone_planner and len(per_drone_planner) != n:
        raise ValueError(
            f"planner.per_drone has {len(per_drone_planner)} entries but the "
            f"scenario has {n} drones"
        )

    sensor_cfg = dict(cfg.sensor)
    sensor_cfg.setdefault("dt", cfg.simulator.get("dt", 0.05))
    sensor_cls = SENSOR_REGISTRY.get(sensor_cfg.get("type", "perfect"))

    # Per-drone vehicle names — only the airsim bridge cares about this.
    # When `simulator.vehicles: [Drone1, Drone2, ...]` is provided, bridge i
    # is bound to vehicles[i]; otherwise every backend uses whatever default
    # the config carries (`vehicle: Drone1` for airsim, ignored for dummy /
    # ros2). Length must match `scenario.n_drones`.
    vehicles_cfg = list(cfg.simulator.get("vehicles", []))
    if vehicles_cfg and len(vehicles_cfg) != n:
        raise ValueError(
            f"simulator.vehicles has {len(vehicles_cfg)} entries but the "
            f"scenario has {n} drones"
        )
    sims: list[Any] = []
    planners: list[Planner] = []
    sensors: list[Any] = []
    for i in range(n):
        # Only sim 0 advances the shared scenario; the rest are passive.
        sim = sim_cls.from_config(cfg.simulator, scenario)
        if i > 0 and hasattr(sim, "_advance_scenario"):
            sim._advance_scenario = False
        if vehicles_cfg and hasattr(sim, "vehicle"):
            sim.vehicle = vehicles_cfg[i]
        sim.set_goal(scenario.drones[i].goal)
        sims.append(sim)
        pcfg = base_planner
        if per_drone_planner and per_drone_planner[i]:
            pcfg = {**base_planner, **per_drone_planner[i]}
        planner_cls = PLANNER_REGISTRY.get(pcfg.get("type", "straight"))
        planners.append(planner_cls.from_config(pcfg))
        sensors.append(sensor_cls.from_config(sensor_cfg))
    return scenario, sims, planners, sensors
