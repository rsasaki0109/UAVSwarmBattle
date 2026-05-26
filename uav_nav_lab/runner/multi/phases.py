"""Per-drone, per-tick phase helpers used by :mod:`.episode`.

The runner's main loop in :func:`.episode.run_episode_multi` calls
these once per drone per step:

- :func:`_replan_one_drone`  — perception + planner.plan() for one drone.
- :func:`_log_step_for_drone` — recorder.log_step + optional frame dump
  for one drone (used by both two-phase and single-phase stepping).

These were extracted out of ``episode.py`` so the orchestrator stays
small and so each phase can be read (and, eventually, tested) without
holding the whole loop in your head.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...planner import Plan, Planner
from ...recorder import EpisodeRecorder
from .peers import _peers_view


def _replan_one_drone(
    *,
    drone_idx: int,
    t: float,
    scenario: Any,
    sims: list[Any],
    planners: list[Planner],
    sensors: list[Any],
    states: list[Any],
    finished: list[bool],
    radii: list[float],
    obs_i: Any,
) -> Plan:
    """Run one drone's perception + replan and return the new plan.

    Side-effect-free w.r.t. peers and the loop's `plans` list — the
    caller stores the returned plan back into `plans[i]` and updates
    `last_replan_t[i]`.
    """
    perceived_map = sensors[drone_idx].observe_map(
        t, states[drone_idx].position, sims[drone_idx].obstacle_map,
        sim_extra=states[drone_idx].extra or None,
    )
    scenario_dyn = scenario.dynamic_obstacles
    peer_dyn = _peers_view(states, radii, finished, me=drone_idx)
    # Filter through the sensor — a range-limited sensor will drop
    # peers / scene-dyn obstacles beyond its range.
    perceived_dyn = sensors[drone_idx].observe_dynamics(
        t, states[drone_idx].position, scenario_dyn + peer_dyn
    )
    # Aerobatic / choreography scenarios provide a *time-varying* goal
    # via `dynamic_goal_at(drone_idx, t)`; non-aerobatic scenarios use
    # the static sim.goal as before.
    if hasattr(scenario, "dynamic_goal_at"):
        cur_goal = scenario.dynamic_goal_at(drone_idx, t)
        sims[drone_idx].set_goal(cur_goal)
    else:
        cur_goal = sims[drone_idx].goal
    planners[drone_idx].set_current_state(
        states[drone_idx].position,
        states[drone_idx].velocity,
    )
    return planners[drone_idx].plan(
        obs_i,
        cur_goal,
        perceived_map,
        dynamic_obstacles=perceived_dyn,
    )


def _log_step_for_drone(
    *,
    drone_idx: int,
    step: int,
    t: float,
    scenario: Any,
    states: list[Any],
    new_states: list[Any],
    observations: list[Any],
    cmd: np.ndarray,
    info: Any,
    recorder: EpisodeRecorder,
    frame_dirs: list[Path | None] | None,
) -> None:
    """Common log + frame-dump path for both two-phase and single-phase steps."""
    ref_pos = (
        scenario.reference_position(drone_idx, t)
        if hasattr(scenario, "reference_position")
        else None
    )
    ns = new_states[drone_idx]
    recorder.log_step(
        t=t,
        true_pos=states[drone_idx].position,
        true_vel=states[drone_idx].velocity,
        observed_pos=observations[drone_idx],
        cmd=cmd,
        info={"collision": info.collision, "goal_reached": info.goal_reached},
        sim_extra=dict(ns.extra) if ns.extra else None,
        reference_pos=ref_pos,
    )
    if frame_dirs is not None and frame_dirs[drone_idx] is not None and ns.extra:
        cam_imgs = ns.extra.get("camera_images") or {}
        for cam_name, png_bytes in cam_imgs.items():
            if not png_bytes:
                continue
            fname = f"step_{step:04d}_{cam_name}.png"
            (frame_dirs[drone_idx] / fname).write_bytes(bytes(png_bytes))
