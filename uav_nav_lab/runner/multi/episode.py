"""Per-episode loop for multi-drone runs.

`run_episode_multi` orchestrates the per-step loop. The previously
in-lined sub-phases — single-drone replan, per-drone log-step (with
camera-frame dump), master hand-off, outcome resolution — are extracted
as small helpers so each phase is independently readable and testable.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ...planner import Plan, Planner
from ...recorder import EpisodeRecorder
from ..experiment import _follow_plan
from .peers import _check_peer_collision, _peers_view


def _drone_radius_from_sim(sim: Any, fallback_radii: list[float]) -> float:
    """`dummy_*` carries the drone radius on `.p.drone_radius`; airsim uses
    scenario.drones[0].radius. Fall back through both, then to 0.4 m as the
    framework default for peer-collision math."""
    return float(
        getattr(getattr(sim, "p", None), "drone_radius", None)
        or (fallback_radii[0] if fallback_radii else 0.4)
    )


def _reset_drones(
    scenario: Any,
    sims: list[Any],
    sensors: list[Any],
    planners: list[Planner],
    seed: int,
) -> list[Any]:
    """Reset all drones; only sim 0 reseeds the scenario (so the static
    layout is reproducible and consistent across all drones in this
    episode). Restore the master-advance flag in case the previous
    episode handed it away when its master died — without this, all
    dynamic obstacles freeze at their initial positions starting with
    the first episode that follows an all-drones-died episode.
    """
    for i, sim in enumerate(sims):
        if hasattr(sim, "_advance_scenario"):
            sim._advance_scenario = (i == 0)
    states: list[Any] = []
    for i, sim in enumerate(sims):
        s = sim.reset(
            seed=seed if i == 0 else None,
            initial_position=scenario.drones[i].start,
        )
        states.append(s)
        sensors[i].reset(seed=seed + 1000 * i)
        planners[i].reset()
        pred = getattr(planners[i], "_predictor", None)
        if pred is not None and hasattr(pred, "reset"):
            pred.reset(seed=seed + 7777 + 1000 * i)
    return states


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


def _handoff_master(sims: list[Any], finished: list[bool]) -> None:
    """Hand the scenario-clock master flag to the next live drone if the
    current master finished this tick.

    Backends with a shared physics clock (airsim) elect one bridge to
    advance time via ``_advance_scenario``; once that bridge finishes,
    the runner stops calling its step() — and the clock freezes for
    everyone else. This hand-off keeps the remaining drones flying.
    """
    n = len(sims)
    master_idx = next(
        (i for i in range(n) if getattr(sims[i], "_advance_scenario", False)),
        None,
    )
    if master_idx is not None and finished[master_idx]:
        sims[master_idx]._advance_scenario = False
        for i in range(n):
            if not finished[i] and hasattr(sims[i], "_advance_scenario"):
                sims[i]._advance_scenario = True
                break


def _resolve_outcomes(
    *,
    scenario: Any,
    new_states: list[Any],
    infos: list[Any],
    peer_hit: list[bool],
    finished: list[bool],
    final_states: list[Any],
    recorders: list[EpisodeRecorder],
) -> None:
    """Mark per-drone outcomes after a step.

    Aerobatic scenarios (`dynamic_goal_at`) suppress goal-reach
    termination — `goal_reached` ticks spuriously whenever the drone
    happens to be near the current lookahead point. Let max_steps end
    the episode instead; the eval is tracking-error based, not binary
    success.
    """
    _aerobatic = hasattr(scenario, "dynamic_goal_at")
    for i, info in enumerate(infos):
        if finished[i]:
            continue
        if info.collision or peer_hit[i]:
            recorders[i].set_outcome("collision", final_t=float(new_states[i].t))
            finished[i] = True
            final_states[i] = new_states[i]
            continue
        if info.goal_reached and not _aerobatic:
            recorders[i].set_outcome("success", final_t=float(new_states[i].t))
            finished[i] = True
            final_states[i] = new_states[i]
            continue


def _finalize_timeouts(
    *,
    scenario: Any,
    states: list[Any],
    finished: list[bool],
    final_states: list[Any],
    recorders: list[EpisodeRecorder],
) -> None:
    """Mark "success"/"timeout" for any drones still running at max_steps.

    Aerobatic / choreography scenarios reach max_steps as the natural
    completion condition (completed all loops without collision), so
    mark "success" rather than "timeout".
    """
    _aerobatic = hasattr(scenario, "dynamic_goal_at")
    for i in range(len(states)):
        if not finished[i]:
            outcome = "success" if _aerobatic else "timeout"
            recorders[i].set_outcome(outcome, final_t=float(states[i].t))
            final_states[i] = states[i]


def run_episode_multi(
    scenario: Any,
    sims: list[Any],
    planners: list[Planner],
    sensors: list[Any],
    *,
    seed: int,
    replan_period: float,
    max_steps: int,
    episode_index: int,
    frame_dirs: list[Path | None] | None = None,
) -> list[EpisodeRecorder]:
    n = len(sims)
    radii = [d.radius for d in scenario.drones]
    drone_radius = _drone_radius_from_sim(sims[0], radii)

    states = _reset_drones(scenario, sims, sensors, planners, seed)

    recorders = [
        EpisodeRecorder(episode_index=episode_index, seed=seed) for _ in range(n)
    ]
    for i, rec in enumerate(recorders):
        rec.meta["drone_id"] = i
        rec.meta["drone_name"] = scenario.drones[i].name

    plans: list[Plan | None] = [None] * n
    last_replan_t = [-float("inf")] * n
    finished = [False] * n
    final_states = list(states)

    t = 0.0
    for step in range(max_steps):
        # 1. observations + replanning (per drone, in parallel order)
        observations: list[Any] = []
        for i in range(n):
            obs_i = sensors[i].observe(t, states[i].position)
            observations.append(obs_i)
            if finished[i]:
                continue
            if plans[i] is None or (t - last_replan_t[i]) >= replan_period:
                t0 = time.perf_counter()
                plans[i] = _replan_one_drone(
                    drone_idx=i, t=t,
                    scenario=scenario, sims=sims, planners=planners,
                    sensors=sensors, states=states, finished=finished,
                    radii=radii, obs_i=obs_i,
                )
                planner_dt_ms = (time.perf_counter() - t0) * 1000.0
                last_replan_t[i] = t
                recorders[i].log_replan(
                    t=t, plan_length=int(plans[i].waypoints.shape[0]),
                    planner_dt_ms=planner_dt_ms,
                    rollouts=plans[i].meta.get("rollouts"),
                    best_rollout_idx=plans[i].meta.get("best_rollout_idx"),
                )

        # 2. step each drone's sim.
        # When the sim backend supports two-phase stepping
        # (step_command / step_readback), the runner issues commands in
        # passive-first order (passive drones queue moveByVelocityAsync
        # while the engine is paused, then the master unpauses, continues
        # time, and re-pauses) and reads every bridge's state afterward —
        # eliminating the 1-tick command lag that the original
        # master-first loop carried.
        _two_phase = all(
            hasattr(s, "step_command") and hasattr(s, "step_readback") for s in sims
        )
        new_states: list[Any] = list(states)
        infos: list[Any] = [None] * n
        if _two_phase:
            # — phase 1: passive-first command dispatch —
            cmds: list[np.ndarray | None] = [None] * n
            for i in range(n):
                if i == 0 or finished[i]:
                    continue
                cmds[i] = _follow_plan(
                    plans[i], observations[i], planners[i].max_speed,
                    t_since_replan=float(t - last_replan_t[i]),
                )
                sims[i].step_command(cmds[i])
            # Master (i=0) handles unpause → continue → pause before
            # queuing its own command, so every passive's queued velocity
            # is processed in the same tick.
            if not finished[0]:
                cmds[0] = _follow_plan(
                    plans[0], observations[0], planners[0].max_speed,
                    t_since_replan=float(t - last_replan_t[0]),
                )
                sims[0].step_command(cmds[0])
            # — phase 2: readback (all bridges, after time advance) —
            for i in range(n):
                if finished[i]:
                    continue
                ns, info = sims[i].step_readback()
                new_states[i] = ns
                infos[i] = info
                # Re-compute the cmd to log it; recreating here matches
                # the pre-split behaviour bit-for-bit (the previous code
                # also called _follow_plan a second time).
                cmd_for_log = _follow_plan(
                    plans[i], observations[i], planners[i].max_speed,
                    t_since_replan=float(t - last_replan_t[i]),
                )
                _log_step_for_drone(
                    drone_idx=i, step=step, t=t, scenario=scenario,
                    states=states, new_states=new_states,
                    observations=observations, cmd=cmd_for_log,
                    info=info, recorder=recorders[i],
                    frame_dirs=frame_dirs,
                )
        else:
            for i in range(n):
                if finished[i]:
                    continue
                cmd = _follow_plan(
                    plans[i], observations[i], planners[i].max_speed,
                    t_since_replan=float(t - last_replan_t[i]),
                )
                ns, info = sims[i].step(cmd)
                new_states[i] = ns
                infos[i] = info
                _log_step_for_drone(
                    drone_idx=i, step=step, t=t, scenario=scenario,
                    states=states, new_states=new_states,
                    observations=observations, cmd=cmd,
                    info=info, recorder=recorders[i],
                    frame_dirs=frame_dirs,
                )

        # 3. peer-vs-peer collision check on the freshly stepped positions
        peer_hit = _check_peer_collision(new_states, radii, drone_radius)

        # 4. resolve outcomes
        _resolve_outcomes(
            scenario=scenario, new_states=new_states, infos=infos,
            peer_hit=peer_hit, finished=finished,
            final_states=final_states, recorders=recorders,
        )

        # 5. master hand-off — keep the shared physics clock alive after
        # the current master finishes.
        _handoff_master(sims, finished)

        states = new_states
        # Use any unfinished drone's t to advance the global clock; if
        # the master finished this tick its `states[master].t` is frozen
        # at the final t and would stall the loop.
        live_t = next(
            (states[i].t for i in range(n) if not finished[i]),
            None,
        )
        if live_t is not None:
            t = live_t
        if all(finished):
            break

    _finalize_timeouts(
        scenario=scenario, states=states, finished=finished,
        final_states=final_states, recorders=recorders,
    )

    return recorders
