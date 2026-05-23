"""Per-episode orchestrator for multi-drone runs.

:func:`run_episode_multi` is the main loop. The per-drone, per-tick
helpers (replan, log-step) live in :mod:`.phases`; the end-of-step
state-machine helpers (outcome resolution, master hand-off,
finalize-timeouts) live in :mod:`.outcomes`. The orchestrator keeps
only the setup helpers it uses once (drone radius, reset) plus the
main loop itself.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ...planner import Plan, Planner
from ...recorder import EpisodeRecorder
from ..experiment import _follow_plan
from .outcomes import _finalize_timeouts, _handoff_master, _resolve_outcomes
from .peers import _check_peer_collision
from .phases import _log_step_for_drone, _replan_one_drone


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
