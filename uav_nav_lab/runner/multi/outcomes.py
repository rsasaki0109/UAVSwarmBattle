"""End-of-step state-machine helpers used by :mod:`.episode`.

After each step, the runner needs to:

1. transfer the scenario-clock master if the current master finished
   (:func:`_handoff_master`),
2. classify per-drone outcomes (collision / goal-reach / still-running)
   from the new states + collision flags (:func:`_resolve_outcomes`),
3. at episode end, mark remaining live drones as "timeout" (or
   "success" for aerobatic loops that complete by running out the
   clock) (:func:`_finalize_timeouts`).

Pulled out of ``episode.py`` so the orchestrator's main loop reads as
"observation → replan → step → outcome resolution → master hand-off"
without 90 LOC of branching in the middle.
"""

from __future__ import annotations

from typing import Any

from ...recorder import EpisodeRecorder


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
