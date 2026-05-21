"""Shared helpers for the 2D / 3D × single / multi animators.

Four things used to live (drifted!) inside every animator function:

* the matplotlib lazy-import shim (4×),
* the sim-fps → render-fps frame down-sampler (4×),
* the dynamic-obstacle replay for 3D scenes (2×),
* the "most recent replan with rollouts at or before t" lookup (2×).

Pulled out here so the per-cell modules stay focused on the artist /
update closures specific to their projection and drone count.
"""

from __future__ import annotations

from typing import Any


PALETTE: list[str] = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:olive",
]


def need_mpl_anim() -> tuple[Any, Any]:
    """Return ``(plt, animation)`` after forcing the headless Agg backend.

    Same shim every animator needs. ``SystemExit`` (not ``ImportError``)
    is raised so the CLI prints a clear install hint instead of a stack
    trace.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
        return plt, animation
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for `uav-nav anim`. "
            "Install with: pip install -e '.[viz]'"
        ) from e


def frame_indices_for_episode(n_steps: int, dt: float, fps: int) -> list[int]:
    """Down-sample ``sim_fps = 1/dt`` to the requested render ``fps``.

    Always includes the final step so animations end on the recorded
    outcome rather than a step short of it. Returns ``[]`` when the
    episode logged zero steps.
    """
    if n_steps <= 0:
        return []
    sim_fps = 1.0 / dt
    stride = max(1, int(round(sim_fps / fps)))
    frame_indices = list(range(0, n_steps, stride))
    if frame_indices[-1] != n_steps - 1:
        frame_indices.append(n_steps - 1)
    return frame_indices


def dynamic_obstacle_positions_at(
    j: int,
    dyn_specs: list[dict],
    dt: float,
    bounds: tuple[float, float, float],
) -> tuple[list[float], list[float], list[float]]:
    """Re-derive ``scenario.dynamic_obstacles`` position at step ``j``.

    The 3D animators don't ``advance()`` the scenario per frame (it is
    expensive on voxel worlds), so they reconstruct each obstacle's
    deterministic motion analytically from its ``start / velocity /
    reflect`` spec. ``bounds`` is the ``(world_x, world_y, world_z)``
    extent used for reflection.
    """
    if not dyn_specs:
        return [], [], []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for spec in dyn_specs:
        pos = list(map(float, spec["start"]))
        vel = list(map(float, spec.get("velocity", [0, 0, 0])))
        reflect = bool(spec.get("reflect", True))
        for _ in range(j):
            for k in range(3):
                pos[k] += vel[k] * dt
                if reflect:
                    if pos[k] < 0:
                        pos[k] = -pos[k]
                        vel[k] = -vel[k]
                    elif pos[k] > bounds[k]:
                        pos[k] = 2 * bounds[k] - pos[k]
                        vel[k] = -vel[k]
        xs.append(pos[0])
        ys.append(pos[1])
        zs.append(pos[2])
    return xs, ys, zs


def replan_at_or_before(replans: list[dict], cur_t: float) -> dict | None:
    """Most recent replan with non-empty ``rollouts`` at or before ``cur_t``.

    Returns ``None`` if no such replan exists (e.g. early in the
    episode, or a non-sampling planner that never logs rollouts).
    """
    chosen: dict | None = None
    for r in replans:
        if r["t"] > cur_t + 1e-9:
            break
        if r.get("rollouts") is not None:
            chosen = r
    return chosen
