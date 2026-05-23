"""Joint success rates from runner output directories.

Thin wrappers around :func:`.joint_stats.load_joint_episodes` +
:func:`.joint_stats.wilson` for the recurring "give me a percentage
Wilson CI for this run directory" pattern that every research script
needs.

Kept separate from :mod:`.joint_stats` so the stdlib-only stats
primitives stay independently importable (no path / dataclass / numpy
dependency creep).
"""

from __future__ import annotations

from pathlib import Path

from .joint_stats import load_joint_episodes, wilson


def joint_outcomes(
    run_dir: str | Path,
    n_eps: int | None = None,
    layout: str = "flat",
) -> list[bool]:
    """Joint per-episode success booleans from a runner output dir.

    Returns ``[]`` for a non-existent directory. ``n_eps`` caps the
    returned list to the first N episodes (sorted by seed); leave it
    ``None`` to return every episode the loader finds.
    """
    p = Path(run_dir)
    if not p.exists():
        return []
    eps = load_joint_episodes(p, layout=layout)  # type: ignore[arg-type]
    if n_eps is not None:
        eps = eps[:n_eps]
    return [e["joint"] for e in eps]


def joint_success_rate(
    run_dir: str | Path,
    n_eps: int | None = None,
    z: float = 1.96,
    layout: str = "flat",
) -> tuple[float, float, float, int, int]:
    """Wilson CI in **percentage** form for a run directory.

    Returns ``(rate_pct, lo_pct, hi_pct, k, n)``. Returns
    ``(nan, nan, nan, 0, 0)`` when no episodes are found, so callers
    can format as ``"—"`` instead of branching on existence first.
    """
    outs = joint_outcomes(run_dir, n_eps=n_eps, layout=layout)
    n = len(outs)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0, 0
    k = sum(1 for o in outs if o)
    p, lo, hi = wilson(k, n, z)
    return p * 100.0, lo * 100.0, hi * 100.0, k, n
