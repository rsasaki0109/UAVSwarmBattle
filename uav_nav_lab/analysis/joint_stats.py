"""Joint-episode statistics shared across paired-analysis scripts.

Three concerns live here together because every consumer needs all
three:

* :func:`wilson`             — score-interval CI for a binomial proportion.
* :func:`mcnemar_exact_p`    — exact two-sided McNemar p-value (via
  :func:`binom_pmf`) for paired-seed disagreement counts.
* :func:`load_joint_episodes` — read ``episode_*_joint.json`` files
  from either a flat layout (``<run>/episode_NNN_joint.json``) or a
  chunked one (``<run>/seed_NNN/episode_000_joint.json``) and return a
  uniform list of dicts.

None of these touch numpy / scipy — they are deliberately
stdlib-only so the analysis package stays cheap to import.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Returns ``(p_hat, lo, hi)`` with ``p_hat = k/n`` and the score CI
    at the given ``z`` (default ≈ 95 %). ``n == 0`` returns all zeros
    (consumers can decide whether to format as N/A).
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def binom_pmf(k: int, n: int, p: float) -> float:
    """Binomial PMF: ``P(X = k)`` for ``X ~ Binom(n, p)``."""
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for paired disagreement counts.

    ``b`` and ``c`` are the off-diagonal cells of the 2×2 paired-
    outcomes table (e.g. ``b`` = A-success ∧ B-failure, ``c`` =
    A-failure ∧ B-success). When ``b + c == 0`` there is no
    disagreement to test, so the function returns ``1.0``.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_one_side = sum(binom_pmf(i, n, 0.5) for i in range(k + 1))
    return min(1.0, 2.0 * p_one_side)


Layout = Literal["flat", "chunked"]


def load_joint_episodes(run_dir: Path, layout: Layout = "flat") -> list[dict]:
    """Load joint-episode summaries from a runner output directory.

    ``layout='flat'`` matches the single-process runner output
    (``<run>/episode_NNN_joint.json``); ``layout='chunked'`` matches
    the per-seed chunked layout used by the AirSim sweep wrapper
    (``<run>/seed_NNN/episode_000_joint.json``).

    Each entry is a dict with the keys expected by every paired-
    analysis script: ``seed``, ``joint`` (bool), ``per_drone``
    (list[bool]), and ``final_t`` (float | None). Results are sorted
    by seed so downstream zip-pair logic is deterministic.
    """
    glob = (
        "episode_*_joint.json"
        if layout == "flat"
        else "seed_*/episode_000_joint.json"
    )
    out: list[dict] = []
    for jp in sorted(Path(run_dir).glob(glob)):
        d = json.loads(jp.read_text())
        out.append(
            {
                "seed": d["meta"]["seed"],
                "joint": d["outcome"] == "success",
                "per_drone": [o == "success" for o in d["per_drone_outcomes"]],
                "final_t": d.get("final_t"),
            }
        )
    return sorted(out, key=lambda r: r["seed"])
