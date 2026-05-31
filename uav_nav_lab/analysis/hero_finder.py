"""Automatic "hero cell" finder for paired baseline-vs-proposed sweeps.

A *hero cell* is the scenario configuration where a proposed method most
dramatically beats a baseline — the cell you want to turn into a GIF or a
headline result. Manually scanning a sweep for "baseline 0/10, proposed 10/10"
is tedious and easy to bias; this ranks every cell objectively.

For each cell we pair two run directories by episode seed (so the comparison
is on identical scenarios), build the McNemar 2x2 table of discordant
outcomes, and score the cell by how decisively the proposed method wins:

    margin       = proposed_rate - baseline_rate         (the effect size)
    significance = McNemar exact two-sided p-value        (is it real?)
    drama        = margin, gated by significance + a baseline-headroom bonus
                   so "0% → 100%" outranks "80% → 100%" at equal margin.

The math is pure and unit-tested; :func:`find_heroes` is the I/O wrapper that
loads run dirs. Outcomes can be compared at the *joint* level (every drone in
the episode succeeds — the coordination metric) or *per-drone* level.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .joint_stats import mcnemar_exact_p, wilson

OutcomeLevel = Literal["joint", "per_drone"]


@dataclass
class HeroScore:
    """Scored comparison of one baseline run vs one proposed run."""

    label: str
    n_paired: int
    baseline_rate: float
    proposed_rate: float
    baseline_ci: tuple[float, float]
    proposed_ci: tuple[float, float]
    margin: float           # proposed_rate - baseline_rate
    b: int                  # baseline win, proposed fail (regressions)
    c: int                  # baseline fail, proposed win (the wins we want)
    mcnemar_p: float
    significant: bool       # p < alpha and c > b
    drama: float            # ranking key (higher = better hero)

    def headline(self) -> str:
        sig = "✓ sig" if self.significant else "  ns "
        return (
            f"[{sig}] {self.label}: "
            f"{self.baseline_rate * 100:5.1f}% → {self.proposed_rate * 100:5.1f}% "
            f"(Δ{self.margin * 100:+5.1f} pp, n={self.n_paired}, "
            f"c={self.c}/b={self.b}, p={self.mcnemar_p:.3g}, drama={self.drama:.3f})"
        )


def _rates(pairs: list[tuple[bool, bool]]) -> tuple[float, float, int, int]:
    """From seed-aligned (baseline, proposed) outcome pairs return
    (baseline_rate, proposed_rate, b, c)."""
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0, 0, 0
    bsucc = sum(1 for base, _ in pairs if base)
    psucc = sum(1 for _, prop in pairs if prop)
    b = sum(1 for base, prop in pairs if base and not prop)
    c = sum(1 for base, prop in pairs if (not base) and prop)
    return bsucc / n, psucc / n, b, c


def score_cell(
    label: str,
    pairs: list[tuple[bool, bool]],
    *,
    alpha: float = 0.05,
) -> HeroScore:
    """Score one cell from seed-aligned (baseline, proposed) outcome pairs.

    The drama score rewards a large margin, but only counts it when the
    result is significant (p < alpha) and in the right direction (c > b);
    a non-significant or wrong-direction cell scores 0 so it sinks to the
    bottom of the ranking. Among significant cells, a low baseline gets a
    headroom bonus so a 0%→100% flip beats an 80%→100% nudge at equal margin.
    """
    n = len(pairs)
    base_rate, prop_rate, b, c = _rates(pairs)
    margin = prop_rate - base_rate
    p = mcnemar_exact_p(b, c)
    significant = (p < alpha) and (c > b)
    if significant:
        # headroom bonus: 1 + (1 - baseline_rate) → up to 2x for a 0% baseline
        drama = margin * (1.0 + (1.0 - base_rate))
    else:
        drama = 0.0
    _, blo, bhi = wilson(int(round(base_rate * n)), n)
    _, plo, phi = wilson(int(round(prop_rate * n)), n)
    return HeroScore(
        label=label,
        n_paired=n,
        baseline_rate=base_rate,
        proposed_rate=prop_rate,
        baseline_ci=(blo, bhi),
        proposed_ci=(plo, phi),
        margin=margin,
        b=b,
        c=c,
        mcnemar_p=p,
        significant=significant,
        drama=drama,
    )


def _outcomes_by_seed(
    run_dir: Path, level: OutcomeLevel
) -> dict[int, bool]:
    """Map episode-seed → success bool for a run dir.

    Multi-drone runs write ``episode_*_joint.json`` (one per episode) with an
    ``outcome`` ("success" iff every drone reached its goal) and a ``meta.seed``.
    Single-drone runs write ``episode_*.json`` with an ``outcome`` and a
    top-level ``seed``. We prefer the joint files when present (the mission-level
    metric); otherwise fall back to single-drone episodes. ``level`` is accepted
    for forward compatibility but both layouts key on episode success here.
    """
    run_dir = Path(run_dir)
    joint_files = sorted(run_dir.glob("episode_*_joint.json"))
    if joint_files:
        out: dict[int, bool] = {}
        for jf in joint_files:
            d = json.loads(jf.read_text())
            out[_seed_of(d)] = d.get("outcome") == "success"
        return out
    # Single-drone (or joint-less) layout: episode_*.json is one file PER DRONE,
    # several sharing a seed. Aggregate to a joint outcome: a seed succeeds iff
    # every drone-episode at that seed succeeded.
    by_seed: dict[int, bool] = {}
    for ef in sorted(run_dir.glob("episode_*.json")):
        if ef.name.endswith("_joint.json"):
            continue
        d = json.loads(ef.read_text())
        seed = _seed_of(d)
        ok = d.get("outcome") == "success"
        by_seed[seed] = ok if seed not in by_seed else (by_seed[seed] and ok)
    return by_seed


def _seed_of(ep: dict) -> int:
    """Episode seed, tolerating both top-level `seed` and `meta.seed` layouts."""
    if "seed" in ep:
        return int(ep["seed"])
    meta = ep.get("meta") or {}
    if "seed" in meta:
        return int(meta["seed"])
    raise KeyError("episode JSON has no seed (checked top-level and meta)")


def pair_by_seed(
    baseline_dir: Path, proposed_dir: Path, level: OutcomeLevel = "joint"
) -> list[tuple[bool, bool]]:
    """Seed-align two run dirs into (baseline_success, proposed_success) pairs.

    Only seeds present in *both* runs are paired (so an aborted run doesn't
    silently bias the comparison). Order is by seed for determinism.
    """
    base = _outcomes_by_seed(Path(baseline_dir), level)
    prop = _outcomes_by_seed(Path(proposed_dir), level)
    common = sorted(set(base) & set(prop))
    return [(base[s], prop[s]) for s in common]


def find_heroes(
    cells: list[tuple[str, Path, Path]],
    *,
    level: OutcomeLevel = "joint",
    alpha: float = 0.05,
) -> list[HeroScore]:
    """Rank a list of (label, baseline_dir, proposed_dir) cells by drama.

    Returns the scores sorted best-hero-first. Cells with no common seeds are
    skipped (they cannot be paired). The caller typically prints the top few
    and renders a GIF of the top cell.
    """
    scores: list[HeroScore] = []
    for label, base_dir, prop_dir in cells:
        pairs = pair_by_seed(base_dir, prop_dir, level)
        if not pairs:
            continue
        scores.append(score_cell(label, pairs, alpha=alpha))
    scores.sort(key=lambda s: (s.drama, s.margin, s.c), reverse=True)
    return scores
