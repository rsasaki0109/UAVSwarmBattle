"""Shared analysis utilities for paired comparison scripts.

Pulled out of ``scripts/paired_analysis_*.py`` so the same Wilson CI,
binomial / McNemar tests, and joint-episode loaders are not
re-implemented (with subtle drift) in every consumer.
"""

from __future__ import annotations

from .joint_stats import (
    binom_pmf,
    load_joint_episodes,
    mcnemar_exact_p,
    wilson,
)

__all__ = [
    "binom_pmf",
    "load_joint_episodes",
    "mcnemar_exact_p",
    "wilson",
]
