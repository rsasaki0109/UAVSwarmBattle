"""Shared analysis utilities for research scripts.

Three layers, intentionally separated so each can be imported without
paying the cost of the others:

- :mod:`.joint_stats`    — stdlib-only stats primitives (Wilson,
  McNemar, joint-episode loader).
- :mod:`.success_rates`  — thin wrappers turning a run directory into
  a percentage CI / outcome list.
- :mod:`.warmup`         — runs warmup_select_mppi warmup episodes
  and dumps the pooled (top2, cvg) signal + auto-pick; numpy + yaml
  + runner internals are confined to this module so scripts never
  import them directly.

Scripts in ``scripts/`` should depend on this package, not on
``runner.multi`` private surface or re-implementations of Wilson CI.
"""

from __future__ import annotations

from .joint_stats import (
    binom_pmf,
    load_joint_episodes,
    mcnemar_exact_p,
    wilson,
)
from .success_rates import joint_outcomes, joint_success_rate
from .warmup import WarmupDiagnostic, diagnose_warmup

__all__ = [
    "WarmupDiagnostic",
    "binom_pmf",
    "diagnose_warmup",
    "joint_outcomes",
    "joint_success_rate",
    "load_joint_episodes",
    "mcnemar_exact_p",
    "wilson",
]
