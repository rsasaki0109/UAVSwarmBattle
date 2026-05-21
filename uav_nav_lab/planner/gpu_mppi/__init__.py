"""GPU MPPI planner subpackage.

Split from a 603-line single-file module into:

- :mod:`.planner`   — orchestration + class definition (registry decorator)
- :mod:`.rollout`   — batched GPU rollout + cost computation
- :mod:`.aggregator` — vanilla / argmin-fallback / mode-aware action selection
- :mod:`.ctg_cache` — Dijkstra cost-to-go cache with tolerance window
"""

from __future__ import annotations

from .planner import GPUMPPIPlanner

__all__ = ["GPUMPPIPlanner"]
