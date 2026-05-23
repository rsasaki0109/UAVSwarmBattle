"""MPPI planner subpackage.

Split from a 317-line single-file module into three single-concern files:

- :mod:`.planner`    — :class:`MPPIPlanner` (class + per-replan orchestration).
- :mod:`.rollout`    — :func:`score_rollouts` per-sample cost compute kernel.
- :mod:`.aggregator` — :func:`softmax_aggregate` action selection rule.

External callers should keep using ``from uav_nav_lab.planner.mppi
import MPPIPlanner`` — the class is re-exported from this package for
backward compatibility.
"""

from __future__ import annotations

from .planner import MPPIPlanner

__all__ = ["MPPIPlanner"]
