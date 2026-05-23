"""MPPI planner subpackage.

Split from a 317-line single-file module into two single-concern files:

- :mod:`.planner`    — :class:`MPPIPlanner` (class + per-replan orchestration).
- :mod:`.aggregator` — :func:`softmax_aggregate` action selection rule.

Per-sample cost compute is shared with MPC in
:mod:`uav_nav_lab.planner._rollout` — the **only** difference between
the two planners is the aggregation rule (argmin vs softmax).

External callers should keep using ``from uav_nav_lab.planner.mppi
import MPPIPlanner`` — the class is re-exported from this package for
backward compatibility.
"""

from __future__ import annotations

from .planner import MPPIPlanner

__all__ = ["MPPIPlanner"]
