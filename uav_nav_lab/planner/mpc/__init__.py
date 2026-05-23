"""Sampling-based MPC planner subpackage.

Split from a 299-line single-file module into three single-concern files:

- :mod:`.planner`    — :class:`SamplingMPCPlanner` (class + per-replan orchestration).
- :mod:`.aggregator` — :func:`argmin_aggregate` action selection rule.

Per-sample cost compute is shared with MPPI in
:mod:`uav_nav_lab.planner._rollout` — the **only** difference between
the two planners is the aggregation rule (argmin vs softmax).

External callers should keep using ``from uav_nav_lab.planner.mpc
import SamplingMPCPlanner`` — the class is re-exported from this
package for backward compatibility.
"""

from __future__ import annotations

from .planner import SamplingMPCPlanner

__all__ = ["SamplingMPCPlanner"]
