"""CHOMP planner subpackage.

The planner class is re-exported here so external callers can keep using
``from uav_nav_lab.planner.chomp import ChompPlanner`` after the package
split. The CHOMP objective helpers live in :mod:`.objective` and are the
shared public surface used by MPC-CHOMP.
"""

from __future__ import annotations

from .planner import ChompPlanner

__all__ = ["ChompPlanner"]
