"""Multi-drone runner subpackage.

Split from a 440-line single-file module into:

- :mod:`.builder`    — :func:`_build_multi` (scenario / sims / planners / sensors)
- :mod:`.peers`      — pure peer helpers (:func:`_peers_view`,
  :func:`_check_peer_collision`)
- :mod:`.episode`    — :func:`run_episode_multi` and its phase helpers
  (replan, log-step, master hand-off, outcome resolution)
- :mod:`.experiment` — :func:`run_experiment_multi` driver

One scenario, N drones. Each drone gets its own simulator / sensor /
planner instance; all instances share the same scenario object so the
static map and any scenario-owned dynamic obstacles stay consistent.
"""

from __future__ import annotations

from .builder import _build_multi
from .episode import run_episode_multi
from .experiment import run_experiment_multi
from .peers import _check_peer_collision, _peers_view

__all__ = [
    "run_experiment_multi",
    "run_episode_multi",
    "_build_multi",
    "_peers_view",
    "_check_peer_collision",
]
