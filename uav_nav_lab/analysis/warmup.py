"""Warmup-episode diagnostic for warmup_select_mppi YAMLs.

Encapsulates the "build multi-drone scenario, run N warmup episodes,
read the pooled (top2, cvg) session, optionally read the auto-picked
temperature" pattern that the X / U / city_* scripts each used to
re-implement. Hides the private ``runner.multi`` internals and the
``_SHARED_SESSIONS`` module-level singleton from script authors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class WarmupDiagnostic:
    """Pooled signals + post-warmup pick from a warmup_select_mppi run.

    ``top2_series`` and ``cvg_series`` are interleaved per-drone
    per-replan in the order ``plan()`` was called by the multi-drone
    runner. Use :attr:`n_drones` if you need to de-interleave.

    ``selected_temperature`` and ``selected_reason`` are populated only
    when the caller asked for ``episodes >= 2`` — the auto-pick fires
    at the start of episode 1.
    """

    top2_series: list[float]
    cvg_series: list[float]
    n_drones: int
    n_samples: int
    top2_mean: float
    cvg_mean: float
    top2_max: float
    cvg_max: float
    selected_temperature: float | None = None
    selected_reason: str | None = None


def _mean(xs: list[float]) -> float:
    arr = np.asarray(xs, float)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def _max(xs: list[float]) -> float:
    arr = np.asarray(xs, float)
    return float(np.nanmax(arr)) if arr.size else float("nan")


def diagnose_warmup(
    yaml_path: str | Path,
    base_seed: int = 42,
    episodes: int = 1,
) -> WarmupDiagnostic:
    """Run ``episodes`` warmup episodes of a warmup_select_mppi YAML
    and return the pooled signal + auto-pick (if ``episodes >= 2``).

    ``base_seed`` is the seed for episode 0; episode ``i`` uses
    ``base_seed + i``. ``replan_period`` and ``max_steps`` are read
    from the YAML's ``planner.replan_period`` and
    ``simulator.max_steps``.

    Hides ``_build_multi`` / ``run_episode_multi`` /
    ``_SHARED_SESSIONS`` from callers — research scripts should depend
    only on this function, not on the runner.multi private surface.
    """
    # Imports here to keep the analysis package import-light: a script
    # that only needs joint_stats / success_rates should not pay the
    # cost of importing the runner machinery.
    from ..config import ExperimentConfig
    from ..planner.warmup_select_mppi import _SHARED_SESSIONS
    from ..runner.multi.builder import _build_multi
    from ..runner.multi.episode import run_episode_multi

    yaml_path = Path(yaml_path)
    _SHARED_SESSIONS.clear()
    cfg = ExperimentConfig.from_yaml(yaml_path)
    cfg.num_episodes = max(1, int(episodes))
    scenario, sims, planners, sensors = _build_multi(cfg)
    raw = yaml.safe_load(open(yaml_path))
    rp = float(raw["planner"]["replan_period"])
    ms = int(raw["simulator"]["max_steps"])
    for ep in range(int(episodes)):
        run_episode_multi(
            scenario, sims, planners, sensors,
            seed=base_seed + ep, replan_period=rp, max_steps=ms,
            episode_index=ep, frame_dirs=[None] * scenario.n_drones,
        )

    sess_list = list(_SHARED_SESSIONS.values())
    if not sess_list:
        return WarmupDiagnostic(
            top2_series=[], cvg_series=[],
            n_drones=scenario.n_drones, n_samples=0,
            top2_mean=float("nan"), cvg_mean=float("nan"),
            top2_max=float("nan"), cvg_max=float("nan"),
        )
    sess = sess_list[0]
    top2 = list(sess.top2)
    cvg = list(sess.cvg)
    diag = WarmupDiagnostic(
        top2_series=top2,
        cvg_series=cvg,
        n_drones=scenario.n_drones,
        n_samples=len(top2),
        top2_mean=_mean(top2),
        cvg_mean=_mean(cvg),
        top2_max=_max(top2),
        cvg_max=_max(cvg),
    )
    if int(episodes) >= 2 and planners:
        diag.selected_temperature = float(planners[0].temperature)
        diag.selected_reason = str(getattr(planners[0], "_selected_reason", ""))
    return diag
