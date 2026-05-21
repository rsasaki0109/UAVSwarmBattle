"""Shared test helpers extracted from the old monolithic ``test_smoke.py``."""

from __future__ import annotations

from pathlib import Path

import pytest

from uav_nav_lab.config import ExperimentConfig

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _require_mplot3d() -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("mpl_toolkits.mplot3d.axes3d")


def _basic_cfg(overrides: dict | None = None) -> ExperimentConfig:
    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_basic.yaml")
    cfg.num_episodes = 2
    cfg.simulator["max_steps"] = 600
    if overrides:
        cfg.raw.update(overrides)
    return cfg
