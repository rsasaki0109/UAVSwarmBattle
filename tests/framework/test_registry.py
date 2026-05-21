"""Registry-population and optional-dependency smoke tests."""

from __future__ import annotations

import json  # noqa: F401
import subprocess  # noqa: F401
import sys  # noqa: F401
from pathlib import Path  # noqa: F401

import numpy as np  # noqa: F401
import pytest  # noqa: F401

from uav_nav_lab.cli import build_parser, main  # noqa: F401
from uav_nav_lab.config import ExperimentConfig  # noqa: F401
from uav_nav_lab.eval import evaluate_run  # noqa: F401
from uav_nav_lab.planner import PLANNER_REGISTRY  # noqa: F401
from uav_nav_lab.runner import expand_sweep, run_experiment  # noqa: F401

from tests._helpers import EXAMPLES, _basic_cfg, _require_mplot3d  # noqa: F401


def test_registries_populated() -> None:
    assert "astar" in PLANNER_REGISTRY.names()
    assert "straight" in PLANNER_REGISTRY.names()
    assert "mpc" in PLANNER_REGISTRY.names()


def test_gpu_mppi_is_optional_when_torch_is_missing() -> None:
    """Importing planner registry should not require optional GPU deps."""
    code = """
import builtins
real_import = builtins.__import__

def blocked_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ImportError("blocked torch for smoke test")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
from uav_nav_lab.planner import PLANNER_REGISTRY

assert "gpu_mppi" in PLANNER_REGISTRY.names()
planner_cls = PLANNER_REGISTRY.get("gpu_mppi")
try:
    planner_cls()
except SystemExit as exc:
    assert "requires PyTorch" in str(exc)
else:
    raise AssertionError("gpu_mppi should fail clearly only when instantiated")
"""
    res = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parent.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr


def test_bridge_stubs_registered() -> None:
    """AirSim and ROS2 backends register at import time but should fail with
    a clear message if their heavy deps are not installed."""
    from uav_nav_lab.sim import SIM_REGISTRY

    assert "airsim" in SIM_REGISTRY.names()
    assert "ros2" in SIM_REGISTRY.names()
