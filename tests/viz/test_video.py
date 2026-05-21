"""Frame writer + ffmpeg stitcher coverage."""

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


def test_runner_saves_camera_frames_to_disk_when_configured(tmp_path: Path) -> None:
    """When `output.save_camera_frames: true`, the runner should write each
    step's camera_images bytes to `<run_dir>/frames_NNN/step_NNNN_<name>.png`."""
    from uav_nav_lab.planner.base import Plan
    from uav_nav_lab.runner.experiment import _run_episode

    class FakeSim:
        # Point-mass-like stub that reports collision/goal_reached on a fixed
        # step so the episode terminates predictably.
        dt = 0.05
        def __init__(self) -> None:
            self.scenario = SimpleNamespace(  # noqa: F821
                dynamic_obstacles=[], ndim=2,
            )
            self.obstacle_map = np.zeros((10, 10), dtype=bool)
            self.goal = np.array([9.0, 9.0])
            self._t = 0.0
            self._step = 0
        def reset(self, *, seed=None):  # noqa: ARG002
            self._t = 0.0
            self._step = 0
            from uav_nav_lab.sim.base import SimState
            return SimState(t=0.0, position=np.array([1.0, 1.0]),
                            velocity=np.zeros(2),
                            extra={"camera_images": {"cam0": b"PNG_INIT"}})
        def step(self, _cmd):
            from uav_nav_lab.sim.base import SimState, SimStepInfo
            self._t += self.dt
            self._step += 1
            ns = SimState(
                t=self._t, position=np.array([1.0, 1.0]), velocity=np.zeros(2),
                extra={"camera_images": {"cam0": f"PNG_{self._step:04d}".encode()}},
            )
            done = self._step >= 3
            return ns, SimStepInfo(collision=False, goal_reached=done, truncated=False)

    class SpyPlanner:
        max_speed = 1.0
        def reset(self): pass
        def plan(self, observation, goal, perceived_map, dynamic_obstacles=None):  # noqa: ARG002
            wpts = np.array([observation, goal[: observation.shape[0]]])
            return Plan(waypoints=wpts, meta={"status": "ok"})

    from uav_nav_lab.sensor import SENSOR_REGISTRY

    sensor = SENSOR_REGISTRY.get("perfect").from_config({})
    from types import SimpleNamespace  # noqa: F811
    sim = FakeSim()
    fdir = tmp_path / "frames_000"
    rec = _run_episode(
        sim=sim, planner=SpyPlanner(), sensor=sensor,
        seed=0, replan_period=0.05, max_steps=10, episode_index=0,
        frame_dir=fdir,
    )
    assert rec.outcome == "success"
    # 3 steps logged, 3 frame files written (step indices 0000-0002).
    assert sorted(p.name for p in fdir.iterdir()) == [
        "step_0000_cam0.png", "step_0001_cam0.png", "step_0002_cam0.png",
    ]
    # Bytes round-trip — recorder writes verbatim, no re-encoding.
    assert (fdir / "step_0000_cam0.png").read_bytes() == b"PNG_0001"
    assert (fdir / "step_0002_cam0.png").read_bytes() == b"PNG_0003"


def test_video_stitch_run_produces_one_mp4_per_episode_camera(tmp_path: Path) -> None:
    """Smoke test the `uav-nav video` end of the pipeline: write a few
    valid PNGs into a frames_NNN/ directory and verify `stitch_run`
    emits the expected mp4 path. Skip if ffmpeg or PIL is missing —
    both are realistic prereqs for the actual user workflow."""
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed; skipping video stitching test")
    pil_image = pytest.importorskip("PIL.Image")

    from uav_nav_lab.video import stitch_run

    # h264 needs even dimensions; 16×16 is small enough to keep the test
    # fast but real enough to encode cleanly.
    run_dir = tmp_path / "run"
    fdir = run_dir / "frames_000"
    fdir.mkdir(parents=True)
    for i in range(3):
        img = pil_image.new("RGB", (16, 16), color=(i * 80, 100, 200))
        img.save(fdir / f"step_{i:04d}_cam0.png")

    saved = stitch_run(run_dir, fps=10)
    assert len(saved) == 1
    assert saved[0] == run_dir / "episode_000_cam0.mp4"
    assert saved[0].stat().st_size > 0
