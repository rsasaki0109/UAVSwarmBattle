"""Cross-axis end-to-end smoke tests."""

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


def test_run_then_eval(tmp_path: Path) -> None:
    cfg = _basic_cfg()
    run_dir = run_experiment(cfg, tmp_path / "run")
    eps = sorted(run_dir.glob("episode_*.json"))
    assert len(eps) == 2
    summary = evaluate_run(run_dir)
    assert summary["n_episodes"] == 2
    assert 0.0 <= summary["success_rate"] <= 1.0
    assert (run_dir / "summary.json").exists()


def test_3d_runs(tmp_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_3d.yaml")
    cfg.num_episodes = 1
    cfg.simulator["max_steps"] = 400
    run_dir = run_experiment(cfg, tmp_path / "3d")
    summary = evaluate_run(run_dir)
    assert summary["n_episodes"] == 1
    # confirm the run was actually 3D (3 components per logged position)
    ep0 = json.loads((run_dir / "episode_000.json").read_text())
    assert len(ep0["steps"][0]["true_pos"]) == 3


def test_airsim_discriminating_sweep_configs_are_consistent() -> None:
    """The AirSim static-cube sweep keeps planner occupancy and meshes paired."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY

    pairs = [
        ("exp_airsim_multi_discriminating_n30.yaml", "exp_airsim_multi_discriminating_n30_gpu_mppi.yaml", 5),
        (
            "exp_airsim_multi_discriminating_central_n30.yaml",
            "exp_airsim_multi_discriminating_central_n30_gpu_mppi.yaml",
            6,
        ),
        (
            "exp_airsim_multi_discriminating_central_soft_n30.yaml",
            "exp_airsim_multi_discriminating_central_soft_n30_gpu_mppi.yaml",
            6,
        ),
        (
            "exp_airsim_multi_discriminating_central_west_n30.yaml",
            "exp_airsim_multi_discriminating_central_west_n30_gpu_mppi.yaml",
            6,
        ),
        (
            "exp_airsim_multi_discriminating_central_west_thick_n30.yaml",
            "exp_airsim_multi_discriminating_central_west_thick_n30_gpu_mppi.yaml",
            6,
        ),
        (
            "exp_airsim_multi_discriminating_central_north_n30.yaml",
            "exp_airsim_multi_discriminating_central_north_n30_gpu_mppi.yaml",
            6,
        ),
        (
            "exp_airsim_multi_discriminating_central_half_n30.yaml",
            "exp_airsim_multi_discriminating_central_half_n30_gpu_mppi.yaml",
            6,
        ),
        (
            "exp_airsim_multi_discriminating_central_29p25_n30.yaml",
            "exp_airsim_multi_discriminating_central_29p25_n30_gpu_mppi.yaml",
            6,
        ),
        (
            "exp_airsim_multi_discriminating_central_29p375_n30.yaml",
            "exp_airsim_multi_discriminating_central_29p375_n30_gpu_mppi.yaml",
            6,
        ),
        ("exp_airsim_multi_discriminating_mid_n30.yaml", "exp_airsim_multi_discriminating_mid_n30_gpu_mppi.yaml", 6),
        (
            "exp_airsim_multi_discriminating_dense_n30.yaml",
            "exp_airsim_multi_discriminating_dense_n30_gpu_mppi.yaml",
            7,
        ),
        (
            "exp_airsim_multi_discriminating_packed_n30.yaml",
            "exp_airsim_multi_discriminating_packed_n30_gpu_mppi.yaml",
            9,
        ),
    ]

    scenario_cls = SCENARIO_REGISTRY.get("multi_drone_voxel")
    for mpc_name, gpu_name, expected_count in pairs:
        mpc_cfg = ExperimentConfig.from_yaml(EXAMPLES / mpc_name)
        gpu_cfg = ExperimentConfig.from_yaml(EXAMPLES / gpu_name)

        mpc_boxes = mpc_cfg.scenario["obstacles"]["boxes"]
        gpu_boxes = gpu_cfg.scenario["obstacles"]["boxes"]
        mpc_meshes = mpc_cfg.simulator["static_obstacles"]
        gpu_meshes = gpu_cfg.simulator["static_obstacles"]

        assert mpc_boxes == gpu_boxes
        assert mpc_meshes == gpu_meshes
        assert len(mpc_boxes) == expected_count
        assert len(mpc_meshes) == expected_count
        assert len({mesh["name"] for mesh in mpc_meshes}) == expected_count

        scenario = scenario_cls.from_config(mpc_cfg.scenario)
        for drone in mpc_cfg.scenario["drones"]:
            start = tuple(int(round(v)) for v in drone["start"])
            goal = tuple(int(round(v)) for v in drone["goal"])
            assert not scenario.occupancy[start], f"{mpc_name}: start occupied for {drone['name']}"
            assert not scenario.occupancy[goal], f"{mpc_name}: goal occupied for {drone['name']}"


def test_airsim_lidar_to_pointcloud_occupancy_to_planner_pipeline_via_mocks() -> None:
    """Guard rail for the AirSim-LiDAR perception pipeline (PRs #4 / #5 / #6).

    Drives the inner experiment loop with a mock AirSim client + the real
    AirSimBridge + the real pointcloud_occupancy sensor + a spy planner.
    Verifies the per-step LiDAR returns flow:
      AirSim mock → bridge.step → state.extra → recorder JSON summary
      AirSim mock → bridge.step → state.extra → sensor.observe_map →
        perceived_map the spy planner sees
    Each layer is unit-tested separately; this test catches regressions
    where the layers stop composing."""
    from types import SimpleNamespace

    from uav_nav_lab.planner.base import Plan
    from uav_nav_lab.runner.experiment import _run_episode
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sensor import SENSOR_REGISTRY
    from uav_nav_lab.sim.airsim_bridge import AirSimBridge

    grid_cls = SCENARIO_REGISTRY.get("grid_world")
    sc = grid_cls.from_config(
        {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0],
         "obstacles": {"type": "none"}, "resolution": 1.0}
    )

    class FakeKin:
        # Drone parks at NED (1, 1, 0) → ENU (1, 1, 0) every step. The
        # static position keeps the lidar hits landing on the same cells
        # each iteration so the assertions stay simple.
        class _V:
            x_val = 1.0
            y_val = 1.0
            z_val = 0.0
        position = _V()
        linear_velocity = _V()

    class FakeClient:
        def __init__(self) -> None:
            self.lidar_calls: list[str] = []

        def confirmConnection(self): pass
        def enableApiControl(self, _o, _v): pass
        def armDisarm(self, _o, _v): pass
        def reset(self): pass
        def simSetVehiclePose(self, *_a, **_k): pass
        def simPause(self, _o): pass
        def simContinueForTime(self, _dt): pass

        def moveByVelocityAsync(self, *_a, **_k):
            class _F:
                def join(self): pass
            return _F()

        def getMultirotorState(self, vehicle_name=None):  # noqa: ARG002
            return SimpleNamespace(kinematics_estimated=FakeKin())

        def simGetCollisionInfo(self, vehicle_name=None):  # noqa: ARG002
            return SimpleNamespace(has_collided=False)

        def getLidarData(self, name, vehicle_name=None):  # noqa: ARG002
            self.lidar_calls.append(name)
            # NED (1, 0, 0), (0, 1, 0) → ENU (0, 1, 0), (1, 0, 0).
            # Drone at world (1, 1) → world cells [1, 2] and [2, 1].
            return SimpleNamespace(point_cloud=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    fake = FakeClient()
    bridge = AirSimBridge(dt=0.05, scenario=sc, client=fake, lidars=["L1"])

    pc_sensor_cls = SENSOR_REGISTRY.get("pointcloud_occupancy")
    sensor = pc_sensor_cls.from_config({"resolution": 1.0, "memory": True})

    captured = {"perceived_map": None, "calls": 0}

    class SpyPlanner:
        max_speed = 1.0

        def reset(self) -> None: pass

        def plan(self, observation, goal, perceived_map, dynamic_obstacles=None):  # noqa: ARG002
            captured["perceived_map"] = np.asarray(perceived_map).copy()
            captured["calls"] += 1
            wpts = np.array([observation, goal[: observation.shape[0]]])
            return Plan(waypoints=wpts, meta={"status": "ok"})

    rec = _run_episode(
        sim=bridge, planner=SpyPlanner(), sensor=sensor,
        seed=0, replan_period=0.05, max_steps=3, episode_index=0,
    )

    # Lidar polled at every bridge.step (3 steps).
    assert fake.lidar_calls == ["L1", "L1", "L1"]
    # Recorder surfaced the per-step lidar count summary on every row.
    assert len(rec.steps) == 3
    assert all(s["lidar_points"] == {"L1": 2} for s in rec.steps)
    # Spy planner saw an occupancy grid with the lidar-derived hits.
    pm = captured["perceived_map"]
    assert pm is not None
    assert pm.shape == (10, 10)
    assert pm[1, 2] and pm[2, 1]
    assert pm.sum() == 2
    # Replan ran each step (replan_period == dt), so the planner saw the
    # accumulated map every time, not a stale snapshot.
    assert captured["calls"] == 3


def test_airsim_camera_to_video_end_to_end_via_mocks(tmp_path: Path) -> None:
    """End-to-end: a real `AirSimBridge` (with an injected fake airsim
    client returning *valid* PIL-encoded PNG bytes) drives `_run_episode`
    with `frame_dir=tmp_path/frames_000`; the recorder writes those bytes
    verbatim to disk; `stitch_run` then ffmpegs them into an MP4.

    This is the only test that exercises *all three* layers
    (bridge → runner frame writer → ffmpeg encoder) against the same
    bytes, catching any contract drift between them."""
    import io
    import shutil
    from types import ModuleType, SimpleNamespace

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed; skipping camera→video e2e test")
    pil_image = pytest.importorskip("PIL.Image")

    from uav_nav_lab.planner.base import Plan
    from uav_nav_lab.runner.experiment import _run_episode
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sensor import SENSOR_REGISTRY
    from uav_nav_lab.sim.airsim_bridge import AirSimBridge
    from uav_nav_lab.video import stitch_run

    # Build N distinct VALID PNG buffers up front so the fake client just
    # serves them back per step. h264 needs even dimensions — 16×16 is the
    # same size used by the pure-stitch test above.
    def _make_png(color: tuple[int, int, int]) -> bytes:
        buf = io.BytesIO()
        pil_image.new("RGB", (16, 16), color=color).save(buf, format="PNG")
        return buf.getvalue()

    # 4 steps × 1 camera, varying color so the encoded video is non-trivial.
    canned_pngs = [_make_png((i * 60, 100, 200)) for i in range(4)]

    # --- inject minimal fake airsim module (mirrors the AirSim camera test) ---
    # Bridge's `_build_image_requests` looks up every entry of its
    # `image_type` map at module-import time on `airsim.ImageType`, so the
    # fake must carry all of them even though we only request `scene`.
    class _ImgType:
        Scene = 0
        DepthVis = 3
        DepthPerspective = 2
        DepthPlanar = 1
        Segmentation = 5
        SurfaceNormals = 6
        Infrared = 7
    class _ImgReq:
        def __init__(self, camera_name, image_type, pixels_as_float, compress):  # noqa: ARG002
            self.camera_name = camera_name
            self.image_type = image_type
            self.compress = compress
    class _Vec3:
        def __init__(self, x, y, z):
            self.x_val, self.y_val, self.z_val = x, y, z
    class _Pose:
        def __init__(self, position, orientation):
            self.position, self.orientation = position, orientation
    fake_airsim = ModuleType("airsim")
    fake_airsim.ImageType = _ImgType
    fake_airsim.ImageRequest = _ImgReq
    fake_airsim.Vector3r = _Vec3
    fake_airsim.Pose = _Pose
    fake_airsim.to_quaternion = lambda *_a, **_k: object()
    saved_mod = sys.modules.get("airsim")
    sys.modules["airsim"] = fake_airsim

    try:
        # Drive the FakeClient through 4 simGetImages() calls then collide.
        # 4 frames is enough to verify the index→bytes mapping survives
        # bridge→recorder→writer without becoming flaky on fast machines.
        class FakeKin:
            class _V:
                x_val = 0.0
                y_val = 0.0
                z_val = 0.0
            position = _V()
            linear_velocity = _V()

        class FakeClient:
            def __init__(self) -> None:
                self._call = 0
            def confirmConnection(self): pass
            def enableApiControl(self, _o, _v): pass
            def armDisarm(self, _o, _v): pass
            def reset(self): pass
            def simSetVehiclePose(self, *_a, **_k): pass
            def simPause(self, _o): pass
            def simContinueForTime(self, _dt): pass
            def moveByVelocityAsync(self, *_a, **_k):
                class _F:
                    def join(self): pass
                return _F()
            def getMultirotorState(self, vehicle_name=None):  # noqa: ARG002
                return SimpleNamespace(kinematics_estimated=FakeKin())
            def simGetCollisionInfo(self, vehicle_name=None):  # noqa: ARG002
                # End the episode on step 4 so we get exactly 4 frames written.
                hit = self._call >= len(canned_pngs)
                return SimpleNamespace(has_collided=hit)
            def simGetImages(self, requests, vehicle_name=None):  # noqa: ARG002
                idx = min(self._call, len(canned_pngs) - 1)
                self._call += 1
                return [SimpleNamespace(image_data_uint8=canned_pngs[idx])]

        grid_cls = SCENARIO_REGISTRY.get("grid_world")
        sc = grid_cls.from_config(
            {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0],
             "obstacles": {"type": "none"}}
        )

        bridge = AirSimBridge(
            dt=0.05, scenario=sc, client=FakeClient(),
            cameras=[{"name": "front_center", "image_type": "scene"}],
        )

        class StubPlanner:
            max_speed = 1.0
            def reset(self): pass
            def plan(self, observation, goal, perceived_map, dynamic_obstacles=None):  # noqa: ARG002
                wpts = np.array([observation, goal[: observation.shape[0]]])
                return Plan(waypoints=wpts, meta={"status": "ok"})

        sensor = SENSOR_REGISTRY.get("perfect").from_config({})
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        fdir = run_dir / "frames_000"

        rec = _run_episode(
            sim=bridge, planner=StubPlanner(), sensor=sensor,
            seed=0, replan_period=0.05, max_steps=20, episode_index=0,
            frame_dir=fdir,
        )
        # Episode terminated on the simulated collision after 4 successful
        # camera polls; recorder+writer should have produced 4 PNGs.
        assert rec.outcome == "collision"
        png_files = sorted(fdir.glob("step_*_front_center.png"))
        assert len(png_files) == 4
        # Each on-disk PNG decodes back to the canned image of the same step.
        for i, p in enumerate(png_files):
            decoded = pil_image.open(p)
            assert decoded.size == (16, 16)
            assert decoded.getpixel((0, 0)) == (i * 60, 100, 200)

        # Now stitch them into an MP4 — this is the part that catches any
        # bytes-validity bug between the bridge layer and ffmpeg.
        saved = stitch_run(run_dir, fps=10)
        assert len(saved) == 1
        assert saved[0] == run_dir / "episode_000_front_center.mp4"
        assert saved[0].stat().st_size > 0
    finally:
        if saved_mod is None:
            del sys.modules["airsim"]
        else:
            sys.modules["airsim"] = saved_mod


def test_dynamic_run(tmp_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_dynamic.yaml")
    cfg.num_episodes = 1
    cfg.simulator["max_steps"] = 400
    run_dir = run_experiment(cfg, tmp_path / "dyn_run")
    summary = evaluate_run(run_dir)
    assert summary["n_episodes"] == 1


def test_lidar_run(tmp_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_lidar.yaml")
    cfg.num_episodes = 1
    cfg.simulator["max_steps"] = 600
    run_dir = run_experiment(cfg, tmp_path / "lidar_run")
    summary = evaluate_run(run_dir)
    assert summary["n_episodes"] == 1
