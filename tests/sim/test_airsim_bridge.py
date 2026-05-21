"""AirSim bridge unit tests against mock clients."""

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


def test_airsim_bridge_step_round_trips_enu_via_mock_client() -> None:
    """Verify the AirSim bridge's ENU/NED conversions and step plumbing
    against an injected mock client — no AirSim install required."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.airsim_bridge import AirSimBridge, _enu_to_ned, _ned_to_enu

    # Mathematical sanity on the conversion helpers.
    assert np.allclose(_enu_to_ned(np.array([1.0, 2.0, 3.0])), np.array([2.0, 1.0, -3.0]))
    assert np.allclose(_ned_to_enu(np.array([2.0, 1.0, -3.0])), np.array([1.0, 2.0, 3.0]))

    grid_cls = SCENARIO_REGISTRY.get("grid_world")
    sc = grid_cls.from_config(
        {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0], "obstacles": {"type": "none"}}
    )

    class FakeKin:
        # NED kinematics: drone is at NED (4, 3, -1) → ENU (3, 4, 1).
        class _V:
            x_val = 4.0
            y_val = 3.0
            z_val = -1.0
        position = _V()
        linear_velocity = _V()

    class FakeState:
        kinematics_estimated = FakeKin()

    class FakeCollision:
        has_collided = False

    class FakeClient:
        def __init__(self) -> None:
            self.commands = []  # capture moveByVelocityAsync args

        def confirmConnection(self) -> None:
            pass

        def enableApiControl(self, _on: bool, _vehicle: str) -> None:
            pass

        def armDisarm(self, _on: bool, _vehicle: str) -> None:
            pass

        def reset(self) -> None:
            pass

        def simSetVehiclePose(self, *_args, **_kwargs) -> None:  # noqa: D401
            pass

        def simPause(self, _on: bool) -> None:
            pass

        def simContinueForTime(self, _dt: float) -> None:
            pass

        def moveByVelocityAsync(self, vx, vy, vz, dt, vehicle_name=None):
            self.commands.append((vx, vy, vz, dt, vehicle_name))

            class _Future:
                def join(self) -> None:
                    pass

            return _Future()

        def getMultirotorState(self, vehicle_name=None):  # noqa: ARG002
            return FakeState()

        def simGetCollisionInfo(self, vehicle_name=None):  # noqa: ARG002
            return FakeCollision()

    fake = FakeClient()
    bridge = AirSimBridge(dt=0.05, scenario=sc, client=fake)
    state = bridge.reset()
    assert state.position.shape[0] == 2

    # ENU velocity (1, 2) → NED (2, 1, 0). 2D scenario pads vz=0.
    out_state, info = bridge.step(np.array([1.0, 2.0]))
    last = fake.commands[-1]
    assert last[0] == 2.0  # NED x = ENU y
    assert last[1] == 1.0  # NED y = ENU x
    assert last[2] == 0.0  # 2D scenario → vz = 0
    # Returned state in ENU (3, 4) [from FakeKin (4, 3, -1) NED].
    assert np.allclose(out_state.position, np.array([3.0, 4.0]))
    assert info.collision is False


def test_airsim_bridge_spawns_static_obstacles_via_mock_client(monkeypatch) -> None:
    """Static AirSim meshes are spawned once by the master bridge on reset."""
    from types import SimpleNamespace

    fake_airsim = SimpleNamespace()

    class Vector3r:
        def __init__(self, x_val, y_val, z_val) -> None:
            self.x_val = x_val
            self.y_val = y_val
            self.z_val = z_val

    class Pose:
        def __init__(self, position, orientation) -> None:
            self.position = position
            self.orientation = orientation

    fake_airsim.Vector3r = Vector3r
    fake_airsim.Pose = Pose
    fake_airsim.to_quaternion = lambda *_args: object()
    monkeypatch.setitem(sys.modules, "airsim", fake_airsim)

    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.airsim_bridge import AirSimBridge

    grid_cls = SCENARIO_REGISTRY.get("grid_world")
    sc = grid_cls.from_config(
        {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0], "obstacles": {"type": "none"}}
    )

    class FakeKin:
        class _V:
            x_val = 0.0
            y_val = 0.0
            z_val = -1.0

        position = _V()
        linear_velocity = _V()

    class FakeClient:
        def __init__(self) -> None:
            self.spawned = []
            self.destroyed = []

        def confirmConnection(self) -> None:
            pass

        def enableApiControl(self, *_args) -> None:
            pass

        def armDisarm(self, *_args) -> None:
            pass

        def reset(self) -> None:
            pass

        def simPause(self, *_args) -> None:
            pass

        def simSetWind(self, *_args) -> None:
            pass

        def simSetVehiclePose(self, *_args, **_kwargs) -> None:
            pass

        def simDestroyObject(self, name) -> None:
            self.destroyed.append(name)

        def simSpawnObject(self, *args) -> None:
            self.spawned.append(args)

        def getMultirotorState(self, vehicle_name=None):  # noqa: ARG002
            return SimpleNamespace(kinematics_estimated=FakeKin())

        def simGetCollisionInfo(self, vehicle_name=None):  # noqa: ARG002
            return SimpleNamespace(has_collided=False)

    fake = FakeClient()
    bridge = AirSimBridge(
        dt=0.05,
        scenario=sc,
        client=fake,
        static_obstacles=[
            {
                "name": "disc_pillar",
                "asset": "Cube",
                "position": [1.0, 2.0, 3.0],
                "scale": [4.0, 5.0, 6.0],
            }
        ],
    )
    bridge.reset()

    assert fake.destroyed == ["disc_pillar"]
    name, asset, pose, scale, physics_enabled, is_blueprint = fake.spawned[0]
    assert (name, asset, physics_enabled, is_blueprint) == ("disc_pillar", "Cube", False, False)
    assert (pose.position.x_val, pose.position.y_val, pose.position.z_val) == (2.0, 1.0, -3.0)
    assert (scale.x_val, scale.y_val, scale.z_val) == (5.0, 4.0, 6.0)


def test_airsim_bridge_polls_cameras_and_stashes_png_bytes_via_mock_client() -> None:
    """When `cameras: [{name, image_type}]` is configured, AirSimBridge.step()
    should call client.simGetImages() with a list of airsim.ImageRequest
    objects and stash the response bytes at state.extra["camera_images"][name].

    The bridge lazy-imports airsim only when cameras are configured; the
    test therefore injects a minimal fake airsim module into sys.modules
    so the import succeeds and ImageRequest construction works."""
    import sys
    from types import ModuleType, SimpleNamespace

    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.airsim_bridge import AirSimBridge

    grid_cls = SCENARIO_REGISTRY.get("grid_world")
    sc = grid_cls.from_config(
        {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0], "obstacles": {"type": "none"}}
    )

    # Minimal stand-in for `airsim.ImageRequest` / `airsim.ImageType` so
    # `_build_image_requests` does not have to know about CI vs prod.
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

    # `reset()` also touches airsim.Pose / Vector3r / to_quaternion for the
    # teleport step, so the fake module needs those too — otherwise the
    # narrow `except ImportError` around teleport leaks an AttributeError.
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
    saved = sys.modules.get("airsim")
    sys.modules["airsim"] = fake_airsim
    try:
        class FakeKin:
            class _V:
                x_val = 0.0
                y_val = 0.0
                z_val = 0.0
            position = _V()
            linear_velocity = _V()

        captured_requests: list[list[Any]] = []  # noqa: F821

        class FakeClient:
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
            def simGetImages(self, requests, vehicle_name=None):  # noqa: ARG002
                captured_requests.append(list(requests))
                # Two cameras configured → two responses with distinct PNG bytes.
                return [
                    SimpleNamespace(image_data_uint8=b"PNG_BYTES_FRONT"),
                    SimpleNamespace(image_data_uint8=b"PNG_BYTES_DEPTH"),
                ]

        fake = FakeClient()
        bridge = AirSimBridge(
            dt=0.05, scenario=sc, client=fake,
            cameras=[
                {"name": "front_center", "image_type": "scene"},
                {"name": "front_depth", "image_type": "depth_vis"},
            ],
        )
        bridge.reset()
        out_state, _ = bridge.step(np.array([0.0, 0.0]))

        # simGetImages received two ImageRequests with the right names + types.
        assert len(captured_requests) == 1
        reqs = captured_requests[0]
        assert reqs[0].camera_name == "front_center"
        assert reqs[0].image_type == _ImgType.Scene
        assert reqs[0].compress is True
        assert reqs[1].camera_name == "front_depth"
        assert reqs[1].image_type == _ImgType.DepthVis

        # Both PNG payloads landed in state.extra under their camera names.
        cams = out_state.extra["camera_images"]
        assert cams["front_center"] == b"PNG_BYTES_FRONT"
        assert cams["front_depth"] == b"PNG_BYTES_DEPTH"
    finally:
        if saved is None:
            del sys.modules["airsim"]
        else:
            sys.modules["airsim"] = saved


def test_airsim_bridge_polls_depth_cameras_and_stashes_float_depth_via_mock_client() -> None:
    """When `depths: [{name, fov_deg, width, height}]` is configured,
    AirSimBridge.step() should call client.simGetImages() with
    pixels_as_float=True and stash {depth, intrinsics} at
    state.extra["depth_images"][name]. Intrinsics derive from the
    configured fov + image size."""
    import sys
    from types import ModuleType, SimpleNamespace

    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.airsim_bridge import AirSimBridge

    grid_cls = SCENARIO_REGISTRY.get("grid_world")
    sc = grid_cls.from_config(
        {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0], "obstacles": {"type": "none"}}
    )

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
            self.pixels_as_float = pixels_as_float
            self.compress = compress
    class _Vec3:
        def __init__(self, x, y, z): self.x_val, self.y_val, self.z_val = x, y, z
    class _Pose:
        def __init__(self, position, orientation): self.position, self.orientation = position, orientation
    fake_airsim = ModuleType("airsim")
    fake_airsim.ImageType = _ImgType
    fake_airsim.ImageRequest = _ImgReq
    fake_airsim.Vector3r = _Vec3
    fake_airsim.Pose = _Pose
    fake_airsim.to_quaternion = lambda *_a, **_k: object()
    saved = sys.modules.get("airsim")
    sys.modules["airsim"] = fake_airsim
    try:
        captured: list[list[Any]] = []  # noqa: F821

        class FakeKin:
            class _V:
                x_val = 0.0
                y_val = 0.0
                z_val = 0.0
            position = _V()
            linear_velocity = _V()

        class FakeClient:
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
            def simGetImages(self, requests, vehicle_name=None):  # noqa: ARG002
                captured.append(list(requests))
                # Return one float-pixel response per request: 8x6 = 48 floats.
                return [SimpleNamespace(image_data_float=[2.5] * (8 * 6))]

        bridge = AirSimBridge(
            dt=0.05, scenario=sc, client=FakeClient(),
            depths=[{
                "name": "fwd", "image_type": "depth_planar",
                "fov_deg": 90.0, "width": 8, "height": 6,
            }],
        )
        bridge.reset()
        out_state, _ = bridge.step(np.array([0.0, 0.0]))

        # The right ImageRequest was issued: depth_planar + float + uncompressed.
        assert len(captured) == 1
        req = captured[0][0]
        assert req.camera_name == "fwd"
        assert req.image_type == _ImgType.DepthPlanar
        assert req.pixels_as_float is True
        assert req.compress is False

        # Payload landed correctly under state.extra["depth_images"][name].
        depth_bag = out_state.extra["depth_images"]
        payload = depth_bag["fwd"]
        assert payload["depth"].shape == (6, 8)
        assert float(payload["depth"][0, 0]) == pytest.approx(2.5)
        # 90deg fov on 8 px wide → fx = 4 / tan(45) = 4. cx = 4, cy = 3.
        assert payload["intrinsics"]["fx"] == pytest.approx(4.0)
        assert payload["intrinsics"]["fy"] == pytest.approx(4.0)
        assert payload["intrinsics"]["cx"] == pytest.approx(4.0)
        assert payload["intrinsics"]["cy"] == pytest.approx(3.0)
    finally:
        if saved is None:
            del sys.modules["airsim"]
        else:
            sys.modules["airsim"] = saved


def test_airsim_bridge_polls_lidar_and_converts_to_enu_via_mock_client() -> None:
    """When `lidars: [name]` is configured, AirSimBridge.step() should call
    client.getLidarData(name) and stash an (N, 3) ENU point cloud at
    state.extra['lidar_points'][name] — converted from AirSim's NED
    (x, y, z) → (y, x, -z) the same way poses are."""
    from types import SimpleNamespace

    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.airsim_bridge import AirSimBridge, _ned_pointcloud_to_enu

    # Helper sanity: NED [(1,2,-3), (4,5,-6)] flat → ENU [(2,1,3), (5,4,6)].
    pts = _ned_pointcloud_to_enu([1.0, 2.0, -3.0, 4.0, 5.0, -6.0])
    assert pts.shape == (2, 3)
    assert np.allclose(pts, np.array([[2.0, 1.0, 3.0], [5.0, 4.0, 6.0]]))
    # Empty / malformed readouts return shape (0, 3) instead of crashing.
    assert _ned_pointcloud_to_enu([]).shape == (0, 3)
    assert _ned_pointcloud_to_enu([1.0, 2.0]).shape == (0, 3)

    grid_cls = SCENARIO_REGISTRY.get("grid_world")
    sc = grid_cls.from_config(
        {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0], "obstacles": {"type": "none"}}
    )

    class FakeKin:
        class _V:
            x_val = 0.0
            y_val = 0.0
            z_val = 0.0
        position = _V()
        linear_velocity = _V()

    class FakeClient:
        def __init__(self) -> None:
            self.lidar_calls: list[tuple[str, str | None]] = []

        def confirmConnection(self) -> None: pass
        def enableApiControl(self, _on, _vehicle): pass
        def armDisarm(self, _on, _vehicle): pass
        def reset(self): pass
        def simSetVehiclePose(self, *_a, **_kw): pass
        def simPause(self, _on): pass
        def simContinueForTime(self, _dt): pass

        def moveByVelocityAsync(self, *_a, **_kw):
            class _F:
                def join(self): pass
            return _F()

        def getMultirotorState(self, vehicle_name=None):  # noqa: ARG002
            return SimpleNamespace(kinematics_estimated=FakeKin())

        def simGetCollisionInfo(self, vehicle_name=None):  # noqa: ARG002
            return SimpleNamespace(has_collided=False)

        def getLidarData(self, name, vehicle_name=None):
            self.lidar_calls.append((name, vehicle_name))
            # Two NED points that map to predictable ENU rows.
            return SimpleNamespace(point_cloud=[1.0, 2.0, -3.0, 4.0, 5.0, -6.0])

    fake = FakeClient()
    bridge = AirSimBridge(dt=0.05, scenario=sc, client=fake, lidars=["FrontLidar"])
    bridge.reset()
    out_state, _ = bridge.step(np.array([0.0, 0.0]))

    # Lidar polled with the configured name + vehicle.
    assert fake.lidar_calls == [("FrontLidar", "Drone1")]
    # Points landed in state.extra under the lidar name.
    cloud = out_state.extra["lidar_points"]["FrontLidar"]
    assert cloud.shape == (2, 3)
    assert np.allclose(cloud, np.array([[2.0, 1.0, 3.0], [5.0, 4.0, 6.0]]))


# ---------------------------------------------------------------------------
# Characterization tests for the helpers extracted by the subpackage split.
# Lock down current behaviour at the *current* import locations so the
# refactor can verify it preserves the external contract.
# ---------------------------------------------------------------------------


def _install_fake_airsim(monkeypatch) -> object:
    """Inject a minimal `airsim` stand-in into sys.modules.

    `_build_image_requests` / `_build_depth_requests` do
    ``import airsim`` inside the method body, so the tests need ImageType
    and ImageRequest defined. Returns the fake module for inspection.
    """
    import sys
    from types import ModuleType

    class _ImgType:
        Scene = 0
        DepthVis = 3
        DepthPerspective = 2
        DepthPlanar = 1
        Segmentation = 5
        SurfaceNormals = 6
        Infrared = 7

    class _ImgReq:
        def __init__(self, camera_name, image_type, pixels_as_float, compress):
            self.camera_name = camera_name
            self.image_type = image_type
            self.pixels_as_float = pixels_as_float
            self.compress = compress

    fake = ModuleType("airsim")
    fake.ImageType = _ImgType
    fake.ImageRequest = _ImgReq
    monkeypatch.setitem(sys.modules, "airsim", fake)
    return fake


def test_airsim_bridge_enu_extent_to_ned_swaps_xy_keeps_z() -> None:
    from uav_nav_lab.sim.airsim_bridge import _enu_extent_to_ned

    assert np.allclose(
        _enu_extent_to_ned(np.array([2.0, 4.0, 6.0])),
        np.array([4.0, 2.0, 6.0]),
    )
    # Shorter inputs pad to 3D with ones (extent default).
    assert np.allclose(_enu_extent_to_ned(np.array([3.0, 5.0])), np.array([5.0, 3.0, 1.0]))


def test_airsim_bridge_ned_pointcloud_to_enu_handles_normal_and_edge_cases() -> None:
    from uav_nav_lab.sim.airsim_bridge import _ned_pointcloud_to_enu

    cloud = _ned_pointcloud_to_enu([1.0, 2.0, -3.0, 4.0, 5.0, -6.0])
    assert cloud.shape == (2, 3)
    assert np.allclose(cloud, np.array([[2.0, 1.0, 3.0], [5.0, 4.0, 6.0]]))

    # Empty cloud → (0, 3).
    empty = _ned_pointcloud_to_enu([])
    assert empty.shape == (0, 3)

    # Malformed (non-multiple of 3) cloud → (0, 3).
    bad = _ned_pointcloud_to_enu([1.0, 2.0])
    assert bad.shape == (0, 3)


def test_airsim_bridge_normalise_static_obstacle_accepts_center_size_aliases() -> None:
    from uav_nav_lab.sim.airsim_bridge import AirSimBridge

    spec = AirSimBridge._normalise_static_obstacle(
        7, {"center": [1.0, 2.0, 3.0], "size": [4.0, 5.0, 6.0]}
    )
    assert spec["name"] == "uav_nav_static_007"
    assert spec["asset"] == "1M_Cube_Chamfer"  # default
    assert np.allclose(spec["position"], [1.0, 2.0, 3.0])
    assert np.allclose(spec["scale"], [4.0, 5.0, 6.0])
    assert spec["physics_enabled"] is False
    assert spec["is_blueprint"] is False


def test_airsim_bridge_normalise_static_obstacle_requires_position_and_scale() -> None:
    from uav_nav_lab.sim.airsim_bridge import AirSimBridge

    with pytest.raises(ValueError, match="position/center and scale/size"):
        AirSimBridge._normalise_static_obstacle(0, {"name": "x"})


def test_airsim_bridge_intrinsics_from_fov_matches_pinhole_formula() -> None:
    from uav_nav_lab.sim.airsim_bridge import AirSimBridge

    intr = AirSimBridge._intrinsics_from_fov(90.0, 256, 144)
    # fx = (W/2) / tan(fov/2); fov=90° → tan=1 → fx = 128.
    assert intr["fx"] == pytest.approx(128.0)
    assert intr["fy"] == pytest.approx(128.0)
    assert intr["cx"] == pytest.approx(128.0)
    assert intr["cy"] == pytest.approx(72.0)


def test_airsim_bridge_build_image_requests_dispatches_type_map(monkeypatch) -> None:
    fake = _install_fake_airsim(monkeypatch)
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.airsim_bridge import AirSimBridge

    sc = SCENARIO_REGISTRY.get("grid_world").from_config(
        {"size": [4, 4], "start": [0.0, 0.0], "goal": [3.0, 3.0], "obstacles": {"type": "none"}}
    )
    bridge = AirSimBridge(
        dt=0.05, scenario=sc,
        cameras=[
            {"name": "rgb", "image_type": "scene"},
            {"name": "seg", "image_type": "segmentation"},
            {"name": "ir",  "image_type": "infrared"},
            {"name": "weird", "image_type": "unknown_type"},  # falls back to Scene
        ],
    )
    reqs = bridge._build_image_requests()
    assert [r.camera_name for r in reqs] == ["rgb", "seg", "ir", "weird"]
    assert reqs[0].image_type == fake.ImageType.Scene
    assert reqs[1].image_type == fake.ImageType.Segmentation
    assert reqs[2].image_type == fake.ImageType.Infrared
    assert reqs[3].image_type == fake.ImageType.Scene
    # Scene path: compressed PNG bytes, not floats.
    assert all(r.pixels_as_float is False for r in reqs)
    assert all(r.compress is True for r in reqs)


def test_airsim_bridge_build_depth_requests_uses_pixels_as_float(monkeypatch) -> None:
    fake = _install_fake_airsim(monkeypatch)
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.airsim_bridge import AirSimBridge

    sc = SCENARIO_REGISTRY.get("grid_world").from_config(
        {"size": [4, 4], "start": [0.0, 0.0], "goal": [3.0, 3.0], "obstacles": {"type": "none"}}
    )
    bridge = AirSimBridge(
        dt=0.05, scenario=sc,
        depths=[
            {"name": "front_depth", "fov_deg": 90.0, "width": 256, "height": 144},
            {"name": "front_persp", "image_type": "depth_perspective",
             "fov_deg": 60.0, "width": 128, "height": 72},
            {"name": "bogus", "image_type": "depth_vis",  # not in type_map → DepthPlanar
             "fov_deg": 45.0, "width": 64, "height": 48},
        ],
    )
    reqs = bridge._build_depth_requests()
    assert [r.camera_name for r in reqs] == ["front_depth", "front_persp", "bogus"]
    assert reqs[0].image_type == fake.ImageType.DepthPlanar
    assert reqs[1].image_type == fake.ImageType.DepthPerspective
    assert reqs[2].image_type == fake.ImageType.DepthPlanar  # fallback
    # Depth path: floats, uncompressed.
    assert all(r.pixels_as_float is True for r in reqs)
    assert all(r.compress is False for r in reqs)
