"""Characterization tests for uav_nav_lab.recording.airsim_camera.

Injects a fake ``airsim`` module via ``sys.modules`` so the helpers
can be exercised without the real package installed. Records all
calls into the fake client so we can assert which API entry points
were hit with what arguments.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from uav_nav_lab.recording import pitch_front_center, set_topdown_camera


class _FakeClient:
    def __init__(self) -> None:
        self.confirmed = False
        self.resets = 0
        self.pose_calls: list[tuple[str, Any, str | None]] = []

    def confirmConnection(self) -> None:
        self.confirmed = True

    def reset(self) -> None:
        self.resets += 1

    def simSetCameraPose(self, name: str, pose: Any, vehicle_name: str | None = None) -> None:
        self.pose_calls.append((name, pose, vehicle_name))


def _install_fake_airsim(monkeypatch: pytest.MonkeyPatch) -> _FakeClient:
    """Inject a fake airsim module exposing the minimum surface used."""
    fake = ModuleType("airsim")

    class Vector3r:
        def __init__(self, x: float, y: float, z: float) -> None:
            self.x, self.y, self.z = x, y, z

    class Pose:
        def __init__(self, position: Vector3r, orientation: Any) -> None:
            self.position = position
            self.orientation = orientation

    def to_quaternion(pitch: float, roll: float, yaw: float) -> tuple[float, float, float]:
        return (pitch, roll, yaw)  # marker tuple — tests inspect [0] for pitch

    client = _FakeClient()

    def MultirotorClient() -> _FakeClient:
        return client

    fake.Vector3r = Vector3r          # type: ignore[attr-defined]
    fake.Pose = Pose                   # type: ignore[attr-defined]
    fake.to_quaternion = to_quaternion  # type: ignore[attr-defined]
    fake.MultirotorClient = MultirotorClient  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "airsim", fake)
    # Skip the per-call time.sleep delays so tests stay fast
    monkeypatch.setattr("uav_nav_lab.recording.airsim_camera.time.sleep", lambda _s: None)
    return client


def test_pitch_front_center_default_no_vehicle_no_reset(monkeypatch: pytest.MonkeyPatch):
    client = _install_fake_airsim(monkeypatch)
    pitch_front_center()
    assert client.confirmed is True
    assert client.resets == 0
    assert len(client.pose_calls) == 1
    name, pose, vehicle = client.pose_calls[0]
    assert name == "front_center"
    assert vehicle is None
    # Pitch is the first arg to to_quaternion — our fake echoes it
    assert pose.orientation[0] == pytest.approx(-0.30)
    # Forward offset 0.50 m matches the README hero recording geometry
    assert (pose.position.x, pose.position.y, pose.position.z) == (0.50, 0.0, 0.0)


def test_pitch_front_center_with_reset_calls_client_reset(monkeypatch: pytest.MonkeyPatch):
    client = _install_fake_airsim(monkeypatch)
    pitch_front_center(reset=True)
    assert client.resets == 1


def test_pitch_front_center_passes_vehicle_name_when_given(monkeypatch: pytest.MonkeyPatch):
    client = _install_fake_airsim(monkeypatch)
    pitch_front_center(vehicle_name="Drone1")
    _, _, vehicle = client.pose_calls[0]
    assert vehicle == "Drone1"


def test_pitch_front_center_accepts_custom_pitch(monkeypatch: pytest.MonkeyPatch):
    client = _install_fake_airsim(monkeypatch)
    pitch_front_center(pitch_rad=-0.50)
    _, pose, _ = client.pose_calls[0]
    assert pose.orientation[0] == pytest.approx(-0.50)


def test_set_topdown_camera_uses_topdown_pose_and_drone1(monkeypatch: pytest.MonkeyPatch):
    import math

    client = _install_fake_airsim(monkeypatch)
    set_topdown_camera()
    assert len(client.pose_calls) == 1
    name, pose, vehicle = client.pose_calls[0]
    assert name == "topdown"
    assert vehicle == "Drone1"
    # NED 30N, 30E, 55 up — passed through as (x=30, y=30, z=-55)
    assert (pose.position.x, pose.position.y, pose.position.z) == (30.0, 30.0, -55.0)
    # Straight down — pitch = π/2
    assert pose.orientation[0] == pytest.approx(math.pi / 2)


def test_set_topdown_camera_swallows_simset_errors(monkeypatch: pytest.MonkeyPatch):
    # Verify the try/except around simSetCameraPose — some AirSim builds
    # raise on unknown camera names; the original script chose to ignore.
    _install_fake_airsim(monkeypatch)
    # Replace simSetCameraPose with one that raises
    failing_client = sys.modules["airsim"].MultirotorClient()
    failing_client.simSetCameraPose = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("unknown camera"))
    set_topdown_camera(client=failing_client)  # should NOT raise
