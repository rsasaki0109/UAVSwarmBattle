"""AirSim camera setup helpers used by the recording scripts.

The ``airsim`` package is lazy-imported so this module loads cleanly
in CI / lightweight environments. Tests inject a fake ``airsim``
module via ``sys.modules`` (see ``tests/recording/test_airsim_camera.py``).
"""

from __future__ import annotations

import math
import time
from typing import Any


_FRONT_CENTER_OFFSET = (0.50, 0.0, 0.0)  # forward 0.5 m, no lateral / vertical offset
DEFAULT_PITCH_RAD = -0.30                # ~17° down — keeps cube clusters in frame at 30 m
_TOPDOWN_OFFSET = (30.0, 30.0, -55.0)    # NED: 30N, 30E, 55 m up
_TOPDOWN_PITCH = math.pi / 2             # straight down


def pitch_front_center(
    *,
    vehicle_name: str | None = None,
    pitch_rad: float = DEFAULT_PITCH_RAD,
    reset: bool = False,
    settle_s: float = 1.0,
    client: Any = None,
) -> None:
    """Pitch the ``front_center`` camera down on the specified vehicle.

    Default ``pitch_rad = -0.30`` (~17° down) is the angle the README
    hero GIFs were recorded at — keeps cube clusters in frame at 30 m
    altitude. ``reset=True`` calls ``client.reset()`` first to clear
    stale collision flags (single-drone demo only — multi-drone
    experiments handle reset themselves).

    Pass a ``client`` for tests; production callers leave it ``None``
    so a fresh ``MultirotorClient`` is created.
    """
    if client is None:
        import airsim  # type: ignore[import-not-found]
        client = airsim.MultirotorClient()
        client.confirmConnection()
    if reset:
        client.reset()
        time.sleep(2.0)
    else:
        time.sleep(settle_s)
    import airsim  # type: ignore[import-not-found]
    cam_pose = airsim.Pose(
        airsim.Vector3r(*_FRONT_CENTER_OFFSET),
        airsim.to_quaternion(pitch_rad, 0.0, 0.0),
    )
    if vehicle_name is None:
        client.simSetCameraPose("front_center", cam_pose)
    else:
        client.simSetCameraPose("front_center", cam_pose, vehicle_name=vehicle_name)
    time.sleep(0.3)


def set_topdown_camera(
    camera_name: str = "topdown",
    *,
    vehicle_name: str = "Drone1",
    client: Any = None,
) -> None:
    """Set a fixed top-down camera at ``(30, 30, 55 m up)`` pitched
    straight down — used by the top-down capture script.

    Silently swallows ``simSetCameraPose`` errors because some AirSim
    builds only let you reuse a camera name declared in
    ``settings.json``; calling against an unknown name returns an
    error code the original script chose to ignore so the run can
    continue with the existing camera placement.
    """
    if client is None:
        import airsim  # type: ignore[import-not-found]
        client = airsim.MultirotorClient()
        client.confirmConnection()
    import airsim  # type: ignore[import-not-found]
    cam_pose = airsim.Pose(
        airsim.Vector3r(*_TOPDOWN_OFFSET),
        airsim.to_quaternion(_TOPDOWN_PITCH, 0.0, 0.0),
    )
    try:
        client.simSetCameraPose(camera_name, cam_pose, vehicle_name=vehicle_name)
    except Exception:
        pass
