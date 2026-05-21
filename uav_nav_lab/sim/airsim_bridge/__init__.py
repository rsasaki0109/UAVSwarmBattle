"""AirSim bridge subpackage.

Split from a 651-line single-file module into:

- :mod:`.bridge`    — :class:`AirSimBridge` class (reset / step / accessors)
- :mod:`.coords`    — pure ENU ↔ NED conversion helpers
- :mod:`.sensors`   — camera / depth ImageRequest builders + pinhole intrinsics
- :mod:`.obstacles` — static / dynamic obstacle spawn + per-step pose updates

Coordinate-frame helpers are re-exported here under their original names
so existing call sites continue to work after the split.

LiDAR sensors:
  - Configure on the AirSim side via settings.json (one entry per sensor
    with a unique name).
  - List the names in the bridge config (``simulator.lidars: [Lidar1, …]``)
    to have the bridge poll ``getLidarData(name)`` after each step and
    stash the converted (N, 3) ENU point cloud at
    ``state.extra["lidar_points"][name]``. Empty list = no polling.

Cameras:
  - Configure with ``simulator.cameras: [{name, image_type}, …]`` where
    ``image_type`` is one of ``scene`` (default), ``depth_vis``,
    ``depth_perspective``, ``depth_planar``, ``segmentation``,
    ``surface_normals``, ``infrared``. Compressed PNG bytes land at
    ``state.extra["camera_images"][name]`` after each step.
"""

from __future__ import annotations

from .bridge import AirSimBridge
from .coords import (
    _enu_extent_to_ned,
    _enu_to_ned,
    _ned_pointcloud_to_enu,
    _ned_to_enu,
)

__all__ = [
    "AirSimBridge",
    "_enu_to_ned",
    "_enu_extent_to_ned",
    "_ned_to_enu",
    "_ned_pointcloud_to_enu",
]
