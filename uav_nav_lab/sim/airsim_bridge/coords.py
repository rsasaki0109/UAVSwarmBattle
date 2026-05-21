"""ENU ↔ NED coordinate-frame helpers.

The framework is ENU (east-north-up, +z up). AirSim is NED (north-east-down,
+z down). We map ``(x, y, z)_ENU = (y, x, -z)_NED``.

These helpers are pure, depend only on numpy, and have no AirSim
dependency — they are unit-tested directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _enu_to_ned(p: np.ndarray) -> np.ndarray:
    """(x, y, z)_ENU → (y, x, -z)_NED. Pads / truncates to 3D."""
    p = np.asarray(p, dtype=float)
    out = np.zeros(3)
    out[: p.size] = p[:3]
    return np.array([out[1], out[0], -out[2]])


def _enu_extent_to_ned(extent: np.ndarray) -> np.ndarray:
    """Axis-aligned ENU box extent → AirSim NED axis order."""
    e = np.asarray(extent, dtype=float)
    out = np.ones(3)
    out[: e.size] = e[:3]
    return np.array([out[1], out[0], out[2]])


def _ned_to_enu(p: np.ndarray) -> np.ndarray:
    """(y, x, -z)_NED → (x, y, z)_ENU."""
    p = np.asarray(p, dtype=float)
    return np.array([p[1], p[0], -p[2]])


def _ned_pointcloud_to_enu(point_cloud_flat: Any) -> np.ndarray:
    """AirSim's `LidarData.point_cloud` is a flat list of NED triples
    (x, y, z) in vehicle-local frame. Reshape to (N, 3) and convert to
    ENU with the same (x, y, z) → (y, x, -z) flip used for poses.

    Returns shape (N, 3) — empty (0, 3) array if the readout is empty
    or malformed (e.g. lidar not yet populated)."""
    arr = np.asarray(list(point_cloud_flat), dtype=float)
    if arr.size == 0 or arr.size % 3 != 0:
        return np.zeros((0, 3))
    ned = arr.reshape(-1, 3)
    enu = np.empty_like(ned)
    enu[:, 0] = ned[:, 1]
    enu[:, 1] = ned[:, 0]
    enu[:, 2] = -ned[:, 2]
    return enu
