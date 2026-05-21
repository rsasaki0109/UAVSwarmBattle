"""ENU ↔ NED coordinate conversions for the ROS 2 bridge.

ENU (east-north-up) is what the framework speaks internally; NED
(north-east-down) is what AirSim's ROS wrappers publish. The mapping
is the same swap+sign pattern as in ``airsim_bridge.coords`` but
implemented here independently because the input shapes differ — this
module operates on plain 3-vectors, not the (1, 3) point arrays
AirSim returns.
"""

from __future__ import annotations

import numpy as np


def _enu_to_ned(p: np.ndarray) -> np.ndarray:
    """``(x, y, z)_ENU → (y, x, -z)_NED``. Pads / truncates to 3D."""
    p = np.asarray(p, dtype=float)
    out = np.zeros(3)
    out[: p.size] = p[:3]
    return np.array([out[1], out[0], -out[2]])


def _ned_to_enu(p: np.ndarray) -> np.ndarray:
    """``(y, x, -z)_NED → (x, y, z)_ENU``."""
    p = np.asarray(p, dtype=float)
    return np.array([p[1], p[0], -p[2]])
