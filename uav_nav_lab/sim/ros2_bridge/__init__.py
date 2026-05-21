"""ROS 2 simulator bridge subpackage.

Split from a 546-line single-file module into:

- :mod:`.coords`   — pure ENU ↔ NED helpers.
- :mod:`.messages` — ``sensor_msgs/PointCloud2`` and ``sensor_msgs/Image``
  decoders (kept out of the adapter so its heavy imports do not fire
  when the bridge is unit-tested with a fake adapter).
- :mod:`.adapter`  — :class:`_RclpyAdapter`, the rclpy-backed
  production adapter that owns the ROS node + pub / subs.
- :mod:`.bridge`   — :class:`Ros2Bridge`, the ``SimInterface``
  implementation. Only this name is part of the public surface; the
  helpers and the adapter are exported for backward compatibility with
  earlier imports.
"""

from __future__ import annotations

from .adapter import _RclpyAdapter
from .bridge import Ros2Bridge
from .coords import _enu_to_ned, _ned_to_enu
from .messages import _decode_pointcloud2, _encode_image_to_png

__all__ = [
    "Ros2Bridge",
    "_RclpyAdapter",
    "_enu_to_ned",
    "_ned_to_enu",
    "_decode_pointcloud2",
    "_encode_image_to_png",
]
