"""Decoders for the ROS message types the adapter subscribes to.

Kept out of :mod:`.adapter` so the heavy `struct` / `PIL` imports do
not fire when the bridge is only constructed (e.g. unit-tested with a
fake adapter). Both decoders are exercised end-to-end only in a real
ROS 2 environment, hence ``pragma: no cover``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _decode_pointcloud2(msg: Any) -> np.ndarray:  # pragma: no cover
    """Extract (N, 3) float32 ENU points from a ``sensor_msgs/PointCloud2``.

    Reads ``x`` / ``y`` / ``z`` fields by name + offset from
    ``msg.fields``, so it works against the standard PointCloud2 layout
    regardless of whether extra fields (intensity, rgb, …) are present
    in between.
    """
    import struct

    fields = {f.name: f for f in msg.fields}
    if not all(k in fields for k in ("x", "y", "z")):
        return np.zeros((0, 3), dtype=np.float32)
    n = int(msg.width) * int(msg.height)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    fmt_pref = ">" if getattr(msg, "is_bigendian", False) else "<"
    fmt = f"{fmt_pref}f"
    point_step = int(msg.point_step)
    data = bytes(msg.data)
    x_off = int(fields["x"].offset)
    y_off = int(fields["y"].offset)
    z_off = int(fields["z"].offset)
    out = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        base = i * point_step
        out[i, 0] = struct.unpack_from(fmt, data, base + x_off)[0]
        out[i, 1] = struct.unpack_from(fmt, data, base + y_off)[0]
        out[i, 2] = struct.unpack_from(fmt, data, base + z_off)[0]
    return out


def _encode_image_to_png(msg: Any) -> bytes:  # pragma: no cover
    """Convert a ``sensor_msgs/Image`` to PNG bytes via PIL.

    Supports the most common encodings (rgb8, bgr8, mono8, rgba8). For
    less common encodings (depth16, yuyv, …) returns empty bytes so the
    runner can still write the rest of the step's data without
    crashing.
    """
    try:
        from PIL import Image as PILImage
    except ImportError as e:
        raise SystemExit(
            "PIL/Pillow is required to encode ROS Image messages. "
            "Install with `pip install pillow` (already a `[viz]` extra)."
        ) from e
    import io

    enc = str(msg.encoding)
    h, w = int(msg.height), int(msg.width)
    raw = bytes(msg.data)
    if enc == "rgb8":
        img = PILImage.frombytes("RGB", (w, h), raw)
    elif enc == "bgr8":
        img = PILImage.frombytes("RGB", (w, h), raw)
        b, g, r = img.split()
        img = PILImage.merge("RGB", (r, g, b))
    elif enc == "rgba8":
        img = PILImage.frombytes("RGBA", (w, h), raw)
    elif enc == "mono8":
        img = PILImage.frombytes("L", (w, h), raw)
    else:
        return b""  # unsupported encoding — silently skip this frame
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
