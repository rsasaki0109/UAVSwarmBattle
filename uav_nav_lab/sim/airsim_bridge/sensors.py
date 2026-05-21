"""Camera / depth-camera request builders and pinhole intrinsics.

Free helpers that translate the bridge's camera and depth specs to
``airsim.ImageRequest`` lists, plus a pinhole-intrinsics formula for the
depth path.

All three helpers are CI-friendly: the request builders lazy-import
``airsim`` inside the function body so the module imports cleanly when
the AirSim package is absent. Tests inject a fake ``airsim`` module via
``sys.modules``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def build_image_requests(camera_specs: list[Mapping[str, Any]]) -> list[Any]:
    """Translate camera spec dicts to ``airsim.ImageRequest`` objects.

    Each spec has the form ``{"name": str, "image_type": str}`` where
    ``image_type`` is one of the keys in the type map below. Unknown
    types fall back to ``Scene``. The request uses
    ``pixels_as_float=False, compress=True`` so the response carries a
    PNG byte string at ``response.image_data_uint8``.
    """
    import airsim  # type: ignore[import-not-found]

    type_map = {
        "scene": airsim.ImageType.Scene,
        "depth_vis": airsim.ImageType.DepthVis,
        "depth_perspective": airsim.ImageType.DepthPerspective,
        "depth_planar": airsim.ImageType.DepthPlanar,
        "segmentation": airsim.ImageType.Segmentation,
        "surface_normals": airsim.ImageType.SurfaceNormals,
        "infrared": airsim.ImageType.Infrared,
    }
    return [
        airsim.ImageRequest(
            spec["name"],
            type_map.get(spec["image_type"], airsim.ImageType.Scene),
            False,
            True,
        )
        for spec in camera_specs
    ]


def build_depth_requests(depth_specs: list[Mapping[str, Any]]) -> list[Any]:
    """Translate depth-camera spec dicts to ``airsim.ImageRequest`` objects.

    Uses ``pixels_as_float=True`` and ``compress=False`` so the response
    carries ``image_data_float`` (a flat list of metres) rather than
    the colour-mapped PNG that the camera path produces. Image type
    defaults to ``depth_planar``; ``depth_perspective`` is also valid.
    Any other value (including ``depth_vis``) falls back to
    ``depth_planar`` since visualised depth is not raw.
    """
    import airsim  # type: ignore[import-not-found]

    type_map = {
        "depth_planar": airsim.ImageType.DepthPlanar,
        "depth_perspective": airsim.ImageType.DepthPerspective,
    }
    return [
        airsim.ImageRequest(
            spec["name"],
            type_map.get(spec["image_type"], airsim.ImageType.DepthPlanar),
            True,    # pixels_as_float
            False,   # compress
        )
        for spec in depth_specs
    ]


def intrinsics_from_fov(fov_deg: float, width: int, height: int) -> dict[str, float]:
    """AirSim cameras use a horizontal fov; AirSim's pixels are square so
    fy = fx. Optical centre = image centre."""
    fov_rad = float(fov_deg) * np.pi / 180.0
    fx = (width / 2.0) / float(np.tan(fov_rad / 2.0))
    return {"fx": fx, "fy": fx, "cx": width / 2.0, "cy": height / 2.0}
