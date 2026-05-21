"""Static and dynamic obstacle spawn / update against an AirSim client.

Static obstacles are configured via the YAML; the bridge spawns them
(with a destroy-first pass) on reset.

Dynamic obstacles are owned by the scenario object — this module just
mirrors a kinematic cube per scenario obstacle into AirSim so Unreal's
collision detector can see drone-vs-cube hits, then pushes pose updates
every step after :meth:`scenario.advance`.

The airsim package is lazy-imported so this module loads cleanly in
environments where airsim is not installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .coords import _enu_extent_to_ned, _enu_to_ned


def normalise_static_obstacle(idx: int, spec: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalise a single static-obstacle YAML entry.

    Accepts ``center``/``size`` aliases for ``position``/``scale`` and
    fills in default ``name`` / ``asset`` / ``physics_enabled`` /
    ``is_blueprint`` fields. Raises ``ValueError`` if position/scale
    cannot be inferred.
    """
    if "position" not in spec and "center" in spec:
        spec["position"] = spec["center"]
    if "scale" not in spec and "size" in spec:
        spec["scale"] = spec["size"]
    if "position" not in spec or "scale" not in spec:
        raise ValueError(
            "AirSim static_obstacles entries require position/center and scale/size"
        )
    return {
        "name": str(spec.get("name", f"uav_nav_static_{idx:03d}")),
        "asset": str(spec.get("asset", spec.get("asset_name", "1M_Cube_Chamfer"))),
        "position": np.asarray(spec["position"], dtype=float),
        "scale": np.asarray(spec["scale"], dtype=float),
        "physics_enabled": bool(spec.get("physics_enabled", False)),
        "is_blueprint": bool(spec.get("is_blueprint", False)),
    }


def sync_static_obstacles(client: Any, static_obstacles: list[dict[str, Any]]) -> None:
    """Spawn the configured static meshes into the shared AirSim scene.

    No-op if there are no static obstacles configured or if the client
    does not implement ``simSpawnObject`` (covers mock clients that don't
    care about static geometry).
    """
    if not static_obstacles or not hasattr(client, "simSpawnObject"):
        return
    try:
        import airsim  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover
        return
    for spec in static_obstacles:
        name = spec["name"]
        if hasattr(client, "simDestroyObject"):
            try:
                client.simDestroyObject(name)
            except Exception:
                pass
        pos_ned = _enu_to_ned(spec["position"])
        scale_ned = _enu_extent_to_ned(spec["scale"])
        pose = airsim.Pose(
            airsim.Vector3r(float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2])),
            airsim.to_quaternion(0.0, 0.0, 0.0),
        )
        scale = airsim.Vector3r(
            float(scale_ned[0]), float(scale_ned[1]), float(scale_ned[2])
        )
        try:
            spawned = client.simSpawnObject(
                name,
                spec["asset"],
                pose,
                scale,
                spec["physics_enabled"],
                spec["is_blueprint"],
            )
        except TypeError:
            spawned = client.simSpawnObject(
                name,
                spec["asset"],
                pose,
                scale,
                spec["physics_enabled"],
            )
        if spawned is False:
            raise RuntimeError(
                f"AirSim failed to spawn static obstacle {name!r} "
                f"with asset {spec['asset']!r}"
            )


def sync_dynamic_obstacles_initial(client: Any, scenario: Any) -> list[str]:
    """Spawn AirSim visual cubes for each scenario dynamic obstacle.

    The scenario owns the obstacle's physical state (position, velocity,
    reflection). This function registers a corresponding kinematic cube
    in AirSim at the obstacle's initial position so Unreal's collision
    detector can see drone-vs-cube hits.

    Returns the list of spawned obstacle names (empty if no dynamic
    obstacles are configured or the client cannot spawn).
    """
    if not hasattr(scenario, "_dynamic"):
        return []
    dynamics = getattr(scenario, "_dynamic", [])
    if not dynamics or not hasattr(client, "simSpawnObject"):
        return []
    try:
        import airsim  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover
        return []
    names: list[str] = []
    for i, d in enumerate(dynamics):
        name = f"uavnav_dyn_{i:03d}"
        if hasattr(client, "simDestroyObject"):
            try:
                client.simDestroyObject(name)
            except Exception:
                pass
        pos_ned = _enu_to_ned(np.asarray(d.pos, dtype=float))
        radius = float(d.radius)
        scale = airsim.Vector3r(
            2.0 * radius, 2.0 * radius, 2.0 * radius,
        )
        pose = airsim.Pose(
            airsim.Vector3r(
                float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2])
            ),
            airsim.to_quaternion(0.0, 0.0, 0.0),
        )
        try:
            spawned = client.simSpawnObject(
                name, "1M_Cube_Chamfer", pose, scale, False, False,
            )
        except TypeError:
            spawned = client.simSpawnObject(
                name, "1M_Cube_Chamfer", pose, scale, False,
            )
        if spawned is False:
            raise RuntimeError(
                f"AirSim failed to spawn dynamic obstacle {name!r}"
            )
        names.append(name)
    return names


def update_dynamic_obstacle_poses(
    client: Any, names: list[str], scenario: Any
) -> None:
    """Move AirSim cubes to match scenario.dynamic_obstacles current pos.

    Called every step after :meth:`scenario.advance` so the cubes track
    the scenario's authoritative dynamic-obstacle state.
    """
    if not names:
        return
    dynamics = getattr(scenario, "_dynamic", [])
    if not dynamics:
        return
    try:
        import airsim  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover
        return
    for name, d in zip(names, dynamics):
        pos_ned = _enu_to_ned(np.asarray(d.pos, dtype=float))
        pose = airsim.Pose(
            airsim.Vector3r(
                float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2])
            ),
            airsim.to_quaternion(0.0, 0.0, 0.0),
        )
        try:
            client.simSetObjectPose(name, pose, True)
        except Exception:
            pass
