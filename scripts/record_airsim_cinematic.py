"""Record a close cinematic AirSim shot: an external camera orbits the hub while
the four drones cross it, so the quadrotors fill the frame with photorealistic
detail (rotors, shadows) instead of being distant dots.

Uses an AirSim ExternalCamera named "cine" (must be declared in settings.json),
re-posed every frame onto an orbit around the crossing point (ENU 30,30,30) and
aimed at it, captured live while `uav-nav run` flies the fleet.

  python scripts/record_airsim_cinematic.py
"""
from __future__ import annotations

import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from uav_nav_lab.recording import frames_to_gif
from uav_nav_lab.recording.experiment_runner import _build_cli_command

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_YAML = REPO_ROOT / "examples" / "exp_airsim_multi_demo.yaml"
EXP_RUN_DIR = REPO_ROOT / "results" / "airsim_multi_demo"
FRAMES_DIR = REPO_ROOT / "results" / "airsim_cine_frames"
GIF_OUT = REPO_ROOT / "docs" / "images" / "swarm_airsim_cinematic.gif"

CHASE_BACK = 6.5    # metres trailing behind the lead drone
CHASE_UP = 2.0      # metres above it
CHASE_AHEAD = 5.0   # look this far ahead of the drone


def _chase_pose(pos, heading):
    """Camera trailing `pos` along `heading` (unit NED), looking just ahead of it."""
    import airsim
    h = heading / (np.linalg.norm(heading) + 1e-9)
    cam = pos - h * CHASE_BACK + np.array([0.0, 0.0, -CHASE_UP])
    target = pos + h * CHASE_AHEAD
    d = target - cam
    yaw = math.atan2(d[1], d[0])
    pitch = math.atan2(-d[2], math.hypot(d[0], d[1]))
    return airsim.Pose(airsim.Vector3r(*cam), airsim.to_quaternion(pitch, 0.0, yaw))


def main() -> int:
    import airsim
    from PIL import Image

    c = airsim.MultirotorClient()
    c.confirmConnection()
    c.reset()
    time.sleep(1.0)

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for f in FRAMES_DIR.glob("*.png"):
        f.unlink()

    print("[1/2] launch flight + orbit cine camera", flush=True)
    env = {**os.environ, "UAV_NAV_NO_CAMERA": "1"}
    proc = subprocess.Popen(_build_cli_command(EXP_YAML), cwd=REPO_ROOT, env=env)

    fps, i, tail = 12, 0, 0
    interval = 1.0 / fps
    prev = None
    while True:
        t0 = time.perf_counter()
        try:
            pp = c.simGetVehiclePose("Drone1").position
            pos = np.array([pp.x_val, pp.y_val, pp.z_val])
            heading = (pos - prev) if (prev is not None and np.linalg.norm(pos - prev) > 1e-3) \
                else np.array([0.0, 1.0, 0.0])   # Drone1 flies +Y (east) in NED
            prev = pos
            c.simSetCameraPose("cine", _chase_pose(pos, heading), external=True)
            r = c.simGetImages([airsim.ImageRequest("cine", airsim.ImageType.Scene, False, False)],
                               vehicle_name="", external=True)
            if r and r[0].image_data_uint8:
                im = r[0]
                flat = np.frombuffer(im.image_data_uint8, dtype=np.uint8)
                ch = flat.size // (im.width * im.height)
                arr = flat.reshape(im.height, im.width, ch)[:, :, :3][:, :, ::-1]
                Image.fromarray(arr).save(str(FRAMES_DIR / f"frame_{i:04d}.png"))
                i += 1
        except Exception as e:  # noqa: BLE001
            print(f"  frame {i} error: {e}", flush=True)
        if proc.poll() is not None:
            tail += 1
            if tail > fps:
                break
        dt = time.perf_counter() - t0
        if dt < interval:
            time.sleep(interval - dt)
    proc.wait()
    print(f"  {i} frames", flush=True)

    print("[2/2] frames -> GIF", flush=True)
    n = frames_to_gif(FRAMES_DIR, GIF_OUT, fps=fps, width=520,
                      target_seconds=8.0, frame_pattern="frame_%04d.png", name_contains=None)
    print(f"[gif] {GIF_OUT}  ({GIF_OUT.stat().st_size // 1024} KB)  ({n} src frames)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
