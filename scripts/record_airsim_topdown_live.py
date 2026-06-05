"""Record an AirSim multi-drone crossing from a fixed top-down camera, captured
*live* while the drones fly (so all four are in frame as they cross the centre).

The shipped top-down recorder captures only after the experiment finishes, by
which point the drones are parked at their goals. This one launches `uav-nav run`
as a subprocess and grabs top-down frames concurrently for the whole flight.

Run from the project root with an AirSim server up and
~/Documents/AirSim/settings.json declaring Drone1..Drone4 *and* a camera named
"topdown" on Drone1 (any pose — this script re-aims it). The named camera is
required: simGetImages against a camera name not in settings.json SIGSEGVs the
Unreal engine. Minimal settings.json camera entry::

    "topdown": {"CaptureSettings": [{"ImageType": 0, "Width": 720,
                "Height": 405, "FOV_Degrees": 90}],
                "X": 0, "Y": 0, "Z": -40, "Pitch": -90, "Roll": 0, "Yaw": 0}

  python scripts/record_airsim_topdown_live.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import math

from uav_nav_lab.recording import frames_to_gif
from uav_nav_lab.recording.experiment_runner import _build_cli_command

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_YAML = REPO_ROOT / "examples" / "exp_airsim_multi_demo.yaml"
EXP_RUN_DIR = REPO_ROOT / "results" / "airsim_multi_demo"
FRAMES_DIR = REPO_ROOT / "results" / "airsim_topdown_live_frames"
GIF_OUT = REPO_ROOT / "docs" / "images" / "swarm_airsim_topdown.gif"


def main() -> int:
    import airsim
    from PIL import Image

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.reset()
    time.sleep(1.0)
    # Chase-cam 40 m straight above Drone1, looking straight DOWN. Drone1 ("east")
    # flies through the central crossing, so at the crossing all four are in frame.
    # (The shipped set_topdown_camera aims +90deg -> sky on this build; -pi/2 = down,
    # and the "topdown" camera must be declared in settings.json or simGetImages
    # SIGSEGVs the engine.)
    pose = airsim.Pose(airsim.Vector3r(0.0, 0.0, -40.0),
                       airsim.to_quaternion(-math.pi / 2, 0.0, 0.0))
    client.simSetCameraPose("topdown", pose, vehicle_name="Drone1")

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for f in FRAMES_DIR.glob("*.png"):
        f.unlink()

    print("[1/2] launch flight + capture top-down concurrently")
    env = {**os.environ, "UAV_NAV_NO_CAMERA": "1"}
    proc = subprocess.Popen(_build_cli_command(EXP_YAML), cwd=REPO_ROOT, env=env)

    fps, i = 12, 0
    interval = 1.0 / fps
    # keep grabbing while the flight runs, plus a short tail
    tail = 0
    while True:
        t0 = time.perf_counter()
        try:
            resp = client.simGetImages(
                [airsim.ImageRequest("topdown", airsim.ImageType.Scene, False, False)],
                vehicle_name="Drone1",
            )
            if resp and resp[0].image_data_uint8:
                r = resp[0]
                flat = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
                ch = flat.size // (r.height * r.width)        # 3 (BGR) or 4 (BGRA)
                arr = flat.reshape(r.height, r.width, ch)
                Image.fromarray(arr[:, :, :3][:, :, ::-1]).save(str(FRAMES_DIR / f"frame_{i:04d}.png"))
                i += 1
        except Exception as e:  # noqa: BLE001
            print(f"  frame {i} capture error: {e}")
        if proc.poll() is not None:
            tail += 1
            if tail > fps:  # ~1s after the flight ends
                break
        dt = time.perf_counter() - t0
        if dt < interval:
            time.sleep(interval - dt)
    proc.wait()
    print(f"  captured {i} frames")

    print("[2/2] frames -> GIF")
    n = frames_to_gif(FRAMES_DIR, GIF_OUT, fps=fps, width=480,
                      target_seconds=9.0, frame_pattern="frame_%04d.png", name_contains=None)
    print(f"[gif] {GIF_OUT}  ({GIF_OUT.stat().st_size // 1024} KB)  ({n} src frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
