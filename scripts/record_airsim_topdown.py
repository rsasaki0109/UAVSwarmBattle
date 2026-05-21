"""Record AirSim multi-drone demo from a top-down fixed camera.

Places a camera at (30, 30, 55) looking straight down so all 4 drones
are visible as they cross at the centre. Captures frames via AirSim
API directly (not through the framework's camera pipeline) so the
``UAV_NAV_NO_CAMERA=1`` flag is set on the experiment subprocess to
skip in-loop capture.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from uav_nav_lab.recording import (
    frames_to_gif,
    run_uav_nav_experiment,
    set_topdown_camera,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_YAML = REPO_ROOT / "examples" / "exp_airsim_multi_demo.yaml"
EXP_RUN_DIR = REPO_ROOT / "results" / "airsim_multi_demo"
FRAMES_DIR = REPO_ROOT / "results" / "airsim_topdown_frames"
GIF_OUT = REPO_ROOT / "docs" / "images" / "demo_airsim_multi.gif"


def capture_topdown(fps: int = 10, duration_s: float = 8.0) -> list[Path]:
    """Capture frames from a fixed top-down camera via AirSim API."""
    import airsim
    from PIL import Image  # type: ignore[import-not-found]

    client = airsim.MultirotorClient()
    client.confirmConnection()
    # Don't reset — the experiment runner handles that.
    set_topdown_camera(client=client)

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for f in FRAMES_DIR.glob("*.png"):
        f.unlink()

    n_frames = int(fps * duration_s)
    interval = 1.0 / fps
    paths: list[Path] = []
    for i in range(n_frames):
        t0 = time.perf_counter()
        try:
            responses = client.simGetImages(
                [airsim.ImageRequest("topdown", airsim.ImageType.Scene, False, False)],
                vehicle_name="Drone1",
            )
            if responses:
                img_data = responses[0].image_data_uint8
                if img_data:
                    # AirSim returns BGRA
                    arr = np.frombuffer(img_data, dtype=np.uint8).reshape(
                        responses[0].height, responses[0].width, 4
                    )
                    img = Image.fromarray(arr[:, :, :3][:, :, ::-1])  # BGR→RGB
                    path = FRAMES_DIR / f"frame_{i:04d}.png"
                    img.save(str(path))
                    paths.append(path)
        except Exception as e:
            print(f"  frame {i} error: {e}")

        elapsed = time.perf_counter() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)

    return paths


def main() -> int:
    print("[1/3] run experiment (4 drones crossing)")
    run_uav_nav_experiment(
        EXP_YAML, EXP_RUN_DIR,
        repo_root=REPO_ROOT,
        extra_env={"UAV_NAV_NO_CAMERA": "1"},
    )

    print("[2/3] capture top-down frames")
    paths = capture_topdown(fps=10, duration_s=8.0)
    print(f"  captured {len(paths)} frames")

    print("[3/3] frames → GIF")
    n = frames_to_gif(
        FRAMES_DIR, GIF_OUT,
        fps=10, width=640,
        target_seconds=None,        # simple fps mode — no decimation
        frame_pattern="frame_%04d.png",
        name_contains=None,
    )
    print(f"[gif] {GIF_OUT}  ({GIF_OUT.stat().st_size // 1024} KB)  ({n} src frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
