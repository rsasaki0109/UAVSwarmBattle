"""Record an AirSim multi-drone demo flight as a GIF.

End-to-end:
  1. resets the running AirSim server (clears stale state on Drones 1-4),
  2. pitches Drone1's front-center camera ~17° down so the other
     drones stay in frame as they cross at the centre,
  3. runs `uav-nav run examples/exp_airsim_multi_demo.yaml` (the multi
     runner drives all 4 vehicles through their MPC plans + CV peer
     prediction; only Drone1's bridge captures camera frames),
  4. ffmpegs the per-step PNGs into docs/images/demo_airsim_multi.gif.

Run from the project root, with an AirSim server already up *and*
~/Documents/AirSim/settings.json declaring Drone1..Drone4:
  python3 scripts/record_airsim_multi_demo.py
"""

from __future__ import annotations

from pathlib import Path

from uav_nav_lab.recording import (
    frames_to_gif,
    pitch_front_center,
    run_uav_nav_experiment,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
YAML = REPO_ROOT / "examples" / "exp_airsim_multi_demo.yaml"
RUN_DIR = REPO_ROOT / "results" / "airsim_multi_demo"
GIF_OUT = REPO_ROOT / "docs" / "images" / "demo_airsim_multi.gif"


def main() -> int:
    print("[1/2] run experiment (4 drones)")
    # Pitch camera, then run — experiment_runner handles reset + teleport.
    pitch_front_center(vehicle_name="Drone1")
    run_uav_nav_experiment(YAML, RUN_DIR, repo_root=REPO_ROOT)
    frames_dir = RUN_DIR / "frames_000_drone_00"
    if not frames_dir.is_dir():
        frames_dir = RUN_DIR / "frames_000"
    print(f"[2/2] frames → GIF (from {frames_dir.name}/)")
    n = frames_to_gif(
        frames_dir, GIF_OUT,
        fps=15, width=640, target_seconds=7.0,
    )
    print(f"[gif] {GIF_OUT}  ({GIF_OUT.stat().st_size // 1024} KB)  ({n} src frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
