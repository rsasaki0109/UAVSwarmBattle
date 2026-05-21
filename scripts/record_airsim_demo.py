"""Record an AirSim demo flight as a GIF for the README hero.

End-to-end:
  1. resets the running AirSim server (clears stale collision flags),
  2. pitches the front-center camera ~17° down so the cube clusters
     stay visible as we fly over them,
  3. runs `uav-nav run examples/exp_airsim_demo.yaml` (the bridge
     drives the airsim instance through the planner-generated path
     and saves one PNG per step under results/airsim_demo/frames_000/),
  4. ffmpegs those PNGs into docs/images/demo_airsim.gif.

Why a script instead of a CLI feature: this is a one-off recording
operation tied to a specific scene composition (camera pitch + scenario
geometry). Folding it into the bridge would make the bridge carry
demo-specific concerns; a tiny driver script keeps the framework clean.

Run from the project root, with an AirSim server already up:
  python3 scripts/record_airsim_demo.py
"""

from __future__ import annotations

from pathlib import Path

from uav_nav_lab.recording import (
    frames_to_gif,
    pitch_front_center,
    run_uav_nav_experiment,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
YAML = REPO_ROOT / "examples" / "exp_airsim_demo.yaml"
RUN_DIR = REPO_ROOT / "results" / "airsim_demo"
GIF_OUT = REPO_ROOT / "docs" / "images" / "demo_airsim.gif"


def main() -> int:
    print("[1/3] setup AirSim camera")
    pitch_front_center(reset=True)
    print("[2/3] run experiment")
    run_uav_nav_experiment(YAML, RUN_DIR, repo_root=REPO_ROOT)
    print("[3/3] frames → GIF")
    n = frames_to_gif(
        RUN_DIR / "frames_000", GIF_OUT,
        fps=12, width=320, target_seconds=5.5,
    )
    print(f"[gif] {GIF_OUT}  ({GIF_OUT.stat().st_size // 1024} KB)  ({n} src frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
