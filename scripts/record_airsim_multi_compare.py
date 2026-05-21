"""Record an AirSim 4-drone MPC-vs-GPU-MPPI side-by-side GIF.

Drives both planners through the same Blocks crossing scenario on the
same running Blocks server, captures Drone1's FPV from each run, and
ffmpegs them into a side-by-side comparison GIF for the README hero.

Pipeline:
  1. pitch Drone1's front-center camera ~17° down (better FPV angle),
  2. run examples/exp_airsim_multi_demo.yaml (4-drone MPC + CV peer
     prediction baseline),
  3. run examples/exp_airsim_multi_demo_gpu_mppi.yaml (same scenario,
     planner swapped to GPU MPPI at the 3D Pareto cell),
  4. ffmpeg each run's Drone1 FPV frames (frames_000_drone_00/) into a
     single-pane GIF,
  5. side-by-side via scripts/render_compare_gif.py, with labels and
     a max total width that lands the file under ~5 MB for README hero.

Run from the project root, with an AirSim server already up and
~/Documents/AirSim/settings.json declaring Drone1..Drone4 with
front_center camera on Drone1:
  python3 scripts/record_airsim_multi_compare.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from uav_nav_lab.recording import (
    frames_to_gif,
    pitch_front_center,
    run_uav_nav_experiment,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    {
        "tag": "mpc",
        "yaml": REPO_ROOT / "examples" / "exp_airsim_multi_demo.yaml",
        "results": REPO_ROOT / "results" / "airsim_multi_demo",
        "label": "MPC (n=16, h=30) · 12.85s",
    },
    {
        "tag": "gpu_mppi",
        "yaml": REPO_ROOT / "examples" / "exp_airsim_multi_demo_gpu_mppi.yaml",
        "results": REPO_ROOT / "results" / "airsim_multi_demo_gpu_mppi",
        "label": "GPU MPPI (n=64, h=20) · 17.65s",
    },
]
COMPARE_OUT = REPO_ROOT / "docs" / "images" / "compare_airsim_multi_mpc_vs_gpu_mppi.gif"


def main() -> int:
    print("[1/4] camera pitch")
    pitch_front_center(vehicle_name="Drone1")
    pane_gifs: list[Path] = []
    for i, run in enumerate(RUNS, start=2):
        print(f"[{i}/4] run {run['tag']}: {run['yaml'].name}")
        run_uav_nav_experiment(run["yaml"], run["results"], repo_root=REPO_ROOT)
        frames = run["results"] / "frames_000_drone_00"
        if not frames.is_dir():
            raise FileNotFoundError(f"no Drone1 frames at {frames}")
        out = Path("/tmp") / f"_compare_{run['tag']}.gif"
        frames_to_gif(frames, out, fps=15, width=480, target_seconds=7.0)
        pane_gifs.append(out)
    print("[4/4] side-by-side compose")
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "render_compare_gif.py"),
         str(pane_gifs[0]), str(pane_gifs[1]),
         "--out", str(COMPARE_OUT),
         "--left-label", RUNS[0]["label"],
         "--right-label", RUNS[1]["label"],
         "--fps", "12", "--max-total-width", "800", "--frame-stride", "2"],
        cwd=REPO_ROOT, check=True,
    )
    for g in pane_gifs:
        g.unlink(missing_ok=True)
    print(f"[done] {COMPARE_OUT}  ({COMPARE_OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
