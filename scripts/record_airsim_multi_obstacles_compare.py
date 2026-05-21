"""Record an AirSim 4-drone MPC-vs-GPU-MPPI side-by-side GIF with
virtual planner obstacles forcing curving trajectories.

Same pipeline as `record_airsim_multi_compare.py` (Drone1 FPV from
each run, ffmpeg → side-by-side), but the source YAMLs are the
`exp_airsim_multi_obstacles_demo*.yaml` pair, which add a centerline-
gate obstacle set + `planner.inflate: 2` so both planners detour
visibly around the (30, 30) crossing centre.

The original demo GIF (straight-line crossings) lives at
`docs/images/compare_airsim_multi_mpc_vs_gpu_mppi.gif` and remains
the reference for the no-obstacle parity story. This script produces
`docs/images/compare_airsim_multi_obstacles.gif` for README hero use
when the visual emphasis is on planner dynamics rather than
ceiling parity.

Run from the project root, with an AirSim Blocks server already up:
  python3 scripts/record_airsim_multi_obstacles_compare.py
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
        "yaml": REPO_ROOT / "examples" / "exp_airsim_multi_obstacles_demo.yaml",
        "results": REPO_ROOT / "results" / "airsim_multi_obstacles_demo",
        "label": "MPC (n=16, h=30) · 13.40s · path 57-67 m · perp 5.4-9.2 m",
    },
    {
        "tag": "gpu_mppi",
        "yaml": REPO_ROOT / "examples" / "exp_airsim_multi_obstacles_demo_gpu_mppi.yaml",
        "results": REPO_ROOT / "results" / "airsim_multi_obstacles_demo_gpu_mppi",
        "label": "GPU MPPI (n=64, h=20) · 13.20s · path 52-55 m · perp 4.7-5.7 m",
    },
]
COMPARE_OUT = REPO_ROOT / "docs" / "images" / "compare_airsim_multi_obstacles.gif"


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
        out = Path("/tmp") / f"_compare_obs_{run['tag']}.gif"
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
