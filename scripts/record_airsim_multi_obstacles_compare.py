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

import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    {
        "tag": "mpc",
        "yaml": REPO_ROOT / "examples" / "exp_airsim_multi_obstacles_demo.yaml",
        "results": REPO_ROOT / "results" / "airsim_multi_obstacles_demo",
        "label": "MPC (n=16, h=30) · 10.85s · path 56-61 m",
    },
    {
        "tag": "gpu_mppi",
        "yaml": REPO_ROOT / "examples" / "exp_airsim_multi_obstacles_demo_gpu_mppi.yaml",
        "results": REPO_ROOT / "results" / "airsim_multi_obstacles_demo_gpu_mppi",
        "label": "GPU MPPI (n=64, h=20) · 11.55s · path 50-51 m",
    },
]
COMPARE_OUT = REPO_ROOT / "docs" / "images" / "compare_airsim_multi_obstacles.gif"


def _setup_camera() -> None:
    import airsim  # type: ignore[import-not-found]
    c = airsim.MultirotorClient()
    c.confirmConnection()
    time.sleep(1.0)
    cam_pose = airsim.Pose(
        airsim.Vector3r(0.50, 0.0, 0.0),
        airsim.to_quaternion(-0.30, 0.0, 0.0),
    )
    c.simSetCameraPose("front_center", cam_pose, vehicle_name="Drone1")
    time.sleep(0.3)


def _run_experiment(yaml: Path, results: Path) -> None:
    if results.exists():
        shutil.rmtree(results)
    cmd = [sys.executable, "-c",
           "import sys; sys.argv=['uav-nav','run',str(r'%s')];"
           "from uav_nav_lab.cli import main; main()" % yaml]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _frames_to_gif(
    frames_dir: Path,
    out: Path,
    fps: int = 15,
    width: int = 480,
    target_seconds: float = 7.0,
) -> None:
    n_frames = sum(1 for p in frames_dir.iterdir()
                   if p.suffix == ".png" and "front_center" in p.name)
    desired = max(1, int(round(fps * target_seconds)))
    keep_every = max(1, n_frames // desired)
    palette = frames_dir / "_palette.png"
    pattern = str(frames_dir / "step_%04d_front_center.png")
    vf = (
        f"select='not(mod(n,{keep_every}))',"
        f"setpts=N/{fps}/TB,"
        f"scale={width}:-1:flags=lanczos"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", pattern,
         "-vf", f"{vf},palettegen=stats_mode=diff",
         str(palette)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", pattern, "-i", str(palette),
         "-lavfi", f"{vf} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5",
         "-loop", "0",
         str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    palette.unlink(missing_ok=True)


def main() -> int:
    print("[1/4] camera pitch")
    _setup_camera()
    pane_gifs: list[Path] = []
    for i, run in enumerate(RUNS, start=2):
        print(f"[{i}/4] run {run['tag']}: {run['yaml'].name}")
        _run_experiment(run["yaml"], run["results"])
        frames = run["results"] / "frames_000_drone_00"
        if not frames.is_dir():
            raise FileNotFoundError(f"no Drone1 frames at {frames}")
        out = Path("/tmp") / f"_compare_obs_{run['tag']}.gif"
        _frames_to_gif(frames, out)
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
