"""Record a Foxglove-style multi-panel robotics dashboard from one AirSim flight.

Launches `uav-nav run` (the 4-drone crossing) and, concurrently, polls four data
streams off the live sim — Drone1's FPV camera, Drone1's LiDAR, and all four
drone poses — then composes them into a synced 2x2 dashboard (dark Foxglove
aesthetic):

  ┌ FPV camera (Drone1) ───┬ LiDAR (top-down) ──────┐
  ├ Scene — 4 drones ──────┼ Telemetry — min sep ───┤

Real Foxglove Studio needs a GUI to load an MCAP/ROS bag; this reproduces the
*look* headless with matplotlib. Needs an AirSim server up + settings.json with
Drone1's front_center camera and LidarFront sensor. Run from the project root:
  python scripts/record_airsim_dashboard.py
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from uav_nav_lab.recording.experiment_runner import _build_cli_command

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_YAML = REPO_ROOT / "examples" / "exp_airsim_multi_demo.yaml"
EXP_RUN_DIR = REPO_ROOT / "results" / "airsim_multi_demo"
GIF_OUT = REPO_ROOT / "docs" / "images" / "swarm_airsim_dashboard.gif"
VEHICLES = ["Drone1", "Drone2", "Drone3", "Drone4"]
BG = "#0d1117"
PANEL = "#161b22"
ACCENT = "#58a6ff"


def main() -> int:
    import airsim
    from PIL import Image

    c = airsim.MultirotorClient()
    c.confirmConnection()
    c.reset()
    time.sleep(1.0)

    print("[1/2] launch flight + poll camera/lidar/poses", flush=True)
    env = {**os.environ, "UAV_NAV_NO_CAMERA": "1"}
    proc = subprocess.Popen(_build_cli_command(EXP_YAML), cwd=REPO_ROOT, env=env)

    cam_frames, lidar_xyz, poses, times = [], [], [], []
    fps = 10
    interval = 1.0 / fps
    t_start = time.perf_counter()
    tail = 0
    while True:
        t0 = time.perf_counter()
        try:
            img = c.simGetImages([airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)],
                                 vehicle_name="Drone1")[0]
            flat = np.frombuffer(img.image_data_uint8, dtype=np.uint8)
            ch = flat.size // (img.height * img.width) if img.height * img.width else 0
            if ch >= 3:
                arr = flat.reshape(img.height, img.width, ch)[:, :, :3][:, :, ::-1]
                cam_frames.append(np.asarray(Image.fromarray(arr).resize((300, 169))))
            else:
                cam_frames.append(cam_frames[-1] if cam_frames else np.zeros((169, 300, 3), np.uint8))
            ld = c.getLidarData(lidar_name="LidarFront", vehicle_name="Drone1")
            lp = np.array(ld.point_cloud, dtype=np.float32)
            if lp.size >= 3:
                lp = lp.reshape(-1, 3)
                if lp.shape[0] > 250:
                    lp = lp[np.random.default_rng(len(lidar_xyz)).choice(lp.shape[0], 250, replace=False)]
                sp = ld.pose.position
                lidar_xyz.append(lp + np.array([sp.x_val, sp.y_val, sp.z_val]))
            else:
                lidar_xyz.append(np.zeros((0, 3), np.float32))
            poses.append(np.array([[c.simGetVehiclePose(v).position.x_val,
                                    c.simGetVehiclePose(v).position.y_val,
                                    c.simGetVehiclePose(v).position.z_val] for v in VEHICLES]))
            times.append(time.perf_counter() - t_start)
        except Exception as e:  # noqa: BLE001
            print(f"  poll error: {e}", flush=True)
        if proc.poll() is not None:
            tail += 1
            if tail > fps:
                break
        dt = time.perf_counter() - t0
        if dt < interval:
            time.sleep(interval - dt)
    proc.wait()
    nF = len(poses)
    print(f"  {nF} frames captured", flush=True)
    if nF == 0:
        print("nothing captured"); return 1

    poses = np.array(poses)                      # (T,4,3) NED
    times = np.array(times)
    # trim leading reset/teleport frames where the drones are still stacked at the
    # origin (min separation ~0) so the telemetry doesn't start with a 0->spread jump
    sep0 = np.array([min(np.linalg.norm(poses[t, i] - poses[t, j])
                         for i in range(4) for j in range(i + 1, 4)) for t in range(nF)])
    start = int(np.argmax(sep0 > 5.0)) if (sep0 > 5.0).any() else 0
    poses, times = poses[start:], times[start:] - times[start]
    cam_frames, lidar_xyz = cam_frames[start:], lidar_xyz[start:]
    nF = len(poses)
    # min pairwise separation over time
    minsep = []
    for t in range(nF):
        p = poses[t]
        d = min(np.linalg.norm(p[i] - p[j]) for i in range(4) for j in range(i + 1, 4))
        minsep.append(d)
    minsep = np.array(minsep)

    # plot helpers: NED -> (East, North); altitude = -z
    cols = plt.get_cmap("turbo")(np.linspace(0, 1, 4))
    decim = max(1, nF // 90)
    frames = list(range(0, nF, decim))

    fig = plt.figure(figsize=(9.2, 5.4)); fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.16,
                          left=0.04, right=0.985, top=0.92, bottom=0.08)
    axc = fig.add_subplot(gs[0, 0]); axl = fig.add_subplot(gs[0, 1])
    axs = fig.add_subplot(gs[1, 0]); axt = fig.add_subplot(gs[1, 1])
    fig.suptitle("UAVSwarmBattle · AirSim live dashboard", color="#e6edf3",
                 fontsize=14, fontweight="bold", x=0.5, y=0.985)

    def style(ax, title):
        ax.set_facecolor(PANEL)
        for s in ax.spines.values():
            s.set_color("#30363d")
        ax.set_title(title, color=ACCENT, fontsize=10, loc="left", fontweight="bold", family="monospace")
        ax.tick_params(colors="#6e7681", labelsize=7)

    # camera panel
    style(axc, "FPV camera · Drone1"); axc.set_xticks([]); axc.set_yticks([])
    im = axc.imshow(cam_frames[0])
    # lidar panel (top-down)
    style(axl, "LiDAR · top-down"); axl.set_aspect("equal")
    axl.set_xlim(0, 60); axl.set_ylim(0, 60); axl.set_xticks([]); axl.set_yticks([])
    lsc = axl.scatter([], [], s=2, c=[], cmap="turbo", vmin=24, vmax=40)
    # scene panel (4 drones top-down)
    style(axs, "Scene · 4 drones (top-down)"); axs.set_aspect("equal")
    axs.set_xlim(0, 60); axs.set_ylim(0, 60); axs.set_xticks([]); axs.set_yticks([])
    dots = axs.scatter(poses[0, :, 1], poses[0, :, 0], s=80, c=cols, edgecolors="white", linewidths=0.8, zorder=5)
    trails = [axs.plot([], [], "-", color=cols[i], lw=1.5, alpha=0.6)[0] for i in range(4)]
    # telemetry panel
    style(axt, "Telemetry · min separation (m)")
    axt.set_xlim(0, float(times[-1]) if times[-1] > 0 else 1); axt.set_ylim(0, max(minsep.max() * 1.1, 5))
    axt.axhline(0.8, color="#f85149", lw=1.2, ls="--", alpha=0.8)
    axt.text(0.02, 0.06, "collision 0.8 m", color="#f85149", fontsize=7, transform=axt.transAxes, family="monospace")
    tline, = axt.plot([], [], "-", color="#3fb950", lw=2.0)
    tnow = axt.scatter([], [], s=40, c="#3fb950", zorder=5)

    def update(fi):
        t = frames[fi]
        im.set_data(cam_frames[t])
        lp = lidar_xyz[t]
        if lp.shape[0]:
            lsc.set_offsets(np.column_stack([lp[:, 1], lp[:, 0]])); lsc.set_array(-lp[:, 2])
        else:
            lsc.set_offsets(np.empty((0, 2)))
        dots.set_offsets(np.column_stack([poses[t, :, 1], poses[t, :, 0]]))
        for i in range(4):
            trails[i].set_data(poses[:t + 1, i, 1], poses[:t + 1, i, 0])
        tline.set_data(times[:t + 1], minsep[:t + 1])
        tnow.set_offsets([[times[t], minsep[t]]])
        return [im, lsc, dots, tline, tnow] + trails

    anim = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=False)
    GIF_OUT.parent.mkdir(parents=True, exist_ok=True)
    anim.save(GIF_OUT, writer=PillowWriter(fps=fps), dpi=92, savefig_kwargs={"facecolor": BG})
    print(f"[gif] {GIF_OUT}  ({GIF_OUT.stat().st_size // 1024} KB)  ({len(frames)} frames / {nF} captured)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
