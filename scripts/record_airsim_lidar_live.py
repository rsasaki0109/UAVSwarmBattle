"""Record AirSim's onboard LiDAR building a 3-D point cloud of the Blocks world
while the drone flies — a SLAM-style scan that shows off the sensor simulation.

Launches `uav-nav run` (the 4-drone crossing) as a subprocess and, concurrently,
polls Drone1's 16-channel LiDAR. Each sweep's points (sensor-local frame) are
transformed to world coordinates via the sensor pose and accumulated; the result
is animated as a growing point cloud coloured by height, with the drone's track
and an orbiting camera.

Needs an AirSim server up and ~/Documents/AirSim/settings.json declaring Drone1
with a `LidarFront` sensor (16 channels). Run from the project root:
  python scripts/record_airsim_lidar_live.py
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation, PillowWriter

from uav_nav_lab.recording import frames_to_gif  # noqa: F401  (kept for parity / unused)
from uav_nav_lab.recording.experiment_runner import _build_cli_command

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_YAML = REPO_ROOT / "examples" / "exp_airsim_multi_demo.yaml"
EXP_RUN_DIR = REPO_ROOT / "results" / "airsim_multi_demo"
GIF_OUT = REPO_ROOT / "docs" / "images" / "swarm_airsim_lidar.gif"


def _quat_rot(q, v):
    """Rotate vector v (3,) by AirSim quaternion q (w,x,y,z)."""
    w, x, y, z = q
    # rotation matrix from quaternion
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return v @ R.T


def main() -> int:
    import airsim

    c = airsim.MultirotorClient()
    c.confirmConnection()
    c.reset()
    time.sleep(1.0)

    print("[1/2] launch flight + poll LiDAR concurrently")
    env = {**os.environ, "UAV_NAV_NO_CAMERA": "1"}
    proc = subprocess.Popen(_build_cli_command(EXP_YAML), cwd=REPO_ROOT, env=env)

    clouds: list[np.ndarray] = []   # per-sweep world points (subsampled)
    drone_track: list[np.ndarray] = []
    fps = 12
    interval = 1.0 / fps
    tail = 0
    while True:
        t0 = time.perf_counter()
        try:
            ld = c.getLidarData(lidar_name="LidarFront", vehicle_name="Drone1")
            pts = np.array(ld.point_cloud, dtype=np.float32)
            if pts.size >= 3:
                pts = pts.reshape(-1, 3)
                if pts.shape[0] > 90:           # subsample each sweep (3-D scatter is slow)
                    pts = pts[np.random.default_rng(len(clouds)).choice(pts.shape[0], 90, replace=False)]
                p = ld.pose.position
                q = (ld.pose.orientation.w_val, ld.pose.orientation.x_val,
                     ld.pose.orientation.y_val, ld.pose.orientation.z_val)
                sensor_pos = np.array([p.x_val, p.y_val, p.z_val])
                world = sensor_pos + _quat_rot(q, pts)      # NED world
                clouds.append(world)
                drone_track.append(sensor_pos)
        except Exception as e:  # noqa: BLE001
            print(f"  lidar poll error: {e}")
        if proc.poll() is not None:
            tail += 1
            if tail > fps:
                break
        dt = time.perf_counter() - t0
        if dt < interval:
            time.sleep(interval - dt)
    proc.wait()
    print(f"  {len(clouds)} sweeps, {sum(len(x) for x in clouds)} points")
    if not clouds:
        print("no lidar points captured"); return 1

    # NED -> plot frame (East, North, Up)
    def to_plot(a):
        return np.column_stack([a[:, 1], a[:, 0], -a[:, 2]])

    track = to_plot(np.array(drone_track))
    allpts = to_plot(np.vstack(clouds))
    cum = np.cumsum([len(x) for x in clouds])   # cloud[i] ends at cum[i]

    fig = plt.figure(figsize=(6.6, 6.6))
    fig.patch.set_facecolor("#0d1117")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0d1117")
    mn, mx = allpts.min(0), allpts.max(0)
    ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(min(mn[2], 0), mx[2])
    ax.set_box_aspect((mx[0] - mn[0], mx[1] - mn[1], max(mx[2] - min(mn[2], 0), 1)))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#0d1117"); axis.pane.set_edgecolor("#30363d")
    ax.grid(False)
    ax.set_title("AirSim onboard LiDAR — scanning the Blocks world in 3-D",
                 color="#e6edf3", fontsize=13, fontweight="bold", pad=4)

    scat = ax.scatter([], [], [], s=3, c=[], cmap="turbo", vmin=mn[2], vmax=mx[2], depthshade=False)
    tline, = ax.plot([], [], [], "-", color="#ff7b72", lw=2.0)
    dpt = ax.scatter([], [], [], s=80, c="#ffd33d", edgecolors="white", linewidths=0.8)

    n = len(clouds)
    # decimate to <=60 animation frames (each shows the cloud accumulated so far)
    n_out = min(60, n)
    sweep_at = [min(n - 1, round(k * (n - 1) / max(n_out - 1, 1))) for k in range(n_out)]

    def update(f):
        i = sweep_at[min(f, n_out - 1)]
        end = cum[i]
        pp = allpts[:end]
        scat._offsets3d = (pp[:, 0], pp[:, 1], pp[:, 2])
        scat.set_array(pp[:, 2])
        tline.set_data(track[:i + 1, 0], track[:i + 1, 1]); tline.set_3d_properties(track[:i + 1, 2])
        dpt._offsets3d = (track[i:i + 1, 0], track[i:i + 1, 1], track[i:i + 1, 2])
        ax.view_init(elev=28.0, azim=(40.0 + 1.2 * f) % 360.0)
        return [scat, tline, dpt]

    anim = FuncAnimation(fig, update, frames=n_out, interval=1000 / fps, blit=False)
    GIF_OUT.parent.mkdir(parents=True, exist_ok=True)
    anim.save(GIF_OUT, writer=PillowWriter(fps=fps), dpi=80, savefig_kwargs={"facecolor": "#0d1117"})
    print(f"[gif] {GIF_OUT}  ({GIF_OUT.stat().st_size // 1024} KB)  ({n_out} frames, {n} sweeps)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
