"""Record the lab's first LEARNED policy flying in photorealistic AirSim Blocks.

The neta-A teammate-token deep-set (student<-conv, distilled from a right-of-way
convention teacher; scripts/_swarm_policy.py) is rolled out on an N=4 antipodal
swap, then replayed on four AirSim quadrotors while an external "cine" camera
orbits the converging fleet. The learned policy turns the symmetric head-on hub
into a clean roundabout -- here in 3D, with rotors and shadows.

The policy is planar (single-integrator, ego-goal frame); we compute the rollout
offline from the cached model and teleport the drones along it each frame, so the
photoreal shot matches the learned trajectory exactly. The orbit camera tracks the
live fleet centroid (simGetVehiclePose), which sidesteps the per-drone spawn offset.

  python scripts/swarm_bc_symmetry_phase.py --episodes 1   # writes the model cache
  python scripts/record_airsim_swarm_policy.py             # needs Blocks on :41451
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _swarm_policy as sp  # noqa: E402

from uav_nav_lab.recording import frames_to_gif  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "results" / "swarm_bc_models.npz"
FRAMES_DIR = REPO_ROOT / "results" / "airsim_policy_frames"
GIF_OUT = REPO_ROOT / "docs" / "images" / "swarm_airsim_policy.gif"
DRONES = ("Drone1", "Drone2", "Drone3", "Drone4")

PKEYS = ["phi1", "phi1b", "phi2", "phi2b", "ego1", "ego1b",
         "out1", "out1b", "out2", "out2b"]
SKEYS = ["em", "es", "pm", "ps"]

ALT = -30.0          # NED altitude (ENU 30 m), clear of the Blocks cubes
SCALE = 1.15         # world metres per policy unit (ring ~8 -> ~9 m)
ORBIT_R = 9.5        # camera orbit radius around the fleet centroid (m)
ORBIT_UP = 6.0       # camera height above the centroid (m)


def _load(prefix, z):
    P = {k: z[f"{prefix}_{k}"] for k in PKEYS}
    stats = {k: z[f"{prefix}_s_{k}"] for k in SKEYS}
    return sp.make_student_controller(P, stats)


def _aim(cam, target):
    import airsim
    d = np.asarray(target) - np.asarray(cam)
    yaw = math.atan2(d[1], d[0])
    pitch = math.atan2(-d[2], math.hypot(d[0], d[1]))
    return airsim.Pose(airsim.Vector3r(*cam), airsim.to_quaternion(pitch, 0.0, yaw))


def _orbit_pose(angle_deg, center):
    a = math.radians(angle_deg)
    cam = np.asarray(center) + np.array(
        [ORBIT_R * math.cos(a), ORBIT_R * math.sin(a), -ORBIT_UP])
    return _aim(cam, center)


def _smooth(traj, k=2):
    """Light temporal smoothing of the (F,N,2) trajectory for a cleaner glide."""
    F = traj.shape[0]
    out = traj.copy()
    for f in range(F):
        lo, hi = max(0, f - k), min(F, f + k + 1)
        out[f] = traj[lo:hi].mean(axis=0)
    return out


def main() -> int:
    import airsim
    from PIL import Image

    global ORBIT_R, ORBIT_UP, SCALE
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20076)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--orbit-r", type=float, default=ORBIT_R)
    ap.add_argument("--orbit-up", type=float, default=ORBIT_UP)
    ap.add_argument("--scale", type=float, default=SCALE)
    ap.add_argument("--out", default=str(GIF_OUT))
    args = ap.parse_args()
    ORBIT_R, ORBIT_UP, SCALE = args.orbit_r, args.orbit_up, args.scale

    if not CACHE.exists():
        print(f"missing {CACHE}; run swarm_bc_symmetry_phase.py first")
        return 1
    z = np.load(CACHE)
    ctrl = _load("conv", z)

    n = len(DRONES)
    rng = np.random.default_rng(args.seed)
    start, goal = sp.antipodal(n, rng)
    roll = sp.rollout(start, goal, ctrl, record=True)
    print(f"[policy] N={n} seed={args.seed} success={roll.success} frames={len(roll.traj)}",
          flush=True)
    traj = _smooth(np.array(roll.traj))          # (F, N, 2) in policy units
    F = traj.shape[0]

    # NED world targets (each drone at its policy position, at fixed altitude)
    world = np.zeros((F, n, 3))
    world[:, :, 0] = traj[:, :, 0] * SCALE
    world[:, :, 1] = traj[:, :, 1] * SCALE
    world[:, :, 2] = ALT
    # per-drone yaw from velocity (face travel direction)
    yaw = np.zeros((F, n))
    for i in range(n):
        for f in range(F):
            d = world[min(f + 1, F - 1), i, :2] - world[max(f - 1, 0), i, :2]
            yaw[f, i] = math.atan2(d[1], d[0]) if np.linalg.norm(d) > 1e-3 else \
                (yaw[f - 1, i] if f else 0.0)

    c = airsim.MultirotorClient()
    c.confirmConnection()
    c.reset()
    time.sleep(1.0)
    for v in DRONES:
        c.enableApiControl(True, v)
        c.armDisarm(True, v)

    def place(f):
        for i, v in enumerate(DRONES):
            p = world[f, i]
            pose = airsim.Pose(airsim.Vector3r(*p),
                               airsim.to_quaternion(0.0, 0.0, yaw[f, i]))
            c.simSetVehiclePose(pose, True, v)

    place(0)
    time.sleep(0.5)

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for f in FRAMES_DIR.glob("*.png"):
        f.unlink()

    print("[record] orbiting the learned roundabout...", flush=True)
    saved = 0
    for f in range(F):
        place(f)
        centroid = np.mean([[c.simGetVehiclePose(v).position.x_val,
                             c.simGetVehiclePose(v).position.y_val,
                             c.simGetVehiclePose(v).position.z_val]
                            for v in DRONES], axis=0)
        pose = _orbit_pose(35.0 + 0.6 * f, centroid)
        c.simSetCameraPose("cine", pose, external=True)
        try:
            r = c.simGetImages(
                [airsim.ImageRequest("cine", airsim.ImageType.Scene, False, False)],
                vehicle_name="", external=True)
            if r and r[0].image_data_uint8:
                im = r[0]
                flat = np.frombuffer(im.image_data_uint8, dtype=np.uint8)
                ch = flat.size // (im.width * im.height)
                arr = flat.reshape(im.height, im.width, ch)[:, :, :3][:, :, ::-1]
                Image.fromarray(arr).save(str(FRAMES_DIR / f"frame_{saved:04d}.png"))
                saved += 1
        except Exception as e:  # noqa: BLE001
            print(f"  frame {f} error: {e}", flush=True)
    print(f"  {saved} frames", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    nsrc = frames_to_gif(FRAMES_DIR, out, fps=args.fps, width=520,
                         target_seconds=8.0, frame_pattern="frame_%04d.png",
                         name_contains=None)
    print(f"[gif] {out}  ({out.stat().st_size // 1024} KB)  ({nsrc} src frames)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
