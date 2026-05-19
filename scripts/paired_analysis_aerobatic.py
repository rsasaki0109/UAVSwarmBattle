"""Aerobatic / synchronized-loop tracking analysis.

Loads dummy_3d multi-drone aerobatic results from two run directories
(e.g. MPC vs GPU MPPI) and computes per-drone tracking RMSE, per-drone
max-error, inter-drone phase offset RMSE, and collision rate.

Usage:
    python3 scripts/paired_analysis_aerobatic.py \
        results/aerobatic_loop4_mpc \
        results/aerobatic_loop4_gpu_mppi
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

import numpy as np


def load_episode(run_dir: Path, ep_idx: int, n_drones: int = 4):
    drones = []
    for i in range(n_drones):
        path = run_dir / f"episode_{ep_idx:03d}_drone_{i:02d}.json"
        if not path.exists():
            return None
        drones.append(json.loads(path.read_text()))
    return drones


def episode_metrics(drones: list[dict], center: np.ndarray, normal_axis: str = "y"):
    """Return per-drone tracking RMSE/max, per-pair phase RMSE,
    collision boolean per drone. `normal_axis` picks the loop plane:
    'y' = xz (vertical), 'z' = xy (horizontal oval / race)."""
    n = len(drones)
    track_rmse = []
    track_max = []
    collisions = []
    angles_per_drone: list[np.ndarray] = []
    for i, d in enumerate(drones):
        steps = d["steps"]
        errs = []
        ang = []
        for s in steps:
            if "reference_pos" not in s:
                continue
            true_p = np.asarray(s["true_pos"], dtype=float)
            ref_p = np.asarray(s["reference_pos"], dtype=float)
            errs.append(np.linalg.norm(true_p - ref_p))
            rel = true_p - center
            if normal_axis == "y":
                ang.append(math.atan2(rel[2], rel[0]))  # xz plane
            else:  # "z" — horizontal oval
                ang.append(math.atan2(rel[1], rel[0]))  # xy plane
        errs = np.asarray(errs)
        track_rmse.append(float(np.sqrt((errs * errs).mean())) if errs.size else math.nan)
        track_max.append(float(errs.max()) if errs.size else math.nan)
        collisions.append(d.get("outcome") == "collision")
        angles_per_drone.append(np.asarray(ang))
    # adjacent-pair phase offset
    phase_rmse: list[float] = []
    for i in range(n):
        j = (i + 1) % n
        a_i = angles_per_drone[i]
        a_j = angles_per_drone[j]
        m = min(len(a_i), len(a_j))
        if m == 0:
            phase_rmse.append(math.nan)
            continue
        diff = a_j[:m] - a_i[:m]
        # Expected offset = 2π/n (e.g. 90° for n=4); wrap to [-π, π]
        expected = 2.0 * math.pi / n
        err = np.mod(diff - expected + math.pi, 2.0 * math.pi) - math.pi
        phase_rmse.append(float(np.sqrt((err * err).mean())))
    return {
        "track_rmse": track_rmse,
        "track_max": track_max,
        "collisions": collisions,
        "phase_rmse_rad": phase_rmse,
    }


def summarise(name: str, run_dir: Path, n_drones: int, n_eps: int,
              center: np.ndarray, normal_axis: str = "y"):
    per_drone_rmse = [[] for _ in range(n_drones)]
    per_drone_max = [[] for _ in range(n_drones)]
    per_pair_phase = [[] for _ in range(n_drones)]
    collision_rate = [0 for _ in range(n_drones)]
    n_loaded = 0
    for ep in range(n_eps):
        drones = load_episode(run_dir, ep, n_drones)
        if drones is None:
            break
        m = episode_metrics(drones, center, normal_axis=normal_axis)
        for i in range(n_drones):
            per_drone_rmse[i].append(m["track_rmse"][i])
            per_drone_max[i].append(m["track_max"][i])
            per_pair_phase[i].append(m["phase_rmse_rad"][i])
            collision_rate[i] += int(m["collisions"][i])
        n_loaded += 1
    print(f"{name} ({run_dir}, n={n_loaded} eps):")
    overall_rmse = np.asarray([v for col in per_drone_rmse for v in col])
    overall_phase = np.asarray([v for col in per_pair_phase for v in col])
    print(f"  per-drone tracking RMSE: "
          f"{overall_rmse.mean():.3f} m (max episode {overall_rmse.max():.3f})")
    print(f"  per-drone max error:     "
          f"{np.asarray([v for col in per_drone_max for v in col]).mean():.3f} m")
    print(f"  phase-offset RMSE:       "
          f"{math.degrees(overall_phase.mean()):.2f}° "
          f"(max episode {math.degrees(overall_phase.max()):.2f}°)")
    print(f"  collisions: {sum(collision_rate)}/{n_drones * n_loaded} "
          f"drone-eps (= {100.0 * sum(collision_rate) / max(1, n_drones * n_loaded):.1f} %)")
    return {
        "rmse_all": overall_rmse,
        "phase_all": overall_phase,
        "collision_rate": sum(collision_rate),
        "n_drone_eps": n_drones * n_loaded,
    }


def main(mpc_dir: str, mppi_dir: str, n_drones: int = 4, n_eps: int = 30) -> int:
    # Try to extract center + normal_axis from config
    center = np.array([20.0, 20.0, 7.0])
    normal_axis = "y"
    try:
        import yaml as _yaml
        cfg = _yaml.safe_load(Path(mpc_dir, "config.yaml").read_text())
        c = cfg["scenario"].get("center", center)
        center = np.asarray(c, dtype=float)
        normal_axis = str(cfg["scenario"].get("normal_axis", "y"))
    except Exception:
        pass
    print(f"center: {center.tolist()}  normal_axis: {normal_axis}\n")
    a = summarise("MPC", Path(mpc_dir), n_drones, n_eps, center, normal_axis)
    print()
    b = summarise("GPU MPPI", Path(mppi_dir), n_drones, n_eps, center, normal_axis)
    print()
    # Paired-mean diff
    if a["rmse_all"].size and b["rmse_all"].size and a["rmse_all"].size == b["rmse_all"].size:
        d = b["rmse_all"] - a["rmse_all"]
        print(f"Per-(drone, ep) tracking RMSE diff (GPU - MPC): mean {d.mean():+.3f} m")
        if (d < 0).sum() > 0:
            print(f"  GPU better on {int((d < 0).sum())}/{d.size} drone-episodes")
        print(f"Phase RMSE diff (GPU - MPC) mean: "
              f"{math.degrees((b['phase_all'] - a['phase_all']).mean()):+.2f}°")
    return 0


if __name__ == "__main__":
    n_drones = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    n_eps = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    sys.exit(main(sys.argv[1], sys.argv[2], n_drones, n_eps))
