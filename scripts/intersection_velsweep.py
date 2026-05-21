"""Run the intersection_v1 cell at a range of intruder velocities for both
MPC and CPU MPPI, then report success rates + trajectory-level metrics.

Usage:
    python3 scripts/intersection_velsweep.py

Produces:
    results/intersection_velsweep_{mpc,mppi}_v{vx}.{1f}/   per-cell run dirs
    stdout: markdown table of success counts + trajectory metrics

Backs the velocity-sweep block in docs/findings.md (Intersection
coordination section).
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

VELOCITIES = [0.0, 0.3, 0.5, 1.0, 2.0]
N_EPS = 5
N_DRONES = 2


def _reflect(p, lim):
    v = p % (2.0 * lim)
    if v > lim:
        v = 2.0 * lim - v
    return v


def _intruder_pos(start, vel, t, size):
    p = np.asarray(start, dtype=float) + np.asarray(vel, dtype=float) * t
    return np.array([_reflect(p[k], size[k]) for k in range(3)])


def _perp_deviation(traj, start, goal):
    s = np.asarray(start, dtype=float)
    g = np.asarray(goal, dtype=float)
    line = g - s
    norm = np.linalg.norm(line)
    if norm < 1e-9:
        return 0.0
    unit = line / norm
    devs = []
    for p in traj:
        v = p - s
        proj = np.dot(v, unit) * unit
        devs.append(np.linalg.norm(v - proj))
    return max(devs)


def _min_cruise_speed(steps):
    speeds = np.array([np.linalg.norm(s["true_vel"]) for s in steps])
    if len(speeds) > 20:
        speeds = speeds[10:-10]
    return float(speeds.min())


def run_cell(planner: str, vx: float) -> Path:
    base = yaml.safe_load(open(f"examples/exp_intersection_v1_{planner}.yaml"))
    tag = f"velsweep_{planner}_v{vx:.1f}"
    cfg = json.loads(json.dumps(base))
    cfg["name"] = f"intersection_{tag}"
    cfg["scenario"]["dynamic_obstacles"][0]["velocity"] = [vx, 0.0, 0.0]
    cfg["output"]["dir"] = f"results/intersection_{tag}"
    tmp_yaml = Path("/tmp") / f"{tag}.yaml"
    with open(tmp_yaml, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    subprocess.run(["uav-nav", "run", str(tmp_yaml)], check=True)
    return Path(cfg["output"]["dir"])


def cell_metrics(planner: str, vx: float) -> dict:
    base = yaml.safe_load(open(f"examples/exp_intersection_v1_{planner}.yaml"))
    dt = base["simulator"]["dt"]
    size = base["scenario"]["size"]
    drones_cfg = base["scenario"]["drones"]
    intruder_start = base["scenario"]["dynamic_obstacles"][0]["start"]
    run_dir = Path(f"results/intersection_velsweep_{planner}_v{vx:.1f}")

    min_dists, deviations, times, speeds = [], [], [], []
    n_succ = n_coll = n_total = 0
    for ep in range(N_EPS):
        for di in range(N_DRONES):
            d = json.load(open(run_dir / f"episode_{ep:03d}_drone_{di:02d}.json"))
            n_total += 1
            outcome = d["outcome"]
            n_succ += outcome == "success"
            n_coll += outcome == "collision"
            steps = d["steps"]
            traj = np.array([s["true_pos"] for s in steps])
            ts = np.array([s["t"] for s in steps])
            intruder_traj = np.array([
                _intruder_pos(intruder_start, [vx, 0.0, 0.0], t, size) for t in ts
            ])
            min_dists.append(float(np.linalg.norm(traj - intruder_traj, axis=1).min()))
            deviations.append(_perp_deviation(traj, drones_cfg[di]["start"], drones_cfg[di]["goal"]))
            times.append(float(ts[-1]))
            speeds.append(_min_cruise_speed(steps))

    def stats(xs):
        return float(np.mean(xs)), 1.96 * float(np.std(xs)) / np.sqrt(len(xs))

    return dict(
        outcomes=(n_succ, n_coll, n_total),
        min_dist=stats(min_dists), deviation=stats(deviations),
        time=stats(times), min_speed=stats(speeds),
    )


def main() -> int:
    for planner in ("mpc", "mppi"):
        for vx in VELOCITIES:
            run_cell(planner, vx)

    print("\n=== success counts ===")
    print("| vel (m/s) | MPC succ/total | MPC coll | MPPI succ/total | MPPI coll |")
    print("|-----------|----------------|----------|-----------------|-----------|")
    for vx in VELOCITIES:
        m = cell_metrics("mpc", vx)
        p = cell_metrics("mppi", vx)
        print(f"| {vx:.1f}       | {m['outcomes'][0]}/{m['outcomes'][2]}            | "
              f"{m['outcomes'][1]}/{m['outcomes'][2]}      | {p['outcomes'][0]}/{p['outcomes'][2]}             | "
              f"{p['outcomes'][1]}/{p['outcomes'][2]}     |")

    print("\n=== trajectory metrics (mean ± 1.96·SEM over n=10 drone-eps) ===")
    print("| vel (m/s) | MPC min-dist | MPPI min-dist | MPC detour | MPPI detour | MPC min cruise speed | MPPI min cruise speed |")
    print("|---|---|---|---|---|---|---|")
    for vx in VELOCITIES:
        m = cell_metrics("mpc", vx)
        p = cell_metrics("mppi", vx)
        fmt = lambda x: f"{x[0]:.2f} ± {x[1]:.2f}"
        print(f"| {vx:.1f} | {fmt(m['min_dist'])} | {fmt(p['min_dist'])} | "
              f"{fmt(m['deviation'])} | {fmt(p['deviation'])} | "
              f"{fmt(m['min_speed'])} | {fmt(p['min_speed'])} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
