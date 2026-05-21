"""Behavioral-fingerprint metrics for the intersection cells.

After the velocity sweep showed success rate is flat (both planners
collision-free across intruder velocities 0-2 m/s), this script
quantifies the *behavioral* signature each planner leaves:

- min_clearance        — min drone-intruder distance over the episode (m)
- min_ttc              — min time-to-closest-approach when closing in (s)
- max_lateral_dev      — max perpendicular deviation from start→goal line (m)
- path_time            — episode duration to goal (s)
- stop_fraction        — fraction of cruise time with |v| < STOP_THRESHOLD
- max_dcmd             — max |Δcmd| step-to-step (m/s per dt, jerk proxy)
- plan_ms_after_warm   — median planner_dt_ms excluding the first replan
                         (which is Dijkstra-cache warmup)

Aggregates over n=5 episodes × N_drones per cell. Prints a markdown
table with mean ± 1.96·SEM. Run after intersection_v1_{mpc,mppi} and
intersection_4way_{mpc,mppi} have produced their results dirs.

Usage:
    python3 scripts/intersection_fingerprint.py
"""
import json
from pathlib import Path

import numpy as np
import yaml

STOP_THRESHOLD = 1.0   # m/s — below this counts as "stopped"
TTC_HORIZON    = 5.0   # s   — max look-ahead for closest-approach time
CELLS = [
    ("v1",         "results/intersection_v1_mpc",                "results/intersection_v1_mppi",                "examples/exp_intersection_v1_mpc.yaml",                2),
    ("4way",       "results/intersection_4way_mpc",              "results/intersection_4way_mppi",              "examples/exp_intersection_4way_mpc.yaml",              4),
    ("chokepoint", "results/intersection_chokepoint_v1_mpc",     "results/intersection_chokepoint_v1_mppi",     "examples/exp_intersection_chokepoint_v1_mpc.yaml",     2),
    ("wave",       "results/intersection_wave_v1_mpc",           "results/intersection_wave_v1_mppi",           "examples/exp_intersection_wave_v1_mpc.yaml",           2),
]
N_EPS = 5


def _reflect(p, lim):
    v = p % (2.0 * lim)
    if v > lim:
        v = 2.0 * lim - v
    return v


def _intruder_pos_vel(start, vel, t, size):
    """Return (pos, vel) of a reflecting intruder at time t."""
    p = np.asarray(start, dtype=float) + np.asarray(vel, dtype=float) * t
    v_out = np.asarray(vel, dtype=float).copy()
    pos_out = np.zeros(3)
    for k in range(3):
        lim = size[k]
        n_reflects = int(p[k] // lim)
        if n_reflects % 2 == 0:
            pos_out[k] = p[k] - n_reflects * lim
        else:
            pos_out[k] = lim - (p[k] - n_reflects * lim)
            v_out[k] = -v_out[k]
    return pos_out, v_out


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
    return float(max(devs))


def _episode_metrics(d, drone_cfg, intruder_cfg, size, dt):
    steps = d["steps"]
    traj = np.array([s["true_pos"] for s in steps])
    vels = np.array([s["true_vel"] for s in steps])
    cmds = np.array([s["cmd"] for s in steps])
    ts = np.array([s["t"] for s in steps])

    intruder_pos = np.zeros_like(traj)
    intruder_vel = np.zeros_like(vels)
    for k, t in enumerate(ts):
        intruder_pos[k], intruder_vel[k] = _intruder_pos_vel(
            intruder_cfg["start"], intruder_cfg["velocity"], t, size)

    clearances = np.linalg.norm(traj - intruder_pos, axis=1)
    min_clearance = float(clearances.min())

    # TTC: closest-approach time when relative motion is closing.
    ttcs = []
    for k in range(len(traj)):
        rp = intruder_pos[k] - traj[k]
        rv = intruder_vel[k] - vels[k]
        rv_sq = float(np.dot(rv, rv))
        if rv_sq < 1e-9:
            continue
        t_min = float(-np.dot(rp, rv) / rv_sq)
        if 0 < t_min < TTC_HORIZON:
            ttcs.append(t_min)
    min_ttc = float(min(ttcs)) if ttcs else float(TTC_HORIZON)

    max_dev = _perp_deviation(traj, drone_cfg["start"], drone_cfg["goal"])
    path_time = float(ts[-1])

    # stop_fraction over the cruise portion (drop first/last 10 frames)
    if len(vels) > 20:
        cruise = vels[10:-10]
    else:
        cruise = vels
    speeds = np.linalg.norm(cruise, axis=1)
    stop_frac = float(np.mean(speeds < STOP_THRESHOLD))

    # max |Δcmd| step-to-step (m/s difference between successive cmd vectors)
    dcmd = np.linalg.norm(np.diff(cmds, axis=0), axis=1)
    max_dcmd = float(dcmd.max()) if len(dcmd) else 0.0

    # planner_dt_ms median excluding first replan (warmup)
    pm = [float(r["planner_dt_ms"]) for r in d.get("replans", []) if "planner_dt_ms" in r]
    plan_ms = float(np.median(pm[1:])) if len(pm) > 1 else float("nan")

    return dict(
        min_clearance=min_clearance,
        min_ttc=min_ttc,
        max_dev=max_dev,
        path_time=path_time,
        stop_frac=stop_frac,
        max_dcmd=max_dcmd,
        plan_ms=plan_ms,
    )


def cell_fingerprint(run_dir, yaml_path, n_drones):
    cfg = yaml.safe_load(open(yaml_path))
    size = cfg["scenario"]["size"]
    dt = cfg["simulator"]["dt"]
    drones_cfg = cfg["scenario"]["drones"]
    intruder_cfg = cfg["scenario"]["dynamic_obstacles"][0]
    run = Path(run_dir)
    metrics = {k: [] for k in
               ("min_clearance", "min_ttc", "max_dev", "path_time",
                "stop_frac", "max_dcmd", "plan_ms")}
    for ep in range(N_EPS):
        for di in range(n_drones):
            d = json.load(open(run / f"episode_{ep:03d}_drone_{di:02d}.json"))
            m = _episode_metrics(d, drones_cfg[di], intruder_cfg, size, dt)
            for k, v in m.items():
                metrics[k].append(v)
    return metrics


def _stats(xs):
    xs = np.asarray([x for x in xs if not np.isnan(x)])
    if len(xs) == 0:
        return float("nan"), float("nan")
    return float(np.mean(xs)), 1.96 * float(np.std(xs)) / np.sqrt(len(xs))


def fmt(stat):
    m, h = stat
    return f"{m:.2f} ± {h:.2f}"


def main() -> int:
    rows = []
    for tag, mpc_dir, mppi_dir, yaml_path, n_drones in CELLS:
        mpc = cell_fingerprint(mpc_dir, yaml_path, n_drones)
        mppi_yaml = yaml_path.replace("_mpc.yaml", "_mppi.yaml")
        mppi = cell_fingerprint(mppi_dir, mppi_yaml, n_drones)
        rows.append((tag, n_drones, mpc, mppi))

    print("| cell | n drones | metric | MPC | MPPI |")
    print("|------|----------|--------|-----|------|")
    METRIC_LABELS = [
        ("min_clearance", "min clearance (m)"),
        ("min_ttc",       "min TTC (s)"),
        ("max_dev",       "max lateral dev (m)"),
        ("path_time",     "path time (s)"),
        ("stop_frac",     "stop fraction"),
        ("max_dcmd",      "max |Δcmd| (m/s)"),
        ("plan_ms",       "plan time (ms)"),
    ]
    for tag, n_drones, mpc, mppi in rows:
        for key, label in METRIC_LABELS:
            print(f"| {tag} | {n_drones} | {label} | {fmt(_stats(mpc[key]))} | {fmt(_stats(mppi[key]))} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
