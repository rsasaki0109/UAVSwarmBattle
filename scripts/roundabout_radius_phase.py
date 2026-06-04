"""The Merry-Go-Round ring-radius trade-off: a smaller ring is faster but has a
lower capacity.

The explicit roundabout pays a ~63 % makespan premium because it rides a
half-circumference arc of the radius-R ring. A *smaller* ring is a shorter arc
(faster), but it packs the same N drones onto a shorter circumference (tighter,
so it collides at high density). So ring_radius should trade makespan against
capacity, and the safe radius should grow with N.

Sweep ring_radius x N on `planner.type: roundabout` (antipodal swap), reporting
success and makespan (ideal free-flight 2R_start/speed = 8.0 s, R_start=20).

  python scripts/roundabout_radius_phase.py --radius-list 8 12 16 20 --n-list 6 12 24 --episodes 20
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import math
import tempfile
from multiprocessing import Pool
from pathlib import Path

from uav_nav_lab.config import ExperimentConfig
from uav_nav_lab.runner.multi.experiment import run_experiment_multi

SPEED = 5.0
CX, CY = 25.0, 25.0
START_R = 20.0
IDEAL = 2 * START_R / SPEED


def _drones(n):
    out = []
    for k in range(n):
        a = 2.0 * math.pi * k / n
        out.append({"name": f"d{k}",
                    "start": [round(CX + START_R * math.cos(a), 3), round(CY + START_R * math.sin(a), 3)],
                    "goal": [round(CX - START_R * math.cos(a), 3), round(CY - START_R * math.sin(a), 3)],
                    "radius": 0.4, "start_jitter": 0.8})
    return out


def _cfg(n, ring, seed, n_eps, max_steps):
    return {
        "name": f"rad_n{n}_r{ring}", "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50],
                     "resolution": 1.0, "obstacles": {"type": "none"}, "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": {"type": "roundabout", "max_speed": SPEED, "replan_period": 0.05,
                    "center": [CX, CY], "ring_radius": ring, "exit_angle": 0.35,
                    "time_step": 0.05, "goal_radius": 1.5},
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/rad_tmp"},
    }


def _run_cell(job):
    ring, n, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, ring, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = (d["outcome"], float(d.get("final_t", 0.0)))
    return (ring, n, by_seed)


def _succ(bs):
    return sum(o == "success" for o, _ in bs.values())


def _mksp(bs):
    v = [t for o, t in bs.values() if o == "success"]
    return sum(v) / len(v) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius-list", type=float, nargs="+", default=[8, 12, 16, 20])
    ap.add_argument("--n-list", type=int, nargs="+", default=[6, 12, 24])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--out", default="results/roundabout_radius_phase.json")
    args = ap.parse_args()

    jobs = [(r, n, args.seed, args.episodes, args.max_steps)
            for n in args.n_list for r in args.radius_list]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    grid = {}
    for ring, n, bs in res:
        grid[(ring, n)] = bs

    report = {"ideal": IDEAL, "episodes": args.episodes, "cells": []}
    print(f"\nMerry-Go-Round ring radius x N (antipodal, n={args.episodes}; succ ; makespan, "
          f"ideal {IDEAL:.1f}s, arc=pi*ring/speed)")
    header = "ring |" + "".join(f" {('N='+str(n)):>14} |" for n in sorted(args.n_list))
    print(header)
    print("-" * len(header))
    for r in sorted(args.radius_list):
        arc_t = math.pi * r / SPEED
        row = f"{r:>4.0f} |"
        rec = {"ring": r, "arc_makespan_est": arc_t, "by_n": {}}
        for n in sorted(args.n_list):
            bs = grid[(r, n)]
            m = len(bs)
            row += f" {_succ(bs):>2}/{m:<2} {_mksp(bs):>6.2f}s |"
            rec["by_n"][n] = {"succ": _succ(bs), "makespan": _mksp(bs)}
        print(row + f"  (arc~{arc_t:.1f}s)")
        report["cells"].append(rec)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
