"""The in-plane right-of-way convention rescues a REACTIVE planner in 3-D, where
the extra dimension alone could not.

The [3-D dissolution result](docs/findings.md) showed the reactive CBF deadlock
does NOT dissolve in a voxel world: its goal-seeking nominal stays in-plane, so
the vertical escape goes unused (cbf_3d = 0/40 at every N). That implies the cure
for a reactive controller in 3-D is the SAME in-plane right-of-way that works in
2-D — not the dimension. This tests it: apply `pairwise_bias` (now active on the
horizontal components in 3-D too) to the 3-D CBF and see whether the deadlock the
vertical axis could not break is broken by the in-plane roundabout.

Arms per N, 3-D voxel, paired by seed, McNemar exact, collision/timeout split:
  cbf_3d        stock (the 3-D deadlock)
  cbf_3d_pw     + pairwise right-of-way (in-plane)

  python scripts/antipodal_cbf_3d_convention_phase.py --n-list 4 6 8 --episodes 40
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
from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p

CX, CY = 25.0, 25.0
RADIUS = 20.0
Z_SIZE = 16
Z_MID = 8.0
SPEED = 5.0


def _drones(n):
    out = []
    for k in range(n):
        ang = 2.0 * math.pi * k / n
        sx, sy = CX + RADIUS * math.cos(ang), CY + RADIUS * math.sin(ang)
        gx, gy = CX - RADIUS * math.cos(ang), CY - RADIUS * math.sin(ang)
        out.append({"name": f"d{k}",
                    "start": [round(sx, 3), round(sy, 3), Z_MID],
                    "goal": [round(gx, 3), round(gy, 3), Z_MID],
                    "radius": 0.4, "start_jitter": 0.8})
    return out


def _cfg(n, pw, seed, n_eps, max_steps):
    planner = {"type": "cbf", "max_speed": SPEED, "replan_period": 0.05,
               "radius": 0.4, "safety_margin": 0.1, "alpha": 2.0,
               "neighbor_dist": 15.0, "time_step": 0.05, "goal_radius": 1.5}
    if pw > 0.0:
        planner["pairwise_bias"] = pw
        planner["pairwise_radius"] = 8.0
    return {"name": f"cbf3dconv_n{n}_pw{pw}", "seed": seed, "num_episodes": n_eps,
            "scenario": {"type": "multi_drone_voxel", "size": [50, 50, Z_SIZE],
                         "resolution": 1.0, "obstacles": {"type": "none"},
                         "drones": _drones(n)},
            "simulator": {"type": "dummy_3d", "dt": 0.05, "max_steps": max_steps,
                          "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
            "planner": planner, "sensor": {"type": "perfect"},
            "output": {"dir": "results/cbf3dconv_tmp"}}


def _run_cell(job):
    label, n, pw, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, pw, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (label, n, by_seed)


def _succ(bs, seeds):
    return sum(bs[s] == "success" for s in seeds)


def _brk(bs, seeds):
    return (sum(bs[s] == "collision" for s in seeds),
            sum(bs[s] == "timeout" for s in seeds))


def _mc(a, b, seeds):
    bb = sum(a[s] == "success" and b[s] != "success" for s in seeds)
    cc = sum(a[s] != "success" and b[s] == "success" for s in seeds)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--pairwise-bias", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--out", default="results/antipodal_cbf_3d_convention_phase.json")
    args = ap.parse_args()

    jobs = []
    for n in args.n_list:
        jobs.append(("cbf_3d", n, 0.0, args.seed, args.episodes, args.max_steps))
        jobs.append(("cbf_3d_pw", n, args.pairwise_bias, args.seed, args.episodes, args.max_steps))
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    cells = {}
    for label, n, bs in res:
        cells.setdefault(n, {})[label] = bs

    report = {"radius": RADIUS, "pairwise_bias": args.pairwise_bias,
              "episodes": args.episodes, "cells": []}
    print(f"\n3-D CBF: does the in-plane convention rescue the reactive deadlock the "
          f"vertical axis could not? (pw={args.pairwise_bias}, n={args.episodes}, paired)")
    print(f"{'N':>2} | {'cbf_3d':>14} | {'cbf_3d_pw':>14} | {'pw vs stock (b/c,p)':>22}")
    print("-" * 64)
    for n in sorted(cells):
        c = cells[n]
        st, pw = c["cbf_3d"], c["cbf_3d_pw"]
        seeds = sorted(set(st) & set(pw))
        m = len(seeds)
        cs, ts = _brk(st, seeds)
        cp, tp = _brk(pw, seeds)
        b, cc, p = _mc(st, pw, seeds)
        print(f"{n:>2} | {_succ(st,seeds):>2}/{m:<2}[{cs:>2}c/{ts:>2}t] | "
              f"{_succ(pw,seeds):>2}/{m:<2}[{cp:>2}c/{tp:>2}t] | "
              f"b={b:>2} c={cc:>2} p={p:>8.5f}")
        report["cells"].append({"n": n, "m": m,
            "cbf_3d": _succ(st, seeds), "cbf_3d_ct": [cs, ts],
            "cbf_3d_pw": _succ(pw, seeds), "cbf_3d_pw_ct": [cp, tp],
            "pw_vs_stock": {"b": b, "c": cc, "p": p}})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
