"""Decentralized *triggered* Merry-Go-Round vs the always-on conventions and the
fixed-centre roundabout, on the antipodal swap (Q1: can agents agree on a ring
without being handed the centre?).

The lab's `roundabout` planner is the simplified Merry-Go-Round: an always-on ring
around a FIXED centre handed in by symmetry. The new `mgr` planner implements the
decentralized parts (Zhou et al. 2025, arXiv:2503.05848) the `roundabout` punts on
— triggered on locally-detected deadlock, ring centre negotiated from the ego's
own conflict cluster (no global hub knowledge), capacity-tiered radius, clear-exit
peel-off — layered on the ORCA reciprocal LP as the base avoider.

Q1 asks whether that *local* negotiation reproduces the clean fixed-centre result
on the symmetric hub: do independent drones converge on a common ring from sensing
alone? Arms (all ORCA-family at the same replan cadence; the fixed-centre
`roundabout` at its documented 0.05):
  orca           stock ORCA (reproduces the antipodal hub deadlock)
  orca_global    ORCA + lateral_bias  (always-on global convention)
  orca_pairwise  ORCA + pairwise_bias (always-on conditional convention)
  mgr            decentralized triggered Merry-Go-Round (this work)
  roundabout     fixed-centre always-on ring (the symmetry-cheat reference)

  python scripts/antipodal_mgr_phase.py --n-list 4 6 8 12 --episodes 40
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

SPEED = 5.0
CX, CY = 25.0, 25.0
RADIUS = 20.0
IDEAL = 2 * RADIUS / SPEED
ARMS = ["cbf", "cbf_global", "cbf_pairwise", "mgr", "mgr_sym", "roundabout"]


def _cbf_base(replan):
    return {"type": "cbf", "max_speed": SPEED, "replan_period": replan,
            "radius": 0.4, "alpha": 2.0, "time_step": 0.1,
            "neighbor_dist": 15.0, "safety_margin": 0.1, "goal_radius": 1.5}


def _planner(arm, gb, pb, replan):
    if arm == "cbf":
        return _cbf_base(replan)
    if arm == "cbf_global":
        p = _cbf_base(replan); p["lateral_bias"] = gb; return p
    if arm == "cbf_pairwise":
        p = _cbf_base(replan); p["pairwise_bias"] = pb; p["pairwise_radius"] = 8.0; return p
    if arm == "mgr":
        p = _cbf_base(replan); p["type"] = "mgr"; return p
    if arm == "mgr_sym":
        p = _cbf_base(replan); p["type"] = "mgr"; p["require_convergence"] = True; return p
    if arm == "roundabout":
        return {"type": "roundabout", "max_speed": SPEED, "replan_period": 0.05,
                "center": [CX, CY], "ring_radius": RADIUS, "exit_angle": 0.35,
                "time_step": 0.05, "goal_radius": 1.5}
    raise ValueError(arm)


def _drones(n):
    out = []
    for k in range(n):
        a = 2.0 * math.pi * k / n
        out.append({"name": f"d{k}",
                    "start": [round(CX + RADIUS * math.cos(a), 3), round(CY + RADIUS * math.sin(a), 3)],
                    "goal": [round(CX - RADIUS * math.cos(a), 3), round(CY - RADIUS * math.sin(a), 3)],
                    "radius": 0.4, "start_jitter": 0.8})
    return out


def _cfg(n, arm, gb, pb, replan, seed, n_eps, max_steps):
    return {
        "name": f"mgr_n{n}_{arm}", "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50],
                     "resolution": 1.0, "obstacles": {"type": "none"}, "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": _planner(arm, gb, pb, replan),
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/mgr_tmp"},
    }


def _run_cell(job):
    arm, n, gb, pb, replan, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, arm, gb, pb, replan, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = (d["outcome"], float(d.get("final_t", 0.0)))
    return (arm, n, by_seed)


def _succ(bs, seeds):
    return sum(bs[s][0] == "success" for s in seeds)


def _brk(bs, seeds):
    return (sum(bs[s][0] == "collision" for s in seeds),
            sum(bs[s][0] == "timeout" for s in seeds))


def _mksp(bs, seeds):
    v = [bs[s][1] for s in seeds if bs[s][0] == "success"]
    return sum(v) / len(v) if v else float("nan")


def _mc(a, b, seeds):
    bb = sum(a[s][0] == "success" and b[s][0] != "success" for s in seeds)
    cc = sum(a[s][0] != "success" and b[s][0] == "success" for s in seeds)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[4, 6, 8, 12])
    ap.add_argument("--global-bias", type=float, default=0.5)
    ap.add_argument("--pairwise-bias", type=float, default=1.0)
    ap.add_argument("--replan", type=float, default=0.1)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--out", default="results/antipodal_mgr_phase.json")
    args = ap.parse_args()

    jobs = [(arm, n, args.global_bias, args.pairwise_bias, args.replan, args.seed,
             args.episodes, args.max_steps)
            for n in args.n_list for arm in ARMS]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    cells = {}
    for arm, n, bs in res:
        cells.setdefault(n, {})[arm] = bs

    report = {"global_bias": args.global_bias, "pairwise_bias": args.pairwise_bias,
              "replan": args.replan, "episodes": args.episodes, "ideal_makespan": IDEAL,
              "cells": []}
    print(f"\nDecentralized triggered Merry-Go-Round on the antipodal swap "
          f"(n={args.episodes}, paired; succ [coll/timeout] ; makespan, ideal {IDEAL:.1f}s)")
    hdr = " | ".join(f"{a:>16}" for a in ARMS)
    print(f"{'N':>3} | {hdr} | {'mgr vs cbf':>16} | {'mgr vs round':>16}")
    print("-" * (8 + 19 * len(ARMS) + 38))
    for n in sorted(cells):
        c = cells[n]
        seeds = sorted(set.intersection(*[set(c[a]) for a in ARMS]))
        m = len(seeds)
        def cell(a):
            cc, to = _brk(c[a], seeds)
            return f"{_succ(c[a],seeds):>2}/{m}[{cc}c/{to}t]{_mksp(c[a],seeds):>5.1f}"
        b1, c1, p1 = _mc(c["mgr"], c["cbf"], seeds)
        b2, c2, p2 = _mc(c["mgr"], c["roundabout"], seeds)
        row_cells = " | ".join(f"{cell(a):>16}" for a in ARMS)
        print(f"{n:>3} | {row_cells} | b={b1:>2} c={c1:>2} {p1:>6.4f} | b={b2:>2} c={c2:>2} {p2:>6.4f}")
        row = {"n": n, "m": m,
               "mgr_vs_cbf": {"b": b1, "c": c1, "p": p1},
               "mgr_vs_roundabout": {"b": b2, "c": c2, "p": p2}}
        for a in ARMS:
            cc, to = _brk(c[a], seeds)
            row[a] = _succ(c[a], seeds)
            row[a + "_ct"] = [cc, to]
            row[a + "_makespan"] = _mksp(c[a], seeds)
        report["cells"].append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
