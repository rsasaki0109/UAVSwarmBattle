"""Q2: is the *triggered* Merry-Go-Round the off-switch the always-on conventions
lack? Unstructured random-derangement traffic, CBF family.

`findings.md` "The convention is for symmetric convergence only — on unstructured
traffic it is a net liability" showed the always-on right-of-way (global lateral /
conditional pairwise) HARMS dense unstructured traffic and concluded the convention
"must not be left always-on." But that tested only the always-on rules. The
Merry-Go-Round here engages its roundabout ONLY on a locally-detected deadlock
(ego braked to a stop with a peer ahead); with no deadlock it is plain CBF. So it
should be HARMLESS on unstructured traffic — where there is no symmetric hub to
trigger it — while still curing the antipodal hub (see antipodal_mgr_phase.py).
Triggering would then be the "off-switch" the always-on conventions cannot supply.

N drones at random positions crossing to a random derangement of those positions
(many uncorrelated pairwise crossings, no symmetric hub), CBF base, paired by seed:
  cbf            stock CBF
  cbf_global     CBF + lateral_bias  (always-on global convention)
  cbf_pairwise   CBF + pairwise_bias (always-on conditional convention)
  mgr            decentralized triggered Merry-Go-Round (this work)

  python scripts/unstructured_mgr_phase.py --n-list 8 12 16 --episodes 40
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import random
import tempfile
from multiprocessing import Pool
from pathlib import Path

from uav_nav_lab.config import ExperimentConfig
from uav_nav_lab.runner.multi.experiment import run_experiment_multi
from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p

SPEED = 5.0
LO, HI = 8.0, 42.0
ARMS = ["cbf", "cbf_global", "cbf_pairwise", "mgr", "mgr_sym"]


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
    raise ValueError(arm)


def _drones(n):
    # Fixed random layout (seeded by n so all arms share it): n positions, goals
    # a derangement of them -> uncorrelated crossings, no symmetric hub.
    rng = random.Random(1234 + n)
    pts = []
    while len(pts) < n:
        x = round(rng.uniform(LO, HI), 3)
        y = round(rng.uniform(LO, HI), 3)
        if all((x - px) ** 2 + (y - py) ** 2 > 16.0 for px, py in pts):
            pts.append((x, y))
    perm = list(range(n))
    for _ in range(200):
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            break
    return [{"name": f"d{k}", "start": list(pts[k]), "goal": list(pts[perm[k]]),
             "radius": 0.4, "start_jitter": 0.5} for k in range(n)]


def _cfg(n, arm, gb, pb, replan, seed, n_eps, max_steps):
    return {
        "name": f"unstrmgr_n{n}_{arm}", "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50],
                     "resolution": 1.0, "obstacles": {"type": "none"}, "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": _planner(arm, gb, pb, replan),
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/unstrmgr_tmp"},
    }


def _run_cell(job):
    arm, n, gb, pb, replan, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, arm, gb, pb, replan, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (arm, n, by_seed)


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
    ap.add_argument("--n-list", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--global-bias", type=float, default=0.5)
    ap.add_argument("--pairwise-bias", type=float, default=1.0)
    ap.add_argument("--replan", type=float, default=0.1)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--out", default="results/unstructured_mgr_phase.json")
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
              "replan": args.replan, "episodes": args.episodes, "cells": []}
    print(f"\nUnstructured random-derangement traffic: is triggered MGR the off-switch? "
          f"(CBF, n={args.episodes}, paired; succ [coll/timeout])")
    print(f"{'N':>2} | {'cbf':>13} | {'cbf_pairwise':>13} | {'mgr':>13} | {'mgr_sym':>13} | "
          f"{'pw vs cbf':>15} | {'mgr vs cbf':>15} | {'mgrsym vs cbf':>15}")
    print("-" * 120)
    for n in sorted(cells):
        c = cells[n]
        seeds = sorted(set.intersection(*[set(c[a]) for a in ARMS]))
        m = len(seeds)
        def cell(x):
            cc, to = _brk(c[x], seeds)
            return f"{_succ(c[x],seeds):>2}/{m:<2}[{cc:>2}c/{to:>2}t]"
        bg, cg, pg = _mc(c["cbf_global"], c["cbf"], seeds)
        bp, cp, pp = _mc(c["cbf_pairwise"], c["cbf"], seeds)
        bm, cm, pm = _mc(c["mgr"], c["cbf"], seeds)
        bs, cs, ps = _mc(c["mgr_sym"], c["cbf"], seeds)
        print(f"{n:>2} | {cell('cbf'):>13} | {cell('cbf_pairwise'):>13} | {cell('mgr'):>13} | "
              f"{cell('mgr_sym'):>13} | b={bp:>2} c={cp:>2} {pp:>6.4f} | b={bm:>2} c={cm:>2} {pm:>6.4f} | "
              f"b={bs:>2} c={cs:>2} {ps:>6.4f}")
        row = {"n": n, "m": m,
               "global_vs_cbf": {"b": bg, "c": cg, "p": pg},
               "pairwise_vs_cbf": {"b": bp, "c": cp, "p": pp},
               "mgr_vs_cbf": {"b": bm, "c": cm, "p": pm},
               "mgr_sym_vs_cbf": {"b": bs, "c": cs, "p": ps}}
        for a in ARMS:
            cc, to = _brk(c[a], seeds)
            row[a] = _succ(c[a], seeds)
            row[a + "_ct"] = [cc, to]
        report["cells"].append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
