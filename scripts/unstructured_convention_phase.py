"""Where does the convention NOT help? The right-of-way is a fix for SYMMETRIC
convergence; on unstructured random traffic the unconditional global rule should
be a no-op or a harm, while the conditional pairwise rule stays neutral.

Every convention result lives on a structured symmetric task (antipodal hub,
doorway, perpendicular crossing). This is the boundary test: N drones at random
positions cross to a random derangement of those positions — many uncorrelated
pairwise crossings, no single symmetric hub to break. There is no global rotation
to organise, so:
  - the GLOBAL veer-right (lateral_bias) tilts every drone unconditionally even
    where no symmetry exists -> it should not help and may HARM (gratuitous detours
    /induced conflicts, like its 3-D no-deadlock harm);
  - the PAIRWISE rule only tilts near a real conflict -> it should stay neutral.

Random-derangement swarm (2D grid), MPC, paired by seed, McNemar exact:
  stock / global (lateral_bias) / pairwise (pairwise_bias)

  python scripts/unstructured_convention_phase.py --n-list 8 12 16 --episodes 40
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import math
import random
import tempfile
from multiprocessing import Pool
from pathlib import Path

from uav_nav_lab.config import ExperimentConfig
from uav_nav_lab.runner.multi.experiment import run_experiment_multi
from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p

SPEED = 5.0
LO, HI = 8.0, 42.0


def _base():
    return {"type": "mpc", "max_speed": SPEED, "replan_period": 0.2, "horizon": 40,
            "dt_plan": 0.05, "n_samples": 32, "resolution": 1.0, "inflate": 1,
            "goal_radius": 1.5, "safety_margin": 0.5, "use_prediction": True,
            "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
            "predictor": {"type": "constant_velocity"}}


def _planner(arm, gbias, pbias):
    p = _base()
    if arm == "global":
        p["lateral_bias"] = gbias
    elif arm == "pairwise":
        p["pairwise_bias"] = pbias
        p["pairwise_radius"] = 8.0
    return p


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


def _cfg(n, arm, gbias, pbias, seed, n_eps, max_steps):
    return {
        "name": f"unstr_n{n}_{arm}", "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50],
                     "resolution": 1.0, "obstacles": {"type": "none"}, "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": _planner(arm, gbias, pbias),
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/unstr_tmp"},
    }


def _run_cell(job):
    arm, n, gbias, pbias, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, arm, gbias, pbias, seed, n_eps, max_steps))
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


ARMS = ["stock", "global", "pairwise"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--global-bias", type=float, default=2.0)
    ap.add_argument("--pairwise-bias", type=float, default=10.0)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--out", default="results/unstructured_convention_phase.json")
    args = ap.parse_args()

    jobs = [(arm, n, args.global_bias, args.pairwise_bias, args.seed, args.episodes, args.max_steps)
            for n in args.n_list for arm in ARMS]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    cells = {}
    for arm, n, bs in res:
        cells.setdefault(n, {})[arm] = bs

    report = {"global_bias": args.global_bias, "pairwise_bias": args.pairwise_bias,
              "episodes": args.episodes, "cells": []}
    print(f"\nUnstructured random-derangement traffic: does the convention help? "
          f"(MPC, n={args.episodes}, paired; succ [coll/timeout])")
    print(f"{'N':>2} | {'stock':>14} | {'global':>14} | {'pairwise':>14} | "
          f"{'global vs stock':>16} | {'pw vs stock':>16}")
    print("-" * 92)
    for n in sorted(cells):
        c = cells[n]
        st, gl, pw = c["stock"], c["global"], c["pairwise"]
        seeds = sorted(set(st) & set(gl) & set(pw))
        m = len(seeds)
        def cell(x):
            cc, to = _brk(x, seeds)
            return f"{_succ(x,seeds):>2}/{m:<2}[{cc:>2}c/{to:>2}t]"
        b1, c1, p1 = _mc(st, gl, seeds)
        b2, c2, p2 = _mc(st, pw, seeds)
        print(f"{n:>2} | {cell(st):>14} | {cell(gl):>14} | {cell(pw):>14} | "
              f"b={b1:>2} c={c1:>2} p={p1:>6.4f} | b={b2:>2} c={c2:>2} p={p2:>6.4f}")
        row = {"n": n, "m": m, "global_vs_stock": {"b": b1, "c": c1, "p": p1},
               "pairwise_vs_stock": {"b": b2, "c": c2, "p": p2}}
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
