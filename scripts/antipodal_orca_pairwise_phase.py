"""On ORCA, does the PAIRWISE right-of-way avoid the GLOBAL one's over-rotation
timeout cliff?

PR #85 ported the MPC global `lateral_bias` right-of-way to the ORCA reciprocal
baseline and found it rescues the antipodal deadlock in an INVERTED-U band:
too little (stock) COLLIDES at the hub, too much (>~0.5) TIMES OUT — every drone
over-rotates and orbits without ever converging on goal. The global tilt fires
unconditionally (it veers right of the goal heading even with no neighbour
around), which is exactly why a large value orbits.

This is the ORCA twin of the MPC finding (#84) that a PAIRWISE, neighbour-
conditional right-of-way removes the global rule's harm. Here the new ORCA
`pairwise_bias` tilts the preferred velocity toward the sum of "pass each nearby
neighbour on the right", weighted exp(-d/radius); with no neighbour the tilt
vanishes, so a drone that has cleared the hub re-aims at its goal instead of
orbiting. Hypothesis: pairwise has NO (or a far higher) over-rotation timeout
cliff — it rescues the deadlock across a much wider strength range than global.

Two views, paired by seed, McNemar exact, collision-vs-timeout breakdown:
  --mode strength : fixed N, sweep global {..} and pairwise {..} strengths
                    (shows global's upper timeout cliff vs pairwise's)
  --mode nscale   : best global vs best pairwise across N

  python scripts/antipodal_orca_pairwise_phase.py --mode strength --n 8 --episodes 40
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
SPEED = 5.0


def _drones(n):
    out = []
    for k in range(n):
        ang = 2.0 * math.pi * k / n
        sx, sy = CX + RADIUS * math.cos(ang), CY + RADIUS * math.sin(ang)
        gx, gy = CX - RADIUS * math.cos(ang), CY - RADIUS * math.sin(ang)
        out.append({"name": f"d{k}",
                    "start": [round(sx, 3), round(sy, 3)],
                    "goal": [round(gx, 3), round(gy, 3)],
                    "radius": 0.4, "start_jitter": 0.8})
    return out


def _planner(kind, strength):
    # Operating point matched to PR #85 (antipodal_orca_convention_phase.py):
    # default replan cadence (NOT every-step), neighbor_dist 15, time_step 0.25,
    # so stock ORCA actually reproduces the antipodal deadlock.
    p = {"type": "orca", "max_speed": SPEED,
         "radius": 0.4, "safety_margin": 0.1, "time_horizon": 2.0,
         "time_step": 0.25, "neighbor_dist": 15.0, "goal_radius": 1.5}
    if kind == "global":
        p["lateral_bias"] = strength
    elif kind == "pairwise":
        p["pairwise_bias"] = strength
        p["pairwise_radius"] = 8.0
    return p


def _cfg(n, kind, strength, seed, n_eps, max_steps):
    return {
        "name": f"orcapw_n{n}_{kind}_{strength}", "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50],
                     "resolution": 1.0, "obstacles": {"type": "none"},
                     "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": _planner(kind, strength),
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/orcapw_tmp"},
    }


def _run_cell(job):
    label, n, kind, strength, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, kind, strength, seed, n_eps, max_steps))
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
    ap.add_argument("--mode", choices=["strength", "nscale"], default="strength")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--n-list", type=int, nargs="+", default=[6, 8, 12])
    ap.add_argument("--global-list", type=float, nargs="+", default=[0.2, 0.5, 1.0, 2.0])
    ap.add_argument("--pairwise-list", type=float, nargs="+", default=[2.0, 5.0, 10.0, 20.0])
    ap.add_argument("--best-global", type=float, default=0.2)
    ap.add_argument("--best-pairwise", type=float, default=10.0)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--out", default="results/antipodal_orca_pairwise_phase.json")
    args = ap.parse_args()

    jobs = []
    if args.mode == "strength":
        n = args.n
        jobs.append(("stock", n, "global", 0.0, args.seed, args.episodes, args.max_steps))
        for g in args.global_list:
            jobs.append((f"global_{g}", n, "global", g, args.seed, args.episodes, args.max_steps))
        for p in args.pairwise_list:
            jobs.append((f"pairwise_{p}", n, "pairwise", p, args.seed, args.episodes, args.max_steps))
    else:
        for n in args.n_list:
            jobs.append(("stock", n, "global", 0.0, args.seed, args.episodes, args.max_steps))
            jobs.append(("global", n, "global", args.best_global, args.seed, args.episodes, args.max_steps))
            jobs.append(("pairwise", n, "pairwise", args.best_pairwise, args.seed, args.episodes, args.max_steps))

    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    report = {"mode": args.mode, "radius": RADIUS, "speed": SPEED,
              "episodes": args.episodes, "rows": []}

    if args.mode == "strength":
        by_label = {label: bs for label, n, bs in res}
        seeds = sorted(set.intersection(*[set(bs) for bs in by_label.values()]))
        m = len(seeds)
        stock = by_label["stock"]
        print(f"\nORCA pairwise vs global right-of-way @ N={args.n} "
              f"(n={m}, paired; succ [coll/timeout])")
        print(f"{'arm':>14} | {'succ':>10} | {'vs stock (b/c,p)':>22}")
        print("-" * 56)
        for label, n, bs in res:
            co, to = _brk(bs, seeds)
            b, c, p = _mc(stock, bs, seeds)
            print(f"{label:>14} | {_succ(bs,seeds):>2}/{m:<2} [{co:>2}c/{to:>2}t] | "
                  f"b={b:>2} c={c:>2} p={p:>8.5f}")
            report["rows"].append({"arm": label, "succ": _succ(bs, seeds),
                                   "coll": co, "timeout": to,
                                   "vs_stock": {"b": b, "c": c, "p": p}})
    else:
        cells = {}
        for label, n, bs in res:
            cells.setdefault(n, {})[label] = bs
        print(f"\nORCA best-global vs best-pairwise across N "
              f"(global={args.best_global}, pairwise={args.best_pairwise}, n={args.episodes})")
        print(f"{'N':>2} | {'stock':>12} | {'global':>12} | {'pairwise':>12} | "
              f"{'pw vs global':>20}")
        print("-" * 76)
        for n in sorted(cells):
            c = cells[n]
            st, gl, pw = c["stock"], c["global"], c["pairwise"]
            seeds = sorted(set(st) & set(gl) & set(pw))
            m = len(seeds)
            cs, ts = _brk(st, seeds); cg, tg = _brk(gl, seeds); cp, tp = _brk(pw, seeds)
            b, cc, p = _mc(gl, pw, seeds)
            print(f"{n:>2} | {_succ(st,seeds):>2}/{m:<2}[{cs}c/{ts}t] | "
                  f"{_succ(gl,seeds):>2}/{m:<2}[{cg}c/{tg}t] | "
                  f"{_succ(pw,seeds):>2}/{m:<2}[{cp}c/{tp}t] | b={b:>2} c={cc:>2} p={p:>7.4f}")
            report["rows"].append({"n": n, "m": m,
                "stock": _succ(st, seeds), "stock_ct": [cs, ts],
                "global": _succ(gl, seeds), "global_ct": [cg, tg],
                "pairwise": _succ(pw, seeds), "pairwise_ct": [cp, tp],
                "pw_vs_global": {"b": b, "c": cc, "p": p}})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
