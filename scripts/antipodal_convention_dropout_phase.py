"""Under peer-observation DROPOUT, the simple comms-free convention overtakes the
"better" one that must see each neighbour.

Two conventions break the antipodal hub deadlock:
  * GLOBAL right-of-way (`lateral_bias`): each drone tilts off its OWN goal
    heading and veers right. It reads NO peer — it is COMMS-FREE.
  * PAIRWISE winding (`pairwise_bias`): each drone accumulates a preferred pass
    side from the positions of its nearby NEIGHBOURS. It STRICTLY DOMINATES the
    global rule in a fully-observed arena (docs/findings.md "A pairwise
    winding-number right-of-way strictly dominates the global veer-right") — but
    it must SEE each neighbour to choose the side, so it is COMMS-DEPENDENT.

Both share the same MPC avoider, so when a peer is unobserved both lose it for
collision avoidance equally. The difference is the SYMMETRY-BREAKER on top: the
global rule keeps tilting every drone the same way no matter how many peers
vanish, while the pairwise rule's pass-side decision is computed from whatever
neighbours happen to be visible this replan. So the prediction is that
intermittent peer dropout should erode the pairwise convention faster than the
global one — and the pairwise DOMINANCE should INVERT once the channel is lossy
enough, the simpler comms-free rule winning exactly where sensing fails.

New `dropout` sensor: each replan, each dynamic obstacle is independently missing
with probability p (a Bernoulli packet-loss model; ego pose stays ground truth).

Arms at fixed N, MPC + constant_velocity, swept over dropout p, paired by seed,
McNemar exact:
  stock     no convention            (the deadlock floor)
  global    lateral_bias  = B        (comms-free)
  pairwise  pairwise_bias = PB       (comms-dependent, dominates at p=0)

Headline: McNemar(global vs pairwise) per p — expect a tie (or pairwise-favoured)
at p=0 that flips to global-favoured as p grows.

  python scripts/antipodal_convention_dropout_phase.py --n 8 --p-list 0 0.3 0.5 0.7 --episodes 40
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
ARMS = ["stock", "global", "pairwise"]


def _base():
    return {"type": "mpc", "max_speed": SPEED, "replan_period": 0.2, "horizon": 40,
            "dt_plan": 0.05, "n_samples": 32, "resolution": 1.0, "inflate": 1,
            "goal_radius": 1.5, "safety_margin": 0.5, "use_prediction": True,
            "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
            "predictor": {"type": "constant_velocity"}}


def _planner(arm, gb, pb):
    p = _base()
    if arm == "global":
        p["lateral_bias"] = gb
    elif arm == "pairwise":
        p["pairwise_bias"] = pb
        p["pairwise_radius"] = 8.0
    return p


def _drones(n):
    out = []
    for k in range(n):
        a = 2.0 * math.pi * k / n
        out.append({"name": f"d{k}",
                    "start": [round(CX + RADIUS * math.cos(a), 3),
                              round(CY + RADIUS * math.sin(a), 3)],
                    "goal": [round(CX - RADIUS * math.cos(a), 3),
                             round(CY - RADIUS * math.sin(a), 3)],
                    "radius": 0.4, "start_jitter": 0.8})
    return out


def _cfg(n, arm, p, gb, pb, seed, n_eps, max_steps):
    return {
        "name": f"drop_n{n}_{arm}_p{p}", "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50],
                     "resolution": 1.0, "obstacles": {"type": "none"},
                     "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": _planner(arm, gb, pb),
        "sensor": {"type": "dropout", "dropout_prob": p},
        "output": {"dir": "results/drop_tmp"},
    }


def _run_cell(job):
    arm, n, p, gb, pb, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, arm, p, gb, pb, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return ((arm, p), by_seed)


def _succ(bs, seeds):
    return sum(bs[s] == "success" for s in seeds)


def _brk(bs, seeds):
    return (sum(bs[s] == "collision" for s in seeds),
            sum(bs[s] == "timeout" for s in seeds))


def _mc(a, b, seeds):
    # c-b>0 => b better than a
    bb = sum(a[s] == "success" and b[s] != "success" for s in seeds)
    cc = sum(a[s] != "success" and b[s] == "success" for s in seeds)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--p-list", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7])
    ap.add_argument("--global-bias", type=float, default=2.0)
    ap.add_argument("--pairwise-bias", type=float, default=2.0)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--out", default="results/antipodal_convention_dropout_phase.json")
    args = ap.parse_args()

    jobs = [(arm, args.n, p, args.global_bias, args.pairwise_bias, args.seed,
             args.episodes, args.max_steps)
            for p in args.p_list for arm in ARMS]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)
    cells = dict(res)

    seeds = sorted(set.intersection(*[set(bs) for bs in cells.values()]))
    report = {"n": args.n, "global_bias": args.global_bias,
              "pairwise_bias": args.pairwise_bias, "episodes": args.episodes,
              "m": len(seeds), "cells": {}, "tests": {}}
    print(f"\nConvention robustness to peer dropout @ N={args.n}, "
          f"global_bias={args.global_bias}, pairwise_bias={args.pairwise_bias}, "
          f"paired m={len(seeds)}")
    print(f"{'p':>5} | " + " | ".join(f"{a:>11}" for a in ARMS)
          + " | global vs pairwise (c-b>0 => pairwise better)")
    print("-" * 86)
    for p in args.p_list:
        row = []
        for arm in ARMS:
            bs = cells[(arm, p)]
            s = _succ(bs, seeds); co, to = _brk(bs, seeds)
            row.append(f"{s:>2}/{len(seeds):<2} {co}c{to}t")
            report["cells"][f"{arm}_p{p}"] = {"success": s, "collision": co, "timeout": to}
        b, c, pv = _mc(cells[("global", p)], cells[("pairwise", p)], seeds)
        report["tests"][f"global_vs_pairwise_p{p}"] = {"b": b, "c": c, "p": pv}
        print(f"{p:>5} | " + " | ".join(f"{x:>11}" for x in row)
              + f" | b={b} c={c} p={pv:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
