"""Does the right-of-way convention survive non-holonomic drones?

The entire convention arc — the antipodal deadlock, the global ``lateral_bias``
roundabout, the density cliff — was measured on HOLONOMIC point-mass drones
(`dummy_2d`): a drone changes its velocity vector in any direction, so "veer
right" is a free sideways nudge. A real fixed-wing UAV or wheeled robot is
NON-HOLONOMIC: it cannot strafe; to veer right it must *turn*, spending heading
change and forward progress. This swaps only the simulator (`dummy_unicycle`,
forward drive + rate-limited turn) and keeps the MPC + lateral_bias controller
identical, then sweeps the turn-rate limit ``turn_rate_max`` from sluggish
(strongly non-holonomic) to fast (≈ holonomic).

Antipodal swap, N drones, MPC. Two arms paired by seed (McNemar-exact):

  bias 0   no convention (the deadlock baseline)
  bias 2   global lateral_bias (the right-of-way roundabout)

For each turn rate: success of each arm and the convention's effect (c - b).
The holonomic point-mass (`dummy_2d`) is the ∞-turn-rate reference.

  python scripts/nonholonomic_convention_phase.py --n 6 --episodes 40
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


def _drones(n):
    out = []
    for k in range(n):
        a = 2.0 * math.pi * k / n
        out.append({"name": f"d{k}",
                    "start": [round(CX + RADIUS * math.cos(a), 3), round(CY + RADIUS * math.sin(a), 3)],
                    "goal": [round(CX - RADIUS * math.cos(a), 3), round(CY - RADIUS * math.sin(a), 3)],
                    "radius": 0.4, "start_jitter": 0.8})
    return out


def _planner(bias):
    p = {"type": "mpc", "max_speed": SPEED, "replan_period": 0.2, "horizon": 40,
         "dt_plan": 0.05, "n_samples": 32, "resolution": 1.0, "inflate": 1,
         "goal_radius": 1.5, "safety_margin": 0.5, "use_prediction": True,
         "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
         "predictor": {"type": "constant_velocity"}}
    if bias:
        p["lateral_bias"] = bias
    return p


def _sim(turn_rate, max_steps):
    if turn_rate is None:   # holonomic point-mass reference
        return {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4}
    return {"type": "dummy_unicycle", "dt": 0.05, "max_steps": max_steps,
            "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4,
            "turn_rate_max": turn_rate}


def _cfg(n, turn_rate, bias, seed, n_eps, max_steps):
    return {
        "name": f"nh_n{n}", "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50], "resolution": 1.0,
                     "obstacles": {"type": "none"}, "drones": _drones(n)},
        "simulator": _sim(turn_rate, max_steps),
        "planner": _planner(bias),
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/nh_tmp"},
    }


def _run_cell(job):
    turn_rate, bias, n, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, turn_rate, bias, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        bits = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            bits[d["meta"]["seed"]] = (d["outcome"] == "success")
    key = "holo" if turn_rate is None else f"{turn_rate}"
    return (key, bias, bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--turn-rates", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--bias", type=float, default=2.0)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--out", default="results/nonholonomic_convention_phase.json")
    args = ap.parse_args()

    turn_rates = [None] + list(args.turn_rates)   # None = holonomic reference
    jobs = [(tr, bias, args.n, args.seed, args.episodes, args.max_steps)
            for tr in turn_rates for bias in (0.0, args.bias)]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)

    by = {}
    for key, bias, bits in res:
        by[(key, bias)] = bits

    print(f"Right-of-way convention under non-holonomic dynamics — antipodal N={args.n}, "
          f"bias={args.bias}, m={args.episodes}")
    print("  turn_rate | no-conv | convention |  b  c  |    p     (c = convention-only rescue)")
    print("-" * 74)
    rows = ["holo"] + [f"{tr}" for tr in args.turn_rates]
    labels = {"holo": "holo(inf)"}
    out_rows = []
    for key in rows:
        b0 = by[(key, 0.0)]
        b2 = by[(key, args.bias)]
        seeds = sorted(set(b0) & set(b2))
        s0 = sum(b0[s] for s in seeds)
        s2 = sum(b2[s] for s in seeds)
        b = sum(1 for s in seeds if b0[s] and not b2[s])
        c = sum(1 for s in seeds if b2[s] and not b0[s])
        p = mcnemar_exact_p(b, c)
        m = len(seeds)
        print(f" {labels.get(key,key):>9} | {s0:>2}/{m}  |   {s2:>2}/{m}    | {b:>2} {c:>2}  | {p:.2e}")
        out_rows.append({"turn_rate": key, "no_conv": s0, "conv": s2, "m": m, "b": b, "c": c, "p": p})
    print("-" * 74)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"n": args.n, "bias": args.bias,
                                          "episodes": args.episodes, "rows": out_rows}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
