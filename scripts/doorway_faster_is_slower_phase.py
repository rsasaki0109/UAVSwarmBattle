"""Faster-is-slower at a doorway: does raising every drone's desired speed make
the bottleneck WORSE, not better?

The "faster-is-slower" effect is a classic counter-intuitive result of crowd
dynamics (Helbing, Farkas & Vicsek, *Simulating dynamical features of escape
panic*, Nature 2000): at a bottleneck, increasing each agent's desired speed
REDUCES the flow, because greedier agents arrive at the gap faster than they can
be deconflicted and clog it. Every prior doorway result here fixed the speed and
varied the *rule*; this fixes the rule and varies the desired speed.

The mechanism transfers to this lab through the acceleration limit. `max_accel`
is held fixed (the "friction"): a drone that wants to go faster needs more room
to brake, so at the gap it either cannot decelerate in time (collision) or the
planner must slow it anyway (no makespan gain) — both are faster-is-slower. We
hold the right-of-way convention ON so the head-on deadlock is already solved and
the ONLY variable is the desired speed.

Doorway: a 2-cell wall with a centred gap; N drones cross left->right and N
right->left, all funnelling through the gap. MPC + global `lateral_bias`,
`max_accel`=6 fixed, sweep `max_speed`. Per speed we report joint success, the
collision/timeout split, and the makespan (final_t) of the SUCCESSFUL episodes.
McNemar(success) pairs the peak-success speed against each higher speed: a
significant drop is faster-is-slower in success; a flat/rising makespan despite a
higher desired speed is faster-is-slower in time.

  python scripts/doorway_faster_is_slower_phase.py --n 4 --gap 4 \
      --speeds 2 4 6 8 10 12 --episodes 40
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import tempfile
from multiprocessing import Pool
from pathlib import Path

from uav_nav_lab.config import ExperimentConfig
from uav_nav_lab.runner.multi.experiment import run_experiment_multi
from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p

SIZE = 50
WALL_X = (24, 25)


def _wall_cells(gap):
    half = gap // 2
    gap_lo, gap_hi = 25 - half, 25 + half
    cells = []
    for x in WALL_X:
        for y in range(SIZE):
            if not (gap_lo <= y < gap_hi):
                cells.append([x, y])
    return cells


def _drones(n):
    ys = [round(20.0 + i * (10.0 / max(1, n - 1)), 3) for i in range(n)]
    out = []
    for i, y in enumerate(ys):
        out.append({"name": f"lr{i}", "start": [6.0, y], "goal": [44.0, y],
                    "radius": 0.4, "start_jitter": 0.5})
    for i, y in enumerate(ys):
        out.append({"name": f"rl{i}", "start": [44.0, y], "goal": [6.0, y],
                    "radius": 0.4, "start_jitter": 0.5})
    return out


def _cfg(n, gap, speed, bias, max_accel, seed, n_eps, max_steps):
    return {
        "name": f"fis_n{n}_g{gap}_v{speed}", "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [SIZE, SIZE],
                     "resolution": 1.0,
                     "obstacles": {"type": "none", "cells": _wall_cells(gap)},
                     "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": max_accel, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": {"type": "mpc", "max_speed": speed, "replan_period": 0.2,
                    "horizon": 40, "dt_plan": 0.05, "n_samples": 32,
                    "resolution": 1.0, "inflate": 1, "goal_radius": 1.5,
                    "safety_margin": 0.5, "use_prediction": True,
                    "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
                    "lateral_bias": bias, "predictor": {"type": "constant_velocity"}},
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/fis_tmp"},
    }


def _run_cell(job):
    n, gap, speed, bias, max_accel, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(
            _cfg(n, gap, speed, bias, max_accel, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = (d["outcome"], float(d.get("final_t", 0.0)))
    return (speed, by_seed)


def _succ(bs, seeds):
    return sum(bs[s][0] == "success" for s in seeds)


def _brk(bs, seeds):
    return (sum(bs[s][0] == "collision" for s in seeds),
            sum(bs[s][0] == "timeout" for s in seeds))


def _makespan(bs, seeds):
    v = [bs[s][1] for s in seeds if bs[s][0] == "success"]
    return sum(v) / len(v) if v else float("nan")


def _mc(a, b, seeds):
    bb = sum(a[s][0] == "success" and b[s][0] != "success" for s in seeds)
    cc = sum(a[s][0] != "success" and b[s][0] == "success" for s in seeds)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--gap", type=int, default=4)
    ap.add_argument("--speeds", type=float, nargs="+", default=[2, 4, 6, 8, 10, 12])
    ap.add_argument("--bias", type=float, default=2.0)
    ap.add_argument("--max-accel", type=float, default=6.0)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--out", default="results/doorway_faster_is_slower_phase.json")
    args = ap.parse_args()

    jobs = [(args.n, args.gap, v, args.bias, args.max_accel, args.seed,
             args.episodes, args.max_steps) for v in args.speeds]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)
    cells = dict(res)

    seeds = sorted(set.intersection(*[set(bs) for bs in cells.values()]))
    succ = {v: _succ(cells[v], seeds) for v in args.speeds}
    peak = max(args.speeds, key=lambda v: succ[v])

    report = {"n": args.n, "gap": args.gap, "bias": args.bias,
              "max_accel": args.max_accel, "episodes": args.episodes,
              "m": len(seeds), "peak_speed": peak, "cells": {}, "tests": {}}
    print(f"\nFaster-is-slower @ doorway N={args.n}+{args.n}, gap={args.gap}, "
          f"max_accel={args.max_accel}, bias={args.bias}, paired m={len(seeds)}")
    print(f"{'speed':>6} | {'success':>8} | {'coll/to':>8} | {'makespan(ok)':>12} | "
          f"{'vs peak v=%g (c-b>0=>peak better)' % peak:>34}")
    print("-" * 84)
    for v in args.speeds:
        bs = cells[v]
        s = succ[v]; co, to = _brk(bs, seeds); mk = _makespan(bs, seeds)
        report["cells"][f"v{v}"] = {"success": s, "collision": co, "timeout": to,
                                    "makespan": mk}
        if v == peak:
            print(f"{v:>6g} | {s:>3}/{len(seeds):<3} | {co:>3}/{to:<3} | "
                  f"{mk:>12.2f} | (peak)")
        else:
            b, c, p = _mc(cells[peak], bs, seeds)
            report["tests"][f"v{v}_vs_peak"] = {"b": b, "c": c, "p": p}
            print(f"{v:>6g} | {s:>3}/{len(seeds):<3} | {co:>3}/{to:<3} | "
                  f"{mk:>12.2f} | b={b} c={c} p={p:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
