"""Does the right-of-way convention survive a fleet of NON-interchangeable
drones — i.e. drones that fly at different speeds?

scripts/antipodal_rightofway_phase.py proved that a decentralized right-of-way
`planner.lateral_bias` (veer RIGHT of the goal heading) breaks the antipodal
hub deadlock by turning the symmetric head-on convergence into a clockwise
roundabout, lifting an N=6 antipodal swap to ~100%. But that proof used a
HOMOGENEOUS fleet: every drone identical, same max_speed. A roundabout only
works if everyone circulates at a compatible pace — the convention implicitly
assumes interchangeable agents. Real swarms are heterogeneous.

This script stress-tests that assumption along the cleanest axis: SPEED spread,
with the fleet MEAN speed held fixed so the comparison is not confounded by an
overall faster/slower fleet. With N=6 drones we alternate fast/slow around the
ring: speeds = base ± spread/2, three of each, mean == base. `planner.per_drone`
(wired in the multi-drone builder) carries the per-index max_speed override.

Question: does a fast drone lap a slow one into the hub and re-shatter the
roundabout that the convention builds? Arms at fixed N=6, paired by seed,
McNemar exact, all in the 2D antipodal swap:

  homo_b2     spread 0, bias B   the homogeneous roundabout (reference, ~100%)
  het<S>_b2   spread S, bias B   does the convention survive speed spread S?
  hetmax_b0   spread Smax, bias 0  control: heterogeneity WITHOUT the convention
                                   (is mixed speed alone enough? expect collapse)

Reports het<S>_b2 vs homo_b2 (does heterogeneity DEGRADE the working convention?)
and hetmax_b2 vs hetmax_b0 (does the convention still RESCUE the mixed fleet?).
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

CENTER = (25.0, 25.0)
RADIUS = 20.0
BASE_SPEED = 5.0


def _drones(n):
    out = []
    for k in range(n):
        ang = 2.0 * math.pi * k / n
        sx = CENTER[0] + RADIUS * math.cos(ang)
        sy = CENTER[1] + RADIUS * math.sin(ang)
        gx = CENTER[0] - RADIUS * math.cos(ang)
        gy = CENTER[1] - RADIUS * math.sin(ang)
        out.append({"name": f"d{k}",
                    "start": [round(sx, 3), round(sy, 3)],
                    "goal": [round(gx, 3), round(gy, 3)],
                    "radius": 0.4, "start_jitter": 0.8})
    return out


def _per_drone(n, spread):
    """Alternate fast/slow around the ring with mean speed == BASE_SPEED.
    spread 0 returns [] (homogeneous — let the shared planner config stand)."""
    if spread == 0:
        return []
    out = []
    for k in range(n):
        sp = BASE_SPEED + (spread / 2.0 if k % 2 == 0 else -spread / 2.0)
        out.append({"max_speed": round(sp, 3)})
    return out


def _cfg(n, spread, bias, seed, n_eps, max_steps):
    return {
        "name": f"antipodal_n{n}_spread{spread}_b{bias}", "seed": seed,
        "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50],
                     "resolution": 1.0, "obstacles": {"type": "none"},
                     "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": {"type": "mpc", "max_speed": BASE_SPEED, "replan_period": 0.2,
                    "horizon": 40, "dt_plan": 0.05, "n_samples": 32,
                    "resolution": 1.0, "inflate": 1, "goal_radius": 1.5,
                    "safety_margin": 0.5, "use_prediction": True,
                    "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
                    "lateral_bias": bias, "predictor": {"type": "constant_velocity"},
                    "per_drone": _per_drone(n, spread)},
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/antipodal_hetdyn_tmp"},
    }


def _run_cell(job):
    label, n, spread, bias, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(n, spread, bias, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return (label, by_seed)


def _succ(bs, seeds):
    return sum(bs[s] == "success" for s in seeds)


def _breakdown(bs, seeds):
    """(n_collision, n_timeout) — separates a coordination breakdown (collision)
    from a slow drone simply running out of clock (timeout)."""
    coll = sum(bs[s] == "collision" for s in seeds)
    to = sum(bs[s] == "timeout" for s in seeds)
    return coll, to


def _mc(a, b, seeds):
    # c-b>0 => b better than a
    bb = sum(a[s] == "success" and b[s] != "success" for s in seeds)
    cc = sum(a[s] != "success" and b[s] == "success" for s in seeds)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--spreads", type=float, nargs="+", default=[2.0, 4.0, 6.0],
                    help="speed spreads to test with the convention on (besides homo 0)")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--bias", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=6)
    # The slowest arm (speed BASE - max_spread/2, e.g. 2 at spread 6) must have
    # ample clock so a FREE slow drone never times out — otherwise "timeout"
    # conflates a coordination jam with a drone simply running out of steps.
    # A speed-2 drone over a ~1.4x roundabout of the 40-unit diameter needs
    # ~560 steps; 1000 gives ~1.8x margin. The coll/to breakdown confirms it.
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--out", default="results/antipodal_hetero_dynamics_phase.json")
    args = ap.parse_args()

    n = args.n
    smax = max(args.spreads)
    jobs = [("homo_b2", n, 0.0, args.bias, args.seed, args.episodes, args.max_steps)]
    for s in args.spreads:
        jobs.append((f"het{s}_b2", n, s, args.bias, args.seed, args.episodes, args.max_steps))
    jobs.append(("hetmax_b0", n, smax, 0.0, args.seed, args.episodes, args.max_steps))

    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)
    cells = dict(res)

    seeds = sorted(set.intersection(*[set(bs) for bs in cells.values()]))
    homo = cells["homo_b2"]
    hetmax_b2 = cells[f"het{smax}_b2"]
    hetmax_b0 = cells["hetmax_b0"]

    report = {"n": n, "base_speed": BASE_SPEED, "bias": args.bias,
              "episodes": args.episodes, "m": len(seeds), "arms": {}, "tests": {}}
    print(f"\nHeterogeneous-speed right-of-way @ N={n}, bias={args.bias}, "
          f"mean speed={BASE_SPEED}, paired m={len(seeds)}")
    print(f"{'arm':>10} | {'succ':>6} | {'coll/to':>8} | {'vs homo_b2 (c-b>0 => worse homo)':>34}")
    print("-" * 72)
    for label, bs in res:
        s = _succ(bs, seeds)
        coll, to = _breakdown(bs, seeds)
        report["arms"][label] = {"success": s, "collision": coll, "timeout": to}
        if label == "homo_b2":
            print(f"{label:>10} | {s:>3}/{len(seeds):<2} | {coll:>3}/{to:<3} | (reference)")
        else:
            b, c, p = _mc(homo, bs, seeds)
            print(f"{label:>10} | {s:>3}/{len(seeds):<2} | {coll:>3}/{to:<3} | "
                  f"b={b} c={c} p={p:.4f}")
            report["tests"][f"{label}_vs_homo_b2"] = {"b": b, "c": c, "p": p}

    # Does the convention still rescue the maximally-heterogeneous fleet?
    b, c, p = _mc(hetmax_b0, hetmax_b2, seeds)
    report["tests"]["hetmax_b2_vs_hetmax_b0"] = {"b": b, "c": c, "p": p}
    print(f"\nconvention rescue @ spread {smax}: hetmax_b2 vs hetmax_b0  "
          f"b={b} c={c} p={p:.4f}  (c-b>0 => bias rescues mixed fleet)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
