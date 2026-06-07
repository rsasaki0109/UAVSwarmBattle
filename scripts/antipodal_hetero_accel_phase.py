"""Does the right-of-way convention survive a fleet of drones with different
ACCELERATION limits — i.e. mixed *agility*, not mixed cruise speed?

scripts/antipodal_hetero_dynamics_phase.py proved the convention is robust to
SPEED heterogeneity (a 4x spread is fine), because the roundabout it builds is
SPATIAL not temporal — a fast drone keeps its own lane rather than lapping a slow
one into the hub. But speed only sets how fast a drone *cruises*; it is the
ACCELERATION limit that sets how fast it can *turn*. And a roundabout is made of
turning: every drone must continuously bend its velocity onto a curved lane. A
sluggish drone (low `max_accel`) commanded to veer cannot execute the turn in
time and drifts wide / cuts toward the centre — a failure mode speed spread never
exposes. So acceleration is the heterogeneity axis that most directly stresses
the roundabout mechanism.

This is also a literature anchor. Acceleration-Velocity Obstacles (van den Berg
et al. 2011, https://gamma.cs.unc.edu/AVO/) is the canonical reciprocal-avoidance
framework under acceleration constraints; and a recent predictive-CBF study
(arXiv:2501.10447) reports that "heterogeneity in the controller parameters is
not sufficient to prevent deadlock on its own" — i.e. mixing dynamics does NOT
substitute for an explicit symmetry-breaker. We test exactly that here.

Setup mirrors the speed study so the two are directly comparable: N=6 antipodal
swap, MPC + right-of-way `lateral_bias`, fleet MEAN `max_accel` held fixed so the
comparison is not confounded by an overall more/less agile fleet. We alternate
agile/sluggish around the ring via the new `simulator.per_drone` override
(wired in the multi-drone builder, mirroring `planner.per_drone`):
  max_accel = base +/- spread/2, three of each, mean == base.

Arms at fixed N=6, paired by seed, McNemar exact, all in the 2D antipodal swap:

  homo_b2     spread 0, bias B    the homogeneous roundabout (reference, ~100%)
  het<S>_b2   spread S, bias B    does acceleration spread S DEGRADE the convention?
  hetmax_b0   spread Smax, bias 0 control: mixed agility WITHOUT the convention
                                  (is mixed accel alone enough? expect collapse)

Reports het<S>_b2 vs homo_b2 (does heterogeneity degrade the working convention?)
and hetmax_b2 vs hetmax_b0 (does the convention still RESCUE the mixed fleet?).
The coll/to breakdown separates a coordination jam (collision) from a sluggish
drone simply running out of clock (timeout).
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


def _per_drone(n, base_accel, spread):
    """Alternate agile/sluggish around the ring with mean max_accel == base.
    spread 0 returns [] (homogeneous — let the shared simulator config stand)."""
    if spread == 0:
        return []
    out = []
    for k in range(n):
        a = base_accel + (spread / 2.0 if k % 2 == 0 else -spread / 2.0)
        out.append({"max_accel": round(a, 3)})
    return out


def _cfg(n, base_accel, spread, bias, seed, n_eps, max_steps, base_speed):
    sim = {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
           "max_accel": base_accel, "goal_radius": 1.5, "drone_radius": 0.4,
           "per_drone": _per_drone(n, base_accel, spread)}
    return {
        "name": f"antipodal_n{n}_accelspread{spread}_b{bias}", "seed": seed,
        "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50],
                     "resolution": 1.0, "obstacles": {"type": "none"},
                     "drones": _drones(n)},
        "simulator": sim,
        "planner": {"type": "mpc", "max_speed": base_speed, "replan_period": 0.2,
                    "horizon": 40, "dt_plan": 0.05, "n_samples": 32,
                    "resolution": 1.0, "inflate": 1, "goal_radius": 1.5,
                    "safety_margin": 0.5, "use_prediction": True,
                    "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
                    "lateral_bias": bias, "predictor": {"type": "constant_velocity"}},
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/antipodal_hetaccel_tmp"},
    }


def _run_cell(job):
    label, n, base_accel, spread, bias, seed, n_eps, max_steps, base_speed = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(
            _cfg(n, base_accel, spread, bias, seed, n_eps, max_steps, base_speed))
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
    from a sluggish drone simply running out of clock (timeout)."""
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
    ap.add_argument("--base-accel", type=float, default=6.0,
                    help="fleet-mean max_accel (held fixed across spreads)")
    ap.add_argument("--base-speed", type=float, default=5.0,
                    help="cruise speed (raising it raises the roundabout's "
                         "centripetal demand v^2/r — the knob that makes the "
                         "sluggish drones unable to hold their lane)")
    ap.add_argument("--spreads", type=float, nargs="+", default=[3.0, 6.0, 9.0],
                    help="accel spreads to test with the convention on (besides homo 0)")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--bias", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=6)
    # A FREE sluggish drone (accel base-Smax/2) must have ample clock so it never
    # times out on its own — otherwise "timeout" conflates a jam with a drone
    # simply taking longer to reach cruise. The coll/to breakdown confirms it.
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--out", default="results/antipodal_hetero_accel_phase.json")
    args = ap.parse_args()

    n = args.n
    ba = args.base_accel
    bsp = args.base_speed
    smax = max(args.spreads)
    def _job(label, spread, bias):
        return (label, n, ba, spread, bias, args.seed, args.episodes,
                args.max_steps, bsp)
    # homo_b0 = the true deadlock floor (homogeneous fleet, no convention);
    # homo_b2 = the working homogeneous roundabout (reference).
    jobs = [_job("homo_b0", 0.0, 0.0), _job("homo_b2", 0.0, args.bias)]
    for s in args.spreads:
        jobs.append(_job(f"het{s}_b2", s, args.bias))
    jobs.append(_job("hetmax_b0", smax, 0.0))

    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)
    cells = dict(res)

    seeds = sorted(set.intersection(*[set(bs) for bs in cells.values()]))
    homo = cells["homo_b2"]
    homo_b0 = cells["homo_b0"]
    hetmax_b2 = cells[f"het{smax}_b2"]
    hetmax_b0 = cells["hetmax_b0"]

    report = {"n": n, "base_accel": ba, "base_speed": bsp, "bias": args.bias,
              "episodes": args.episodes, "m": len(seeds), "arms": {}, "tests": {}}
    print(f"\nHeterogeneous-acceleration right-of-way @ N={n}, bias={args.bias}, "
          f"mean accel={ba}, speed={bsp}, paired m={len(seeds)}")
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

    # Value of the convention on a homogeneous fleet (floor -> reference).
    b, c, p = _mc(homo_b0, homo, seeds)
    report["tests"]["homo_b2_vs_homo_b0"] = {"b": b, "c": c, "p": p}
    print(f"\nconvention value (homogeneous): homo_b2 vs homo_b0  "
          f"b={b} c={c} p={p:.4f}  (c-b>0 => bias helps)")

    # Does the convention still rescue the maximally-heterogeneous fleet?
    b, c, p = _mc(hetmax_b0, hetmax_b2, seeds)
    report["tests"]["hetmax_b2_vs_hetmax_b0"] = {"b": b, "c": c, "p": p}
    print(f"convention rescue @ accel spread {smax}: hetmax_b2 vs hetmax_b0  "
          f"b={b} c={c} p={p:.4f}  (c-b>0 => bias rescues mixed-agility fleet)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
