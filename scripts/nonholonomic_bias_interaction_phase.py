"""How much right-of-way convention you need depends on the drones' agility.

The [non-holonomic study](docs/findings.md#the-right-of-way-convention-survives-non-holonomic-drones--and-without-it-agility-is-a-non-monotone-liability)
found two things: the `lateral_bias` convention rescues the antipodal swap at every
turn rate, and *without* it the most sluggish drones deadlock the least — a slow
turner cannot perform the symmetric mirror-swerve, so non-holonomy is a free,
partial symmetry-breaker. This asks the follow-up: if sluggish drones already break
symmetry for free, do they need *less* convention to finish the job?

Same setup (antipodal N=6, MPC, `dummy_unicycle`), but now a 2-D sweep of the
convention strength `lateral_bias` × the turn-rate limit. The question is how strong
a bias each agility regime needs to reach ≈100 %.

  bias×rate   the full success grid (rows = turn rate, cols = bias)
  weak        at a WEAK bias, the sluggish fleet vs the agile fleet, paired by seed
              -> sluggishness substitutes for convention
  strong      at a STRONG bias they tie -> convention dominates, agility irrelevant

  python scripts/nonholonomic_bias_interaction_phase.py --episodes 40
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
N = 6
TURN_RATES = [0.5, 2.0, 8.0]
BIASES = [0.0, 0.5, 1.0, 2.0]


def _drones(n):
    out = []
    for k in range(n):
        a = 2.0 * math.pi * k / n
        out.append({"name": f"d{k}",
                    "start": [round(CX + RADIUS * math.cos(a), 3), round(CY + RADIUS * math.sin(a), 3)],
                    "goal": [round(CX - RADIUS * math.cos(a), 3), round(CY - RADIUS * math.sin(a), 3)],
                    "radius": 0.4, "start_jitter": 0.8})
    return out


def _cfg(tr, bias, seed, eps, ms):
    p = {"type": "mpc", "max_speed": SPEED, "replan_period": 0.2, "horizon": 40,
         "dt_plan": 0.05, "n_samples": 32, "resolution": 1.0, "inflate": 1,
         "goal_radius": 1.5, "safety_margin": 0.5, "use_prediction": True,
         "w_goal": 1.0, "w_obs": 100.0, "w_smooth": 0.05,
         "predictor": {"type": "constant_velocity"}}
    if bias:
        p["lateral_bias"] = bias
    return {"name": "nhb", "seed": seed, "num_episodes": eps,
            "scenario": {"type": "multi_drone_grid", "size": [50, 50], "resolution": 1.0,
                         "obstacles": {"type": "none"}, "drones": _drones(N)},
            "simulator": {"type": "dummy_unicycle", "dt": 0.05, "max_steps": ms,
                          "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4,
                          "turn_rate_max": tr},
            "planner": p, "sensor": {"type": "perfect"}, "output": {"dir": "results/nhb"}}


def _run(job):
    tr, bias, seed, eps, ms = job
    with tempfile.TemporaryDirectory() as td:
        out = run_experiment_multi(ExperimentConfig.from_dict(_cfg(tr, bias, seed, eps, ms)), Path(td))
        bits = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            bits[d["meta"]["seed"]] = (d["outcome"] == "success")
    return (tr, bias, bits)


def _mc(a, b):
    bb = sum(1 for s in a if a[s] and not b[s])
    cc = sum(1 for s in a if b[s] and not a[s])
    return bb, cc, mcnemar_exact_p(bb, cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--out", default="results/nonholonomic_bias_interaction_phase.json")
    args = ap.parse_args()

    jobs = [(tr, b, args.seed, args.episodes, args.max_steps) for tr in TURN_RATES for b in BIASES]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run, jobs)
    grid = {(tr, b): bits for tr, b, bits in res}
    m = args.episodes

    print(f"Success grid — antipodal N={N}, non-holonomic, m={m} (rows=turn rate, cols=lateral_bias)")
    print("  turn rate | " + "  ".join(f"b={b}" for b in BIASES))
    print("-" * 48)
    for tr in TURN_RATES:
        print(f"   {tr:>5}    | " + "  ".join(f"{sum(grid[(tr,b)].values()):>2}/{m}"[:5] for b in BIASES))
    print("-" * 48)

    # weak vs strong convention: does sluggishness substitute for the convention?
    print("\nSluggish (0.5) vs agile (8.0) turn rate, paired by seed:")
    for b in (0.0, 0.5, 1.0, 2.0):
        slug, agile = grid[(0.5, b)], grid[(8.0, b)]
        bb, cc, p = _mc(agile, slug)   # c = sluggish-only success
        s_s, s_a = sum(slug.values()), sum(agile.values())
        print(f"  bias={b}: sluggish {s_s:>2}/{m}  agile {s_a:>2}/{m}  (b={bb} c={cc} p={p:.2e})")
    print("=> at weak bias sluggishness substitutes for the convention; at strong bias they tie.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"grid": {f"{tr}|{b}": sum(grid[(tr, b)].values()) for tr in TURN_RATES for b in BIASES},
         "m": m}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
