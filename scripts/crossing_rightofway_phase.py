#!/usr/bin/env python3
"""Is the goal-aware predictor's 2-drone crossing win really a symmetry-breaking win?

`scripts/crossing_predictor_angle_phase.py` (findings.md, "bimodal in encounter
angle") proved the goal-aware `game_theoretic` predictor beats the myopic
`constant_velocity` predictor on a 2-drone crossing at EXACTLY two angles: the 90°
perpendicular crossing and the 180° head-on swap. At every oblique angle both hit
the 100 % ceiling and the predictor is irrelevant.

Separately, `scripts/antipodal_rightofway_phase.py` (findings.md, "a decentralized
right-of-way lateral bias lifts the antipodal swap to 100 %") proved that on N>=3
SYMMETRIC congestion the failure is a coordination/symmetry problem, fixed by a
tiny `planner.lateral_bias` (a global "veer right" cost) at zero forecast cost.

This script asks whether those two stories are the SAME story at N=2. Hypothesis:
the two angles where constant_velocity fails (90°, 180°) are the two MOST symmetric
mutual-maneuver geometries, so cv's failure there is really a symmetry failure —
and a passive right-of-way convention on the DUMB cv predictor should recover those
cells just like the smart predictor does, making goal-aware prediction *substitutable*
by a 10-line decentralized convention on a 2-drone crossing.

Three arms, paired by seed:
  cv     : constant_velocity, lateral_bias 0   (the one that fails at 90°/180°)
  gt     : game_theoretic,    lateral_bias 0   (the proven predictor fix, #69)
  cvrow  : constant_velocity, lateral_bias B   (the proposed cheap fix)

Two McNemar tests: cvrow vs cv (does the passive convention fix the dumb predictor?)
and cvrow vs gt (is the convention as good as the smart predictor?). We deliberately
include oblique CONTROL angles (60°, 150°) where cv already succeeds, to check the
right-of-way bias does not INTRODUCE collisions where it is not needed (a real risk:
the bias is a standing cost, not a conditional rule).

Geometry, jitter, accel and margin are identical to crossing_predictor_angle_phase
so the cv/gt columns reproduce the shipped bimodal result.

Calibrate B at the largest-gap angle (180°) first:
    python scripts/crossing_rightofway_phase.py --bias-sweep 0.5 1 1.5 2 3 4 --angles 180 --n 30
Then the full comparison:
    python scripts/crossing_rightofway_phase.py            # angles 60 90 150 180, n=60
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import copy
import json
import math
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXAMPLE = REPO / "examples/exp_multi_drone_crossing_game_theoretic.yaml"
DEFAULT_ANGLES = [60.0, 90.0, 150.0, 180.0]   # 90 & 180 = stress; 60 & 150 = ceiling controls
SEED_BASE = 42
START_JITTER = 0.8
SAFETY_MARGIN = 0.5
MAX_ACCEL = 6.0
CENTER = (25.0, 25.0)
RADIUS = 21.0
# (label, predictor, lateral_bias-from-args?) — bias filled per run
ARM_SPECS = ("cv", "gt", "cvrow")
BASELINE_ARM = "cv"


def _base_config() -> dict:
    from uav_nav_lab.config import ExperimentConfig

    return ExperimentConfig.from_yaml(EXAMPLE).to_dict()


def _drone_endpoints(alpha_deg: float):
    cx, cy = CENTER
    a_start = [cx - RADIUS, cy]
    a_goal = [cx + RADIUS, cy]
    a = math.radians(alpha_deg)
    dx, dy = math.cos(a), math.sin(a)
    b_start = [cx - RADIUS * dx, cy - RADIUS * dy]
    b_goal = [cx + RADIUS * dx, cy + RADIUS * dy]
    return a_start, a_goal, b_start, b_goal


def _arm_params(arm: str, bias: float) -> tuple[str, float]:
    """Return (predictor_type, lateral_bias) for an arm label."""
    if arm == "cv":
        return "constant_velocity", 0.0
    if arm == "gt":
        return "game_theoretic", 0.0
    if arm == "cvrow":
        return "constant_velocity", bias
    raise ValueError(arm)


def _cell_config(base: dict, arm: str, alpha_deg: float, bias: float) -> dict:
    predictor, lateral_bias = _arm_params(arm, bias)
    cfg = copy.deepcopy(base)
    cfg["planner"]["predictor"] = {"type": predictor}
    cfg["planner"]["lateral_bias"] = lateral_bias
    cfg["planner"]["safety_margin"] = SAFETY_MARGIN
    cfg["simulator"]["max_accel"] = MAX_ACCEL
    a_start, a_goal, b_start, b_goal = _drone_endpoints(alpha_deg)
    drones = cfg["scenario"]["drones"]
    assert len(drones) == 2, "crossing sweep assumes a 2-drone example"
    drones[0]["start"], drones[0]["goal"] = a_start, a_goal
    drones[1]["start"], drones[1]["goal"] = b_start, b_goal
    for d in drones:
        d["start_jitter"] = START_JITTER
    cfg["name"] = f"{arm}_ang{alpha_deg:g}_b{lateral_bias:g}"
    cfg.pop("output", None)
    return cfg


def _run_cell_chunk(job: dict) -> dict:
    from uav_nav_lab.config import ExperimentConfig
    from uav_nav_lab.runner.experiment import run_experiment

    cfg = ExperimentConfig.from_dict(job["config"])
    cfg.seed = job["seed_start"]
    cfg.num_episodes = job["count"]
    out = Path(tempfile.mkdtemp(prefix="rowphase_"))
    run_experiment(cfg, out)
    episodes = []
    for p in sorted(out.glob("episode_*_joint.json")):
        d = json.loads(p.read_text())
        episodes.append({"seed": d["meta"]["seed"], "outcome": d["outcome"]})
    for p in out.glob("*"):
        p.unlink()
    out.rmdir()
    return {"angle": job["angle"], "arm": job["arm"], "episodes": episodes}


def _wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def _build_jobs(base, arms, angles, n, chunk, bias):
    jobs = []
    for angle in angles:
        for arm in arms:
            cfg = _cell_config(base, arm, angle, bias)
            seed = SEED_BASE
            remaining = n
            while remaining > 0:
                count = min(chunk, remaining)
                jobs.append({"angle": angle, "arm": arm, "config": cfg,
                             "seed_start": seed, "count": count})
                seed += count
                remaining -= count
    return jobs


def _collect(chunk_results, arms):
    by_cell = {}
    for r in chunk_results:
        cell = by_cell.setdefault(r["angle"], {a: {} for a in arms})
        for ep in r["episodes"]:
            cell[r["arm"]][ep["seed"]] = ep["outcome"]
    return by_cell


def _bias_sweep(base, args):
    from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p

    angle = args.angles[0]
    print(f"[rowcal] bias calibration at angle={angle:g}° (n={args.n}); "
          f"cvrow vs cv baseline\n")
    # baseline cv once
    cv_jobs = _build_jobs(base, ("cv",), [angle], args.n, args.chunk, 0.0)
    with Pool(processes=args.workers) as pool:
        cv_res = pool.map(_run_cell_chunk, cv_jobs)
    cv = _collect(cv_res, ("cv",))[angle]["cv"]
    cv_succ = sum(v == "success" for v in cv.values())
    print(f"  cv (bias 0): {cv_succ}/{len(cv)} = {cv_succ/len(cv)*100:.1f}%\n")
    print(f"{'bias':>6} | {'cvrow succ':>11} | vs cv (b/c, p)")
    print("-" * 44)
    for b in args.bias_sweep:
        jobs = _build_jobs(base, ("cvrow",), [angle], args.n, args.chunk, b)
        with Pool(processes=args.workers) as pool:
            res = pool.map(_run_cell_chunk, jobs)
        row = _collect(res, ("cvrow",))[angle]["cvrow"]
        seeds = sorted(set(cv) & set(row))
        succ = sum(row[s] == "success" for s in seeds)
        bb = sum(cv[s] == "success" and row[s] != "success" for s in seeds)  # row lost
        cc = sum(cv[s] != "success" and row[s] == "success" for s in seeds)  # row won
        p = mcnemar_exact_p(bb, cc)
        print(f"{b:>6.2f} | {succ:>4}/{len(seeds):<6} = {succ/len(seeds)*100:>5.1f}% "
              f"| {cc}/{bb}  p={p:.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--angles", type=float, nargs="+", default=DEFAULT_ANGLES)
    ap.add_argument("--bias", type=float, default=2.0)
    ap.add_argument("--bias-sweep", type=float, nargs="+", default=None,
                    help="calibrate B at args.angles[0] for the cvrow arm")
    ap.add_argument("--out", default=str(REPO / "results/crossing_rightofway_phase"))
    args = ap.parse_args()

    from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p

    base = _base_config()

    if args.bias_sweep is not None:
        return _bias_sweep(base, args)

    jobs = _build_jobs(base, ARM_SPECS, args.angles, args.n, args.chunk, args.bias)
    print(f"[rowphase] {len(args.angles)} angles x {len(ARM_SPECS)} arms x n={args.n} "
          f"= {len(jobs)} jobs on {args.workers} workers (bias={args.bias})")
    t0 = time.perf_counter()
    with Pool(processes=args.workers) as pool:
        chunk_results = pool.map(_run_cell_chunk, jobs)
    dt = time.perf_counter() - t0
    print(f"[rowphase] sweep done in {dt:.0f}s")

    by_cell = _collect(chunk_results, ARM_SPECS)
    rows, raw = [], []
    for angle in sorted(by_cell):
        cells = by_cell[angle]
        seeds = sorted(set(cells["cv"]) & set(cells["gt"]) & set(cells["cvrow"]))
        n = len(seeds)
        cv, gt, row = cells["cv"], cells["gt"], cells["cvrow"]
        succ = {a: sum(cells[a][s] == "success" for s in seeds) for a in ARM_SPECS}
        coll = {a: sum(cells[a][s] == "collision" for s in seeds) for a in ARM_SPECS}

        def _mc(ref, prop):
            b = sum(ref[s] == "success" and prop[s] != "success" for s in seeds)  # prop lost
            c = sum(ref[s] != "success" and prop[s] == "success" for s in seeds)  # prop won
            return {"b": b, "c": c, "p": mcnemar_exact_p(b, c)}

        cis = {a: _wilson(succ[a], n)[1:] for a in ARM_SPECS}
        rows.append({
            "angle": angle, "n": n,
            "success": {a: succ[a] / n for a in ARM_SPECS},
            "success_ci": {a: list(cis[a]) for a in ARM_SPECS},
            "collision": {a: coll[a] / n for a in ARM_SPECS},
            "cvrow_vs_cv": _mc(cv, row),   # does the passive convention fix dumb cv?
            "cvrow_vs_gt": _mc(gt, row),   # is the convention as good as the smart predictor?
        })
        raw.append({"angle": angle, "outcomes": {a: cells[a] for a in ARM_SPECS}})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"n_per_cell": args.n, "angles": args.angles, "arms": list(ARM_SPECS),
              "bias": args.bias, "start_jitter": START_JITTER,
              "safety_margin": SAFETY_MARGIN, "max_accel": MAX_ACCEL,
              "center": list(CENTER), "radius": RADIUS, "seconds": dt, "rows": rows}
    (out_dir / "phase.json").write_text(json.dumps(result, indent=2))
    (out_dir / "phase_raw.json").write_text(json.dumps(raw, indent=2))

    _print_table(rows, args.bias)
    _plot(rows, args.bias, out_dir / "phase.png")
    print(f"[rowphase] wrote {out_dir/'phase.json'}, phase_raw.json and {out_dir/'phase.png'}")
    return 0


def _print_table(rows, bias) -> None:
    print()
    print(f"{'angle':>6} {'n':>4} | {'cv':>6} {'gt':>6} {'cvrow':>6} | "
          f"{'row vs cv (c/b,p)':>20} | {'row vs gt (c/b,p)':>20}")
    print("-" * 84)
    for r in rows:
        s = r["success"]
        a, g = r["cvrow_vs_cv"], r["cvrow_vs_gt"]
        print(f"{r['angle']:>5g}° {r['n']:>4} | "
              f"{s['cv']*100:>5.1f} {s['gt']*100:>5.1f} {s['cvrow']*100:>5.1f} | "
              f"{a['c']:>3}/{a['b']:<3} p={a['p']:>6.4f} | "
              f"{g['c']:>3}/{g['b']:<3} p={g['p']:>6.4f}")
    print(f"\n  cv = constant_velocity bias 0; gt = game_theoretic bias 0; "
          f"cvrow = constant_velocity bias {bias:g}.")
    print("  90/180 = stress cells (cv fails); 60/150 = ceiling controls.")
    print("  row vs cv: c = cvrow won where cv failed (fix works); b = cvrow broke a cv win.")
    print("  row vs gt: c = cvrow won where gt failed; b = cvrow lost where gt won.")


def _plot(rows, bias, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["angle"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    series = [("cv", "constant_velocity (bias 0)", "#777", "o-"),
              ("gt", "game_theoretic (bias 0)", "#8e44ad", "s-"),
              ("cvrow", f"constant_velocity + right-of-way (bias {bias:g})", "#27ae60", "^-")]
    for key, label, color, style in series:
        ys = [r["success"][key] * 100 for r in rows]
        lo = [r["success_ci"][key][0] * 100 for r in rows]
        hi = [r["success_ci"][key][1] * 100 for r in rows]
        ax1.plot(xs, ys, style, color=color, label=label)
        ax1.fill_between(xs, lo, hi, color=color, alpha=0.12)
    ax1.axvline(90, color="#2980b9", ls=":", lw=1)
    ax1.axvline(180, color="#c0392b", ls=":", lw=1)
    ax1.set_xlabel("encounter angle (deg): 90 = perpendicular, 180 = head-on")
    ax1.set_ylabel("joint success rate (%)")
    ax1.set_title("Does a passive right-of-way convention substitute for the\n"
                  "goal-aware predictor on a 2-drone crossing?")
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(alpha=0.3)

    width = 0.38
    idx = list(range(len(rows)))
    net_cv = [r["cvrow_vs_cv"]["c"] - r["cvrow_vs_cv"]["b"] for r in rows]
    net_gt = [r["cvrow_vs_gt"]["c"] - r["cvrow_vs_gt"]["b"] for r in rows]
    c_cv = ["#27ae60" if r["cvrow_vs_cv"]["p"] < 0.05 else "#bdc3c7" for r in rows]
    c_gt = ["#2980b9" if r["cvrow_vs_gt"]["p"] < 0.05 else "#bdc3c7" for r in rows]
    ax2.bar([i - width / 2 for i in idx], net_cv, width, color=c_cv,
            label="cvrow vs cv (fix works)")
    ax2.bar([i + width / 2 for i in idx], net_gt, width, color=c_gt,
            label="cvrow vs gt (vs smart)")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(idx)
    ax2.set_xticklabels([f"{x:g}°" for x in xs])
    ax2.set_xlabel("encounter angle")
    ax2.set_ylabel("net paired wins for cvrow (c - b)")
    ax2.set_title("Paired gain (solid = McNemar p<0.05)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("Right-of-way as a substitute for goal-aware prediction on a 2-drone crossing")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    raise SystemExit(main())
