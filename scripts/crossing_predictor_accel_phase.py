#!/usr/bin/env python3
"""Phase sweep: when does the game-theoretic peer predictor beat constant velocity?

`game_theoretic` models a peer drone as taking one best-response step toward its
OWN goal, where `constant_velocity` just coasts the peer's current velocity in a
straight line. The shipped example pair
(`examples/exp_multi_drone_crossing_{const_vel,game_theoretic}.yaml`) could not
tell them apart: a perfectly symmetric 2-drone perpendicular crossing with
near-instant acceleration (max_accel=80) makes both drones mirror-swerve and
re-collide, so BOTH predictors score 100% collision on every seed (and the
geometry is seed-invariant, so n>1 is meaningless).

Two fixes turn it into a real test:
  1. `start_jitter` on each drone (a new multi_drone_grid knob) breaks the mirror
     symmetry and makes the encounter vary per seed.
  2. Lowering `max_accel` removes the option to dodge reactively at the last
     instant, so the planner must COMMIT on its forecast — which is exactly when
     forecast quality (the predictor) can matter.

This sweep holds the tight-conflict setup fixed (start_jitter=0.8,
safety_margin=0.5) and sweeps `max_accel` from ballistic (4) to near-instant
(40), running both predictors paired by seed. Hypothesis: the game-theoretic
edge is largest at low accel (must commit on the forecast) and vanishes at high
accel (reactive dodging makes the forecast irrelevant).

Pairing is valid because the runner seeds every per-drone sim/sensor and the
spawn jitter from the same episode seed, so for a given seed both predictor arms
see the same spawn geometry; the only difference is the peer forecast.

Output: ``results/crossing_predictor_accel_phase/phase.json`` (+ per-seed
``phase_raw.json``) and a two-panel ``phase.png``.

Run (defaults: 6 accel levels x 2 predictors x n=60, ~25 min on 16 workers):
    python scripts/crossing_predictor_accel_phase.py
    python scripts/crossing_predictor_accel_phase.py --n 40 --workers 12
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import copy
import json
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXAMPLE = REPO / "examples/exp_multi_drone_crossing_game_theoretic.yaml"
DEFAULT_LEVELS = [4.0, 6.0, 8.0, 12.0, 20.0, 40.0]
SEED_BASE = 42
START_JITTER = 0.8       # breaks the crossing's mirror symmetry, per seed
SAFETY_MARGIN = 0.5
ARMS = ("constant_velocity", "game_theoretic")
BASELINE_ARM = "constant_velocity"


def _base_config() -> dict:
    from uav_nav_lab.config import ExperimentConfig

    return ExperimentConfig.from_yaml(EXAMPLE).to_dict()


def _cell_config(base: dict, predictor: str, max_accel: float) -> dict:
    cfg = copy.deepcopy(base)
    cfg["planner"]["predictor"] = {"type": predictor}
    cfg["planner"]["safety_margin"] = SAFETY_MARGIN
    cfg["simulator"]["max_accel"] = float(max_accel)
    for d in cfg["scenario"]["drones"]:
        d["start_jitter"] = START_JITTER
    cfg["name"] = f"{predictor}_a{max_accel:g}"
    cfg.pop("output", None)
    return cfg


def _run_cell_chunk(job: dict) -> dict:
    from uav_nav_lab.config import ExperimentConfig
    from uav_nav_lab.runner.experiment import run_experiment

    cfg = ExperimentConfig.from_dict(job["config"])
    cfg.seed = job["seed_start"]
    cfg.num_episodes = job["count"]
    out = Path(tempfile.mkdtemp(prefix="xphase_"))
    run_experiment(cfg, out)
    episodes = []
    for p in sorted(out.glob("episode_*_joint.json")):
        d = json.loads(p.read_text())
        episodes.append({"seed": d["meta"]["seed"], "outcome": d["outcome"]})
    for p in out.glob("*"):
        p.unlink()
    out.rmdir()
    return {"level": job["level"], "arm": job["arm"], "episodes": episodes}


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60, help="episodes per (level, predictor) cell")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--levels", type=float, nargs="+", default=DEFAULT_LEVELS,
                    help="max_accel levels to sweep")
    ap.add_argument("--out", default=str(REPO / "results/crossing_predictor_accel_phase"))
    args = ap.parse_args()

    from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p

    base = _base_config()
    jobs: list[dict] = []
    for level in args.levels:
        for arm in ARMS:
            cfg = _cell_config(base, arm, level)
            seed = SEED_BASE
            remaining = args.n
            while remaining > 0:
                count = min(args.chunk, remaining)
                jobs.append({"level": level, "arm": arm, "config": cfg,
                             "seed_start": seed, "count": count})
                seed += count
                remaining -= count

    print(f"[xphase] {len(args.levels)} accel levels x {len(ARMS)} predictors x n={args.n} "
          f"= {len(jobs)} jobs on {args.workers} workers")
    t0 = time.perf_counter()
    with Pool(processes=args.workers) as pool:
        chunk_results = pool.map(_run_cell_chunk, jobs)
    dt = time.perf_counter() - t0
    print(f"[xphase] sweep done in {dt:.0f}s")

    by_cell: dict[float, dict[str, dict[int, str]]] = {}
    for r in chunk_results:
        cell = by_cell.setdefault(r["level"], {a: {} for a in ARMS})
        for ep in r["episodes"]:
            cell[r["arm"]][ep["seed"]] = ep["outcome"]

    rows, raw = [], []
    for level in sorted(by_cell):
        cells = by_cell[level]
        seeds = sorted(set(cells[ARMS[0]]) & set(cells[ARMS[1]]))
        n = len(seeds)
        cv, gt = cells[BASELINE_ARM], cells["game_theoretic"]
        cv_succ = sum(cv[s] == "success" for s in seeds)
        gt_succ = sum(gt[s] == "success" for s in seeds)
        cv_coll = sum(cv[s] == "collision" for s in seeds)
        gt_coll = sum(gt[s] == "collision" for s in seeds)
        b = sum(cv[s] == "success" and gt[s] != "success" for s in seeds)  # gt lost
        c = sum(cv[s] != "success" and gt[s] == "success" for s in seeds)  # gt won
        p = mcnemar_exact_p(b, c)
        _, cv_lo, cv_hi = _wilson(cv_succ, n)
        _, gt_lo, gt_hi = _wilson(gt_succ, n)
        rows.append({
            "level": level, "n": n,
            "cv_success": cv_succ / n, "gt_success": gt_succ / n,
            "cv_success_ci": [cv_lo, cv_hi], "gt_success_ci": [gt_lo, gt_hi],
            "cv_collision": cv_coll / n, "gt_collision": gt_coll / n,
            "success_mcnemar": {"b": b, "c": c, "p": p},
        })
        raw.append({"level": level, "outcomes": {a: cells[a] for a in ARMS}})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"n_per_cell": args.n, "levels": args.levels, "arms": list(ARMS),
              "start_jitter": START_JITTER, "safety_margin": SAFETY_MARGIN,
              "seconds": dt, "rows": rows}
    (out_dir / "phase.json").write_text(json.dumps(result, indent=2))
    (out_dir / "phase_raw.json").write_text(json.dumps(raw, indent=2))

    _print_table(rows)
    _plot(rows, out_dir / "phase.png")
    print(f"[xphase] wrote {out_dir/'phase.json'}, phase_raw.json and {out_dir/'phase.png'}")
    return 0


def _print_table(rows: list[dict]) -> None:
    print()
    print(f"{'max_accel':>9} {'n':>4} | {'const_vel':>9} {'game_theo':>9} {'dSucc':>6} "
          f"{'gt won/lost':>11} {'p':>7}")
    print("-" * 64)
    for r in rows:
        m = r["success_mcnemar"]
        d = (r["gt_success"] - r["cv_success"]) * 100
        print(f"{r['level']:>9g} {r['n']:>4} | "
              f"{r['cv_success']*100:>8.1f}% {r['gt_success']*100:>8.1f}% {d:>+5.1f} "
              f"{m['c']:>4}/{m['b']:<6} {m['p']:>7.4f}")
    print("\n  dSucc = game_theoretic minus constant_velocity joint success (pp).")
    print("  gt won/lost = paired seeds game_theoretic succeeded where const_vel failed (c)")
    print("  / failed where const_vel succeeded (b). p = exact McNemar.")


def _plot(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["level"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for key, label, color in [("cv", "constant_velocity", "#777"),
                              ("gt", "game_theoretic", "#8e44ad")]:
        ys = [r[f"{key}_success"] * 100 for r in rows]
        lo = [r[f"{key}_success_ci"][0] * 100 for r in rows]
        hi = [r[f"{key}_success_ci"][1] * 100 for r in rows]
        ax1.plot(xs, ys, "o-", color=color, label=label)
        ax1.fill_between(xs, lo, hi, color=color, alpha=0.15)
    ax1.set_xscale("log")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{x:g}" for x in xs])
    ax1.set_xlabel("max_accel (low = must commit on forecast → high = reactive dodge)")
    ax1.set_ylabel("joint success rate (%)")
    ax1.set_title("Crossing joint success vs acceleration headroom")
    ax1.legend()
    ax1.grid(alpha=0.3)

    net = [r["success_mcnemar"]["c"] - r["success_mcnemar"]["b"] for r in rows]
    colors = ["#27ae60" if r["success_mcnemar"]["p"] < 0.05 else "#95a5a6" for r in rows]
    bars = ax2.bar([f"{x:g}" for x in xs], net, color=colors)
    for r, bar in zip(rows, bars):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"p={r['success_mcnemar']['p']:.2f}", ha="center",
                 va="bottom" if bar.get_height() >= 0 else "top", fontsize=8)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("max_accel")
    ax2.set_ylabel("net paired wins for game_theoretic (c - b)")
    ax2.set_title("Paired success gain (green = McNemar p<0.05)")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("When does the game-theoretic peer predictor beat constant velocity? "
                 "(2-drone crossing)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    raise SystemExit(main())
