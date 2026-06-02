#!/usr/bin/env python3
"""Phase sweep: the game-theoretic peer-predictor win is an inverted-U over encounter ANGLE.

The proven crossing win (`scripts/crossing_predictor_accel_phase.py`,
findings.md) is measured at ONE geometry: a 90-degree perpendicular crossing. It
sweeps acceleration headroom but holds the encounter angle fixed. This sweep asks
the orthogonal question — *for which encounter geometries does goal-aware peer
prediction actually help?* — and predicts a mechanistic answer.

`game_theoretic` models a peer as taking one best-response step toward its OWN
goal; `constant_velocity` coasts the peer's current velocity straight. These two
forecasts DIVERGE only when the peer's goal direction differs from its current
heading. At a head-on (antipodal) swap the peer's goal sits directly behind the
ego drone, so "steer toward your goal" and "coast straight" point the same way:
the two predictors issue the SAME forecast and therefore the SAME avoidance
decision. At a perpendicular crossing the goal direction is 90 degrees off the
instantaneous closing velocity, so the forecasts diverge most. Hypothesis: the
paired game_theoretic advantage is ~0 at head-on (180 deg), peaks near
perpendicular (90 deg), and shrinks again at shallow angles where the paths
barely conflict — an inverted-U over the encounter angle.

Geometry: both drones fly a diameter of a circle of radius R about the world
centre, so they are GUARANTEED to conflict at the centre and every arm has an
identical path length (2R). Drone A flies +x (angle 0). Drone B's travel
direction is the swept encounter angle alpha; B start = C - R*(cos a, sin a),
goal = C + R*(cos a, sin a). alpha=90 reproduces the shipped perpendicular
crossing; alpha=180 is a pure head-on swap. `start_jitter` breaks the mirror per
seed (so pairing is honest) and `max_accel` is held at the proven sweet spot (6),
where constant_velocity is most stressed.

Pairing is valid because the runner seeds every per-drone sim/sensor and the
spawn jitter from the same episode seed, so for a given seed both predictor arms
see the same spawn geometry; only the peer forecast differs.

Output: ``results/crossing_predictor_angle_phase/phase.json`` (+ per-seed
``phase_raw.json``) and a two-panel ``phase.png``.

Run (defaults: 7 angles x 2 predictors x n=60):
    python scripts/crossing_predictor_angle_phase.py
    python scripts/crossing_predictor_angle_phase.py --angles 90 180 --n 20 --workers 4
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
DEFAULT_ANGLES = [30.0, 60.0, 90.0, 120.0, 150.0, 165.0, 180.0]
SEED_BASE = 42
START_JITTER = 0.8       # breaks the crossing's mirror symmetry, per seed
SAFETY_MARGIN = 0.5
MAX_ACCEL = 6.0          # the proven sweet spot where constant_velocity is most stressed
CENTER = (25.0, 25.0)
RADIUS = 21.0            # matches the shipped crossing (start [4,25] -> goal [46,25])
ARMS = ("constant_velocity", "game_theoretic")
BASELINE_ARM = "constant_velocity"


def _base_config() -> dict:
    from uav_nav_lab.config import ExperimentConfig

    return ExperimentConfig.from_yaml(EXAMPLE).to_dict()


def _drone_endpoints(alpha_deg: float) -> tuple[list[float], list[float], list[float], list[float]]:
    """Two drones on a diameter through CENTER; B's travel direction is alpha from A's (+x).

    Returns (a_start, a_goal, b_start, b_goal). Both pass through CENTER, both have
    path length 2*RADIUS, so only the encounter angle changes across the sweep.
    """
    cx, cy = CENTER
    a_start = [cx - RADIUS, cy]
    a_goal = [cx + RADIUS, cy]
    a = math.radians(alpha_deg)
    dx, dy = math.cos(a), math.sin(a)
    b_start = [cx - RADIUS * dx, cy - RADIUS * dy]
    b_goal = [cx + RADIUS * dx, cy + RADIUS * dy]
    return a_start, a_goal, b_start, b_goal


def _cell_config(base: dict, predictor: str, alpha_deg: float) -> dict:
    cfg = copy.deepcopy(base)
    cfg["planner"]["predictor"] = {"type": predictor}
    cfg["planner"]["safety_margin"] = SAFETY_MARGIN
    cfg["simulator"]["max_accel"] = MAX_ACCEL
    a_start, a_goal, b_start, b_goal = _drone_endpoints(alpha_deg)
    drones = cfg["scenario"]["drones"]
    assert len(drones) == 2, "angle sweep assumes a 2-drone crossing example"
    drones[0]["start"], drones[0]["goal"] = a_start, a_goal
    drones[1]["start"], drones[1]["goal"] = b_start, b_goal
    for d in drones:
        d["start_jitter"] = START_JITTER
    cfg["name"] = f"{predictor}_ang{alpha_deg:g}"
    cfg.pop("output", None)
    return cfg


def _run_cell_chunk(job: dict) -> dict:
    from uav_nav_lab.config import ExperimentConfig
    from uav_nav_lab.runner.experiment import run_experiment

    cfg = ExperimentConfig.from_dict(job["config"])
    cfg.seed = job["seed_start"]
    cfg.num_episodes = job["count"]
    out = Path(tempfile.mkdtemp(prefix="angphase_"))
    run_experiment(cfg, out)
    episodes = []
    for p in sorted(out.glob("episode_*_joint.json")):
        d = json.loads(p.read_text())
        episodes.append({"seed": d["meta"]["seed"], "outcome": d["outcome"]})
    for p in out.glob("*"):
        p.unlink()
    out.rmdir()
    return {"angle": job["angle"], "arm": job["arm"], "episodes": episodes}


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
    ap.add_argument("--n", type=int, default=60, help="episodes per (angle, predictor) cell")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--angles", type=float, nargs="+", default=DEFAULT_ANGLES,
                    help="encounter angles (deg) to sweep; 90=perpendicular, 180=head-on")
    ap.add_argument("--out", default=str(REPO / "results/crossing_predictor_angle_phase"))
    args = ap.parse_args()

    from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p

    base = _base_config()
    jobs: list[dict] = []
    for angle in args.angles:
        for arm in ARMS:
            cfg = _cell_config(base, arm, angle)
            seed = SEED_BASE
            remaining = args.n
            while remaining > 0:
                count = min(args.chunk, remaining)
                jobs.append({"angle": angle, "arm": arm, "config": cfg,
                             "seed_start": seed, "count": count})
                seed += count
                remaining -= count

    print(f"[angphase] {len(args.angles)} angles x {len(ARMS)} predictors x n={args.n} "
          f"= {len(jobs)} jobs on {args.workers} workers")
    t0 = time.perf_counter()
    with Pool(processes=args.workers) as pool:
        chunk_results = pool.map(_run_cell_chunk, jobs)
    dt = time.perf_counter() - t0
    print(f"[angphase] sweep done in {dt:.0f}s")

    by_cell: dict[float, dict[str, dict[int, str]]] = {}
    for r in chunk_results:
        cell = by_cell.setdefault(r["angle"], {a: {} for a in ARMS})
        for ep in r["episodes"]:
            cell[r["arm"]][ep["seed"]] = ep["outcome"]

    rows, raw = [], []
    for angle in sorted(by_cell):
        cells = by_cell[angle]
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
            "angle": angle, "n": n,
            "cv_success": cv_succ / n, "gt_success": gt_succ / n,
            "cv_success_ci": [cv_lo, cv_hi], "gt_success_ci": [gt_lo, gt_hi],
            "cv_collision": cv_coll / n, "gt_collision": gt_coll / n,
            "success_mcnemar": {"b": b, "c": c, "p": p},
        })
        raw.append({"angle": angle, "outcomes": {a: cells[a] for a in ARMS}})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"n_per_cell": args.n, "angles": args.angles, "arms": list(ARMS),
              "start_jitter": START_JITTER, "safety_margin": SAFETY_MARGIN,
              "max_accel": MAX_ACCEL, "center": list(CENTER), "radius": RADIUS,
              "seconds": dt, "rows": rows}
    (out_dir / "phase.json").write_text(json.dumps(result, indent=2))
    (out_dir / "phase_raw.json").write_text(json.dumps(raw, indent=2))

    _print_table(rows)
    _plot(rows, out_dir / "phase.png")
    print(f"[angphase] wrote {out_dir/'phase.json'}, phase_raw.json and {out_dir/'phase.png'}")
    return 0


def _print_table(rows: list[dict]) -> None:
    print()
    print(f"{'angle':>6} {'n':>4} | {'const_vel':>9} {'game_theo':>9} {'dSucc':>6} "
          f"{'gt won/lost':>11} {'p':>7}")
    print("-" * 62)
    for r in rows:
        m = r["success_mcnemar"]
        d = (r["gt_success"] - r["cv_success"]) * 100
        print(f"{r['angle']:>5g}° {r['n']:>4} | "
              f"{r['cv_success']*100:>8.1f}% {r['gt_success']*100:>8.1f}% {d:>+5.1f} "
              f"{m['c']:>4}/{m['b']:<6} {m['p']:>7.4f}")
    print("\n  angle: 90=perpendicular crossing, 180=head-on swap.")
    print("  dSucc = game_theoretic minus constant_velocity joint success (pp).")
    print("  gt won/lost = paired seeds gt succeeded where cv failed (c) / vice versa (b).")
    print("  p = exact McNemar.")


def _plot(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["angle"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for key, label, color in [("cv", "constant_velocity", "#777"),
                              ("gt", "game_theoretic", "#8e44ad")]:
        ys = [r[f"{key}_success"] * 100 for r in rows]
        lo = [r[f"{key}_success_ci"][0] * 100 for r in rows]
        hi = [r[f"{key}_success_ci"][1] * 100 for r in rows]
        ax1.plot(xs, ys, "o-", color=color, label=label)
        ax1.fill_between(xs, lo, hi, color=color, alpha=0.15)
    ax1.axvline(90, color="#2980b9", ls=":", lw=1, label="perpendicular")
    ax1.axvline(180, color="#c0392b", ls=":", lw=1, label="head-on")
    ax1.set_xlabel("encounter angle (deg): 90 = perpendicular, 180 = head-on")
    ax1.set_ylabel("joint success rate (%)")
    ax1.set_title("Crossing joint success vs encounter angle")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    net = [r["success_mcnemar"]["c"] - r["success_mcnemar"]["b"] for r in rows]
    colors = ["#27ae60" if r["success_mcnemar"]["p"] < 0.05 else "#95a5a6" for r in rows]
    bars = ax2.bar([f"{x:g}" for x in xs], net, color=colors)
    for r, bar in zip(rows, bars):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"p={r['success_mcnemar']['p']:.2f}", ha="center",
                 va="bottom" if bar.get_height() >= 0 else "top", fontsize=8)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("encounter angle (deg)")
    ax2.set_ylabel("net paired wins for game_theoretic (c - b)")
    ax2.set_title("Paired success gain (green = McNemar p<0.05)")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("Where does goal-aware peer prediction help? "
                 "game_theoretic vs constant_velocity over encounter angle")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    raise SystemExit(main())
