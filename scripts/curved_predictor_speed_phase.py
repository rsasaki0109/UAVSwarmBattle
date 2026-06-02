#!/usr/bin/env python3
"""Phase sweep: does modelling the hunter's TURN beat a straight-line forecast?

The pursuit-evasion study (`scripts/pursuit_prediction_speed_phase.py`) showed
that anticipating an `intercept` hunter with a *constant-velocity* forecast is a
large, escapability-gated evasion win over reacting. But the hunter does not move
in a straight line — proportional-navigation lead makes it curve — so a
constant-velocity forecast systematically points the wrong way (an offline check
shows it over-predicts the hunter's 1 s position by ~0.15 m mean / ~0.5 m p90).
The `constant_turn` predictor estimates the hunter's turn rate from its velocity
rotation and rolls it along the matching arc, cutting that forecast error
~60-90%.

This sweep asks the next question: does the better *forecast* translate into
better *evasion*, or does the planner's safety margin already absorb a
sub-metre constant-velocity error? Both arms run with prediction ON; the only
difference is `predictor.type` (constant_velocity vs constant_turn). We sweep the
hunter speed across the escapable band (above the drone's max_speed 3.0 nobody
escapes, so there is no signal there) with the same per-seed obstacle jitter that
made the scenario a valid paired test, and pair by seed.

Output: ``results/curved_predictor_speed_phase/phase.json`` (+ per-seed
``phase_raw.json``) and a two-panel ``phase.png``.

Run (defaults: 6 speed levels x 2 predictors x n=60, ~14 min on 16 workers):
    python scripts/curved_predictor_speed_phase.py
    python scripts/curved_predictor_speed_phase.py --n 40 --workers 12
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
EXAMPLE = REPO / "examples/exp_pursuit_evasion_mppi.yaml"
DEFAULT_LEVELS = [2.0, 2.4, 2.7, 2.8, 2.85, 2.9]  # hunter speed; drone max_speed = 3.0
SEED_BASE = 1000
START_JITTER = 3.0
TURN_RATE = 1.2
REPLAN_DT = 0.1          # = planner.replan_period; constant_turn needs this for ω
ARMS = ("constant_velocity", "constant_turn")
BASELINE_ARM = "constant_velocity"


def _base_config() -> dict:
    from uav_nav_lab.config import ExperimentConfig

    return ExperimentConfig.from_yaml(EXAMPLE).to_dict()


def _cell_config(base: dict, predictor: str, speed: float) -> dict:
    cfg = copy.deepcopy(base)
    cfg["planner"]["use_prediction"] = True
    pred = {"type": predictor}
    if predictor == "constant_turn":
        pred["dt"] = REPLAN_DT
    cfg["planner"]["predictor"] = pred
    for d in cfg["scenario"]["dynamic_obstacles"]:
        d["start_jitter"] = START_JITTER
        if d.get("policy") == "intercept":
            d["speed"] = float(speed)
            d["turn_rate"] = TURN_RATE
    cfg["name"] = f"{predictor}_s{speed:g}"
    cfg.pop("output", None)
    return cfg


def _run_cell_chunk(job: dict) -> dict:
    from uav_nav_lab.config import ExperimentConfig
    from uav_nav_lab.runner.experiment import run_experiment

    cfg = ExperimentConfig.from_dict(job["config"])
    cfg.seed = job["seed_start"]
    cfg.num_episodes = job["count"]
    out = Path(tempfile.mkdtemp(prefix="cphase_"))
    run_experiment(cfg, out)
    episodes = []
    for p in sorted(out.glob("episode_*.json")):
        d = json.loads(p.read_text())
        episodes.append({"seed": d["meta"]["seed"], "outcome": d["outcome"]})
    for p in out.glob("*"):
        p.unlink()
    out.rmdir()
    return {"level": job["level"], "arm": job["arm"], "episodes": episodes}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60, help="episodes per (level, predictor) cell")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--levels", type=float, nargs="+", default=DEFAULT_LEVELS,
                    help="hunter speed levels to sweep")
    ap.add_argument("--out", default=str(REPO / "results/curved_predictor_speed_phase"))
    args = ap.parse_args()

    from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p, wilson

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

    print(f"[cphase] {len(args.levels)} speed levels x {len(ARMS)} predictors x n={args.n} "
          f"= {len(jobs)} jobs on {args.workers} workers")
    t0 = time.perf_counter()
    with Pool(processes=args.workers) as pool:
        chunk_results = pool.map(_run_cell_chunk, jobs)
    dt = time.perf_counter() - t0
    print(f"[cphase] sweep done in {dt:.0f}s")

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
        cv, ct = cells[BASELINE_ARM], cells["constant_turn"]
        cv_succ = sum(cv[s] == "success" for s in seeds)
        ct_succ = sum(ct[s] == "success" for s in seeds)
        b = sum(cv[s] == "success" and ct[s] != "success" for s in seeds)  # turn lost
        c = sum(cv[s] != "success" and ct[s] == "success" for s in seeds)  # turn won
        p = mcnemar_exact_p(b, c)
        _, cv_lo, cv_hi = wilson(cv_succ, n)
        _, ct_lo, ct_hi = wilson(ct_succ, n)
        rows.append({
            "level": level, "n": n,
            "cv_success": cv_succ / n, "ct_success": ct_succ / n,
            "cv_success_ci": [cv_lo, cv_hi], "ct_success_ci": [ct_lo, ct_hi],
            "success_mcnemar": {"b": b, "c": c, "p": p},
        })
        raw.append({"level": level, "outcomes": {a: cells[a] for a in ARMS}})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"n_per_cell": args.n, "levels": args.levels, "arms": list(ARMS),
              "start_jitter": START_JITTER, "turn_rate": TURN_RATE,
              "drone_max_speed": base["planner"]["max_speed"],
              "seconds": dt, "rows": rows}
    (out_dir / "phase.json").write_text(json.dumps(result, indent=2))
    (out_dir / "phase_raw.json").write_text(json.dumps(raw, indent=2))

    _print_table(rows, base["planner"]["max_speed"])
    _plot(rows, base["planner"]["max_speed"], out_dir / "phase.png")
    print(f"[cphase] wrote {out_dir/'phase.json'}, phase_raw.json and {out_dir/'phase.png'}")
    return 0


def _print_table(rows: list[dict], max_speed: float) -> None:
    print()
    print(f"  drone max_speed = {max_speed:g}; both arms predict, only the model differs")
    print(f"{'hunter_v':>8} {'n':>4} | {'const_vel':>9} {'const_turn':>10} {'dSucc':>6} "
          f"{'turn won/lost':>13} {'p':>7}")
    print("-" * 66)
    for r in rows:
        m = r["success_mcnemar"]
        d = (r["ct_success"] - r["cv_success"]) * 100
        print(f"{r['level']:>8g} {r['n']:>4} | "
              f"{r['cv_success']*100:>8.1f}% {r['ct_success']*100:>9.1f}% {d:>+5.1f} "
              f"{m['c']:>4}/{m['b']:<8} {m['p']:>7.4f}")
    print("\n  dSucc = constant_turn minus constant_velocity success (pp).")
    print("  turn won/lost = paired seeds constant_turn escaped where const_vel was caught (c)")
    print("  / caught where const_vel escaped (b). p = exact McNemar.")


def _plot(rows: list[dict], max_speed: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["level"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for key, label, color in [("cv", "constant_velocity forecast", "#777"),
                              ("ct", "constant_turn forecast", "#16a085")]:
        ys = [r[f"{key}_success"] * 100 for r in rows]
        lo = [r[f"{key}_success_ci"][0] * 100 for r in rows]
        hi = [r[f"{key}_success_ci"][1] * 100 for r in rows]
        ax1.plot(xs, ys, "o-", color=color, label=label)
        ax1.fill_between(xs, lo, hi, color=color, alpha=0.15)
    ax1.axvline(max_speed, color="k", ls="--", lw=0.8)
    ax1.set_xlabel("hunter speed (escapable band; >= drone max_speed = unwinnable)")
    ax1.set_ylabel("evasion success rate (%)")
    ax1.set_title("Does modelling the hunter's turn improve evasion?")
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
    ax2.set_xlabel("hunter speed")
    ax2.set_ylabel("net paired escapes for constant_turn (c - b)")
    ax2.set_title("Paired evasion gain (green = McNemar p<0.05)")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("Constant-turn vs constant-velocity forecast of a curving hunter "
                 "(pursuit-evasion)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    raise SystemExit(main())
