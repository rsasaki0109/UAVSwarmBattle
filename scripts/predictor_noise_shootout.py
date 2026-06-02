#!/usr/bin/env python3
"""Three forecasting philosophies vs a curving hunter under noisy velocity.

`curved_predictor_noise_phase.py` (#60) showed that modelling the hunter's turn
(`constant_turn`) beats a straight-line forecast at the escapability cliff, but
its win decays once the *velocity* channel it reads ω from gets noisy. That
raises the obvious counter: instead of modelling the curve from the noisy
velocity, what if we *ignore* the velocity field entirely and estimate motion
from the (clean) position stream with a filter? That is exactly what
`kalman_velocity` does — a constant-velocity Kalman filter that observes
position only. So this is a three-way shootout of distinct philosophies, all
with prediction ON, only `predictor.type` differing:

    constant_velocity  — trust the reported velocity, extrapolate linearly
    constant_turn      — model the curve, reading ω from the velocity field
    kalman_velocity    — ignore the velocity field; filter velocity from positions

Offline (steady-ω surrogate, 1 s forecast error, mean) sets up the tension:

    velocity_noise_std    0.0    0.1    0.2    0.3    0.5
    constant_velocity    0.64   0.65   0.68   0.72   0.87
    constant_turn        0.08   0.63   1.17   1.59   2.07   (best clean, collapses)
    kalman_velocity      1.18   1.19   1.19   1.19   1.20   (flat: velocity-noise immune)

By that metric constant_turn wins clean but kalman overtakes it past ≈0.2 —
its position-only estimate is *immune* to velocity noise (flat ~1.2), at the
cost of a high curve-blind floor. But #60 showed the steady-ω offline metric
INVERTS in the closed loop (the real hunter maneuvers), so whether that
crossover survives is an open question — that is what this sweep settles.

Same rig as #60: escapability cliff (hunter speed 2.85), `noisy_tracker`
corrupting only the velocity channel (delay 0, position noise 0), paired by
seed, n=60. kalman uses dt = replan_period like constant_turn.

Output: ``results/predictor_noise_shootout/phase.json`` (+ ``phase_raw.json``)
and a two-panel ``phase.png``.

Run (defaults: 5 noise levels x 3 arms x n=60, ~18 min on 16 workers):
    python scripts/predictor_noise_shootout.py
    python scripts/predictor_noise_shootout.py --n 40 --workers 12
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
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXAMPLE = REPO / "examples/exp_pursuit_evasion_mppi.yaml"
DEFAULT_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5]  # sensor.velocity_noise_std
HUNTER_SPEED = 2.85      # the escapability cliff (drone max_speed = 3.0)
SEED_BASE = 1000
START_JITTER = 3.0
TURN_RATE = 1.2
REPLAN_DT = 0.1          # = planner.replan_period
SIM_DT = 0.05            # = simulator.dt; sizes the tracker delay buffer

ARMS = ("constant_velocity", "constant_turn", "kalman_velocity")
ARM_PREDICTOR = {
    "constant_velocity": {"type": "constant_velocity"},
    "constant_turn": {"type": "constant_turn", "dt": REPLAN_DT},
    "kalman_velocity": {"type": "kalman_velocity", "dt": REPLAN_DT},
}
BASELINE_ARM = "constant_velocity"


def _base_config() -> dict:
    from uav_nav_lab.config import ExperimentConfig

    return ExperimentConfig.from_yaml(EXAMPLE).to_dict()


def _cell_config(base: dict, arm: str, noise: float) -> dict:
    cfg = copy.deepcopy(base)
    cfg["planner"]["use_prediction"] = True
    cfg["planner"]["predictor"] = copy.deepcopy(ARM_PREDICTOR[arm])
    cfg["sensor"] = {
        "type": "noisy_tracker",
        "delay": 0.0,
        "dt": SIM_DT,
        "position_noise_std": 0.0,
        "velocity_noise_std": float(noise),
    }
    for d in cfg["scenario"]["dynamic_obstacles"]:
        d["start_jitter"] = START_JITTER
        if d.get("policy") == "intercept":
            d["speed"] = float(HUNTER_SPEED)
            d["turn_rate"] = TURN_RATE
    cfg["name"] = f"{arm}_n{noise:g}"
    cfg.pop("output", None)
    return cfg


def _run_cell_chunk(job: dict) -> dict:
    from uav_nav_lab.config import ExperimentConfig
    from uav_nav_lab.runner.experiment import run_experiment

    cfg = ExperimentConfig.from_dict(job["config"])
    cfg.seed = job["seed_start"]
    cfg.num_episodes = job["count"]
    out = Path(tempfile.mkdtemp(prefix="shootout_"))
    run_experiment(cfg, out)
    episodes = []
    for p in sorted(out.glob("episode_*.json")):
        d = json.loads(p.read_text())
        episodes.append({"seed": d["meta"]["seed"], "outcome": d["outcome"]})
    for p in out.glob("*"):
        p.unlink()
    out.rmdir()
    return {"level": job["level"], "arm": job["arm"], "episodes": episodes}


def _paired(a_map: dict, b_map: dict, seeds: list, mcnemar) -> dict:
    """McNemar of arm A vs arm B over shared seeds (c = A wins, b = B wins)."""
    c = sum(a_map[s] == "success" and b_map[s] != "success" for s in seeds)
    b = sum(a_map[s] != "success" and b_map[s] == "success" for s in seeds)
    return {"c": c, "b": b, "p": mcnemar(b, c)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60, help="episodes per (noise, arm) cell")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--levels", type=float, nargs="+", default=DEFAULT_LEVELS,
                    help="sensor.velocity_noise_std levels to sweep")
    ap.add_argument("--out", default=str(REPO / "results/predictor_noise_shootout"))
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

    print(f"[shootout] {len(args.levels)} noise levels x {len(ARMS)} arms x n={args.n} "
          f"= {len(jobs)} jobs on {args.workers} workers (hunter speed {HUNTER_SPEED})")
    t0 = time.perf_counter()
    with Pool(processes=args.workers) as pool:
        chunk_results = pool.map(_run_cell_chunk, jobs)
    dt = time.perf_counter() - t0
    print(f"[shootout] sweep done in {dt:.0f}s")

    by_cell: dict[float, dict[str, dict[int, str]]] = {}
    for r in chunk_results:
        cell = by_cell.setdefault(r["level"], {a: {} for a in ARMS})
        for ep in r["episodes"]:
            cell[r["arm"]][ep["seed"]] = ep["outcome"]

    rows, raw = [], []
    for level in sorted(by_cell):
        cells = by_cell[level]
        seeds = set(cells[ARMS[0]])
        for a in ARMS[1:]:
            seeds &= set(cells[a])
        seeds = sorted(seeds)
        n = len(seeds)
        row = {"level": level, "n": n, "success": {}, "success_ci": {},
               "vs_baseline": {}, "head_to_head": {}}
        for a in ARMS:
            succ = sum(cells[a][s] == "success" for s in seeds)
            _, lo, hi = wilson(succ, n)
            row["success"][a] = succ / n if n else 0.0
            row["success_ci"][a] = [lo, hi]
        for a in ARMS:
            if a == BASELINE_ARM:
                continue
            row["vs_baseline"][a] = _paired(cells[a], cells[BASELINE_ARM],
                                            seeds, mcnemar_exact_p)
        for a, b in combinations(ARMS, 2):
            row["head_to_head"][f"{a}__vs__{b}"] = _paired(cells[a], cells[b],
                                                          seeds, mcnemar_exact_p)
        rows.append(row)
        raw.append({"level": level, "outcomes": {a: cells[a] for a in ARMS}})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"n_per_cell": args.n, "levels": args.levels, "arms": list(ARMS),
              "hunter_speed": HUNTER_SPEED, "start_jitter": START_JITTER,
              "turn_rate": TURN_RATE, "drone_max_speed": base["planner"]["max_speed"],
              "seconds": dt, "rows": rows}
    (out_dir / "phase.json").write_text(json.dumps(result, indent=2))
    (out_dir / "phase_raw.json").write_text(json.dumps(raw, indent=2))

    _print_table(rows)
    _plot(rows, out_dir / "phase.png")
    print(f"[shootout] wrote {out_dir/'phase.json'}, phase_raw.json and {out_dir/'phase.png'}")
    return 0


def _print_table(rows: list[dict]) -> None:
    print()
    print(f"  hunter speed = {HUNTER_SPEED} (escapability cliff); all arms predict, "
          f"only the forecast model differs. Success rate (%) and McNemar vs const_vel.")
    print(f"{'velnoise':>8} {'n':>4} | {'const_vel':>9} | "
          f"{'const_turn':>10} {'p':>7} | {'kalman':>8} {'p':>7} | "
          f"{'CT vs KF p':>10}")
    print("-" * 78)
    for r in rows:
        cv = r["success"]["constant_velocity"] * 100
        ct = r["success"]["constant_turn"] * 100
        kf = r["success"]["kalman_velocity"] * 100
        pct = r["vs_baseline"]["constant_turn"]["p"]
        pkf = r["vs_baseline"]["kalman_velocity"]["p"]
        h = r["head_to_head"]["constant_turn__vs__kalman_velocity"]
        print(f"{r['level']:>8g} {r['n']:>4} | {cv:>8.1f}% | "
              f"{ct:>9.1f}% {pct:>7.4f} | {kf:>7.1f}% {pkf:>7.4f} | "
              f"{h['p']:>10.4f}")
    print("\n  CT vs KF p = head-to-head exact McNemar (constant_turn vs kalman_velocity).")


def _plot(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    xs = [r["level"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = {"constant_velocity": "#777", "constant_turn": "#16a085",
              "kalman_velocity": "#8e44ad"}
    for a in ARMS:
        ys = [r["success"][a] * 100 for r in rows]
        lo = [r["success_ci"][a][0] * 100 for r in rows]
        hi = [r["success_ci"][a][1] * 100 for r in rows]
        ax1.plot(xs, ys, "o-", color=colors[a], label=a)
        ax1.fill_between(xs, lo, hi, color=colors[a], alpha=0.15)
    ax1.set_xlabel("sensor velocity_noise_std (m/s)")
    ax1.set_ylabel("evasion success rate (%)")
    ax1.set_title(f"Curve vs filter vs linear (hunter {HUNTER_SPEED})")
    ax1.legend()
    ax1.grid(alpha=0.3)

    x = np.arange(len(rows))
    width = 0.38
    for off, a in [(-width / 2, "constant_turn"), (width / 2, "kalman_velocity")]:
        net = [r["vs_baseline"][a]["c"] - r["vs_baseline"][a]["b"] for r in rows]
        bars = ax2.bar(x + off, net, width, color=colors[a], label=f"{a} vs const_vel")
        for r, bar in zip(rows, bars):
            bar.set_alpha(1.0 if r["vs_baseline"][a]["p"] < 0.05 else 0.4)
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{r['vs_baseline'][a]['p']:.2f}", ha="center",
                     va="bottom" if bar.get_height() >= 0 else "top", fontsize=7)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{r['level']:g}" for r in rows])
    ax2.set_xlabel("sensor velocity_noise_std (m/s)")
    ax2.set_ylabel("net paired escapes vs const_vel (c - b)")
    ax2.set_title("Paired gain over const_velocity (solid = McNemar p<0.05)")
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("Predictor shootout vs a curving hunter under noisy velocity: "
                 "model the curve, filter it out, or trust it linearly")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    raise SystemExit(main())
