#!/usr/bin/env python3
"""Phase sweep: when does anticipating the hunter (use_prediction) help evasion?

The pursuit-evasion example (`examples/exp_pursuit_evasion_mppi.yaml`) pits a
single drone crossing to its goal against an `intercept` hunter that leads the
drone's motion (proportional navigation). The planner knob under test is
`use_prediction`: with it on, the MPPI cost forecasts the hunter's future
positions and the drone commits to a decisive evasive juke; with it off, the
drone only reacts to where the hunter is *now*.

Two traps had to be cleared before this is a real statistical test:

  1. As shipped the scenario is DETERMINISTIC — single drone, perfect sensor,
     fixed geometry — so the episode seed varies nothing and every outcome is
     all-or-nothing (0 % or 100 % across a whole cell). McNemar on 24 identical
     replays reports p=2^-24, which is statistical theatre: it is really n=1.
     Fix: `start_jitter` on the dynamic obstacles (a knob already shipped with
     the noisy_tracker work) gives each seed a genuinely different chase
     geometry, so success rates become graded and the paired test is honest.

  2. The win is gated by ESCAPABILITY. The drone's max_speed is 3.0; the hunter
     speed is the swept axis. Below it the drone *can* outrun the hunter and
     prediction converts that speed margin into an actual escape (reactive
     squanders it by reacting late); at/above it the chase is unwinnable and
     prediction cannot help (and may even cost a little). So the value window is
     "hunter slower than evader" — outside it the feature is null by physics,
     not by failure.

This sweep holds jitter + turn_rate fixed and sweeps the hunter `speed` from
slow (easy escape) to the drone's own max_speed (no escape), running both
prediction arms paired by seed. Pairing is valid because the runner seeds the
sim, the sensor, and the per-episode obstacle jitter from the same episode seed,
so for a given seed both arms see the identical chase; only the forecast differs.

Output: ``results/pursuit_prediction_speed_phase/phase.json`` (+ per-seed
``phase_raw.json``) and a two-panel ``phase.png``.

Run (defaults: 6 speed levels x 2 arms x n=60, ~12 min on 16 workers):
    python scripts/pursuit_prediction_speed_phase.py
    python scripts/pursuit_prediction_speed_phase.py --n 40 --workers 12
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
DEFAULT_LEVELS = [2.0, 2.4, 2.7, 2.85, 2.9, 3.0]  # hunter speed; drone max_speed = 3.0
SEED_BASE = 1000
START_JITTER = 3.0       # per-seed spawn variance on every dynamic obstacle
TURN_RATE = 1.2
ARMS = ("reactive", "predict")
ARM_USE_PRED = {"reactive": False, "predict": True}
BASELINE_ARM = "reactive"


def _base_config() -> dict:
    from uav_nav_lab.config import ExperimentConfig

    return ExperimentConfig.from_yaml(EXAMPLE).to_dict()


def _cell_config(base: dict, arm: str, speed: float) -> dict:
    cfg = copy.deepcopy(base)
    cfg["planner"]["use_prediction"] = ARM_USE_PRED[arm]
    for d in cfg["scenario"]["dynamic_obstacles"]:
        d["start_jitter"] = START_JITTER
        if d.get("policy") == "intercept":
            d["speed"] = float(speed)
            d["turn_rate"] = TURN_RATE
    cfg["name"] = f"{arm}_s{speed:g}"
    cfg.pop("output", None)
    return cfg


def _run_cell_chunk(job: dict) -> dict:
    from uav_nav_lab.config import ExperimentConfig
    from uav_nav_lab.runner.experiment import run_experiment

    cfg = ExperimentConfig.from_dict(job["config"])
    cfg.seed = job["seed_start"]
    cfg.num_episodes = job["count"]
    out = Path(tempfile.mkdtemp(prefix="pphase_"))
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
    ap.add_argument("--n", type=int, default=60, help="episodes per (level, arm) cell")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--levels", type=float, nargs="+", default=DEFAULT_LEVELS,
                    help="hunter speed levels to sweep")
    ap.add_argument("--out", default=str(REPO / "results/pursuit_prediction_speed_phase"))
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

    print(f"[pphase] {len(args.levels)} speed levels x {len(ARMS)} arms x n={args.n} "
          f"= {len(jobs)} jobs on {args.workers} workers")
    t0 = time.perf_counter()
    with Pool(processes=args.workers) as pool:
        chunk_results = pool.map(_run_cell_chunk, jobs)
    dt = time.perf_counter() - t0
    print(f"[pphase] sweep done in {dt:.0f}s")

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
        re_, pr = cells[BASELINE_ARM], cells["predict"]
        re_succ = sum(re_[s] == "success" for s in seeds)
        pr_succ = sum(pr[s] == "success" for s in seeds)
        b = sum(re_[s] == "success" and pr[s] != "success" for s in seeds)  # predict lost
        c = sum(re_[s] != "success" and pr[s] == "success" for s in seeds)  # predict won
        p = mcnemar_exact_p(b, c)
        _, re_lo, re_hi = wilson(re_succ, n)
        _, pr_lo, pr_hi = wilson(pr_succ, n)
        rows.append({
            "level": level, "n": n,
            "reactive_success": re_succ / n, "predict_success": pr_succ / n,
            "reactive_success_ci": [re_lo, re_hi], "predict_success_ci": [pr_lo, pr_hi],
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
    print(f"[pphase] wrote {out_dir/'phase.json'}, phase_raw.json and {out_dir/'phase.png'}")
    return 0


def _print_table(rows: list[dict], max_speed: float) -> None:
    print()
    print(f"  drone max_speed = {max_speed:g} (escape impossible once hunter reaches it)")
    print(f"{'hunter_v':>8} {'n':>4} | {'reactive':>9} {'predict':>9} {'dSucc':>6} "
          f"{'pred won/lost':>13} {'p':>7}")
    print("-" * 64)
    for r in rows:
        m = r["success_mcnemar"]
        d = (r["predict_success"] - r["reactive_success"]) * 100
        print(f"{r['level']:>8g} {r['n']:>4} | "
              f"{r['reactive_success']*100:>8.1f}% {r['predict_success']*100:>8.1f}% {d:>+5.1f} "
              f"{m['c']:>4}/{m['b']:<8} {m['p']:>7.4f}")
    print("\n  dSucc = predict minus reactive success (pp).")
    print("  pred won/lost = paired seeds predict escaped where reactive was caught (c)")
    print("  / caught where reactive escaped (b). p = exact McNemar.")


def _plot(rows: list[dict], max_speed: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["level"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for key, label, color in [("reactive", "reactive (no prediction)", "#777"),
                              ("predict", "predict (anticipate hunter)", "#c0392b")]:
        ys = [r[f"{key}_success"] * 100 for r in rows]
        lo = [r[f"{key}_success_ci"][0] * 100 for r in rows]
        hi = [r[f"{key}_success_ci"][1] * 100 for r in rows]
        ax1.plot(xs, ys, "o-", color=color, label=label)
        ax1.fill_between(xs, lo, hi, color=color, alpha=0.15)
    ax1.axvline(max_speed, color="k", ls="--", lw=0.8)
    ax1.text(max_speed, 50, " drone max_speed\n (escape ceiling)", fontsize=8, va="center")
    ax1.set_xlabel("hunter speed (slow = easy escape → fast = unwinnable)")
    ax1.set_ylabel("evasion success rate (%)")
    ax1.set_title("Evasion success vs hunter speed")
    ax1.legend()
    ax1.grid(alpha=0.3)

    net = [r["success_mcnemar"]["c"] - r["success_mcnemar"]["b"] for r in rows]
    colors = ["#27ae60" if r["success_mcnemar"]["p"] < 0.05 else "#95a5a6" for r in rows]
    bars = ax2.bar([f"{x:g}" for x in xs], net, color=colors)
    for r, bar in zip(rows, bars):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"p={r['success_mcnemar']['p']:.3f}", ha="center",
                 va="bottom" if bar.get_height() >= 0 else "top", fontsize=8)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("hunter speed")
    ax2.set_ylabel("net paired escapes for predict (c - b)")
    ax2.set_title("Paired evasion gain (green = McNemar p<0.05)")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("When does anticipating the hunter help? Prediction's edge is "
                 "gated by escapability (pursuit-evasion)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    raise SystemExit(main())
