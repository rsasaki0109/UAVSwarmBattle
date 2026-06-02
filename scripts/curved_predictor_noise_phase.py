#!/usr/bin/env python3
"""Phase sweep: does the constant_turn evasion win survive NOISY perception, and
does its `smoothing` knob earn its keep?

`scripts/curved_predictor_speed_phase.py` showed that under a *perfect* sensor,
modelling the hunter's turn (constant_turn) beats a straight-line forecast right
at the escapability cliff (hunter speed 2.85: evasion 18% -> 63%). But that win
rests on a clean turn-rate estimate, and constant_turn reads ω from the rotation
of the obstacle's *velocity* between calls — exactly the channel `noisy_tracker`
corrupts. An offline check (curving target, ω=0.6, dt=0.1) shows velocity noise
wrecks the *default* forecast fast:

    1 s-ahead forecast error (m), mean
    vel_noise_std        0.0    0.1    0.2    0.3    0.5
    constant_velocity   0.642  0.650  0.675  0.719  0.865   (robust, degrades slowly)
    ct smoothing=1.0    0.082  0.623  1.161  1.583  2.069   (best clean, WORST noisy)
    ct smoothing=0.15   0.107  0.216  0.376  0.542  0.878   (holds the lead to ~0.3)

So the shipped default (smoothing=1.0, "trust the latest ω") is a trap once the
velocity is noisy: a single noisy velocity flips the estimated turn and flings
the arc. Low smoothing (EMA over many samples) tames it. This sweep closes the
loop on the *outcome*: at the fixed escapability cliff (hunter speed 2.85) it
sweeps `sensor.velocity_noise_std` and compares three arms, all with prediction
ON, pairing by seed:

    constant_velocity  — the robust straight-line baseline
    ct_default         — constant_turn, smoothing=1.0 (as shipped)
    ct_tuned           — constant_turn, smoothing=0.15 (noise-robust)

Position is reported at ground truth (delay=0, position_noise=0) so the only
corruption is the velocity channel the turn-rate estimate depends on — this
isolates the predictor effect rather than mixing in avoidance/association noise.

Output: ``results/curved_predictor_noise_phase/phase.json`` (+ ``phase_raw.json``)
and a two-panel ``phase.png``.

Run (defaults: 5 noise levels x 3 arms x n=60, ~18 min on 16 workers):
    python scripts/curved_predictor_noise_phase.py
    python scripts/curved_predictor_noise_phase.py --n 40 --workers 12
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
DEFAULT_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5]  # sensor.velocity_noise_std
HUNTER_SPEED = 2.85      # the escapability cliff (drone max_speed = 3.0)
SEED_BASE = 1000
START_JITTER = 3.0
TURN_RATE = 1.2
REPLAN_DT = 0.1          # = planner.replan_period; constant_turn needs this for ω
SIM_DT = 0.05            # = simulator.dt; sizes the tracker delay buffer
TUNED_SMOOTHING = 0.15

# arm -> planner.predictor config (all arms predict; only the model/knob differs)
ARMS = ("constant_velocity", "ct_default", "ct_tuned")
ARM_PREDICTOR = {
    "constant_velocity": {"type": "constant_velocity"},
    "ct_default": {"type": "constant_turn", "dt": REPLAN_DT},
    "ct_tuned": {"type": "constant_turn", "dt": REPLAN_DT, "smoothing": TUNED_SMOOTHING},
}
BASELINE_ARM = "constant_velocity"
CT_ARMS = ("ct_default", "ct_tuned")


def _base_config() -> dict:
    from uav_nav_lab.config import ExperimentConfig

    return ExperimentConfig.from_yaml(EXAMPLE).to_dict()


def _cell_config(base: dict, arm: str, noise: float) -> dict:
    cfg = copy.deepcopy(base)
    cfg["planner"]["use_prediction"] = True
    cfg["planner"]["predictor"] = copy.deepcopy(ARM_PREDICTOR[arm])
    # swap in the noisy tracker; isolate the velocity channel
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
    out = Path(tempfile.mkdtemp(prefix="cnphase_"))
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
    ap.add_argument("--n", type=int, default=60, help="episodes per (noise, arm) cell")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--levels", type=float, nargs="+", default=DEFAULT_LEVELS,
                    help="sensor.velocity_noise_std levels to sweep")
    ap.add_argument("--out", default=str(REPO / "results/curved_predictor_noise_phase"))
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

    print(f"[cnphase] {len(args.levels)} noise levels x {len(ARMS)} arms x n={args.n} "
          f"= {len(jobs)} jobs on {args.workers} workers (hunter speed {HUNTER_SPEED})")
    t0 = time.perf_counter()
    with Pool(processes=args.workers) as pool:
        chunk_results = pool.map(_run_cell_chunk, jobs)
    dt = time.perf_counter() - t0
    print(f"[cnphase] sweep done in {dt:.0f}s")

    by_cell: dict[float, dict[str, dict[int, str]]] = {}
    for r in chunk_results:
        cell = by_cell.setdefault(r["level"], {a: {} for a in ARMS})
        for ep in r["episodes"]:
            cell[r["arm"]][ep["seed"]] = ep["outcome"]

    rows, raw = [], []
    for level in sorted(by_cell):
        cells = by_cell[level]
        # pair on seeds present in the baseline AND every CT arm
        seeds = set(cells[BASELINE_ARM])
        for a in CT_ARMS:
            seeds &= set(cells[a])
        seeds = sorted(seeds)
        n = len(seeds)
        cv = cells[BASELINE_ARM]
        cv_succ = sum(cv[s] == "success" for s in seeds)
        _, cv_lo, cv_hi = wilson(cv_succ, n)
        row = {"level": level, "n": n,
               "cv_success": cv_succ / n if n else 0.0,
               "cv_success_ci": [cv_lo, cv_hi], "arms": {}}
        for a in CT_ARMS:
            ct = cells[a]
            ct_succ = sum(ct[s] == "success" for s in seeds)
            b = sum(cv[s] == "success" and ct[s] != "success" for s in seeds)  # CT lost
            c = sum(cv[s] != "success" and ct[s] == "success" for s in seeds)  # CT won
            p = mcnemar_exact_p(b, c)
            _, lo, hi = wilson(ct_succ, n)
            row["arms"][a] = {"success": ct_succ / n if n else 0.0,
                              "success_ci": [lo, hi],
                              "mcnemar": {"b": b, "c": c, "p": p}}
        rows.append(row)
        raw.append({"level": level, "outcomes": {a: cells[a] for a in ARMS}})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"n_per_cell": args.n, "levels": args.levels, "arms": list(ARMS),
              "hunter_speed": HUNTER_SPEED, "start_jitter": START_JITTER,
              "turn_rate": TURN_RATE, "tuned_smoothing": TUNED_SMOOTHING,
              "drone_max_speed": base["planner"]["max_speed"],
              "seconds": dt, "rows": rows}
    (out_dir / "phase.json").write_text(json.dumps(result, indent=2))
    (out_dir / "phase_raw.json").write_text(json.dumps(raw, indent=2))

    _print_table(rows)
    _plot(rows, out_dir / "phase.png")
    print(f"[cnphase] wrote {out_dir/'phase.json'}, phase_raw.json and {out_dir/'phase.png'}")
    return 0


def _print_table(rows: list[dict]) -> None:
    print()
    print(f"  hunter speed = {HUNTER_SPEED} (escapability cliff); all arms predict, "
          f"only the forecast model/knob differs")
    print(f"{'velnoise':>8} {'n':>4} | {'const_vel':>9} | "
          f"{'ct_default':>10} {'won/lost':>9} {'p':>7} | "
          f"{'ct_tuned':>9} {'won/lost':>9} {'p':>7}")
    print("-" * 84)
    for r in rows:
        cv = r["cv_success"] * 100
        dflt = r["arms"]["ct_default"]
        tuned = r["arms"]["ct_tuned"]
        md, mt = dflt["mcnemar"], tuned["mcnemar"]
        print(f"{r['level']:>8g} {r['n']:>4} | {cv:>8.1f}% | "
              f"{dflt['success']*100:>9.1f}% {md['c']:>3}/{md['b']:<5} {md['p']:>7.4f} | "
              f"{tuned['success']*100:>8.1f}% {mt['c']:>3}/{mt['b']:<5} {mt['p']:>7.4f}")
    print("\n  won/lost = paired seeds where the CT arm escaped but const_vel was caught (c)")
    print("  / caught but const_vel escaped (b). p = exact McNemar vs const_velocity.")


def _plot(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["level"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    series = [("cv", "constant_velocity", "#777", lambda r: r),
              ("ct_default", "constant_turn (smoothing=1.0, shipped)", "#e67e22",
               lambda r: r["arms"]["ct_default"]),
              ("ct_tuned", f"constant_turn (smoothing={TUNED_SMOOTHING})", "#16a085",
               lambda r: r["arms"]["ct_tuned"])]
    for key, label, color, sel in series:
        if key == "cv":
            ys = [r["cv_success"] * 100 for r in rows]
            lo = [r["cv_success_ci"][0] * 100 for r in rows]
            hi = [r["cv_success_ci"][1] * 100 for r in rows]
        else:
            ys = [sel(r)["success"] * 100 for r in rows]
            lo = [sel(r)["success_ci"][0] * 100 for r in rows]
            hi = [sel(r)["success_ci"][1] * 100 for r in rows]
        ax1.plot(xs, ys, "o-", color=color, label=label)
        ax1.fill_between(xs, lo, hi, color=color, alpha=0.15)
    ax1.set_xlabel("sensor velocity_noise_std (m/s)")
    ax1.set_ylabel("evasion success rate (%)")
    ax1.set_title(f"Does the turn model survive noisy velocity? (hunter {HUNTER_SPEED})")
    ax1.legend()
    ax1.grid(alpha=0.3)

    width = 0.38
    import numpy as np
    x = np.arange(len(rows))
    for off, a, color in [(-width / 2, "ct_default", "#e67e22"),
                          (width / 2, "ct_tuned", "#16a085")]:
        net = [r["arms"][a]["mcnemar"]["c"] - r["arms"][a]["mcnemar"]["b"] for r in rows]
        bars = ax2.bar(x + off, net, width, color=color, label=a)
        for r, bar in zip(rows, bars):
            bar.set_alpha(1.0 if r["arms"][a]["mcnemar"]["p"] < 0.05 else 0.4)
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{r['arms'][a]['mcnemar']['p']:.2f}", ha="center",
                     va="bottom" if bar.get_height() >= 0 else "top", fontsize=7)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{r['level']:g}" for r in rows])
    ax2.set_xlabel("sensor velocity_noise_std (m/s)")
    ax2.set_ylabel("net paired escapes vs const_vel (c - b)")
    ax2.set_title("Paired evasion gain (solid = McNemar p<0.05)")
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("constant_turn under noisy velocity perception: the shipped "
                 "default is a trap, smoothing recovers it")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    raise SystemExit(main())
