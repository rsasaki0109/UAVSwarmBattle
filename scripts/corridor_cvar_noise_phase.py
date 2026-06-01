#!/usr/bin/env python3
"""Phase sweep: when does CVaR-MPPI's collision reduction earn its keep?

The merged ``noisy_tracker`` sensor turns *forecast uncertainty* into a knob:
each moving obstacle is reported with a fixed delay plus Gaussian
position/velocity noise, so a constant-velocity forecast genuinely errs and a
risk-aware planner that hedges the bad tail (CVaR-MPPI) finally has something to
win. The single canonical scenario (n=200) showed CVaR cuts collisions ~30%
(15.0% -> 10.5%) but the success gain was not significant. That is one point on
a curve. This script sweeps the *actual* perception-noise level while holding
the planner's *assumed* uncertainty (``pred_noise_std=1.5``) fixed, and measures
the paired MPPI -> CVaR difference at each level.

The hypothesis is a mismatch story:
  - low actual noise  -> CVaR over-hedges a forecast that is already good; the
    risk term only adds timidity, so it should NOT help (matches the earlier
    "no benefit under perfect sensing" finding).
  - actual ~= assumed -> CVaR's hedge is calibrated; collision reduction peaks.
  - high actual noise -> everything degrades and a fixed hedge under-shoots; the
    edge should shrink again.

Pairing is valid because the runner seeds the simulator and the sensor from the
same per-episode seed, so for a given seed both planners see the *same* obstacle
trajectory and the *same* sensor-noise realization. We pair by seed and run an
exact McNemar test on both the success and the (no-)collision outcomes.

Output: ``results/corridor_cvar_noise_phase/phase.json`` plus a two-panel
``phase.png`` (collision rate vs noise per planner with Wilson bands; paired
success/collision deltas with significance markers).

Run (defaults: 5 levels x 2 planners x n=80, ~8 min on 16 workers):
    python scripts/corridor_cvar_noise_phase.py
    python scripts/corridor_cvar_noise_phase.py --n 40 --workers 12
"""

from __future__ import annotations

# Single-thread each numpy worker so 16 forked processes don't oversubscribe the
# 20 cores. Must precede any (transitive) numpy import.
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
CANON_POS_NOISE = 1.2   # examples/exp_corridor_tracker_cvar_mppi.yaml
CANON_VEL_NOISE = 1.5
DEFAULT_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0]
SEED_BASE = 7           # matches the example so seed 7.. is reproducible
CVAR_ONLY_KEYS = ("n_scenarios", "risk_alpha", "pred_noise_std")

# Three arms decompose the CVaR edge into its two ingredients:
#   mppi       — single deterministic forecast: no spread, no risk aversion.
#   cvar_mean  — 12 sampled futures (pred_noise_std spread) averaged
#                (risk_alpha=1.0 recovers the expected case, see cvar.py): this
#                ARM HAS THE SPREAD BUT NO RISK AVERSION. The honest control.
#   cvar       — the same spread, but the worst-10% tail (risk_alpha=0.1).
# If cvar_mean already captures the win, the lever is "acknowledge forecast
# uncertainty at all" (margin), not the risk-averse tail. If cvar beats
# cvar_mean, the tail itself earns its keep.
ARMS = ("mppi", "cvar_mean", "cvar")
BASELINE_ARM = "mppi"


def _base_config() -> dict:
    from uav_nav_lab.config import ExperimentConfig

    cfg = ExperimentConfig.from_yaml(REPO / "examples/exp_corridor_tracker_cvar_mppi.yaml")
    return cfg.to_dict()


def _cell_config(base: dict, arm: str, scale: float) -> dict:
    """A scenario config with noise scaled and the planner arm selected.

    All arms keep the identical scenario/sensor/shared-MPPI knobs; the only
    difference at a given level is the cost aggregation (see ARMS).
    """
    cfg = copy.deepcopy(base)
    cfg["sensor"]["position_noise_std"] = round(CANON_POS_NOISE * scale, 6)
    cfg["sensor"]["velocity_noise_std"] = round(CANON_VEL_NOISE * scale, 6)
    if arm == "mppi":
        cfg["planner"]["type"] = "mppi"
        for k in CVAR_ONLY_KEYS:
            cfg["planner"].pop(k, None)
        # constant_velocity predictor is shared and harmless for plain MPPI; the
        # earlier example keeps it so the only delta stays the cost aggregation.
    elif arm == "cvar_mean":
        cfg["planner"]["type"] = "cvar_mppi"
        cfg["planner"]["risk_alpha"] = 1.0   # average all futures = expected case
    elif arm == "cvar":
        cfg["planner"]["type"] = "cvar_mppi"
        # keep the example's risk_alpha (0.1)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    cfg["name"] = f"{arm}_s{scale:g}"
    cfg.pop("output", None)
    return cfg


def _run_cell_chunk(job: dict) -> dict:
    """Worker: run ``count`` episodes from ``seed_start`` for one cell config."""
    from uav_nav_lab.config import ExperimentConfig
    from uav_nav_lab.runner.experiment import run_experiment

    cfg = ExperimentConfig.from_dict(job["config"])
    cfg.seed = job["seed_start"]
    cfg.num_episodes = job["count"]
    out = Path(tempfile.mkdtemp(prefix="phase_"))
    run_experiment(cfg, out)
    episodes = []
    for p in sorted(out.glob("episode_*.json")):
        d = json.loads(p.read_text())
        episodes.append({"seed": d["meta"]["seed"], "outcome": d["outcome"]})
    # clean the temp run dir; we only keep the outcome tuples
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
    ap.add_argument("--n", type=int, default=80, help="episodes per (level, planner) cell")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=10, help="episodes per parallel job")
    ap.add_argument(
        "--levels", type=float, nargs="+", default=DEFAULT_LEVELS,
        help="noise scale factors applied to canonical pos/vel noise",
    )
    ap.add_argument(
        "--out", default=str(REPO / "results/corridor_cvar_noise_phase"),
    )
    args = ap.parse_args()

    from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p

    base = _base_config()

    # Build the flat job list: each cell split into chunks for parallelism.
    jobs: list[dict] = []
    for level in args.levels:
        for arm in ARMS:
            cfg = _cell_config(base, arm, level)
            seed = SEED_BASE
            remaining = args.n
            while remaining > 0:
                count = min(args.chunk, remaining)
                jobs.append({
                    "level": level, "arm": arm, "config": cfg,
                    "seed_start": seed, "count": count,
                })
                seed += count
                remaining -= count

    print(f"[phase] {len(args.levels)} levels x {len(ARMS)} arms x n={args.n} "
          f"= {len(jobs)} jobs on {args.workers} workers")
    t0 = time.perf_counter()
    with Pool(processes=args.workers) as pool:
        chunk_results = pool.map(_run_cell_chunk, jobs)
    dt = time.perf_counter() - t0
    print(f"[phase] sweep done in {dt:.0f}s")

    # Gather episodes per (level, arm): {level: {arm: {seed: outcome}}}
    by_cell: dict[float, dict[str, dict[int, str]]] = {}
    for r in chunk_results:
        cell = by_cell.setdefault(r["level"], {a: {} for a in ARMS})
        for ep in r["episodes"]:
            cell[r["arm"]][ep["seed"]] = ep["outcome"]

    def _paired(a: dict[int, str], b: dict[int, str], event: str) -> dict:
        """McNemar of arm b vs arm a on `event` (event=outcome string).

        Returns counts where `gain` = episodes where b achieved the GOOD side
        and a did not. For 'success' good=success; for 'collision' good=NOT
        colliding, so we report collisions avoided by b.
        """
        seeds = sorted(set(a) & set(b))
        if event == "success":
            ba = sum(a[s] == "success" and b[s] != "success" for s in seeds)  # b lost
            ca = sum(a[s] != "success" and b[s] == "success" for s in seeds)  # b won
        else:  # collision: good outcome is NOT colliding
            ba = sum(a[s] != "collision" and b[s] == "collision" for s in seeds)  # b newly collided
            ca = sum(a[s] == "collision" and b[s] != "collision" for s in seeds)  # b avoided
        return {"b": ba, "c": ca, "p": mcnemar_exact_p(ba, ca)}

    rows = []
    raw = []
    for level in sorted(by_cell):
        cells = by_cell[level]
        seeds = sorted(set.intersection(*[set(cells[a]) for a in ARMS]))
        n = len(seeds)
        row: dict = {"level": level, "n": n}
        for a in ARMS:
            succ = sum(cells[a][s] == "success" for s in seeds)
            coll = sum(cells[a][s] == "collision" for s in seeds)
            tout = sum(cells[a][s] == "timeout" for s in seeds)
            _, s_lo, s_hi = _wilson(succ, n)
            _, c_lo, c_hi = _wilson(coll, n)
            row[a] = {
                "success": succ / n, "success_ci": [s_lo, s_hi],
                "collision": coll / n, "collision_ci": [c_lo, c_hi],
                "timeout": tout / n,
            }
        # Paired tests vs the baseline (mppi) and the decomposition (cvar vs mean)
        base_cells = cells[BASELINE_ARM]
        row["vs_mppi"] = {
            a: {
                "success": _paired(base_cells, cells[a], "success"),
                "collision": _paired(base_cells, cells[a], "collision"),
            }
            for a in ARMS if a != BASELINE_ARM
        }
        # The decisive decomposition: does the risk tail beat the spread alone?
        row["cvar_vs_mean"] = {
            "success": _paired(cells["cvar_mean"], cells["cvar"], "success"),
            "collision": _paired(cells["cvar_mean"], cells["cvar"], "collision"),
        }
        rows.append(row)
        raw.append({"level": level, "outcomes": {a: cells[a] for a in ARMS}})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"n_per_cell": args.n, "levels": args.levels, "arms": list(ARMS),
              "seconds": dt, "rows": rows}
    (out_dir / "phase.json").write_text(json.dumps(result, indent=2))
    (out_dir / "phase_raw.json").write_text(json.dumps(raw, indent=2))

    _print_table(rows)
    _plot(rows, out_dir / "phase.png")
    print(f"[phase] wrote {out_dir/'phase.json'}, phase_raw.json and {out_dir/'phase.png'}")
    return 0


_ARM_LABEL = {
    "mppi": "MPPI (no spread)",
    "cvar_mean": "CVaR-mean (spread, risk-neutral)",
    "cvar": "CVaR (spread + worst-10% tail)",
}


def _print_table(rows: list[dict]) -> None:
    print()
    print("Success / collision rate per arm (paired, n per cell):")
    print(f"{'scale':>6} {'n':>4} | "
          f"{'MPPI':>13} | {'CVaR-mean':>13} | {'CVaR':>13}   (succ% / coll%)")
    print("-" * 74)
    for r in rows:
        def cell(a: str) -> str:
            return f"{r[a]['success']*100:5.1f}/{r[a]['collision']*100:<5.1f}"
        print(f"{r['level']:>6g} {r['n']:>4} | "
              f"{cell('mppi'):>13} | {cell('cvar_mean'):>13} | {cell('cvar'):>13}")

    print("\nPaired collisions avoided vs MPPI baseline (c=avoided / b=newly caused, exact-McNemar p):")
    print(f"{'scale':>6} | {'CVaR-mean vs MPPI':>22} | {'CVaR vs MPPI':>22} | {'CVaR vs CVaR-mean':>22}")
    print("-" * 84)
    for r in rows:
        cm = r["vs_mppi"]["cvar_mean"]["collision"]
        cv = r["vs_mppi"]["cvar"]["collision"]
        dec = r["cvar_vs_mean"]["collision"]
        def fmt(d: dict) -> str:
            return f"{d['c']:>3}/{d['b']:<3} p={d['p']:.3f}"
        print(f"{r['level']:>6g} | {fmt(cm):>22} | {fmt(cv):>22} | {fmt(dec):>22}")
    print("\n  CVaR-mean vs MPPI isolates the SPREAD effect (acknowledging forecast")
    print("  uncertainty). CVaR vs CVaR-mean isolates the RISK-AVERSE TAIL effect.")


def _plot(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["level"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = {"mppi": "#777", "cvar_mean": "#2980b9", "cvar": "#c0392b"}
    # Panel 1: collision rate vs noise, all 3 arms, Wilson bands
    for a in ARMS:
        ys = [r[a]["collision"] * 100 for r in rows]
        lo = [r[a]["collision_ci"][0] * 100 for r in rows]
        hi = [r[a]["collision_ci"][1] * 100 for r in rows]
        ax1.plot(xs, ys, "o-", color=colors[a], label=_ARM_LABEL[a])
        ax1.fill_between(xs, lo, hi, color=colors[a], alpha=0.12)
    ax1.axvline(1.0, ls=":", color="k", alpha=0.5)
    ax1.text(1.0, ax1.get_ylim()[1] * 0.97, " assumed=actual\n (pred_noise_std=1.5)",
             fontsize=8, va="top")
    ax1.set_xlabel("actual sensor-noise scale (xcanonical pos/vel)")
    ax1.set_ylabel("collision rate (%)")
    ax1.set_title("Collision rate vs perception noise")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Panel 2: decomposition — net collisions avoided vs MPPI, split into the
    # spread contribution (CVaR-mean) and the extra tail contribution (CVaR).
    width = 0.38
    import numpy as np
    x = np.arange(len(rows))
    spread = [r["vs_mppi"]["cvar_mean"]["collision"]["c"]
              - r["vs_mppi"]["cvar_mean"]["collision"]["b"] for r in rows]
    total = [r["vs_mppi"]["cvar"]["collision"]["c"]
             - r["vs_mppi"]["cvar"]["collision"]["b"] for r in rows]
    b1 = ax2.bar(x - width / 2, spread, width, color="#2980b9",
                 label="CVaR-mean vs MPPI (spread only)")
    b2 = ax2.bar(x + width / 2, total, width, color="#c0392b",
                 label="CVaR vs MPPI (spread + tail)")
    for r, bar in zip(rows, b1):
        p = r["vs_mppi"]["cvar_mean"]["collision"]["p"]
        if p < 0.05:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "*",
                     ha="center", va="bottom", fontsize=12)
    for r, bar in zip(rows, b2):
        p = r["vs_mppi"]["cvar"]["collision"]["p"]
        if p < 0.05:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "*",
                     ha="center", va="bottom", fontsize=12)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{r['level']:g}" for r in rows])
    ax2.set_xlabel("actual sensor-noise scale")
    ax2.set_ylabel("net collisions avoided vs MPPI (paired c - b)")
    ax2.set_title("Decomposition: spread vs risk-averse tail (* = McNemar p<0.05)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("What drives CVaR-MPPI's edge — the forecast spread or the risk tail? "
                 "(corridor noisy_tracker)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    raise SystemExit(main())
