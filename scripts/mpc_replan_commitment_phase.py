#!/usr/bin/env python3
"""Does more-frequent replanning help or hurt? The replan_period commitment sweet-spot.

Two shipped findings ASSERT a "commitment" mechanism but never sweep the knob that
controls it:

  * The CHOMP-smoothing wash (docs/findings.md, "MPC + CHOMP smoothing ... is a wash")
    explains MPC's low per-step |Δcmd| by noting it "commits to one velocity for the
    whole replan_period (0.2 s = 4 control steps) so the controller has nothing to
    chase between replans." That is stated as mechanism, never measured.
  * The classical-planner ladder (docs/findings.md) swept replan_period as a
    planner-COMPARISON axis at max_accel=80 — a *reactive* regime where the drone can
    dodge late regardless of plan age, so the period "barely mattered" beyond the
    planner ordering.

Neither characterises whether MPC ALONE has a replan_period sweet-spot, and in
particular whether **more-frequent replanning is counterproductive**. The naive
intuition is monotone: replan more often → fresher obstacle positions in the plan →
safer. This script tests the opposite — that commitment, not just freshness, does
safety work, so the curve is an inverted-U and the shortest period (replan every
control step) collides MORE.

Why it could invert: between replans the runner holds the planned velocity constant
(uav_nav_lab/runner/experiment.py: the plan is recomputed only when
`t - last_replan_t >= replan_period`). With a long period the drone commits to one
coherent dodge; with a very short period it re-solves every control step against a
near-symmetric head-on geometry and can flip between left/right avoidance (chatter),
never committing to a side. The effect only bites when the drone CANNOT dodge late,
i.e. at LOW max_accel (the crossing study's "must commit on the forecast" regime) —
which is exactly the regime the ladder did not test.

Planning is instantaneous in sim time here (the runner only logs planner_dt), so this
is a pure commitment/reactivity trade, not a compute-budget effect.

Calibrate the operating point first — find the max_accel that pushes MPC off the
100 % ceiling so the period can discriminate:
    python scripts/mpc_replan_commitment_phase.py --accel-sweep 2 3 4 6 10 20 --n 30
Then sweep the period at that accel (paired by seed, McNemar each cell vs the peak):
    python scripts/mpc_replan_commitment_phase.py --max-accel 4 --n 60
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
EXAMPLE = REPO / "examples/exp_compare_mpc.yaml"
DEFAULT_PERIODS = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]   # planner.replan_period (s); dt=0.05 is the floor
DEFAULT_MAX_ACCEL = 4.0     # calibrated commitment regime (MPC off the ceiling)
SEED_BASE = 200


def _base_config() -> dict:
    from uav_nav_lab.config import ExperimentConfig

    return ExperimentConfig.from_yaml(EXAMPLE).to_dict()


def _cell_config(base: dict, period: float, max_accel: float,
                 obs_speed_mult: float = 1.0, obs_radius: float | None = None) -> dict:
    cfg = copy.deepcopy(base)
    cfg.pop("output", None)
    cfg["planner"]["replan_period"] = float(period)
    cfg["simulator"]["max_accel"] = float(max_accel)
    if obs_speed_mult != 1.0 or obs_radius is not None:
        for ob in cfg["scenario"].get("dynamic_obstacles", []):
            if obs_speed_mult != 1.0:
                ob["velocity"] = [v * obs_speed_mult for v in ob["velocity"]]
            if obs_radius is not None:
                ob["radius"] = float(obs_radius)
    cfg["name"] = f"mpc_rp{period:g}_a{max_accel:g}_x{obs_speed_mult:g}"
    return cfg


def _run_cell_chunk(job: dict) -> dict:
    from uav_nav_lab.config import ExperimentConfig
    from uav_nav_lab.runner.experiment import run_experiment

    cfg = ExperimentConfig.from_dict(job["config"])
    cfg.seed = job["seed_start"]
    cfg.num_episodes = job["count"]
    out = Path(tempfile.mkdtemp(prefix="rpcommit_"))
    run_experiment(cfg, out)
    episodes = []
    for p in sorted(out.glob("episode_*.json")):
        d = json.loads(p.read_text())
        # exclude the multi-drone joint files (none here, but be safe)
        if "_joint" in p.name:
            continue
        episodes.append({"seed": d["meta"]["seed"], "outcome": d["outcome"]})
    for p in out.glob("*"):
        p.unlink()
    out.rmdir()
    return {"key": job["key"], "episodes": episodes}


def _wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def _build_jobs(base, keyed_cfgs, n, chunk):
    """keyed_cfgs: list of (key, cfg)."""
    jobs = []
    for key, cfg in keyed_cfgs:
        seed = SEED_BASE
        remaining = n
        while remaining > 0:
            count = min(chunk, remaining)
            jobs.append({"key": key, "config": cfg, "seed_start": seed, "count": count})
            seed += count
            remaining -= count
    return jobs


def _collect(chunk_results):
    by_key = {}
    for r in chunk_results:
        d = by_key.setdefault(r["key"], {})
        for ep in r["episodes"]:
            d[ep["seed"]] = ep["outcome"]
    return by_key


def _accel_sweep(base, args):
    """Calibration: at fixed replan_period, find the operating point off the ceiling.

    With --accel-sweep, vary max_accel (at --obs-speed-mult). With --obs-sweep, vary
    the dynamic-obstacle speed multiplier (at --max-accel).
    """
    rp = args.periods[0] if args.periods else 0.2
    if args.obs_sweep is not None:
        keyed = [(m, _cell_config(base, rp, args.max_accel, obs_speed_mult=m,
                                  obs_radius=args.obs_radius)) for m in args.obs_sweep]
        label = "obs_speed_mult"
        print(f"[rpcal] obstacle-speed calibration at replan_period={rp:g}, "
              f"max_accel={args.max_accel:g} (n={args.n})\n")
    else:
        keyed = [(a, _cell_config(base, rp, a, obs_speed_mult=args.obs_speed_mult,
                                  obs_radius=args.obs_radius)) for a in args.accel_sweep]
        label = "max_accel"
        print(f"[rpcal] max_accel calibration at replan_period={rp:g}, "
              f"obs_speed_mult={args.obs_speed_mult:g} (n={args.n})\n")
    jobs = _build_jobs(base, keyed, args.n, args.chunk)
    with Pool(processes=args.workers) as pool:
        res = pool.map(_run_cell_chunk, jobs)
    by_key = _collect(res)
    print(f"{label:>14} | {'success':>8} | {'collision':>9} | {'timeout':>8}")
    print("-" * 48)
    sweep_vals = args.obs_sweep if args.obs_sweep is not None else args.accel_sweep
    for a in sweep_vals:
        d = by_key[a]
        n = len(d)
        succ = sum(v == "success" for v in d.values())
        coll = sum(v == "collision" for v in d.values())
        tout = sum(v == "timeout" for v in d.values())
        print(f"{a:>14g} | {succ:>3}/{n:<3} {succ/n*100:>3.0f}% | "
              f"{coll/n*100:>7.1f}% | {tout/n*100:>6.1f}%")
    print("\n  Pick the max_accel where success is ~50-85% (off the ceiling, above the "
          "floor) so replan_period can discriminate.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60, help="episodes per replan_period cell")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--periods", type=float, nargs="+", default=DEFAULT_PERIODS,
                    help="planner.replan_period values to sweep (s)")
    ap.add_argument("--max-accel", type=float, default=DEFAULT_MAX_ACCEL)
    ap.add_argument("--obs-speed-mult", type=float, default=1.0,
                    help="scale all dynamic-obstacle velocities (harder dynamic threat)")
    ap.add_argument("--obs-radius", type=float, default=None,
                    help="override all dynamic-obstacle radii")
    ap.add_argument("--accel-sweep", type=float, nargs="+", default=None,
                    help="calibration: sweep max_accel at periods[0] instead of the main run")
    ap.add_argument("--obs-sweep", type=float, nargs="+", default=None,
                    help="calibration: sweep obs-speed-mult at periods[0] x max-accel")
    ap.add_argument("--overlay", nargs="+", default=None,
                    help="dirs with phase.json to overlay into one figure (no run)")
    ap.add_argument("--overlay-labels", nargs="+", default=None,
                    help="legend labels matching --overlay dirs")
    ap.add_argument("--out", default=str(REPO / "results/mpc_replan_commitment_phase"))
    args = ap.parse_args()

    if args.overlay is not None:
        return _plot_overlay(args.overlay, args.overlay_labels, args.out)

    from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p

    base = _base_config()

    if args.accel_sweep is not None or args.obs_sweep is not None:
        return _accel_sweep(base, args)

    keyed = [(p, _cell_config(base, p, args.max_accel, obs_speed_mult=args.obs_speed_mult,
                              obs_radius=args.obs_radius)) for p in args.periods]
    jobs = _build_jobs(base, keyed, args.n, args.chunk)
    print(f"[rpcommit] {len(args.periods)} periods x n={args.n} = {len(jobs)} jobs "
          f"on {args.workers} workers (max_accel={args.max_accel:g})")
    t0 = time.perf_counter()
    with Pool(processes=args.workers) as pool:
        chunk_results = pool.map(_run_cell_chunk, jobs)
    dt = time.perf_counter() - t0
    print(f"[rpcommit] sweep done in {dt:.0f}s")

    by_key = _collect(chunk_results)
    # common seeds across all periods -> honest pairing
    common = sorted(set.intersection(*[set(by_key[p]) for p in args.periods]))
    n = len(common)

    rows = []
    for p in args.periods:
        d = by_key[p]
        succ = sum(d[s] == "success" for s in common)
        coll = sum(d[s] == "collision" for s in common)
        tout = sum(d[s] == "timeout" for s in common)
        _, lo, hi = _wilson(succ, n)
        rows.append({"period": p, "n": n, "success": succ / n,
                     "success_ci": [lo, hi], "collision": coll / n, "timeout": tout / n,
                     "_succ": succ})

    # peak cell = highest success (ties → mid period)
    peak = max(rows, key=lambda r: r["_succ"])
    peak_p = peak["period"]
    peak_d = by_key[peak_p]
    for r in rows:
        d = by_key[r["period"]]
        # McNemar of this cell vs the peak: b = peak succ & this fail, c = peak fail & this succ
        b = sum(peak_d[s] == "success" and d[s] != "success" for s in common)
        c = sum(peak_d[s] != "success" and d[s] == "success" for s in common)
        r["vs_peak"] = {"b": b, "c": c, "p": mcnemar_exact_p(b, c)}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"n_per_cell": args.n, "periods": args.periods, "max_accel": args.max_accel,
              "peak_period": peak_p, "seconds": dt, "rows": rows}
    (out_dir / "phase.json").write_text(json.dumps(result, indent=2))
    (out_dir / "phase_raw.json").write_text(
        json.dumps({str(p): by_key[p] for p in args.periods}, indent=2))

    _print_table(rows, peak_p, args.max_accel)
    _plot(rows, peak_p, args.max_accel, out_dir / "phase.png")
    print(f"[rpcommit] wrote {out_dir/'phase.json'}, phase_raw.json and {out_dir/'phase.png'}")
    return 0


def _print_table(rows, peak_p, max_accel) -> None:
    print()
    print(f"  max_accel={max_accel:g}, paired by seed (n={rows[0]['n']}); "
          f"peak period = {peak_p:g}s")
    print(f"{'period':>7} | {'success':>13} | {'coll':>6} {'tout':>6} | {'vs peak (c/b, p)':>18}")
    print("-" * 60)
    for r in rows:
        v = r["vs_peak"]
        ci = r["success_ci"]
        mark = "  <- peak" if r["period"] == peak_p else ""
        print(f"{r['period']:>6g}s | {r['success']*100:>5.1f}% "
              f"[{ci[0]*100:>3.0f},{ci[1]*100:>3.0f}] | "
              f"{r['collision']*100:>5.1f}% {r['timeout']*100:>5.1f}% | "
              f"{v['c']:>3}/{v['b']:<3} p={v['p']:>6.4f}{mark}")
    print("\n  period = planner.replan_period; smaller = more frequent replanning.")
    print("  vs peak: c = this cell won where peak failed; b = this cell lost where peak won.")
    print("  An inverted-U with the SHORTEST period significantly below peak = "
          "frequent replanning is counterproductive (commitment does safety work).")


def _plot(rows, peak_p, max_accel, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["period"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ys = [r["success"] * 100 for r in rows]
    lo = [r["success_ci"][0] * 100 for r in rows]
    hi = [r["success_ci"][1] * 100 for r in rows]
    ax1.plot(xs, ys, "o-", color="#2c3e50", lw=2)
    ax1.fill_between(xs, lo, hi, color="#2c3e50", alpha=0.15)
    ax1.set_xscale("log")
    ax1.axvline(peak_p, color="#27ae60", ls=":", lw=1.5, label=f"peak ({peak_p:g}s)")
    ax1.set_xlabel("replan_period (s, log) — smaller = more frequent replanning")
    ax1.set_ylabel("joint success rate (%)")
    ax1.set_title(f"MPC dynamic-obstacle success vs replan cadence\n"
                  f"(commitment regime, max_accel={max_accel:g})")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, which="both")

    net = [r["vs_peak"]["c"] - r["vs_peak"]["b"] for r in rows]
    colors = ["#27ae60" if r["period"] == peak_p else
              ("#c0392b" if r["vs_peak"]["p"] < 0.05 else "#95a5a6") for r in rows]
    bars = ax2.bar([f"{x:g}" for x in xs], net, color=colors)
    for r, bar in zip(rows, bars):
        if r["period"] != peak_p:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"p={r['vs_peak']['p']:.3f}", ha="center",
                     va="top" if bar.get_height() < 0 else "bottom", fontsize=8)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("replan_period (s)")
    ax2.set_ylabel("net paired wins vs peak (c - b)")
    ax2.set_title("Paired deficit vs the peak period (red = McNemar p<0.05 worse)")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("Is more-frequent replanning counterproductive? "
                 "The replan_period commitment sweet-spot")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def _plot_overlay(dirs, labels, out) -> int:
    """Overlay several operating points' success-vs-replan_period curves on one axis.

    Each dir must contain a phase.json written by a prior run. The point of the
    figure: the long-period (stale-commitment) decay is consistent, while the deep
    0.4 s valley appears at exactly ONE operating point (obs x2.0) and does not move
    along the period axis as the obstacle speed changes -- so it is a pathological
    artifact, not a commitment/obstacle resonance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if labels is None:
        labels = [Path(d).name for d in dirs]
    colors = ["#2c3e50", "#2980b9", "#c0392b", "#27ae60", "#8e44ad", "#d35400"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, (d, lab) in enumerate(zip(dirs, labels)):
        data = json.loads((Path(d) / "phase.json").read_text())
        rows = sorted(data["rows"], key=lambda r: r["period"])
        xs = [r["period"] for r in rows]
        ys = [r["success"] * 100 for r in rows]
        lo = [r["success_ci"][0] * 100 for r in rows]
        hi = [r["success_ci"][1] * 100 for r in rows]
        c = colors[i % len(colors)]
        ax.plot(xs, ys, "o-", color=c, lw=2, label=lab, zorder=3)
        ax.fill_between(xs, lo, hi, color=c, alpha=0.12, zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel("replan_period (s, log) — smaller = more frequent replanning")
    ax.set_ylabel("success rate (%)")
    ax.set_title("MPC dynamic-obstacle success vs replan cadence\n"
                 "Frequent replanning never hurts; long commitment decays; "
                 "the 0.4 s valley is a single-point artifact")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, title="operating point")
    fig.tight_layout()
    out_path = Path(out)
    if out_path.is_dir() or out_path.suffix == "":
        out_path = out_path / "phase_overlay.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"[rpcommit] wrote overlay figure {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
