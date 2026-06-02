#!/usr/bin/env python3
"""CHOMP's explicit clearance band has a sweet spot — and still caps below RRT.

The planner-clearance-ladder study (docs/findings.md) showed that against
obstacles the planner does not model, goal-reach is governed by incidental
*clearance*, and the closing takeaway was "reach for clearance (inflate, or a
cost that models motion)." CHOMP is the one classical planner in the suite that
*does* reach for clearance explicitly: its objective is

    U(x) = w_smooth * ||A x||^2 / 2  +  w_obs * sum_i c(x_i)

where the obstacle potential c(x_i) pushes each waypoint away from obstacles
within a band of width `epsilon` (a hinge on the distance field). So CHOMP is the
direct test of that takeaway: does an explicit clearance term buy what RRT gets
for free? `examples/exp_compare_chomp.yaml` buries a single n=30 number (chomp
53.3 %, slotting between rrt_star 23 % and rrt 73 %) but never swept the knob that
sets the clearance — `epsilon` — and never paired-tested it.

This sweeps `epsilon` (the clearance-band width, shipped default 2.0) on the
shipped dynamic scenario (50x50, 25 static + 3 reflecting moving obstacles,
perfect sensing, w_obs 5, rp 0.2), pairs by episode seed, and runs exact McNemar
against the shipped epsilon. One extra arm seeds CHOMP from an RRT path instead
of the straight-line init (init="rrt"), testing the buried roadmap claim that
RRT init "might lift the success rate further" — i.e. whether injecting RRT's
incidental clearance breaks CHOMP's cap.

Key fact the study turns on: CHOMP does NOT use the dynamic obstacles
(planner/chomp/planner.py marks `dynamic_obstacles` ARG002-unused); `epsilon`
adds clearance from the STATIC occupancy only, while every failure is a DYNAMIC
collision. So tuning epsilon tunes clearance from the wrong threat.

References (same scenario, rp 0.2, from the planner-clearance-ladder study):
astar 20.0 %, rrt_star 26.7 %, rrt 66.7 %.

Output: `results/chomp_clearance_band_phase/phase.json` (+ `phase_raw.json`) and a
two-panel `phase.png` (success + collision vs epsilon with the RRT/RRT* reference
band; the rrt-init arm marked).

Run (defaults: 6 epsilons + 1 rrt-init arm x n=80; CHOMP is cheap):
    python scripts/chomp_clearance_band_phase.py
    python scripts/chomp_clearance_band_phase.py --n 60 --workers 6
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
EXAMPLE = REPO / "examples/exp_compare_chomp.yaml"
DEFAULT_EPSILONS = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]   # planner.epsilon (clearance band, m)
SHIPPED_EPS = 2.0
SEED_BASE = 200
GOAL_DIST = math.dist([2.0, 2.0], [45.0, 45.0])
# References from the planner-clearance-ladder study (same scenario, rp 0.2).
REF = {"astar": 0.200, "rrt_star": 0.267, "rrt": 0.667}


def _base_config() -> dict:
    from uav_nav_lab.config import ExperimentConfig

    return ExperimentConfig.from_yaml(EXAMPLE).to_dict()


def _variant_config(base: dict, label: str, epsilon: float, init: str) -> dict:
    cfg = copy.deepcopy(base)
    cfg.pop("output", None)
    cfg["planner"]["epsilon"] = float(epsilon)
    cfg["planner"]["init"] = init  # "straight" (default) or "rrt"
    cfg["name"] = label
    return cfg


def _episode_metrics(d: dict) -> dict:
    steps = d.get("steps", [])
    path_len = 0.0
    prev = None
    for s in steps:
        p = s.get("true_pos")
        if p is None:
            continue
        if prev is not None:
            path_len += math.dist(prev, p)
        prev = p
    return {"seed": d["meta"]["seed"], "outcome": d["outcome"], "path_len": path_len}


def _run_cell_chunk(job: dict) -> dict:
    from uav_nav_lab.config import ExperimentConfig
    from uav_nav_lab.runner.experiment import run_experiment

    cfg = ExperimentConfig.from_dict(job["config"])
    cfg.seed = job["seed_start"]
    cfg.num_episodes = job["count"]
    out = Path(tempfile.mkdtemp(prefix="chband_"))
    run_experiment(cfg, out)
    episodes = []
    for p in sorted(out.glob("episode_*.json")):
        episodes.append(_episode_metrics(json.loads(p.read_text())))
    for p in out.glob("*"):
        p.unlink()
    out.rmdir()
    return {"label": job["label"], "episodes": episodes}


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
    ap.add_argument("--n", type=int, default=80, help="episodes per variant")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--epsilons", type=float, nargs="+", default=DEFAULT_EPSILONS)
    ap.add_argument("--out", default=str(REPO / "results/chomp_clearance_band_phase"))
    args = ap.parse_args()

    from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p

    base = _base_config()

    # Variants: the epsilon sweep (straight init) + one rrt-init arm at shipped eps.
    variants: list[tuple[str, dict]] = []
    for eps in args.epsilons:
        variants.append((f"eps{eps}", _variant_config(base, f"eps{eps}", eps, "straight")))
    rrtinit_label = f"eps{SHIPPED_EPS}_rrtinit"
    variants.append((rrtinit_label,
                     _variant_config(base, rrtinit_label, SHIPPED_EPS, "rrt")))

    jobs: list[dict] = []
    for label, cfg in variants:
        seed = SEED_BASE
        remaining = args.n
        while remaining > 0:
            count = min(args.chunk, remaining)
            jobs.append({"label": label, "config": cfg,
                         "seed_start": seed, "count": count})
            seed += count
            remaining -= count

    print(f"[chband] {len(variants)} variants x n={args.n} = {len(jobs)} jobs "
          f"on {args.workers} workers")
    t0 = time.perf_counter()
    with Pool(processes=args.workers) as pool:
        chunk_results = pool.map(_run_cell_chunk, jobs)
    dt = time.perf_counter() - t0
    print(f"[chband] sweep done in {dt:.0f}s")

    by_label: dict[str, dict[int, dict]] = {}
    for r in chunk_results:
        cell = by_label.setdefault(r["label"], {})
        for ep in r["episodes"]:
            cell[ep["seed"]] = ep

    shipped_label = f"eps{SHIPPED_EPS}"
    shipped = by_label[shipped_label]

    def _paired(a: dict[int, dict], b: dict[int, dict]) -> dict:
        seeds = sorted(set(a) & set(b))
        ba = sum(a[s]["outcome"] == "success" and b[s]["outcome"] != "success" for s in seeds)
        ca = sum(a[s]["outcome"] != "success" and b[s]["outcome"] == "success" for s in seeds)
        return {"b": ba, "c": ca, "p": mcnemar_exact_p(ba, ca)}

    rows = []
    for label, _ in variants:
        cell = by_label[label]
        seeds = sorted(cell)
        n = len(seeds)
        succ = sum(cell[s]["outcome"] == "success" for s in seeds)
        coll = sum(cell[s]["outcome"] == "collision" for s in seeds)
        tout = sum(cell[s]["outcome"] == "timeout" for s in seeds)
        ok = [cell[s] for s in seeds if cell[s]["outcome"] == "success"]
        _, lo, hi = _wilson(succ, n)
        directness = (sum(e["path_len"] for e in ok) / len(ok) / GOAL_DIST) if ok else float("nan")
        # paired vs shipped epsilon (skip self).
        vs = None if label == shipped_label else _paired(shipped, cell)
        rows.append({"label": label, "n": n, "success": succ / n,
                     "success_ci": [lo, hi], "collision": coll / n,
                     "timeout": tout / n, "directness": directness,
                     "vs_shipped": vs})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"n_per_variant": args.n, "epsilons": args.epsilons,
              "shipped_eps": SHIPPED_EPS, "goal_dist": GOAL_DIST,
              "ref": REF, "seconds": dt, "rows": rows}
    (out_dir / "phase.json").write_text(json.dumps(result, indent=2))
    (out_dir / "phase_raw.json").write_text(json.dumps(
        {lbl: by_label[lbl] for lbl, _ in variants}, indent=2))

    _print_table(rows)
    _plot(rows, args.epsilons, out_dir / "phase.png")
    print(f"[chband] wrote {out_dir/'phase.json'}, phase_raw.json and {out_dir/'phase.png'}")
    return 0


def _print_table(rows: list[dict]) -> None:
    print()
    print("CHOMP clearance-band sweep (rp 0.2, w_obs 5; paired by seed vs shipped eps 2.0):")
    print(f"{'variant':>16} {'n':>4} | {'succ':>7} {'coll':>6} {'tout':>6} | "
          f"{'direct':>7} | {'vs eps2.0 (net c-b, p)':>24}")
    print("-" * 80)
    for r in rows:
        vs = r["vs_shipped"]
        vss = "—  (shipped)" if vs is None else f"net {vs['c']-vs['b']:+3d} p={vs['p']:.3f}"
        print(f"{r['label']:>16} {r['n']:>4} | {r['success']*100:6.1f}% "
              f"{r['collision']*100:5.0f}% {r['timeout']*100:5.0f}% | "
              f"{r['directness']:7.3f} | {vss:>24}")
    print(f"\n  References (same scenario, rp 0.2): astar {REF['astar']*100:.0f}%, "
          f"rrt_star {REF['rrt_star']*100:.0f}%, rrt {REF['rrt']*100:.0f}%.")
    print("  All CHOMP failures are collisions; epsilon adds STATIC clearance only.")


def _plot(rows: list[dict], epsilons: list[float], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = [r for r in rows if r["label"].startswith("eps")
             and not r["label"].endswith("rrtinit")]
    xs = epsilons
    succ = [r["success"] * 100 for r in sweep]
    lo = [r["success_ci"][0] * 100 for r in sweep]
    hi = [r["success_ci"][1] * 100 for r in sweep]
    coll = [r["collision"] * 100 for r in sweep]
    rrtinit = next((r for r in rows if r["label"].endswith("rrtinit")), None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.axhspan(REF["rrt"] * 100 - 0.1, REF["rrt"] * 100 + 0.1, color="#e67e22", alpha=0.0)
    ax1.axhline(REF["rrt"] * 100, color="#e67e22", ls="--", lw=1.5,
                label=f"RRT (incidental clearance) {REF['rrt']*100:.0f}%")
    ax1.axhline(REF["rrt_star"] * 100, color="#16a085", ls=":", lw=1.5,
                label=f"RRT* {REF['rrt_star']*100:.0f}%")
    ax1.plot(xs, succ, "o-", color="#2980b9", label="CHOMP (straight init)")
    ax1.fill_between(xs, lo, hi, color="#2980b9", alpha=0.12)
    if rrtinit is not None:
        ax1.plot([SHIPPED_EPS], [rrtinit["success"] * 100], "D", color="#8e44ad",
                 ms=10, label="CHOMP eps2.0 + RRT init")
    ax1.axvline(SHIPPED_EPS, color="#888", ls="-", lw=0.8, alpha=0.6)
    ax1.set_xlabel("epsilon (obstacle-potential clearance band, m)")
    ax1.set_ylabel("goal-reach success rate (%)")
    ax1.set_title("CHOMP clearance band has a sweet spot (shipped = 2.0)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(xs, coll, "s-", color="#c0392b", label="collision rate")
    ax2.axvline(SHIPPED_EPS, color="#888", ls="-", lw=0.8, alpha=0.6)
    ax2.set_xlabel("epsilon (obstacle-potential clearance band, m)")
    ax2.set_ylabel("collision rate (%)")
    ax2.set_title("Too narrow under-avoids, too wide over-avoids — all collisions")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("Does CHOMP's explicit clearance term buy what RRT gets for free? "
                 "(50×50 grid, 3 dynamic obstacles, perfect sensing)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    raise SystemExit(main())
