#!/usr/bin/env python3
"""RRT* rewiring vs RRT: does asymptotic path optimality survive the replan loop?

`examples/exp_compare_rrt{,_star}.yaml` ship two planners that differ by exactly
one mechanism: RRT returns the FIRST collision-free path it samples (a zigzag),
while RRT* additionally rewires the neighbourhood of every new node toward
lowest cost-from-start, returning an asymptotically *optimal* (shortest) path.
The textbook promise is "RRT* gives you a better path." Neither planner had ever
been run through a paired significance test in this repo, and the more pointed
question is whether that better PATH becomes a better OUTCOME once it is dropped
into a fast closed-loop replanner.

Two planning arms, identical scenario / sensor / dynamics, one mechanism apart:

    rrt       — single-tree RRT, returns the first path to the goal region
    rrt_star  — same sampling, plus best-parent + neighbourhood rewiring

A structural property of the two implementations makes this an unusually clean
ablation: for a given layout and planner seed, both draw the *same* RNG stream
and grow the *same* node positions, so they discover the goal region on the same
iteration. The rewiring uses no randomness. On the FIRST replan they therefore
return paths through an identical tree -- RRT the first-found chain, RRT* the
rewired-shortest chain. The only thing that varies is *which path is returned*,
which is exactly the quantity RRT* is supposed to improve. (They diverge on later
replans only because the drone has, by then, executed different paths.)

Mechanism axis: `replan_period`. A fast replanner executes only the PREFIX of
each plan before throwing the rest away and planning afresh, so the global path
optimisation RRT* pays for is mostly discarded; a slow replanner executes more of
each plan, so the optimisation has a chance to matter. Sweeping the period tests
whether any RRT* benefit is gated by how much of its optimised path the loop
actually flies. The dynamic-obstacle reactivity that `replan_period` also
controls hits BOTH arms equally, so it cancels in the paired (rrt_star - rrt)
contrast at each period.

We pair by episode seed (same `episode_seed ^ obstacles.seed` random layout for
both arms) and run an exact McNemar test on the goal-reach outcome at each
period. Alongside success we record the offline quantities the rewiring is
supposed to move -- per-replan planner_dt (the cost of rewiring), executed path
length and time-to-goal -- so an "the path got better but the outcome did not"
result can be shown rather than asserted.

Output: `results/rrt_rewire_replan_phase/phase.json` (+ `phase_raw.json`) and a
two-panel `phase.png` (success vs replan_period per arm with Wilson bands; the
paired rewiring effect on success with significance markers).

Run (defaults: 5 periods x 2 arms x n=80; RRT is cheap, RRT* less so):
    python scripts/rrt_rewire_replan_phase.py
    python scripts/rrt_rewire_replan_phase.py --n 60 --workers 6
"""

from __future__ import annotations

# Single-thread each numpy worker so the forked pool does not oversubscribe.
# Must precede any (transitive) numpy import.
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
EXAMPLE = REPO / "examples/exp_compare_rrt.yaml"
DEFAULT_LEVELS = [0.1, 0.2, 0.4, 0.8, 1.6]   # planner.replan_period (s)
SEED_BASE = 200
REWIRE_RADIUS = 4.0

# rrt is the greedy-first baseline; rrt_star adds rewiring (the mechanism).
ARMS = ("rrt", "rrt_star")


def _base_config() -> dict:
    from uav_nav_lab.config import ExperimentConfig

    return ExperimentConfig.from_yaml(EXAMPLE).to_dict()


def _cell_config(base: dict, arm: str, period: float) -> dict:
    """Same scenario/sensor; only the planner family and replan_period change."""
    cfg = copy.deepcopy(base)
    cfg.pop("output", None)
    cfg["planner"]["type"] = arm
    cfg["planner"]["replan_period"] = float(period)
    if arm == "rrt_star":
        cfg["planner"]["rewire_radius"] = REWIRE_RADIUS
    cfg["name"] = f"{arm}_rp{period}"
    return cfg


def _episode_metrics(d: dict) -> dict:
    """Extract outcome + the offline quantities rewiring is meant to improve."""
    steps = d.get("steps", [])
    replans = d.get("replans", [])
    path_len = 0.0
    prev = None
    for s in steps:
        p = s.get("true_pos")
        if p is None:
            continue
        if prev is not None:
            path_len += math.dist(prev, p)
        prev = p
    dts = [r.get("planner_dt_ms", 0.0) for r in replans]
    plens = [r.get("plan_length", 0) for r in replans]
    return {
        "seed": d["meta"]["seed"],
        "outcome": d["outcome"],
        "path_len": path_len,
        "final_t": d.get("summary", {}).get("final_t", float("nan")),
        "mean_dt_ms": (sum(dts) / len(dts)) if dts else 0.0,
        "mean_plan_len": (sum(plens) / len(plens)) if plens else 0.0,
        "n_replans": len(replans),
    }


def _run_cell_chunk(job: dict) -> dict:
    """Worker: run `count` episodes from `seed_start` for one cell config."""
    from uav_nav_lab.config import ExperimentConfig
    from uav_nav_lab.runner.experiment import run_experiment

    cfg = ExperimentConfig.from_dict(job["config"])
    cfg.seed = job["seed_start"]
    cfg.num_episodes = job["count"]
    out = Path(tempfile.mkdtemp(prefix="rrt_"))
    run_experiment(cfg, out)
    episodes = []
    for p in sorted(out.glob("episode_*.json")):
        episodes.append(_episode_metrics(json.loads(p.read_text())))
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
    ap.add_argument("--n", type=int, default=80, help="episodes per (period, arm) cell")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=5, help="episodes per parallel job")
    ap.add_argument(
        "--levels", type=float, nargs="+", default=DEFAULT_LEVELS,
        help="planner.replan_period values to sweep (s)",
    )
    ap.add_argument("--out", default=str(REPO / "results/rrt_rewire_replan_phase"))
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

    print(f"[rrt] {len(args.levels)} periods x {len(ARMS)} arms x n={args.n} "
          f"= {len(jobs)} jobs on {args.workers} workers")
    t0 = time.perf_counter()
    with Pool(processes=args.workers) as pool:
        chunk_results = pool.map(_run_cell_chunk, jobs)
    dt = time.perf_counter() - t0
    print(f"[rrt] sweep done in {dt:.0f}s")

    by_cell: dict[float, dict[str, dict[int, dict]]] = {}
    for r in chunk_results:
        cell = by_cell.setdefault(r["level"], {a: {} for a in ARMS})
        for ep in r["episodes"]:
            cell[r["arm"]][ep["seed"]] = ep

    def _paired(a: dict[int, dict], b: dict[int, dict]) -> dict:
        """McNemar of arm b vs arm a on success. c = b succeeds where a fails,
        b = b fails where a succeeds (so c-b > 0 means b is better)."""
        seeds = sorted(set(a) & set(b))
        ba = sum(a[s]["outcome"] == "success" and b[s]["outcome"] != "success" for s in seeds)
        ca = sum(a[s]["outcome"] != "success" and b[s]["outcome"] == "success" for s in seeds)
        return {"b": ba, "c": ca, "p": mcnemar_exact_p(ba, ca)}

    rows, raw = [], []
    for level in sorted(by_cell):
        cells = by_cell[level]
        seeds = sorted(set.intersection(*[set(cells[a]) for a in ARMS]))
        n = len(seeds)
        row: dict = {"level": level, "n": n}
        for a in ARMS:
            eps = [cells[a][s] for s in seeds]
            succ = sum(e["outcome"] == "success" for e in eps)
            coll = sum(e["outcome"] == "collision" for e in eps)
            tout = sum(e["outcome"] == "timeout" for e in eps)
            _, lo, hi = _wilson(succ, n)
            # offline quantities on the SUCCEEDING episodes (path length / time
            # are only comparable when the goal was actually reached).
            ok = [e for e in eps if e["outcome"] == "success"]
            row[a] = {
                "success": succ / n, "success_ci": [lo, hi],
                "collision": coll / n, "timeout": tout / n,
                "mean_dt_ms": sum(e["mean_dt_ms"] for e in eps) / len(eps),
                "mean_plan_len": sum(e["mean_plan_len"] for e in eps) / len(eps),
                "path_len_ok": (sum(e["path_len"] for e in ok) / len(ok)) if ok else float("nan"),
                "final_t_ok": (sum(e["final_t"] for e in ok) / len(ok)) if ok else float("nan"),
            }
        # The rewiring effect on the OUTCOME: rrt -> rrt_star.
        row["rewire_effect"] = _paired(cells["rrt"], cells["rrt_star"])
        rows.append(row)
        raw.append({"level": level,
                    "episodes": {a: {s: cells[a][s] for s in seeds} for a in ARMS}})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"n_per_cell": args.n, "levels": args.levels, "arms": list(ARMS),
              "rewire_radius": REWIRE_RADIUS, "seconds": dt, "rows": rows}
    (out_dir / "phase.json").write_text(json.dumps(result, indent=2))
    (out_dir / "phase_raw.json").write_text(json.dumps(raw, indent=2))

    _print_table(rows)
    _plot(rows, out_dir / "phase.png")
    print(f"[rrt] wrote {out_dir/'phase.json'}, phase_raw.json and {out_dir/'phase.png'}")
    return 0


def _print_table(rows: list[dict]) -> None:
    print()
    print("Goal-reach success per planner (paired by seed, n per cell):")
    print(f"{'rp(s)':>6} {'n':>4} | {'rrt':>8} | {'rrt*':>8} | "
          f"{'dt rrt':>8} {'dt rrt*':>8} | {'len rrt':>8} {'len rrt*':>8}")
    print("-" * 78)
    for r in rows:
        print(f"{r['level']:>6} {r['n']:>4} | "
              f"{r['rrt']['success']*100:6.1f}% | {r['rrt_star']['success']*100:6.1f}% | "
              f"{r['rrt']['mean_dt_ms']:7.1f} {r['rrt_star']['mean_dt_ms']:7.1f} | "
              f"{r['rrt']['path_len_ok']:8.1f} {r['rrt_star']['path_len_ok']:8.1f}")

    print("\nRewiring effect on outcome (rrt -> rrt*; net = c-b; exact-McNemar p):")
    print(f"{'rp(s)':>6} | {'net successes':>14} | {'p':>8}")
    print("-" * 38)
    for r in rows:
        d = r["rewire_effect"]
        print(f"{r['level']:>6} | net {d['c']-d['b']:+3d} (c{d['c']}/b{d['b']}) | {d['p']:>8.3f}")
    print("\n  dt = mean per-replan planner time (ms); len = mean executed path")
    print("  length on SUCCEEDING episodes. A positive net = rrt* reaches more goals.")


def _plot(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    xs = [r["level"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = {"rrt": "#e67e22", "rrt_star": "#16a085"}
    labels = {"rrt": "RRT (first path)", "rrt_star": "RRT* (rewired path)"}
    for a in ARMS:
        ys = [r[a]["success"] * 100 for r in rows]
        lo = [r[a]["success_ci"][0] * 100 for r in rows]
        hi = [r[a]["success_ci"][1] * 100 for r in rows]
        ax1.plot(xs, ys, "o-", color=colors[a], label=labels[a])
        ax1.fill_between(xs, lo, hi, color=colors[a], alpha=0.12)
    ax1.set_xscale("log")
    ax1.set_xlabel("replan_period (s, log scale)")
    ax1.set_ylabel("goal-reach success rate (%)")
    ax1.set_title("Success vs replan cadence: RRT vs RRT*")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    x = np.arange(len(rows))
    net = [r["rewire_effect"]["c"] - r["rewire_effect"]["b"] for r in rows]
    bars = ax2.bar(x, net, 0.55, color="#16a085")
    for r, bar in zip(rows, bars):
        if r["rewire_effect"]["p"] < 0.05:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "*",
                     ha="center", va="bottom", fontsize=14)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(r["level"]) for r in rows])
    ax2.set_xlabel("replan_period (s)")
    ax2.set_ylabel("net goal-reaches gained by rewiring (paired c-b)")
    ax2.set_title("Rewiring's effect on outcome (* = McNemar p<0.05)")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("Does RRT* rewiring convert to goal-reach in a replan loop? "
                 "(50×50 grid, 3 dynamic obstacles, perfect sensing)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    raise SystemExit(main())
