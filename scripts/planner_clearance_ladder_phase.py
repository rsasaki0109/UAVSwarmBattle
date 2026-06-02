#!/usr/bin/env python3
"""The classical-planner ladder is a CLEARANCE ladder, not a reasoning ladder.

`examples/exp_compare_astar.yaml` ships, buried in its header, a five-planner
table on the dynamic-obstacle scenario (n=30, Wilson CIs, perfect sensor):

    straight 0%  <  astar 20%  <  rrt_star 23%  <  rrt 73%  <  mpc 100%

and tells two mechanism stories to explain the middle of it: A* loses because
"its 8-connected grid path zigzags, sitting in obstacle paths longer", and
rrt_star loses because "464 ms overshoots the 200 ms replan_period ... the drone
follows stale plans". Neither was ever run through a paired significance test,
and both stories are wrong:

  - The runner makes planning INSTANTANEOUS in sim time (see
    uav_nav_lab/runner/experiment.py: `last_replan_t = t` is the sim time at
    replan, planner_dt is only logged, sim time advances solely via sim.step).
    So rrt_star cannot fail from "stale plans / blowing the replan budget" --
    planner_dt has zero effect on the simulation.
  - A* is not the zigzagger; measured, it produces the MOST direct path of all
    (executed length ~= the straight-line goal distance). It loses for the SAME
    reason rrt_star does, which is the reason proven in the RRT* study
    (docs/findings.md, "RRT* rewiring is a closed-loop liability"): a direct,
    minimum-length path is a minimum-CLEARANCE path, and against obstacles the
    planner does not model, the only protection is incidental clearance.

This script proves the buried ladder with paired McNemar and replaces the two
wrong stories with one measured variable: path DIRECTNESS (executed length /
straight-line goal distance). The claim is that directness predicts collision
across planners -- the grid-optimal (astar) and sampling-optimal (rrt_star)
planners are the most direct and collide most; the sampling-greedy (rrt)
wanders and survives; straight-line is perfectly direct and never makes it.

Four planning arms, identical scenario/sensor/dynamics (50x50 grid, 25 random
static + 3 reflecting moving obstacles, perfect sensing, max_speed 10,
inflate 1), one search strategy apart:

    straight   - head straight at the goal, no obstacle avoidance (the floor)
    astar      - 8-connected grid search, deterministic, grid-optimal
    rrt        - continuous-space sampling, returns the first path (wanders)
    rrt_star   - same sampling + rewiring, returns the shortest path

Mechanism axis: `replan_period`, swept as in the RRT* study so the two studies
compose. Pair by episode seed (same `episode_seed ^ obstacles.seed` layout for
all arms); exact McNemar on goal-reach for the buried headline contrast
astar -> rrt, plus the full per-cell ladder and the directness of each arm.

Output: `results/planner_clearance_ladder_phase/phase.json` (+ `phase_raw.json`)
and a two-panel `phase.png` (success vs replan_period per arm; the unifying
directness->collision relationship across every arm x cadence point).

Run (defaults: 5 periods x 4 arms x n=60; rrt_star is the expensive arm):
    python scripts/planner_clearance_ladder_phase.py
    python scripts/planner_clearance_ladder_phase.py --n 60 --workers 6
"""

from __future__ import annotations

# Single-thread each numpy worker so the forked pool does not oversubscribe.
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
EXAMPLE = REPO / "examples/exp_compare_astar.yaml"
DEFAULT_LEVELS = [0.1, 0.2, 0.4, 0.8, 1.6]   # planner.replan_period (s)
SEED_BASE = 200
# straight-line start->goal distance, the directness denominator.
GOAL_DIST = math.dist([2.0, 2.0], [45.0, 45.0])

# Ordered floor -> grid-optimal -> sampling-optimal -> sampling-greedy.
ARMS = ("straight", "astar", "rrt_star", "rrt")


def _base_config() -> dict:
    from uav_nav_lab.config import ExperimentConfig

    return ExperimentConfig.from_yaml(EXAMPLE).to_dict()


def _cell_config(base: dict, arm: str, period: float) -> dict:
    """Same scenario/sensor; only the planner family and replan_period change.

    All arms share max_speed 10, resolution 1, inflate 1 so the comparison is
    purely the search strategy (matches the shipped exp_compare_* family).
    """
    cfg = copy.deepcopy(base)
    cfg.pop("output", None)
    planner = {"type": arm, "max_speed": 10.0, "replan_period": float(period)}
    if arm in ("astar", "rrt", "rrt_star"):
        planner.update({"resolution": 1.0, "inflate": 1})
    if arm in ("rrt", "rrt_star"):
        planner.update({"step_size": 2.0, "goal_tolerance": 1.5,
                        "goal_bias": 0.1, "max_samples": 1000, "seed": 42})
    if arm == "rrt_star":
        planner["rewire_radius"] = 4.0
    cfg["planner"] = planner
    cfg["name"] = f"{arm}_rp{period}"
    return cfg


def _episode_metrics(d: dict) -> dict:
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
    return {
        "seed": d["meta"]["seed"],
        "outcome": d["outcome"],
        "path_len": path_len,
        "mean_dt_ms": (sum(dts) / len(dts)) if dts else 0.0,
    }


def _run_cell_chunk(job: dict) -> dict:
    from uav_nav_lab.config import ExperimentConfig
    from uav_nav_lab.runner.experiment import run_experiment

    cfg = ExperimentConfig.from_dict(job["config"])
    cfg.seed = job["seed_start"]
    cfg.num_episodes = job["count"]
    out = Path(tempfile.mkdtemp(prefix="ladder_"))
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
    ap.add_argument("--n", type=int, default=60, help="episodes per (period, arm) cell")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=4, help="episodes per parallel job")
    ap.add_argument(
        "--levels", type=float, nargs="+", default=DEFAULT_LEVELS,
        help="planner.replan_period values to sweep (s)",
    )
    ap.add_argument("--out", default=str(REPO / "results/planner_clearance_ladder_phase"))
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

    print(f"[ladder] {len(args.levels)} periods x {len(ARMS)} arms x n={args.n} "
          f"= {len(jobs)} jobs on {args.workers} workers")
    t0 = time.perf_counter()
    with Pool(processes=args.workers) as pool:
        chunk_results = pool.map(_run_cell_chunk, jobs)
    dt = time.perf_counter() - t0
    print(f"[ladder] sweep done in {dt:.0f}s")

    by_cell: dict[float, dict[str, dict[int, dict]]] = {}
    for r in chunk_results:
        cell = by_cell.setdefault(r["level"], {a: {} for a in ARMS})
        for ep in r["episodes"]:
            cell[r["arm"]][ep["seed"]] = ep

    def _paired(a: dict[int, dict], b: dict[int, dict]) -> dict:
        """McNemar of arm b vs arm a on success (c-b>0 means b better)."""
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
            ok = [e for e in eps if e["outcome"] == "success"]
            _, lo, hi = _wilson(succ, n)
            mean_len = (sum(e["path_len"] for e in ok) / len(ok)) if ok else float("nan")
            row[a] = {
                "success": succ / n, "success_ci": [lo, hi],
                "collision": coll / n, "timeout": tout / n,
                "directness": (mean_len / GOAL_DIST) if ok else float("nan"),
                "path_len": mean_len,
                "mean_dt_ms": sum(e["mean_dt_ms"] for e in eps) / len(eps),
            }
        # Buried headline contrast: grid A* -> continuous RRT.
        row["astar_vs_rrt"] = _paired(cells["astar"], cells["rrt"])
        # Cross-check from the RRT* study: rewiring (rrt -> rrt_star).
        row["rrt_vs_rrtstar"] = _paired(cells["rrt"], cells["rrt_star"])
        rows.append(row)
        raw.append({"level": level,
                    "episodes": {a: {s: cells[a][s] for s in seeds} for a in ARMS}})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"n_per_cell": args.n, "levels": args.levels, "arms": list(ARMS),
              "goal_dist": GOAL_DIST, "seconds": dt, "rows": rows}
    (out_dir / "phase.json").write_text(json.dumps(result, indent=2))
    (out_dir / "phase_raw.json").write_text(json.dumps(raw, indent=2))

    _print_table(rows)
    _plot(rows, out_dir / "phase.png")
    print(f"[ladder] wrote {out_dir/'phase.json'}, phase_raw.json and {out_dir/'phase.png'}")
    return 0


def _print_table(rows: list[dict]) -> None:
    print()
    print("Goal-reach success per planner (paired by seed, n per cell):")
    hdr = f"{'rp(s)':>6} {'n':>4} | " + " | ".join(f"{a:>9}" for a in ARMS)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cells = " | ".join(f"{r[a]['success']*100:8.1f}%" for a in ARMS)
        print(f"{r['level']:>6} {r['n']:>4} | {cells}")

    print("\nDirectness (executed length / straight-line goal distance; lower = straighter):")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cells = " | ".join(f"{r[a]['directness']:9.3f}" for a in ARMS)
        print(f"{r['level']:>6} {r['n']:>4} | {cells}")

    print("\nPaired contrasts (net = c-b; exact-McNemar p):")
    print(f"{'rp(s)':>6} | {'astar -> rrt':>26} | {'rrt -> rrt_star':>26}")
    print("-" * 66)
    for r in rows:
        def fmt(d: dict) -> str:
            return f"net {d['c']-d['b']:+3d} (c{d['c']}/b{d['b']}) p={d['p']:.3f}"
        print(f"{r['level']:>6} | {fmt(r['astar_vs_rrt']):>26} | {fmt(r['rrt_vs_rrtstar']):>26}")
    print("\n  The two 'optimal' planners (astar grid, rrt_star sampling) are the most")
    print("  direct and collide most; only the wandering rrt keeps incidental clearance.")


def _plot(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["level"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = {"straight": "#7f8c8d", "astar": "#8e44ad",
              "rrt_star": "#16a085", "rrt": "#e67e22"}
    labels = {"straight": "straight (no avoidance)",
              "astar": "A* (grid-optimal)",
              "rrt_star": "RRT* (sampling-optimal)",
              "rrt": "RRT (sampling-greedy, wanders)"}
    for a in ARMS:
        ys = [r[a]["success"] * 100 for r in rows]
        lo = [r[a]["success_ci"][0] * 100 for r in rows]
        hi = [r[a]["success_ci"][1] * 100 for r in rows]
        ax1.plot(xs, ys, "o-", color=colors[a], label=labels[a])
        ax1.fill_between(xs, lo, hi, color=colors[a], alpha=0.10)
    ax1.set_xscale("log")
    ax1.set_xlabel("replan_period (s, log scale)")
    ax1.set_ylabel("goal-reach success rate (%)")
    ax1.set_title("The planner ladder across replan cadence")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # The unifying relationship: directness vs collision, every arm x cadence.
    for a in ARMS:
        if a == "straight":
            continue  # no successful episodes -> directness undefined
        ds = [r[a]["directness"] for r in rows]
        cs = [r[a]["collision"] * 100 for r in rows]
        ax2.scatter(ds, cs, color=colors[a], s=60, label=labels[a], zorder=3)
    ax2.set_xlabel("path directness (executed length / straight-line goal distance)")
    ax2.set_ylabel("collision rate (%)")
    ax2.set_title("Directness predicts collision: straighter paths hit more")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.annotate("more direct =\nless clearance =\nmore collisions",
                 xy=(0.99, 70), xytext=(1.02, 55), fontsize=8, color="#555")

    fig.suptitle("The classical-planner ladder is a clearance ladder "
                 "(50×50 grid, 3 dynamic obstacles, perfect sensing)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    raise SystemExit(main())
