#!/usr/bin/env python3
"""Perception ablation: what does a sensor's FIELD OF VIEW cost, vs its range?

`examples/exp_ablate_sensor_{pointcloud,depth}.yaml` ship a striking
single-config measurement buried in a YAML header: on a 50x50 random-obstacle
grid with A*, an omnidirectional point-cloud sensor (8 m LiDAR) reaches the goal
93.3 % of the time while a forward-facing 90 deg depth camera at essentially the
same per-replan compute reaches it only 63.3 % — a 30 pp gap attributed to the
camera's blind spots. That number was never run through a paired significance
test, replicated, or decomposed. This script does all three, and adds the
mechanism axis the single point cannot show.

Three sensing arms, identical scenario/planner, only the perception differs:

    perfect     — full obstacle knowledge (the ceiling; no perception limit)
    pointcloud  — omni LiDAR, range-limited to 8 m, 360 deg coverage
    depth       — forward pinhole depth camera, 90 deg FOV, 8 m range

The two paired contrasts decompose the perception cost cleanly:

    perfect  -> pointcloud   = the RANGE cost (can't see past 8 m, but all around)
    pointcloud -> depth      = the FOV cost  (same range, but blind outside 90 deg)

Both planners build an occupancy map with `memory: true`, so a cell once seen
stays known; the depth arm's loss is therefore specifically the obstacles that
were never surfaced because they sat outside the forward cone as the drone moved.

Mechanism axis: obstacle DENSITY. A single density cannot tell whether the FOV
gap is incidental or structural. Sweeping the random-obstacle count shows
whether the FOV cost GROWS with clutter (more obstacles fall into the blind
spot) while the omni arm degrades only slowly (it is merely range-limited).

Pairing: `grid_world.reseed(seed)` derives each episode's layout from
`episode_seed ^ obstacles.seed`, so a given seed is the SAME 25/50/... obstacle
map for all three arms. We pair by seed and run an exact McNemar test on the
success outcome at each density.

Output: `results/sensor_fov_density_phase/phase.json` (+ `phase_raw.json`) and a
two-panel `phase.png` (success vs density per arm with Wilson bands; paired
range-cost vs FOV-cost decomposition with significance markers).

Run (defaults: 5 densities x 3 arms x n=80, A* is cheap; minutes on a quiet box):
    python scripts/sensor_fov_density_phase.py
    python scripts/sensor_fov_density_phase.py --n 60 --workers 6
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
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXAMPLE = REPO / "examples/exp_ablate_sensor_depth.yaml"
DEFAULT_LEVELS = [15, 30, 50, 75, 100]   # scenario.obstacles.count
SEED_BASE = 200
RANGE_M = 8.0
FOV_DEG = 90

# perfect is the ceiling; the two ordered contrasts decompose the cost.
ARMS = ("perfect", "pointcloud", "depth")


def _base_config() -> dict:
    from uav_nav_lab.config import ExperimentConfig

    return ExperimentConfig.from_yaml(EXAMPLE).to_dict()


def _cell_config(base: dict, arm: str, count: int) -> dict:
    """Scenario with `count` random obstacles and the perception arm selected.

    Every arm keeps the identical scenario/planner; only the simulator's
    synthetic-perception payload and the sensor that consumes it change.
    """
    cfg = copy.deepcopy(base)
    cfg.pop("output", None)
    cfg["scenario"]["obstacles"]["count"] = int(count)

    if arm == "perfect":
        cfg["simulator"].pop("synthetic_perception", None)
        cfg["sensor"] = {"type": "perfect"}
    elif arm == "pointcloud":
        cfg["simulator"]["synthetic_perception"] = {"lidar_range": RANGE_M}
        cfg["sensor"] = {
            "type": "pointcloud_occupancy", "resolution": 1.0,
            "memory": True, "inflate": 0, "range_m": RANGE_M,
        }
    elif arm == "depth":
        cfg["simulator"]["synthetic_perception"] = {
            "depth": {"fov_deg": FOV_DEG, "width": 64, "height": 48,
                      "max_depth": RANGE_M},
        }
        cfg["sensor"] = {
            "type": "depth_image_occupancy", "resolution": 1.0,
            "memory": True, "inflate": 0, "stride": 1, "max_depth": RANGE_M,
        }
    else:
        raise ValueError(f"unknown arm {arm!r}")
    cfg["name"] = f"{arm}_c{count}"
    return cfg


def _run_cell_chunk(job: dict) -> dict:
    """Worker: run `count` episodes from `seed_start` for one cell config."""
    from uav_nav_lab.config import ExperimentConfig
    from uav_nav_lab.runner.experiment import run_experiment

    cfg = ExperimentConfig.from_dict(job["config"])
    cfg.seed = job["seed_start"]
    cfg.num_episodes = job["count"]
    out = Path(tempfile.mkdtemp(prefix="fov_"))
    run_experiment(cfg, out)
    episodes = []
    for p in sorted(out.glob("episode_*.json")):
        d = json.loads(p.read_text())
        episodes.append({"seed": d["meta"]["seed"], "outcome": d["outcome"]})
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
    ap.add_argument("--n", type=int, default=80, help="episodes per (density, arm) cell")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=5, help="episodes per parallel job")
    ap.add_argument(
        "--levels", type=int, nargs="+", default=DEFAULT_LEVELS,
        help="random-obstacle counts to sweep",
    )
    ap.add_argument("--out", default=str(REPO / "results/sensor_fov_density_phase"))
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

    print(f"[fov] {len(args.levels)} densities x {len(ARMS)} arms x n={args.n} "
          f"= {len(jobs)} jobs on {args.workers} workers")
    t0 = time.perf_counter()
    with Pool(processes=args.workers) as pool:
        chunk_results = pool.map(_run_cell_chunk, jobs)
    dt = time.perf_counter() - t0
    print(f"[fov] sweep done in {dt:.0f}s")

    by_cell: dict[int, dict[str, dict[int, str]]] = {}
    for r in chunk_results:
        cell = by_cell.setdefault(r["level"], {a: {} for a in ARMS})
        for ep in r["episodes"]:
            cell[r["arm"]][ep["seed"]] = ep["outcome"]

    def _paired(a: dict[int, str], b: dict[int, str]) -> dict:
        """McNemar of arm b vs arm a on success. c = b succeeds where a fails,
        b = b fails where a succeeds (so c-b > 0 means b is better)."""
        seeds = sorted(set(a) & set(b))
        ba = sum(a[s] == "success" and b[s] != "success" for s in seeds)
        ca = sum(a[s] != "success" and b[s] == "success" for s in seeds)
        return {"b": ba, "c": ca, "p": mcnemar_exact_p(ba, ca)}

    rows, raw = [], []
    for level in sorted(by_cell):
        cells = by_cell[level]
        seeds = sorted(set.intersection(*[set(cells[a]) for a in ARMS]))
        n = len(seeds)
        row: dict = {"level": level, "n": n}
        for a in ARMS:
            succ = sum(cells[a][s] == "success" for s in seeds)
            coll = sum(cells[a][s] == "collision" for s in seeds)
            tout = sum(cells[a][s] == "timeout" for s in seeds)
            _, lo, hi = _wilson(succ, n)
            row[a] = {"success": succ / n, "success_ci": [lo, hi],
                      "collision": coll / n, "timeout": tout / n}
        # Decompose: range cost (perfect->pointcloud) and FOV cost (pointcloud->depth).
        # _paired(better, worse) gives c-b < 0 (the worse arm loses net successes).
        row["range_cost"] = _paired(cells["perfect"], cells["pointcloud"])
        row["fov_cost"] = _paired(cells["pointcloud"], cells["depth"])
        rows.append(row)
        raw.append({"level": level, "outcomes": {a: cells[a] for a in ARMS}})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"n_per_cell": args.n, "levels": args.levels, "arms": list(ARMS),
              "range_m": RANGE_M, "fov_deg": FOV_DEG, "seconds": dt, "rows": rows}
    (out_dir / "phase.json").write_text(json.dumps(result, indent=2))
    (out_dir / "phase_raw.json").write_text(json.dumps(raw, indent=2))

    _print_table(rows)
    _plot(rows, out_dir / "phase.png")
    print(f"[fov] wrote {out_dir/'phase.json'}, phase_raw.json and {out_dir/'phase.png'}")
    return 0


def _print_table(rows: list[dict]) -> None:
    print()
    print("Success rate per sensing arm (paired by seed, n per cell):")
    print(f"{'count':>6} {'n':>4} | {'perfect':>9} | {'omni-pc':>9} | {'depth-fwd':>9}")
    print("-" * 50)
    for r in rows:
        def cell(a: str) -> str:
            return f"{r[a]['success']*100:6.1f}%"
        print(f"{r['level']:>6} {r['n']:>4} | {cell('perfect'):>9} | "
              f"{cell('pointcloud'):>9} | {cell('depth'):>9}")

    print("\nPaired success decomposition (net = c-b; exact-McNemar p):")
    print(f"{'count':>6} | {'RANGE cost (perfect->omni)':>28} | {'FOV cost (omni->depth)':>26}")
    print("-" * 70)
    for r in rows:
        def fmt(d: dict) -> str:
            return f"net {d['c']-d['b']:+3d} (c{d['c']}/b{d['b']}) p={d['p']:.3f}"
        print(f"{r['level']:>6} | {fmt(r['range_cost']):>28} | {fmt(r['fov_cost']):>26}")
    print("\n  RANGE cost isolates omni LiDAR's 8 m horizon; FOV cost isolates the")
    print("  forward camera's blind spots at the SAME range.")


def _plot(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    xs = [r["level"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = {"perfect": "#777", "pointcloud": "#2980b9", "depth": "#c0392b"}
    labels = {"perfect": "perfect (full knowledge)",
              "pointcloud": "omni LiDAR (8 m, 360°)",
              "depth": "depth cam (8 m, 90° FOV)"}
    for a in ARMS:
        ys = [r[a]["success"] * 100 for r in rows]
        lo = [r[a]["success_ci"][0] * 100 for r in rows]
        hi = [r[a]["success_ci"][1] * 100 for r in rows]
        ax1.plot(xs, ys, "o-", color=colors[a], label=labels[a])
        ax1.fill_between(xs, lo, hi, color=colors[a], alpha=0.12)
    ax1.set_xlabel("random-obstacle count (50×50 grid)")
    ax1.set_ylabel("goal-reach success rate (%)")
    ax1.set_title("Success vs obstacle density per sensing arm")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    width = 0.38
    x = np.arange(len(rows))
    rng = [-(r["range_cost"]["c"] - r["range_cost"]["b"]) for r in rows]  # losses positive
    fov = [-(r["fov_cost"]["c"] - r["fov_cost"]["b"]) for r in rows]
    b1 = ax2.bar(x - width / 2, rng, width, color="#2980b9",
                 label="RANGE cost (perfect → omni)")
    b2 = ax2.bar(x + width / 2, fov, width, color="#c0392b",
                 label="FOV cost (omni → depth)")
    for r, bar in zip(rows, b1):
        if r["range_cost"]["p"] < 0.05:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "*",
                     ha="center", va="bottom", fontsize=12)
    for r, bar in zip(rows, b2):
        if r["fov_cost"]["p"] < 0.05:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "*",
                     ha="center", va="bottom", fontsize=12)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(r["level"]) for r in rows])
    ax2.set_xlabel("random-obstacle count")
    ax2.set_ylabel("net successes LOST vs the arm above (paired b-c)")
    ax2.set_title("Decomposition: range cost vs FOV cost (* = McNemar p<0.05)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("What a sensor's field of view costs navigation "
                 "(50×50 random grid, A*, memory occupancy)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    raise SystemExit(main())
