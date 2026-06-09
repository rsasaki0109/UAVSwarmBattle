#!/usr/bin/env python3
"""Head-to-head swarm policy battle on antipodal YAML geometry.

Runs registered planners through the same framework multi-drone runner on:
  peers    — 50×50 antipodal, no obstacle
  obstacle — hub-crossing dynamic threat

  python scripts/swarm_policy_battle_phase.py
  python scripts/swarm_policy_battle_phase.py --scenario obstacle --episodes 20 --workers 4 --merge
  python scripts/swarm_policy_battle_phase.py --arms orca_conv mpc_gt swarm_transformer
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p, wilson  # noqa: E402
from uav_nav_lab.config import ExperimentConfig  # noqa: E402
from uav_nav_lab.runner.multi.experiment import run_experiment_multi  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "results/swarm_transformer_framework_obstacle_rl_best.npz"

DRONES = [
    {"name": "d0", "start": [45.0, 25.0], "goal": [5.0, 25.0], "radius": 0.4, "start_jitter": 0.8},
    {"name": "d1", "start": [35.0, 42.32], "goal": [15.0, 7.68], "radius": 0.4, "start_jitter": 0.8},
    {"name": "d2", "start": [15.0, 42.32], "goal": [35.0, 7.68], "radius": 0.4, "start_jitter": 0.8},
    {"name": "d3", "start": [5.0, 25.0], "goal": [45.0, 25.0], "radius": 0.4, "start_jitter": 0.8},
    {"name": "d4", "start": [15.0, 7.68], "goal": [35.0, 42.32], "radius": 0.4, "start_jitter": 0.8},
    {"name": "d5", "start": [35.0, 7.68], "goal": [15.0, 42.32], "radius": 0.4, "start_jitter": 0.8},
]
# Triple crossfire through the hub (battle / README GIF).  Single-missile baseline
# for training + champion eval: examples/exp_multi_drone_antipodal_obstacle*.yaml
# (south-edge vertical only — swarm_transformer 20/20, mpc_gt 12/20 on n=20).
DYN_OBS = [
    {"start": [25.0, 2.0], "velocity": [0.0, 4.5], "radius": 1.5, "reflect": True},
    {"start": [2.0, 25.0], "velocity": [4.2, 0.0], "radius": 1.5, "reflect": True},
    {"start": [48.0, 25.0], "velocity": [-4.2, 0.0], "radius": 1.5, "reflect": True},
]

ALL_ARMS = ("orca", "orca_conv", "hrvo", "mgr", "mpc_gt", "swarm_transformer", "navrl")
ARM_META = {
    "orca": {"paper": "ORCA 2011 / RVO2", "oss": "snape/RVO2"},
    "orca_conv": {"paper": "ORCA + lateral_bias (lab)", "oss": "in-repo orca"},
    "hrvo": {"paper": "HRVO side-commitment (lab)", "oss": "in-repo hrvo"},
    "mgr": {"paper": "Merry-Go-Round arXiv:2503.05848", "oss": "in-repo mgr"},
    "mpc_gt": {"paper": "MPC + game_theoretic (lab)", "oss": "in-repo mpc"},
    "swarm_transformer": {"paper": "TeamHOI-style tokens (lab)", "oss": "in-repo swarm_transformer"},
    "navrl": {"paper": "NavRL RA-L 2025", "oss": "Zhefan-Xu/NavRL"},
}


def _orca_base(*, lateral_bias: float = 0.0) -> dict:
    p = {
        "type": "orca",
        "max_speed": 5.0,
        "replan_period": 0.2,
        "radius": 0.4,
        "time_horizon": 2.0,
        "time_step": 0.25,
        "neighbor_dist": 15.0,
        "safety_margin": 0.1,
        "goal_radius": 1.5,
    }
    if lateral_bias:
        p["lateral_bias"] = lateral_bias
    return p


def _planner(arm: str, scenario: str) -> dict:
    bias = 4.0 if scenario == "obstacle" else 0.2
    if arm == "orca":
        return _orca_base()
    if arm == "orca_conv":
        return _orca_base(lateral_bias=bias)
    if arm == "hrvo":
        return {**_orca_base(), "type": "hrvo"}
    if arm == "mgr":
        return {
            "type": "mgr",
            "max_speed": 5.0,
            "replan_period": 0.2,
            "radius": 0.4,
            "alpha": 2.0,
            "time_step": 0.1,
            "neighbor_dist": 15.0,
            "safety_margin": 0.1,
            "goal_radius": 1.5,
        }
    if arm == "mpc_gt":
        return {
            "type": "mpc",
            "max_speed": 5.0,
            "replan_period": 0.2,
            "horizon": 40,
            "dt_plan": 0.05,
            "n_samples": 32,
            "resolution": 1.0,
            "inflate": 1,
            "goal_radius": 1.5,
            "safety_margin": 0.5,
            "use_prediction": True,
            "w_goal": 1.0,
            "w_obs": 100.0,
            "w_smooth": 0.05,
            "lateral_bias": bias,
            "predictor": {"type": "game_theoretic"},
        }
    if arm == "swarm_transformer":
        if not CKPT.is_file():
            raise FileNotFoundError(f"missing checkpoint: {CKPT}")
        return {
            "type": "swarm_transformer",
            "max_speed": 5.0,
            "replan_period": 0.2,
            "neighbor_dist": 15.0,
            "interaction_radius": 4.0,
            "goal_radius": 1.5,
            "predictor": {"type": "game_theoretic"},
            "checkpoint": str(CKPT),
        }
    if arm == "navrl":
        root = ROOT / "third_party" / "NavRL"
        if not (root / "quick-demos" / "ckpts" / "navrl_checkpoint.pt").is_file():
            raise FileNotFoundError(
                f"NavRL checkpoint missing under {root}. "
                "Run: bash scripts/setup_navrl_adapter.sh"
            )
        return {
            "type": "navrl",
            "navrl_root": str(root),
            "max_speed": 5.0,
            "replan_period": 0.2,
            "goal_radius": 1.5,
            "lidar_range": 4.0,
            "device": "cpu",
        }
    raise ValueError(arm)


def _cfg(scenario: str, arm: str, *, seed: int, n_eps: int) -> dict:
    sc = {
        "type": "multi_drone_grid",
        "size": [50, 50],
        "resolution": 1.0,
        "obstacles": {"type": "none"},
        "drones": DRONES,
    }
    if scenario == "obstacle":
        sc["dynamic_obstacles"] = DYN_OBS
    return {
        "name": f"battle_{scenario}_{arm}",
        "seed": seed,
        "num_episodes": n_eps,
        "scenario": sc,
        "simulator": {
            "type": "dummy_2d",
            "dt": 0.05,
            "max_steps": 1000,
            "max_accel": 6.0,
            "goal_radius": 1.5,
            "drone_radius": 0.4,
        },
        "planner": _planner(arm, scenario),
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/swarm_policy_battle/_tmp"},
    }


def _run_cell(scenario: str, arm: str, seed: int, n_eps: int) -> dict[int, str]:
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(_cfg(scenario, arm, seed=seed, n_eps=n_eps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed: dict[int, str] = {}
        for jf in sorted(out.glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[int(d["meta"]["seed"])] = str(d["outcome"])
        return by_seed


def _run_cell_job(payload: tuple[str, str, int, int]) -> tuple[str, dict[int, str]]:
    scenario, arm, seed, n_eps = payload
    return arm, _run_cell(scenario, arm, seed, n_eps)


def _run_arms(
    scenario: str,
    arms: list[str],
    seed: int,
    n_eps: int,
    *,
    workers: int,
) -> dict[str, dict[int, str]]:
    jobs = [(scenario, arm, seed, n_eps) for arm in arms]
    if workers <= 1 or len(arms) <= 1:
        return {arm: _run_cell(scenario, arm, seed, n_eps) for arm in arms}
    results: dict[str, dict[int, str]] = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(arms))) as pool:
        futs = {pool.submit(_run_cell_job, j): j[1] for j in jobs}
        for fut in as_completed(futs):
            arm, by_seed = fut.result()
            results[arm] = by_seed
            print(f"  done {arm}", flush=True)
    return results


def _rate(outcomes: dict[int, str], seeds: list[int]) -> tuple[int, int]:
    ok = sum(outcomes[s] == "success" for s in seeds)
    return ok, len(seeds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=("peers", "obstacle", "both"), default="both")
    ap.add_argument("--arms", nargs="+", default=list(ALL_ARMS))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel arms per scenario (ProcessPool; navrl loads torch per worker)")
    ap.add_argument("--out", default="results/swarm_policy_battle/phase.json")
    ap.add_argument("--merge", action="store_true",
                    help="merge scenario cells into existing --out (keep other scenarios)")
    args = ap.parse_args()

    scenarios = ["peers", "obstacle"] if args.scenario == "both" else [args.scenario]
    for arm in args.arms:
        if arm not in ALL_ARMS:
            raise SystemExit(f"unknown arm {arm!r}; choose from {ALL_ARMS}")

    report: dict = {
        "episodes": args.episodes,
        "seed0": args.seed,
        "arms": {a: ARM_META[a] for a in args.arms},
        "cells": {},
    }

    for sc in scenarios:
        print(f"\n=== {sc} (seeds {args.seed}…{args.seed + args.episodes - 1}) ===")
        results: dict[str, dict[int, str]] = {}
        if args.workers <= 1:
            for arm in args.arms:
                print(f"  running {arm}...", flush=True)
                results[arm] = _run_cell(sc, arm, args.seed, args.episodes)
        else:
            print(f"  running {len(args.arms)} arms with {min(args.workers, len(args.arms))} workers...", flush=True)
            results = _run_arms(sc, args.arms, args.seed, args.episodes, workers=args.workers)

        seeds = sorted(set.intersection(*[set(results[a]) for a in args.arms]))
        m = len(seeds)
        row: dict = {"scenario": sc, "m": m, "arms": {}, "outcomes_by_seed": {}}
        print(f"{'arm':>18} | joint     | Wilson 95% CI")
        print("-" * 52)
        ref = "swarm_transformer" if "swarm_transformer" in args.arms else args.arms[0]
        for arm in args.arms:
            ok, n = _rate(results[arm], seeds)
            p, lo, hi = wilson(ok, n)
            row["arms"][arm] = {"joint_ok": ok, "n": n, "rate": p, "wilson_lo": lo, "wilson_hi": hi}
            row["outcomes_by_seed"][arm] = {str(s): results[arm][s] for s in seeds}
            print(f"{arm:>18} | {ok:>2}/{n:<6} | [{lo:.2f}, {hi:.2f}]")

        if ref in results and len(args.arms) > 1:
            print(f"\nMcNemar vs {ref}:")
            for arm in args.arms:
                if arm == ref:
                    continue
                b = sum(results[arm][s] == "success" and results[ref][s] != "success" for s in seeds)
                c = sum(results[arm][s] != "success" and results[ref][s] == "success" for s in seeds)
                p = mcnemar_exact_p(b, c)
                row.setdefault("mcnemar_vs_ref", {})[arm] = {"b": b, "c": c, "p": p, "ref": ref}
                print(f"  {arm:>18} vs {ref}: b={b} c={c} p={p:.4f}")

        report["cells"][sc] = row

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.merge and out.is_file():
        prev = json.loads(out.read_text())
        prev.setdefault("cells", {})
        prev.setdefault("arms", {})
        prev["arms"].update(report["arms"])
        for sc, row in report["cells"].items():
            prev["cells"][sc] = row
        prev["episodes"] = max(prev.get("episodes", 0), report["episodes"])
        report = prev

    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
