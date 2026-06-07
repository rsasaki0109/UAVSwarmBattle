"""Does EVACUATING the hub (Merry-Go-Round) beat a PEER-RULE convention when an
external obstacle crosses the hub?

Two prior results frame this:
  * The right-of-way convention is a PEER rule. On the antipodal swap it funnels
    the whole fleet into one shared clockwise current that passes THROUGH the
    central hub; a scene obstacle crossing that hub caps success far below the
    obstacle-free ceiling, and no amount of extra bias buys it back — the bias
    rescues peer-deadlock but cannot pay for a non-peer threat sitting on the
    shared path (docs/findings.md "The right-of-way convention is a peer rule —
    a hub-crossing obstacle defeats the roundabout it builds").
  * The decentralized triggered Merry-Go-Round (`mgr`) builds a different
    geometry: on a detected deadlock each drone orbits the centroid of its local
    conflict cluster on a ring of radius >= ring_min, so the fleet circulates
    AROUND an EVACUATED centre disk rather than converging through it
    (docs/findings.md "A decentralized Merry-Go-Round negotiates its ring from
    sensing alone").

So MGR and the convention fail the peer deadlock at the SAME place (the hub) but
occupy it differently: the convention keeps a current running through the centre;
MGR vacates the centre and rings it. If the cap is really a *wrong-threat-on-the-
shared-path* cap, then the mechanism that vacates the contested space should clear
the hub-crossing obstacle that the peer rule cannot — i.e. the way to beat the cap
is to EVACUATE the contested space, not to add a stronger peer rule.

All three arms share the SAME CBF base avoider so the only variable is the
deadlock-breaking geometry (none / always-on peer convention / triggered ring):

  cbf           plain CBF                         (deadlocks the swap by timeout)
  cbf_pairwise  CBF + pairwise winding convention (the peer rule with the cap)
  mgr           CBF + triggered Merry-Go-Round    (vacates + rings the hub)

crossed with obstacle in {none, hub, far}, N in {6, 8}, paired by seed, McNemar:
  none  match check  — both deadlock-breakers should be ~100 %
  hub   HEADLINE     — mgr vs cbf_pairwise with a body crossing the hub
  far   move-the-stressor control — same obstacle in a corner, expect no effect

  python scripts/antipodal_mgr_obstacle_phase.py --n-list 6 8 --episodes 40
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import math
import tempfile
from multiprocessing import Pool
from pathlib import Path

from uav_nav_lab.config import ExperimentConfig
from uav_nav_lab.runner.multi.experiment import run_experiment_multi
from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p

SPEED = 5.0
CX, CY = 25.0, 25.0
RADIUS = 20.0
ARMS = ["cbf", "cbf_pairwise", "mgr"]

# A body that crosses the central hub vertically and reflects in the 50x50 box so
# it remains a sustained threat near the centre. "far" is the SAME obstacle moved
# to the left edge (x=5), a move-the-stressor control: same dynamics, off the hub.
OBSTACLES = {
    "none": None,
    "hub": {"start": [25.0, 2.0], "velocity": [0.0, 4.5], "radius": 1.5, "reflect": True},
    "far": {"start": [5.0, 2.0], "velocity": [0.0, 4.5], "radius": 1.5, "reflect": True},
}


def _cbf_base(replan):
    return {"type": "cbf", "max_speed": SPEED, "replan_period": replan,
            "radius": 0.4, "alpha": 2.0, "time_step": 0.1,
            "neighbor_dist": 15.0, "safety_margin": 0.1, "goal_radius": 1.5}


def _planner(arm, pb, replan):
    if arm == "cbf":
        return _cbf_base(replan)
    if arm == "cbf_pairwise":
        p = _cbf_base(replan); p["pairwise_bias"] = pb; p["pairwise_radius"] = 8.0
        return p
    if arm == "mgr":
        p = _cbf_base(replan); p["type"] = "mgr"
        return p
    raise ValueError(arm)


def _drones(n):
    out = []
    for k in range(n):
        a = 2.0 * math.pi * k / n
        out.append({"name": f"d{k}",
                    "start": [round(CX + RADIUS * math.cos(a), 3),
                              round(CY + RADIUS * math.sin(a), 3)],
                    "goal": [round(CX - RADIUS * math.cos(a), 3),
                             round(CY - RADIUS * math.sin(a), 3)],
                    "radius": 0.4, "start_jitter": 0.8})
    return out


def _cfg(n, arm, obs, pb, replan, seed, n_eps, max_steps):
    ob = OBSTACLES[obs]
    return {
        "name": f"mgrobs_n{n}_{arm}_{obs}", "seed": seed, "num_episodes": n_eps,
        "scenario": {"type": "multi_drone_grid", "size": [50, 50],
                     "resolution": 1.0, "obstacles": {"type": "none"},
                     "dynamic_obstacles": [dict(ob)] if ob else [],
                     "drones": _drones(n)},
        "simulator": {"type": "dummy_2d", "dt": 0.05, "max_steps": max_steps,
                      "max_accel": 6.0, "goal_radius": 1.5, "drone_radius": 0.4},
        "planner": _planner(arm, pb, replan),
        "sensor": {"type": "perfect"},
        "output": {"dir": "results/mgrobs_tmp"},
    }


def _run_cell(job):
    arm, n, obs, pb, replan, seed, n_eps, max_steps = job
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig.from_dict(
            _cfg(n, arm, obs, pb, replan, seed, n_eps, max_steps))
        out = run_experiment_multi(cfg, Path(td))
        by_seed = {}
        for jf in sorted(Path(out).glob("episode_*_joint.json")):
            d = json.loads(jf.read_text())
            by_seed[d["meta"]["seed"]] = d["outcome"]
    return ((arm, n, obs), by_seed)


def _succ(bs, seeds):
    return sum(bs[s] == "success" for s in seeds)


def _brk(bs, seeds):
    return (sum(bs[s] == "collision" for s in seeds),
            sum(bs[s] == "timeout" for s in seeds))


def _mc(a, b, seeds):
    # c-b>0 => b better than a
    bb = sum(a[s] == "success" and b[s] != "success" for s in seeds)
    cc = sum(a[s] != "success" and b[s] == "success" for s in seeds)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[6, 8])
    ap.add_argument("--obstacles", nargs="+", default=["none", "hub", "far"])
    ap.add_argument("--pairwise-bias", type=float, default=1.0)
    ap.add_argument("--replan", type=float, default=0.1)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--out", default="results/antipodal_mgr_obstacle_phase.json")
    args = ap.parse_args()

    jobs = [(arm, n, obs, args.pairwise_bias, args.replan, args.seed,
             args.episodes, args.max_steps)
            for n in args.n_list for obs in args.obstacles for arm in ARMS]
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_run_cell, jobs)
    cells = dict(res)

    seeds = sorted(set.intersection(*[set(bs) for bs in cells.values()]))
    report = {"pairwise_bias": args.pairwise_bias, "replan": args.replan,
              "episodes": args.episodes, "m": len(seeds), "cells": {}, "tests": {}}
    print(f"\nMGR vs peer-rule convention x hub-crossing obstacle "
          f"(CBF base, pb={args.pairwise_bias}, paired m={len(seeds)})")

    for n in args.n_list:
        print(f"\n=== N={n} ===")
        print(f"{'arm':>13} | " + " | ".join(f"{o:>11}" for o in args.obstacles))
        print("-" * (16 + 14 * len(args.obstacles)))
        for arm in ARMS:
            row = []
            for obs in args.obstacles:
                bs = cells[(arm, n, obs)]
                s = _succ(bs, seeds); co, to = _brk(bs, seeds)
                row.append(f"{s:>2}/{len(seeds):<2} {co}c{to}t")
                report["cells"][f"{arm}_n{n}_{obs}"] = {
                    "success": s, "collision": co, "timeout": to}
            print(f"{arm:>13} | " + " | ".join(f"{x:>11}" for x in row))

        # HEADLINE: does MGR beat the peer-rule convention under the hub obstacle?
        for obs in args.obstacles:
            b, c, p = _mc(cells[("cbf_pairwise", n, obs)], cells[("mgr", n, obs)], seeds)
            report["tests"][f"mgr_vs_pairwise_n{n}_{obs}"] = {"b": b, "c": c, "p": p}
            tag = "  <== HEADLINE" if obs == "hub" else ""
            print(f"   mgr vs cbf_pairwise [{obs:>4}]: b={b} c={c} p={p:.4f}"
                  f"  (c-b>0 => mgr better){tag}")
        # obstacle effect within each deadlock-breaker (hub vs none)
        if "none" in args.obstacles and "hub" in args.obstacles:
            for arm in ("cbf_pairwise", "mgr"):
                b, c, p = _mc(cells[(arm, n, "none")], cells[(arm, n, "hub")], seeds)
                report["tests"][f"{arm}_hub_effect_n{n}"] = {"b": b, "c": c, "p": p}
                print(f"   {arm} hub effect (none->hub): b={b} c={c} p={p:.4f}"
                      f"  (b-c>0 => obstacle hurts)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
