"""Paired n=30 AirSim multi-drone analysis: MPC vs GPU MPPI.

Loads episode JSONs from chunked-output dirs (one seed_NNN subdir per
episode, each with episode_000_*.json). Prints Wilson CI per planner,
McNemar joint-success comparison, per-seed disagreement detail.
"""
from __future__ import annotations
import sys
from pathlib import Path

from uav_nav_lab.analysis import load_joint_episodes, mcnemar_exact_p, wilson


def main(mpc_dir: str, mppi_dir: str) -> int:
    a = load_joint_episodes(Path(mpc_dir), layout="chunked")
    b = load_joint_episodes(Path(mppi_dir), layout="chunked")
    seeds_a = {r["seed"] for r in a}
    seeds_b = {r["seed"] for r in b}
    common = seeds_a & seeds_b
    a = [r for r in a if r["seed"] in common]
    b = [r for r in b if r["seed"] in common]
    n = len(a)
    print(f"n = {n} paired episodes (seeds {a[0]['seed']}..{a[-1]['seed']})\n")

    for name, data in [("MPC", a), ("GPU MPPI", b)]:
        pd_succ = sum(d for ep in data for d in ep["per_drone"])
        pd_tot = 4 * len(data)
        j_succ = sum(ep["joint"] for ep in data)
        j_tot = len(data)
        per = wilson(pd_succ, pd_tot)
        jnt = wilson(j_succ, j_tot)
        indep4 = per[0] ** 4
        delta = jnt[0] - indep4
        ts = [ep["final_t"] for ep in data if ep["final_t"] is not None and ep["joint"]]
        avg_t = sum(ts) / len(ts) if ts else float("nan")
        print(f"{name}:")
        print(f"  per-drone : {pd_succ}/{pd_tot} = {per[0]*100:5.1f}% [{per[1]*100:5.1f}, {per[2]*100:5.1f}]")
        print(f"  joint     : {j_succ}/{j_tot} = {jnt[0]*100:5.1f}% [{jnt[1]*100:5.1f}, {jnt[2]*100:5.1f}]")
        print(f"  indep^4   : {indep4*100:5.1f}%   Δ = {delta*100:+.1f} pp")
        print(f"  final_t   : mean {avg_t:.2f}s over {len(ts)} successes")
        print()

    both = sum(1 for x, y in zip(a, b) if x["joint"] and y["joint"])
    only_a = sum(1 for x, y in zip(a, b) if x["joint"] and not y["joint"])
    only_b = sum(1 for x, y in zip(a, b) if not x["joint"] and y["joint"])
    neither = sum(1 for x, y in zip(a, b) if not x["joint"] and not y["joint"])
    print(f"McNemar paired-seed joint:")
    print(f"  both succ      : {both}")
    print(f"  MPC-only succ  : {only_a}")
    print(f"  GPU-only succ  : {only_b}")
    print(f"  neither succ   : {neither}")
    if only_a + only_b > 0:
        p_val = mcnemar_exact_p(only_a, only_b)
        print(f"  exact McNemar p ≈ {p_val:.3f}")
    print()

    diffs = [(x, y) for x, y in zip(a, b) if x["joint"] != y["joint"]]
    if diffs:
        print("Per-seed disagreement:")
        for x, y in diffs:
            xp = "".join("S" if s else "X" for s in x["per_drone"])
            yp = "".join("S" if s else "X" for s in y["per_drone"])
            print(f"  seed {x['seed']:3d}: MPC[{xp}]={'S' if x['joint'] else 'X'}  "
                  f"GPU[{yp}]={'S' if y['joint'] else 'X'}")
    else:
        print("(no per-seed disagreements)")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
