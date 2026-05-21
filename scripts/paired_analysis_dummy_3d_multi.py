"""Paired dummy_3d multi-drone analysis: MPC vs GPU MPPI.

Counterpart to paired_analysis_airsim_multi.py, but loads from the
non-chunked dummy_3d output layout:
    <run_dir>/episode_NNN_joint.json
instead of
    <run_dir>/seed_NNN/episode_000_joint.json

Prints Wilson CI per planner, indep^N + Δ, McNemar joint comparison.
"""
from __future__ import annotations
import sys
from pathlib import Path

from uav_nav_lab.analysis import load_joint_episodes, mcnemar_exact_p, wilson


def summarise(name: str, data: list[dict]) -> None:
    n = len(data)
    n_drones = len(data[0]["per_drone"]) if data else 0
    pd_succ = sum(d for ep in data for d in ep["per_drone"])
    pd_tot = n_drones * n
    pd_p, pd_lo, pd_hi = wilson(pd_succ, pd_tot)
    j_succ = sum(1 for ep in data if ep["joint"])
    j_p, j_lo, j_hi = wilson(j_succ, n)
    indep_n = pd_p ** n_drones
    delta = j_p - indep_n
    avg_t = sum(ep["final_t"] for ep in data) / n if n else 0.0
    print(f"{name}:")
    print(f"  per-drone {pd_succ}/{pd_tot} = {100*pd_p:.1f} % "
          f"[{100*pd_lo:.1f}, {100*pd_hi:.1f}]")
    print(f"  joint     {j_succ}/{n} = {100*j_p:.1f} % "
          f"[{100*j_lo:.1f}, {100*j_hi:.1f}]")
    print(f"  indep^{n_drones} = {100*indep_n:.1f} %    Δ over indep = {100*delta:+.1f} pp")
    print(f"  mean final_t = {avg_t:.2f} s")
    print()


def main(mpc_dir: str, mppi_dir: str) -> int:
    a = load_joint_episodes(Path(mpc_dir), layout="flat")
    b = load_joint_episodes(Path(mppi_dir), layout="flat")
    seeds_a = {r["seed"] for r in a}
    seeds_b = {r["seed"] for r in b}
    common = seeds_a & seeds_b
    a = [r for r in a if r["seed"] in common]
    b = [r for r in b if r["seed"] in common]
    n = len(a)
    if n == 0:
        print("no common seeds")
        return 1
    print(f"n = {n} paired episodes (seeds {a[0]['seed']}..{a[-1]['seed']})\n")
    summarise("MPC", a)
    summarise("GPU MPPI", b)

    by_seed_a = {r["seed"]: r["joint"] for r in a}
    by_seed_b = {r["seed"]: r["joint"] for r in b}
    both = mpc_only = mppi_only = neither = 0
    mpc_only_seeds: list[int] = []
    mppi_only_seeds: list[int] = []
    for s in sorted(common):
        ja, jb = by_seed_a[s], by_seed_b[s]
        if ja and jb:
            both += 1
        elif ja:
            mpc_only += 1
            mpc_only_seeds.append(s)
        elif jb:
            mppi_only += 1
            mppi_only_seeds.append(s)
        else:
            neither += 1
    p_val = mcnemar_exact_p(mpc_only, mppi_only)
    print("McNemar paired-seed joint:")
    print(f"  both-succ {both}, MPC-only {mpc_only}, GPU-only {mppi_only}, neither {neither}")
    print(f"  exact McNemar p ≈ {p_val:.4f}")
    if mpc_only_seeds:
        print(f"  MPC-only seeds: {mpc_only_seeds}")
    if mppi_only_seeds:
        print(f"  GPU-only seeds: {mppi_only_seeds}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
