"""Paired dummy_3d multi-drone analysis: MPC vs GPU MPPI.

Counterpart to paired_analysis_airsim_multi.py, but loads from the
non-chunked dummy_3d output layout:
    <run_dir>/episode_NNN_joint.json
instead of
    <run_dir>/seed_NNN/episode_000_joint.json

Prints Wilson CI per planner, indep^N + Δ, McNemar joint comparison.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def load_dir(run_dir: Path) -> list[dict]:
    out: list[dict] = []
    for jp in sorted(run_dir.glob("episode_*_joint.json")):
        d = json.loads(jp.read_text())
        out.append({
            "seed": d["meta"]["seed"],
            "joint": d["outcome"] == "success",
            "per_drone": [o == "success" for o in d["per_drone_outcomes"]],
            "final_t": d.get("final_t"),
        })
    return sorted(out, key=lambda r: r["seed"])


def binom_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_one_side = sum(binom_pmf(i, n, 0.5) for i in range(k + 1))
    return min(1.0, 2.0 * p_one_side)


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
    a = load_dir(Path(mpc_dir))
    b = load_dir(Path(mppi_dir))
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
