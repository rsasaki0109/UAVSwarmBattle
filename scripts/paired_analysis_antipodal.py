"""Paired seed McNemar for the AirSim N=4 antipodal convention study.

Two chunked run dirs (seed_NNN/episode_000_joint.json each): the STOCK arm
(lateral_bias=0) and the CONVENTION arm (lateral_bias>0), run over the same
base seed so seed s is paired across arms. Prints Wilson CIs, the joint-
success McNemar, and per-seed disagreement.

  python scripts/paired_analysis_antipodal.py <stock_dir> <conv_dir> [layout]

`layout` is "chunked" (default, seed_NNN/ subdirs) or "flat"
(episode_NNN_joint.json in one dir).
"""
from __future__ import annotations

import sys
from pathlib import Path

from uav_nav_lab.analysis import load_joint_episodes, mcnemar_exact_p, wilson


def main(stock_dir: str, conv_dir: str, layout: str = "chunked") -> int:
    a = load_joint_episodes(Path(stock_dir), layout=layout)
    b = load_joint_episodes(Path(conv_dir), layout=layout)
    by_a = {r["seed"]: r for r in a}
    by_b = {r["seed"]: r for r in b}
    common = sorted(set(by_a) & set(by_b))
    a = [by_a[s] for s in common]
    b = [by_b[s] for s in common]
    n = len(common)
    nd = len(a[0]["per_drone"]) if a else 0
    print(f"n = {n} paired episodes (seeds {common[0]}..{common[-1]}), {nd} drones\n")

    for name, data in [("STOCK (bias=0)", a), ("CONVENTION (bias>0)", b)]:
        pd_succ = sum(d for ep in data for d in ep["per_drone"])
        pd_tot = nd * len(data)
        j_succ = sum(ep["joint"] for ep in data)
        per = wilson(pd_succ, pd_tot)
        jnt = wilson(j_succ, len(data))
        print(f"{name}:")
        print(f"  per-drone : {pd_succ}/{pd_tot} = {per[0]*100:5.1f}% [{per[1]*100:5.1f}, {per[2]*100:5.1f}]")
        print(f"  joint     : {j_succ}/{len(data)} = {jnt[0]*100:5.1f}% [{jnt[1]*100:5.1f}, {jnt[2]*100:5.1f}]")
        print()

    both = sum(1 for x, y in zip(a, b) if x["joint"] and y["joint"])
    only_stock = sum(1 for x, y in zip(a, b) if x["joint"] and not y["joint"])
    only_conv = sum(1 for x, y in zip(a, b) if not x["joint"] and y["joint"])
    neither = sum(1 for x, y in zip(a, b) if not x["joint"] and not y["joint"])
    print("McNemar paired-seed joint success (convention vs stock):")
    print(f"  both succeed       : {both}")
    print(f"  stock-only succeed : {only_stock}   (b)")
    print(f"  conv-only  succeed : {only_conv}   (c)")
    print(f"  neither succeed    : {neither}")
    p_val = mcnemar_exact_p(only_stock, only_conv)
    print(f"  exact McNemar p    : {p_val:.3e}   (c-b = {only_conv - only_stock:+d}, convention better if >0)")
    print()

    diffs = [(x, y) for x, y in zip(a, b) if x["joint"] != y["joint"]]
    if diffs:
        print("Per-seed disagreement (S=success per drone):")
        for x, y in diffs:
            xp = "".join("S" if s else "X" for s in x["per_drone"])
            yp = "".join("S" if s else "X" for s in y["per_drone"])
            print(f"  seed {x['seed']:3d}: stock[{xp}]={'S' if x['joint'] else 'X'}  "
                  f"conv[{yp}]={'S' if y['joint'] else 'X'}")
    else:
        print("(no per-seed disagreements — deterministic split)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2],
                  sys.argv[3] if len(sys.argv) > 3 else "chunked"))
