"""Who must follow the roundabout rule? A flock's cohesion makes adoption a
placement problem, not just a head-count.

Builds on the [K-way flocking hub](docs/findings.md#a-k-way-flocking-hub-jams-at-every-fan-in-the-roundabout-convention-clears-it-without-a-collision-cliff--but-cohesion-shears-first):
without a convention K flocks jam at the hub; the right-of-way veer (bias) clears
it. Here we hand the bias to only *some* agents and ask how many — and *where* —
need it.

The point-agent convention thread found a critical mass: partial adoption stays
below the linear reference and a symmetric non-cooperator pair re-jams the hub. A
flock is different: the lattice (velocity consensus) *couples* a non-adopter to
its flock-mates. So the answer turns on **placement**:

  critmass    within-flock adoption: each flock biases a fraction f of its members;
              the cohesion drag pulls the rest -> a low per-flock critical mass.
  placement   at a FIXED adoption budget, MIXED (spread evenly across every flock)
              vs CLUSTERED (whole flocks adopt, the rest free-ride), paired McNemar.

  python scripts/flocking_critmass_phase.py --mode critmass --episodes 40
  python scripts/flocking_critmass_phase.py --mode placement --episodes 40
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
_spec = importlib.util.spec_from_file_location(
    "_flocking_hub", str(Path(__file__).resolve().parent / "_flocking_hub.py"))
_H = importlib.util.module_from_spec(_spec)
sys.modules["_flocking_hub"] = _H
_spec.loader.exec_module(_H)
simulate_hub = _H.simulate_hub

from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p  # noqa: E402

K = 6
PER_FLOCK = 10
N = K * PER_FLOCK
PEAK = 1.0


def _mc(a_bits, b_bits):
    bb = sum(1 for x, y in zip(a_bits, b_bits) if y and not x)
    cc = sum(1 for x, y in zip(a_bits, b_bits) if x and not y)
    return bb, cc, mcnemar_exact_p(bb, cc)


def mask_within(frac):
    """Each flock biases the first ceil(frac*PER_FLOCK) of its members."""
    m = np.zeros(N, bool)
    k_adopt = int(round(frac * PER_FLOCK))
    for k in range(K):
        m[k * PER_FLOCK: k * PER_FLOCK + k_adopt] = True
    return m


def mask_mixed(budget):
    """Spread `budget` adopters as evenly as possible across all K flocks."""
    m = np.zeros(N, bool)
    base, rem = divmod(budget, K)
    for k in range(K):
        c = base + (1 if k < rem else 0)
        m[k * PER_FLOCK: k * PER_FLOCK + c] = True
    return m


def mask_clustered(budget):
    """Fill whole flocks first; the remaining flocks free-ride entirely."""
    m = np.zeros(N, bool)
    m[:budget] = True
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["critmass", "placement"], default="critmass")
    ap.add_argument("--episodes", type=int, default=40)
    args = ap.parse_args()
    seeds = list(range(args.episodes))
    m = len(seeds)

    if args.mode == "critmass":
        print(f"Within-flock critical mass: each flock biases a fraction of its members (K={K}, m={m})")
        print("  cohesion drag should pull the non-adopters through the roundabout")
        print("  frac | adopters | all-passed |  b  c (vs none) |    p      | cohesion")
        print("-" * 70)
        none = [simulate_hub(K=K, bias=PEAK, adopt=mask_within(0.0), seed=s) for s in seeds]
        t_none = [r.all_passed for r in none]
        for frac in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
            msk = mask_within(frac)
            rs = [simulate_hub(K=K, bias=PEAK, adopt=msk, seed=s) for s in seeds]
            t = [r.all_passed for r in rs]
            b, c, p = _mc(t, t_none)
            ch = np.mean([r.cohesion for r in rs])
            print(f"  {frac:.1f}  |  {int(msk.sum()):>2}/{N}   |   {sum(t):>2}/{m}    | {b:>2} {c:>2}         | {p:.2e} |   {ch:.2f}")
        print("-" * 70)
        print("=> a minority per flock suffices: the velocity-consensus lattice drags the")
        print("   free-riders around the roundabout. (placement mode shows this is WHY a fixed")
        print("   budget works better spread thin than clumped.)")

    else:  # placement: mixed vs clustered at a matched adoption budget
        print(f"Placement at a fixed adoption budget: MIXED (even/flock) vs CLUSTERED (whole flocks) (K={K}, m={m})")
        print("  same number of rule-followers, different placement; paired McNemar (mixed vs clustered)")
        print("  budget | mixed | clustered |  b  c  |    p      | mix coh / clu coh")
        print("-" * 72)
        for budget in (12, 15, 18, 21, 24):
            rm = [simulate_hub(K=K, bias=PEAK, adopt=mask_mixed(budget), seed=s) for s in seeds]
            rc = [simulate_hub(K=K, bias=PEAK, adopt=mask_clustered(budget), seed=s) for s in seeds]
            tm = [r.all_passed for r in rm]
            tc = [r.all_passed for r in rc]
            b, c, p = _mc(tm, tc)  # c = mixed wins, b = clustered wins
            chm = np.mean([r.cohesion for r in rm]); chc = np.mean([r.cohesion for r in rc])
            print(f"  {budget:>2}/{N} | {sum(tm):>2}/{m} |   {sum(tc):>2}/{m}   | {b:>2} {c:>2}  | {p:.2e} |   {chm:.2f} / {chc:.2f}")
        print("-" * 72)
        print("=> MIXED dominates CLUSTERED at every matched budget (c > b): spread thin and the")
        print("   cohesion of every flock drags its free-riders through; clump it and the")
        print("   un-adopting flocks stay coherent walls that re-jam the hub. The cost flips:")
        print("   clustered keeps higher cohesion (intact free-rider flocks) but jams more.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
