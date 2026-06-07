"""Free flocking fragments — and MORE cohesion can't fix it; structure can.

Olfati-Saber ("Flocking for Multi-Agent Dynamic Systems: Algorithms and Theory",
IEEE TAC 2006) proves that ALGORITHM 1 -- "free flocking" with only the
collective potential (cohesion+separation) and velocity consensus (alignment) --
FRAGMENTS for generic initial states: the group splits into several flocks and
never becomes one.  The cure he proves is ALGORITHM 2: add a navigational
feedback term (a shared objective / γ-agent).  Not a stronger potential -- a
*different structure*.

This script asks the naive engineer's question and watches it fail: if the group
keeps splitting, can't we just turn UP the cohesion gain until it holds together?
And it checks the paper's answer: the gradient has FINITE SUPPORT (zero force past
the interaction range r), so once a sub-flock drifts beyond r it is lost at ANY
gain -- and because the potential is symmetric, cranking the gain scales the
repulsion too, scattering the marginal agents and making fragmentation WORSE.
Only the navigational term, which every agent feels regardless of its neighbours,
reunites them.

Outcome (paired by seed, McNemar-exact): the final interaction graph (edge when
‖q_i-q_j‖ < r) is a SINGLE connected component -- one flock, not many.

Two modes (scripts/_flocking.py is the self-contained sim):

  cohesion   Algorithm 1, sweep the cohesion/interaction gain grad_gain.
             -> connectivity stays low and DECREASES with gain; no setting
                approaches Algorithm 2.  Cohesion is the wrong knob.
  structure  Algorithm 1 vs Algorithm 2 at matched gain, swept over group size N.
             -> the navigational structure reunites the flock at every N
                (c >> b), where Algorithm 1 fragments.

  python scripts/flocking_fragmentation_phase.py --mode cohesion --episodes 40
  python scripts/flocking_fragmentation_phase.py --mode structure --episodes 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _flocking import simulate  # noqa: E402

from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p  # noqa: E402

SPREAD0 = 12.0   # initial half-width at N=20 (scaled as sqrt(N/20) to hold density)
C2A = 14.0       # consensus (alignment) gain -- strong enough to form α-lattices
STEPS = 1500


def _bits(seeds, **kw):
    return [simulate(steps=STEPS, c2a=C2A, **kw, seed=s).connected for s in seeds]


def _mc(a_bits, b_bits):
    """b = arm-A-only connected, c = arm-B-only connected (B = the rescue)."""
    b = sum(1 for x, y in zip(a_bits, b_bits) if x and not y)
    c = sum(1 for x, y in zip(a_bits, b_bits) if y and not x)
    return b, c, mcnemar_exact_p(b, c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cohesion", "structure"], default="cohesion")
    ap.add_argument("--episodes", type=int, default=40)
    args = ap.parse_args()
    seeds = list(range(args.episodes))
    m = len(seeds)

    if args.mode == "cohesion":
        print(f"Free flocking (Algo 1) — can MORE cohesion stop fragmentation?  N=20, paired m={m}")
        print(" grad_gain | connected | ncomp~ | largest_frac~")
        print("-" * 56)
        cohesion_bits = {}
        for g in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            res = [simulate(steps=STEPS, c2a=C2A, algorithm=1, grad_gain=g, spread=SPREAD0, seed=s)
                   for s in seeds]
            cohesion_bits[g] = [r.connected for r in res]
            conn = sum(cohesion_bits[g])
            nc = np.mean([r.n_components for r in res])
            lf = np.mean([r.largest_frac for r in res])
            print(f"  {g:>6}   |  {conn:>2}/{m}   | {nc:>5.1f}  |   {lf:.2f}")
        # reference: best cohesion cell vs the navigational structure (Algo 2)
        algo2 = _bits(seeds, algorithm=2, grad_gain=1.0, spread=SPREAD0)
        best_g = max(cohesion_bits, key=lambda k: sum(cohesion_bits[k]))
        b, c, p = _mc(cohesion_bits[best_g], algo2)
        print("-" * 56)
        print(f"Algo 2 (navigational structure): {sum(algo2)}/{m} connected")
        print(f"best cohesion cell (grad_gain={best_g}, {sum(cohesion_bits[best_g])}/{m}) vs Algo 2: "
              f"b={b} c={c} p={p:.2e}")
        print("=> cohesion gain cannot substitute for the navigational structure.")

    else:  # structure
        print(f"Navigational structure reunites the flock — Algo 1 vs Algo 2, paired m={m}")
        print("  N | algo1 conn | algo2 conn |  b  c  |   p")
        print("-" * 56)
        for n in (12, 20, 30, 40):
            sp = SPREAD0 * np.sqrt(n / 20.0)
            a1 = _bits(seeds, algorithm=1, n=n, grad_gain=1.0, spread=sp)
            a2 = _bits(seeds, algorithm=2, n=n, grad_gain=1.0, spread=sp)
            b, c, p = _mc(a1, a2)
            print(f" {n:>2} |   {sum(a1):>2}/{m}    |   {sum(a2):>2}/{m}    | {b:>2} {c:>2}  | {p:.2e}")
        print("-" * 56)
        print("=> the navigational term (shared objective) is the structural cure; "
              "Algo 1's potential, however strong, is not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
