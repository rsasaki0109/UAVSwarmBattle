"""An obstacle splits a migrating flock past a critical size — and the structure
that migrates it cannot re-merge the halves.

Olfati-Saber's Algorithm 3 adds obstacle avoidance (β-agents) to the navigational
flocking of [Algorithm 2](#free-flocking-fragments). A flock with a shared goal
migrates cohesively (proven in the [fragmentation
study](docs/findings.md#free-flocking-fragments--and-you-cannot-cohesion-gain-your-way-out-the-navigational-structure-is-the-fix-not-a-bigger-potential));
this script drives that migrating flock at a disk obstacle on its path and asks two
things:

  radius  sweep the obstacle radius R -> the flock threads a small disk intact but
          a large one SPLITS it; connectivity collapses monotonically past a
          critical R. (Control R=0 stays one flock.)
  heal    once split (fixed large R), turn the navigational gain UP -> it does NOT
          re-merge the halves. The γ-term migrates the pieces but cannot reunite
          them: once the obstacle pushes the two lobes beyond the interaction
          range r, no α-force bridges the gap and the goal-pull is (laterally) too
          weak to close it. Same finite-support wall as the fragmentation study,
          now caused dynamically by an obstacle rather than the initial spread.
  range   the critical R is gated by the interaction range r=1.2d: a flock that
          reaches farther tolerates a bigger obstacle (directional, not a clean
          law — reported as an invariant, not fitted).

Outcome (paired by seed, McNemar-exact): the flock is a SINGLE connected component
after it has passed the obstacle.

  python scripts/flocking_obstacle_split_phase.py --mode radius --episodes 40
  python scripts/flocking_obstacle_split_phase.py --mode heal --episodes 40
  python scripts/flocking_obstacle_split_phase.py --mode range --episodes 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _flocking import simulate  # noqa: E402

from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p  # noqa: E402

# fixed migrating-flock operating point (Algorithm 2 + β-agent obstacle)
BASE = dict(algorithm=2, n=24, steps=1600, c2a=8.0, grad_gain=1.0, c2g=0.6,
            goal=(0.0, 0.0), goal_vel=(5.0, 0.0), goal_moves=True,
            obs_infl=4.0, c_obs=20.0)
OBS_X = 40.0


def _bits(seeds, *, R, c1g=1.0, d=7.0):
    obs = () if R <= 0 else ((OBS_X, 0.0, float(R)),)
    return [simulate(**BASE, d=d, c1g=c1g, spread=d * 2.0, obstacles=obs, seed=s).connected
            for s in seeds]


def _mc(a, b):
    bb = sum(1 for x, y in zip(a, b) if x and not y)
    cc = sum(1 for x, y in zip(a, b) if y and not x)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["radius", "heal", "range"], default="radius")
    ap.add_argument("--episodes", type=int, default=40)
    args = ap.parse_args()
    seeds = list(range(args.episodes))
    m = len(seeds)

    if args.mode == "radius":
        print(f"A disk obstacle splits a migrating flock (d=7, r=8.4) — connectivity after passing, m={m}")
        print("  R | one flock | vs no-obstacle (R=0)")
        print("-" * 50)
        ctrl = _bits(seeds, R=0)
        print(f"  0 |  {sum(ctrl):>2}/{m}   | (control)")
        for R in (2, 3, 4, 6, 9, 14):
            bits = _bits(seeds, R=R)
            b, c, p = _mc(ctrl, bits)
            print(f" {R:>2} |  {sum(bits):>2}/{m}   | b={b} c={c} p={p:.2e}")
        print("-" * 50)
        print("=> below a critical radius the flock threads the obstacle intact; above it, it splits.")

    elif args.mode == "heal":
        R = 8.0
        print(f"Once split (R={R:.0f}), can MORE navigational structure re-merge the halves?  m={m}")
        print(" c1g | one flock | vs intact (R=0, same c1g)")
        print("-" * 50)
        for c1g in (0.6, 1.0, 2.0, 4.0, 8.0):
            split = _bits(seeds, R=R, c1g=c1g)
            intact = _bits(seeds, R=0, c1g=c1g)
            b, c, p = _mc(split, intact)
            print(f" {c1g:>3} |  {sum(split):>2}/{m}   | intact {sum(intact):>2}/{m}  (b={b} c={c} p={p:.2e})")
        print("-" * 50)
        print("=> the γ-term migrates the pieces but does not reunite them at any strength: "
              "the split is permanent once the lobes leave each other's interaction range.")

    else:  # range
        print(f"Is the critical radius gated by the interaction range r=1.2d?  m={m}")
        print("   d (r)    |        connectivity by R         | critical R (>=50%)")
        print("-" * 68)
        Rs = (1, 2, 3, 4, 5, 6, 8, 10)
        for d in (5.0, 7.0, 10.0):
            cells = [(R, sum(_bits(seeds, R=R, d=d))) for R in Rs]
            crit = max([R for R, c in cells if c >= m / 2], default=0)
            body = " ".join(f"R{R}:{c:>2}" for R, c in cells)
            print(f" {d:>4} ({1.2*d:>4.1f}) | {body} | {crit}")
        print("-" * 68)
        print("=> the critical radius grows with the interaction range (a flock that reaches "
              "farther tolerates a bigger obstacle) — directional, not a clean law.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
