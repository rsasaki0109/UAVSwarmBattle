"""A cut flock can be healed — but only by a GLOBAL term, and only if it waits.

The [obstacle study](docs/findings.md#an-obstacle-splits-a-migrating-flock-past-a-critical-radius--r2--and-the-navigational-structure-that-migrates-it-cannot-re-merge-the-halves)
showed that once an obstacle severs a flock, no local force (the finite-support
potential) and no amount of navigational pull re-merges the halves: a cut is
permanent. That finding ended on "the cure would have to be a term that acts
*across* the gap, which a range-limited flock does not have." So build that term —
a **rendezvous** pull toward the flock's global centroid (each agent needs the
whole flock's mean position, i.e. global information, unlike the comms-free local
[convention](docs/findings.md#a-comms-free-rule-beats-a-smarter-one-when-sensing-drops-out)) —
and ask what it takes to heal the cut.

The twist: re-cohesion and obstacle passage CONFLICT. A rendezvous pull that is on
*while* the flock is splitting around the disk tugs the lobes back toward the
centroid sitting at the obstacle, fighting the very detour that gets them past.
So an always-on global pull only heals by brute force (a very strong gain); a pull
that WAITS — gated to act only once each agent has cleared the obstacle — heals
with far weaker coupling. Timing the re-cohesion is the efficient cure.

Outcome (paired by seed, McNemar-exact): a SINGLE connected component after passing
(`scripts/_flocking.py`, Algorithm 2 + β-agent obstacle + centroid rendezvous).

  heal    baseline (no rendezvous) vs gated rendezvous, sweep obstacle radius R
          -> the gated global term rescues the cut where #143 could not.
  timing  gated vs always-on rendezvous at matched gain, sweep the gain
          -> gated dominates at low/moderate coupling; they converge only when the
             always-on pull is strong enough to override the obstacle passage.

  python scripts/flocking_rendezvous_phase.py --mode heal --episodes 40
  python scripts/flocking_rendezvous_phase.py --mode timing --episodes 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _flocking import simulate  # noqa: E402

from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p  # noqa: E402

OBS_X = 40.0
BASE = dict(algorithm=2, n=24, steps=1500, c2a=8.0, grad_gain=1.0, c1g=1.0, c2g=0.6,
            spread=14.0, goal=(0.0, 0.0), goal_vel=(5.0, 0.0), goal_moves=True,
            obs_infl=4.0, c_obs=20.0)
GATE_MARGIN = 5.0   # gate sits past the disk: x > OBS_X + R + margin


def _bits(seeds, *, R, c_rdv=0.0, gated=False):
    obs = ((OBS_X, 0.0, float(R)),)
    gate = (OBS_X + R + GATE_MARGIN) if gated else None
    return [simulate(**BASE, obstacles=obs, c_rdv=c_rdv, rdv_gate_x=gate, seed=s).connected
            for s in seeds]


def _mc(a, b):
    bb = sum(1 for x, y in zip(a, b) if x and not y)
    cc = sum(1 for x, y in zip(a, b) if y and not x)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["heal", "timing"], default="heal")
    ap.add_argument("--episodes", type=int, default=40)
    args = ap.parse_args()
    seeds = list(range(args.episodes))
    m = len(seeds)

    if args.mode == "heal":
        print(f"A gated global rendezvous heals the cut a local flock cannot (c_rdv=3.0), m={m}")
        print("  R | baseline | gated rdv |  b  c  |   p")
        print("-" * 50)
        for R in (3, 4, 6, 9, 12):
            base = _bits(seeds, R=R, c_rdv=0.0)
            gated = _bits(seeds, R=R, c_rdv=3.0, gated=True)
            b, c, p = _mc(base, gated)
            print(f" {R:>2} |  {sum(base):>2}/{m}   |   {sum(gated):>2}/{m}   | {b:>2} {c:>2}  | {p:.2e}")
        print("-" * 50)
        print("=> the cure for a cut is a global term that acts across the gap (#143's missing term).")

    else:  # timing
        R = 6
        print(f"Re-cohesion conflicts with passage — gated vs always-on at matched gain (R={R}), m={m}")
        print(" c_rdv | always-on | gated |  b  c  |   p   (c = gated-only heal)")
        print("-" * 60)
        for g in (1.5, 3.0, 6.0, 10.0):
            always = _bits(seeds, R=R, c_rdv=g, gated=False)
            gated = _bits(seeds, R=R, c_rdv=g, gated=True)
            b, c, p = _mc(always, gated)
            print(f" {g:>5} |   {sum(always):>2}/{m}   | {sum(gated):>2}/{m} | {b:>2} {c:>2}  | {p:.2e}")
        print("-" * 60)
        print("=> gated heals with weak coupling; always-on only catches up by brute force "
              "(strong enough to override the obstacle detour).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
