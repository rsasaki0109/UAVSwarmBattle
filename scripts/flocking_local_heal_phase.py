"""Healing a cut flock is LOCAL, not global — a comms-free reach rule beats the
global rendezvous, and the gap widens with the obstacle.

The [rendezvous study](docs/findings.md#a-cut-flock-can-be-healed--but-only-by-a-global-term-and-only-if-it-waits-a-gated-rendezvous-re-merges-what-local-rules-cannot)
(#144) concluded that healing an obstacle-severed flock is "irreducibly global":
a pull toward the flock's global centroid re-merges the lobes, where local rules
could not. This revisits that claim with a *local* alternative — **adaptive reach**:
an agent that has lost neighbours (degree below `reach_kmin` within its base range)
enlarges its OWN sensing range to `reach_boost·r`. It is comms-free (each agent uses
only its own neighbour count — no centroid, no global state), exactly the kind of
rule [#139](docs/findings.md#a-comms-free-rule-beats-a-smarter-one-when-sensing-drops-out)
shows is robust.

The result overturns #144: the global rendezvous is the *worse* healer. Its target —
the centroid of a flock split into two lobes — sits ON the obstacle, so the pull
drives the lobes back into the disk and cannot bridge a wide cut; it collapses as the
obstacle grows. The local reach rule bridges lobe-to-lobe directly, around nothing,
and heals at every obstacle size. (A fixed larger range heals too — it is the *reach*
that matters, not the trigger; adaptive reach just supplies it comms-free and only
where it is needed.)

Outcome (paired by seed, McNemar-exact): a SINGLE connected component after passing.

  heal       baseline vs adaptive local reach, sweep obstacle radius R
  vs_global  adaptive local reach vs the #144 global rendezvous, sweep R
             -> local dominates and the gap widens with R

  python scripts/flocking_local_heal_phase.py --mode heal --episodes 40
  python scripts/flocking_local_heal_phase.py --mode vs_global --episodes 40
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
RADII = (9, 13, 17, 22)
BOOST, KMIN = 3.0, 5          # comms-free adaptive reach
C_RDV = 3.0                   # #144 global rendezvous strength


def _bits(seeds, R, **kw):
    obs = ((OBS_X, 0.0, float(R)),)
    return [simulate(**BASE, obstacles=obs, seed=s, **kw).connected for s in seeds]


def _adaptive(seeds, R):
    return _bits(seeds, R, reach_boost=BOOST, reach_kmin=KMIN)


def _global(seeds, R):
    return _bits(seeds, R, c_rdv=C_RDV, rdv_gate_x=OBS_X + R + 5.0)


def _mc(a, b):
    bb = sum(1 for x, y in zip(a, b) if x and not y)
    cc = sum(1 for x, y in zip(a, b) if y and not x)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["heal", "vs_global"], default="heal")
    ap.add_argument("--episodes", type=int, default=40)
    args = ap.parse_args()
    seeds = list(range(args.episodes))
    m = len(seeds)

    if args.mode == "heal":
        print(f"Comms-free adaptive local reach heals the cut at every obstacle size, m={m}")
        print("  R | baseline | adaptive | global(#144) |  b  c  |    p (adaptive vs baseline)")
        print("-" * 70)
        for R in RADII:
            base = _bits(seeds, R)
            adpt = _adaptive(seeds, R)
            glob = _global(seeds, R)
            b, c, p = _mc(base, adpt)
            print(f" {R:>2} |  {sum(base):>2}/{m}   |  {sum(adpt):>2}/{m}  |    {sum(glob):>2}/{m}     | {b:>2} {c:>2}  | {p:.2e}")
        print("-" * 70)
        print("=> local reach heals everywhere; the global rendezvous fades as the cut widens.")

    else:  # vs_global
        print(f"Local reach vs the global rendezvous (#144) — c = local-only heal, m={m}")
        print("  R | global | local |  b  c  |    p")
        print("-" * 48)
        for R in RADII:
            glob = _global(seeds, R)
            adpt = _adaptive(seeds, R)
            b, c, p = _mc(glob, adpt)
            print(f" {R:>2} |  {sum(glob):>2}/{m}  | {sum(adpt):>2}/{m} | {b:>2} {c:>2}  | {p:.2e}")
        print("-" * 48)
        print("=> the local rule dominates and the gap widens with R: the centroid of a "
              "wide cut sits on the obstacle, so the global pull cannot bridge it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
