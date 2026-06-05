"""Does cooperative aerial transport scale to ANY team size -- and what makes it?

TeamHOI (CVPR 2026, sail-sg) learns a single decentralized policy for cooperative
human-object interaction that is advertised "team-size- and shape-agnostic": one
policy carries an object with 2..8 (zero-shot 12/16) agents at >97.5 % success.
This script interrogates that claim on the canonical hard case for a RIGID carried
payload -- threading a doorway -- where the GEOMETRY of the formation, not the
policy's cleverness, may decide whether the team gets through.

Setup (scripts/_coop_transport.py): N drones rigidly carry a beam from start to
goal through a wall with a single gap. Two arms, paired by seed and McNemar-exact:

  fixed     beam held perpendicular to travel (its initial pose)
  adaptive  beam reorients to align with travel near the doorway, shrinking the
            cross-aperture footprint so it can slip through

Only the orientation differs; the translational controller (goal + gap y-centring)
is identical, so any gap is attributable to reorientation alone.

Three sweeps:
  --mode gap     fixed N, fixed beam, sweep doorway width
                 -> the geometric cliff: fixed needs gap > bar_len; adaptive down to ~3
  --mode team    fixed doorway, beam length = spacing*(N-1), sweep N
                 -> THE TeamHOI cell: fixed's required aperture SCALES with team size
                    (collapses for N>=3 at a gap N=2 clears); adaptive stays flat
  --mode runway  fixed N/beam/narrow gap, sweep wall distance from start
                 -> adaptive's OWN cost: reorientation needs runway; too close to the
                    wall (or too slow a turn) and the smarter arm also fails -> the
                    benefit is conditional, not a free lunch

  python scripts/coop_transport_doorway_phase.py --mode team --episodes 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _coop_transport import simulate  # noqa: E402

from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p  # noqa: E402

SPACING = 2.5  # m between adjacent drones along the beam (team mode: bar = SPACING*(N-1))


def _bits(adaptive, seeds, **kw):
    return [simulate(adaptive=adaptive, seed=s, **kw).success for s in seeds]


def _mc(fixed_bits, adapt_bits):
    """b = fixed-only success, c = adaptive-only success (the rescue)."""
    b = sum(1 for f, a in zip(fixed_bits, adapt_bits) if f and not a)
    c = sum(1 for f, a in zip(fixed_bits, adapt_bits) if a and not f)
    return b, c, mcnemar_exact_p(b, c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gap", "team", "runway"], default="team")
    ap.add_argument("--episodes", type=int, default=60)
    args = ap.parse_args()
    seeds = list(range(args.episodes))

    print(f"# cooperative transport: fixed vs adaptive orientation "
          f"({args.episodes} seeds, McNemar exact)\n")

    if args.mode == "gap":
        print("doorway-width sweep (N=4, bar_len=10)")
        print(f"{'gap':>5} {'fixed':>7} {'adapt':>7} {'b':>4} {'c':>4} {'p':>10}")
        for gw in [2, 3, 4, 5, 6, 8, 10, 11, 12]:
            kw = dict(n=4, bar_len=10.0, gap_w=float(gw))
            fb = _bits(False, seeds, **kw)
            ab = _bits(True, seeds, **kw)
            b, c, p = _mc(fb, ab)
            print(f"{gw:>5} {sum(fb):>7} {sum(ab):>7} {b:>4} {c:>4} {p:>10.2e}")

    elif args.mode == "team":
        gap = 6.0
        print(f"team-size sweep (doorway={gap}, beam=SPACING*(N-1), SPACING={SPACING})")
        print(f"{'N':>3} {'barlen':>7} {'fixed':>7} {'adapt':>7} {'b':>4} {'c':>4} {'p':>10}")
        for n in [2, 3, 4, 5, 6, 8]:
            bl = SPACING * (n - 1)
            kw = dict(n=n, bar_len=bl, gap_w=gap)
            fb = _bits(False, seeds, **kw)
            ab = _bits(True, seeds, **kw)
            b, c, p = _mc(fb, ab)
            print(f"{n:>3} {bl:>7.1f} {sum(fb):>7} {sum(ab):>7} {b:>4} {c:>4} {p:>10.2e}")

    else:  # runway
        print("runway sweep (N=5, bar_len=10, narrow gap=4): adaptive's own cost")
        print(f"{'wall_x':>7} {'runway':>7} {'fixed':>7} {'adapt':>7} {'b':>4} {'c':>4} {'p':>10}")
        for wx in [11, 13, 15, 16, 17, 18, 22, 30]:
            kw = dict(n=5, bar_len=10.0, gap_w=4.0, wall_x=float(wx))
            fb = _bits(False, seeds, **kw)
            ab = _bits(True, seeds, **kw)
            b, c, p = _mc(fb, ab)
            print(f"{wx:>7} {wx - 8:>7} {sum(fb):>7} {sum(ab):>7} "
                  f"{b:>4} {c:>4} {p:>10.2e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
