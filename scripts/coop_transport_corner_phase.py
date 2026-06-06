"""Reorientation makes a STRAIGHT doorway size-agnostic — an L-CORNER does not.

The doorway study (scripts/coop_transport_doorway_phase.py) showed that letting a
carried beam reorient makes cooperative transport team-size-agnostic: the adaptive
arm holds ~100 % flat across N=2–8. That is a property of a CONVEX aperture. A
right-angle corridor junction is non-convex, and a rigid segment rounding it obeys
the classical "ladder around a corner" bound:

    L_max = (a^(2/3) + b^(2/3))^(3/2)         (corridors of width a, b)

No reorientation beats it — a beam longer than L_max cannot be maneuvered around
the corner in ANY sequence of moves. Since beam length grows with the team, this
caps the maximum team size that can round the corner, however the formation
reshapes. This script measures that ceiling and contrasts it with the doorway.

Two sweeps, paired by seed, McNemar-exact (b = corner-only success, c = doorway-
only success):

  --mode ceiling  beam = 2.5·(N−1), corridor width 4 m, sweep N
                  doorway-adaptive (flat ~100 %) vs corner (cliffs at N≈5–6)
  --mode width    fixed N=6 (beam 12.5 m), sweep corridor width
                  the corner ceiling SCALES with width (N_max = 2.83·w / spacing)

  python scripts/coop_transport_corner_phase.py --mode ceiling --episodes 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _coop_transport import simulate, simulate_corner, corner_Lmax  # noqa: E402

from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p  # noqa: E402

SPACING = 2.5


def _mc(corner_bits, door_bits):
    b = sum(1 for cc, dd in zip(corner_bits, door_bits) if cc and not dd)
    c = sum(1 for cc, dd in zip(corner_bits, door_bits) if dd and not cc)
    return b, c, mcnemar_exact_p(b, c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ceiling", "width"], default="ceiling")
    ap.add_argument("--episodes", type=int, default=60)
    args = ap.parse_args()
    seeds = list(range(args.episodes))

    print(f"# cooperative transport: straight doorway vs L-corner (adaptive both)\n"
          f"# {args.episodes} seeds, McNemar exact (b=corner-only, c=doorway-only)\n")

    if args.mode == "ceiling":
        w = 4.0
        print(f"corridor width {w} m  ->  ladder ceiling L_max = {corner_Lmax(w, w):.2f} m"
              f"  (N_max ≈ {corner_Lmax(w, w) / SPACING + 1:.1f})")
        print(f"{'N':>3} {'beam':>6} {'doorway':>8} {'corner':>7} {'b':>4} {'c':>4} {'p':>10}")
        for n in [2, 3, 4, 5, 6, 7, 8]:
            bl = SPACING * (n - 1)
            door = [simulate(n=n, bar_len=bl, gap_w=6.0, seed=s, adaptive=True).success
                    for s in seeds]
            corner = [simulate_corner(n=n, bar_len=bl, corridor_w=w, seed=s).success
                      for s in seeds]
            b, c, p = _mc(corner, door)
            print(f"{n:>3} {bl:>6.1f} {sum(door):>8} {sum(corner):>7} "
                  f"{b:>4} {c:>4} {p:>10.2e}")

    else:  # width
        n = 6
        bl = SPACING * (n - 1)
        print(f"N={n}, beam={bl} m (L_eff={bl + 1.0:.1f}); sweep corridor width")
        print(f"{'width':>6} {'L_max':>7} {'doorway':>8} {'corner':>7} {'b':>4} {'c':>4} {'p':>10}")
        for w in [3.0, 4.0, 4.5, 5.0, 5.5, 6.0]:
            door = [simulate(n=n, bar_len=bl, gap_w=max(6.0, w + 2), seed=s,
                             adaptive=True).success for s in seeds]
            corner = [simulate_corner(n=n, bar_len=bl, corridor_w=w, seed=s).success
                      for s in seeds]
            b, c, p = _mc(corner, door)
            print(f"{w:>6.1f} {corner_Lmax(w, w):>7.2f} {sum(door):>8} {sum(corner):>7} "
                  f"{b:>4} {c:>4} {p:>10.2e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
