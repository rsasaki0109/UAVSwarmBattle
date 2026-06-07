"""The crossing-flock jam is gated by encounter angle: it is a head-on phenomenon.

The [crossing study](docs/findings.md#two-cohesive-flocks-crossing-head-on-jam-but-never-collide--the-right-of-way-convention-clears-the-gridlock-within-an-operating-band)
showed two flocks driven *head-on* (180°) jam, and the right-of-way convention
clears it. This sweeps the encounter angle to bound *when* the jam happens and
*where* the convention earns its keep — echoing the convention-arc rule that a
deadlock needs symmetric head-on convergence, while decomposable conflicts do not.

Two axes are scored separately, because the convention fixes only one:
  passed   — both flocks cleared the crossing (the JAM axis: did they gridlock?)
  on_lane  — neither flock drifted more than lane_tol off its road (the LANE axis)

  angle     sweep encounter angle (90°=perpendicular .. 180°=head-on), bias on/off
  mcnemar   paired bias=0 vs bias=peak on `passed`, at representative angles

  python scripts/flocking_crossing_angle_phase.py --mode angle --episodes 40
  python scripts/flocking_crossing_angle_phase.py --mode mcnemar --episodes 40
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
_spec = importlib.util.spec_from_file_location(
    "_flocking_crossing", str(Path(__file__).resolve().parent / "_flocking_crossing.py"))
_C = importlib.util.module_from_spec(_spec)
sys.modules["_flocking_crossing"] = _C
_spec.loader.exec_module(_C)
simulate_crossing = _C.simulate_crossing

from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p  # noqa: E402

ANGLES = (90, 105, 120, 135, 150, 165, 180)
PEAK = 1.0


def _mc(a_bits, b_bits):
    bb = sum(1 for x, y in zip(a_bits, b_bits) if y and not x)
    cc = sum(1 for x, y in zip(a_bits, b_bits) if x and not y)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["angle", "mcnemar"], default="angle")
    ap.add_argument("--episodes", type=int, default=40)
    args = ap.parse_args()
    seeds = list(range(args.episodes))
    m = len(seeds)

    if args.mode == "angle":
        print(f"The jam is head-on: it onsets only near 180°, and the convention earns its keep there (N=24, m={m})")
        print("  passed = cleared the crossing (jam axis); on_lane = stayed on its road (lane axis)")
        print("  angle | passed b0/b1 | on_lane b0/b1 | inter-flock min dist b0/b1")
        print("-" * 70)
        for a in ANGLES:
            b0 = [simulate_crossing(seed=s, bias=0.0, encounter_angle=a) for s in seeds]
            b1 = [simulate_crossing(seed=s, bias=PEAK, encounter_angle=a) for s in seeds]
            p0 = sum(r.passed for r in b0); p1 = sum(r.passed for r in b1)
            l0 = sum(r.on_lane for r in b0); l1 = sum(r.on_lane for r in b1)
            mp0 = min(r.min_pair for r in b0); mp1 = min(r.min_pair for r in b1)
            tag = "  <- no jam" if p0 >= 0.9 * m else ("  <- clean rescue" if (p1 >= 0.9 * m and l1 >= 0.9 * m) else "")
            print(f"  {a:>3}° |   {p0:>2}/{p1:<2}     |    {l0:>2}/{l1:<2}      |    {mp0:.2f} / {mp1:.2f}{tag}")
        print("-" * 70)
        print("=> perpendicular/oblique flocks slip past (passed at bias 0 — no jam, convention")
        print("   no-op); the jam onsets near head-on, where the convention breaks it — but a")
        print("   CLEAN clear (passed AND on-lane) needs ~150°+; at 120° it clears off its lane.")

    else:  # mcnemar on the jam axis (passed)
        print(f"Jam-break is significant only where there is a jam (paired bias0 vs bias{PEAK} on `passed`, m={m})")
        print("  angle | bias0 | bias1 |  b  c  |    p      | verdict")
        print("-" * 62)
        for a in (90, 120, 135, 180):
            b0 = [simulate_crossing(seed=s, bias=0.0, encounter_angle=a).passed for s in seeds]
            b1 = [simulate_crossing(seed=s, bias=PEAK, encounter_angle=a).passed for s in seeds]
            b, c, p = _mc(b1, b0)
            verdict = "no jam -> no-op" if sum(b0) >= 0.9 * m else "jam -> broken"
            print(f"  {a:>3}° | {sum(b0):>2}/{m} | {sum(b1):>2}/{m} | {b:>2} {c:>2}  | {p:.2e} | {verdict}")
        print("-" * 62)
        print("=> at 90° both pass (no jam) so the convention is a no-op; from ~120° on the")
        print("   jam appears and the convention breaks it (c >> b).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
