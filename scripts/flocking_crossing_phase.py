"""Two cohesive flocks crossing head-on JAM — but never collide — and a
right-of-way convention clears the jam, within an operating band.

The convention thread of this repo (MPC, ORCA, HRVO) shows the antipodal swap
**deadlocks**: two goal-directed agents aimed through a shared point turn into
each other and collide, and a `lateral_bias` right-of-way convention breaks the
symmetry. This asks whether *cohesive flocking* — Olfati-Saber's α-lattice, where
the inter-agent potential is universal — has the same pathology, and whether the
same cure works.

It does, with a twist. Drive two Olfati-Saber flocks at each other (group 0 left
→ right, group 1 right → left, goals crossing the centre). The flocks **jam**: the
universal α-repulsion makes them a mutual wall, so they grind to a near-halt at the
centre and only squeeze past long after their time budget — yet they NEVER collide
(inter-flock spacing is held by the same repulsion). Cohesive flocking converts the
antipodal *collision* into a non-colliding *gridlock*. The same right-of-way veer
(every agent biases right of its goal heading) clears the jam — but only in a band:
too weak leaves the jam, too strong flings the flocks off their lane.

Outcome scored on_time = both flock centroids cleared the crossing within the time
budget AND stayed within `lane_tol` of their lane.

  band      sweep the bias strength: jam (0) -> clean pass (peak) -> off-lane (0)
  mcnemar   paired bias=0 vs bias=peak, swept over flock size N; the jam-break
            McNemar c, and the inter-flock closest approach (no collision either way)

  python scripts/flocking_crossing_phase.py --mode band --episodes 40
  python scripts/flocking_crossing_phase.py --mode mcnemar --episodes 40
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

BIASES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
PEAK = 1.0


def _mc(a_bits, b_bits):
    """c = a-only (a is the arm we expect to be better)."""
    bb = sum(1 for x, y in zip(a_bits, b_bits) if y and not x)
    cc = sum(1 for x, y in zip(a_bits, b_bits) if x and not y)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["band", "mcnemar"], default="band")
    ap.add_argument("--episodes", type=int, default=40)
    args = ap.parse_args()
    seeds = list(range(args.episodes))
    m = len(seeds)

    if args.mode == "band":
        print(f"Right-of-way clears the crossing jam, but only in a band (N=24, m={m})")
        print("  on_time = cleared the crossing within budget AND stayed on lane")
        print("  bias | passed | on_lane | on_time | inter-flock min dist | mean pass step")
        print("-" * 76)
        for b in BIASES:
            rs = [simulate_crossing(seed=s, bias=b) for s in seeds]
            P = sum(r.passed for r in rs)
            L = sum(r.on_lane for r in rs)
            T = sum(r.on_time for r in rs)
            mp = min(r.min_pair for r in rs)
            ps = np.mean([r.pass_step if r.pass_step >= 0 else 1400 for r in rs])
            tag = "  <- jam" if b == 0.0 else ("  <- peak" if b == PEAK else
                  ("  <- off-lane" if T == 0 and P > 0 else ""))
            print(f"  {b:>4} | {P:>2}/{m}  |  {L:>2}/{m}  | {T:>2}/{m}  |        {mp:.2f}          |    {ps:.0f}{tag}")
        print("-" * 76)
        print("=> jam at 0 (flocks stall, never collide), clean pass at the peak, off-lane")
        print("   past it: the convention is a tunable band, not a switch (cf. the bias cliff).")

    else:  # mcnemar: jam-break paired across flock size N
        print(f"The jam-break is paired-significant and generalizes across flock size (m={m})")
        print(f"  bias=0 vs bias={PEAK}, on_time; min dist = inter-flock closest approach (collision if < ~{0.5*7:.0f})")
        print("   N | jam(b0) | row(b1) |  b  c  |    p     | min dist b0 / b1")
        print("-" * 66)
        for N in (16, 24, 32):
            base = [simulate_crossing(seed=s, bias=0.0, n=N) for s in seeds]
            row = [simulate_crossing(seed=s, bias=PEAK, n=N) for s in seeds]
            tb = [r.on_time for r in base]
            tr = [r.on_time for r in row]
            b, c, p = _mc(tr, tb)
            mpb = min(r.min_pair for r in base)
            mpr = min(r.min_pair for r in row)
            print(f"  {N:>2} |  {sum(tb):>2}/{m}  |  {sum(tr):>2}/{m}  | {b:>2} {c:>2}  | {p:.2e} |   {mpb:.2f} / {mpr:.2f}")
        print("-" * 66)
        print("=> the convention clears the jam at every flock size (c >> b); neither arm")
        print("   collides (min dist stays well above contact) — flocking jams, it doesn't crash.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
