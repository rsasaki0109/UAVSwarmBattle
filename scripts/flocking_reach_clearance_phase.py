"""The comms-free local-reach cure for a cut flock is not free: it buys cohesion
by spending obstacle clearance — and the cost is intrinsic to the mechanism.

The [local-heal study](docs/findings.md#healing-a-cut-flock-is-local-not-global-a-comms-free-reach-rule-beats-the-rendezvous-and-the-gap-widens-with-the-obstacle)
(#147) showed that **adaptive reach** — an agent below `reach_kmin` neighbours
enlarges its OWN sensing range to `reach_boost·r` — heals an obstacle-severed
flock at every obstacle size, comms-free, where the global rendezvous (#144)
collapses. That study scored only the *topology* (a single connected component).
This asks what the topology metric hid: **what does the cure cost?**

It costs obstacle clearance. The boosted attraction reaches *across* the disk to
the far lobe, so it adds a surface-ward pull; the trailing agents re-cohere by
hugging the obstacle. Every healed run passes the disk markedly closer than the
base-range flock — a hidden safety cost the connectivity score never saw.

  tradeoff   baseline vs adaptive reach, sweep obstacle R. Two paired McNemar
             tables, OPPOSITE directions: adaptive HEALS far more (single
             component) AND VIOLATES clearance far more (min surface gap < a
             drone radius rho). The cure trades clearance for cohesion.
  mechanism  the cost is a TAIL, not a mean shift. Adaptive's MEAN min gap can
             even EXCEED the baseline's (many healed runs route wide around the
             disk) — yet its worst-case violates far more often. Scoring safety
             by the mean would HIDE the cost or invert it; only the worst case
             sees it. Plus a within-baseline control: a baseline run that heals
             keeps full clearance, so the cost is not a generic "crossing tax."
  fix        can the cost be removed? Widening the obstacle influence margin
             (obs_infl, a STRUCTURAL lever) does nothing — the min gap is a
             force-balance equilibrium at the barrier. Only raising the
             repulsion GAIN (c_obs, a MAGNITUDE lever) restores clearance while
             keeping the heal. A rare case where the fix is magnitude, not
             structure.

  python scripts/flocking_reach_clearance_phase.py --mode tradeoff --episodes 40
  python scripts/flocking_reach_clearance_phase.py --mode mechanism --episodes 40
  python scripts/flocking_reach_clearance_phase.py --mode fix --episodes 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _flocking import simulate  # noqa: E402

from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p  # noqa: E402

OBS_X = 40.0
BASE = dict(algorithm=2, n=24, steps=1500, c2a=8.0, grad_gain=1.0, c1g=1.0, c2g=0.6,
            spread=14.0, goal=(0.0, 0.0), goal_vel=(5.0, 0.0), goal_moves=True,
            obs_infl=4.0, c_obs=20.0)
RADII = (9, 13, 17, 22)
BOOST, KMIN = 3.0, 5          # comms-free adaptive reach (same op point as #147)
RHO = 0.5                     # drone radius: clearance violated when min surface gap < rho


def _run(seed, R, **kw):
    obs = ((OBS_X, 0.0, float(R)),)
    params = dict(BASE)          # kw overrides BASE defaults (e.g. obs_infl, c_obs)
    params.update(kw)
    return simulate(**params, obstacles=obs, seed=seed, record=True)


def _min_gap(res, R):
    """Smallest centre-to-surface gap to the disk over the whole trajectory."""
    g = np.inf
    for q in res.traj:
        v = q - np.array([OBS_X, 0.0])
        g = min(g, float((np.sqrt((v * v).sum(-1)) - R).min()))
    return g


def _mc(a_bits, b_bits):
    """McNemar with c = a-only (a is the arm we expect to be 'more')."""
    bb = sum(1 for x, y in zip(a_bits, b_bits) if y and not x)
    cc = sum(1 for x, y in zip(a_bits, b_bits) if x and not y)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tradeoff", "mechanism", "fix"], default="tradeoff")
    ap.add_argument("--episodes", type=int, default=40)
    args = ap.parse_args()
    seeds = list(range(args.episodes))
    m = len(seeds)

    if args.mode == "tradeoff":
        print(f"The local-reach cure trades clearance for cohesion (rho={RHO}, m={m})")
        print("  c_heal = adaptive-only heal; c_viol = adaptive-only clearance violation")
        print("  R | HEAL base/adapt  b  c |   p_heal  | VIOL base/adapt  b  c |   p_viol")
        print("-" * 78)
        for R in RADII:
            base = [_run(s, R) for s in seeds]
            adpt = [_run(s, R, reach_boost=BOOST, reach_kmin=KMIN) for s in seeds]
            hb = [r.connected for r in base]
            ha = [r.connected for r in adpt]
            vb = [_min_gap(r, R) < RHO for r in base]
            va = [_min_gap(r, R) < RHO for r in adpt]
            bh, ch, ph = _mc(ha, hb)
            bv, cv, pv = _mc(va, vb)
            print(f" {R:>2} |   {sum(hb):>2}/{sum(ha):<2}      {bh:>2} {ch:>2} | {ph:.2e} |"
                  f"   {sum(vb):>2}/{sum(va):<2}      {bv:>2} {cv:>2} | {pv:.2e}")
        print("-" * 78)
        print("=> adaptive HEALS far more (c_heal) AND VIOLATES clearance far more (c_viol):")
        print("   cohesion is bought with clearance, both legs paired-significant, opposite signs.")

    elif args.mode == "mechanism":
        print(f"The clearance cost is a TAIL, not a mean shift (rho={RHO}, m={m})")
        print("  mean = mean over seeds of each run's worst (min) surface gap")
        print("  R | base mean | adapt mean || base viol | adapt viol  (viol = worst gap < rho)")
        print("-" * 76)
        for R in RADII:
            base = [_run(s, R) for s in seeds]
            adpt = [_run(s, R, reach_boost=BOOST, reach_kmin=KMIN) for s in seeds]
            bg = [_min_gap(r, R) for r in base]
            ag = [_min_gap(r, R) for r in adpt]
            bv = sum(g < RHO for g in bg)
            av = sum(g < RHO for g in ag)
            flag = "  <- adapt mean is LARGER" if np.mean(ag) > np.mean(bg) else ""
            print(f" {R:>2} |   {np.mean(bg):.3f}   |   {np.mean(ag):.3f}    ||   {bv:>2}/{m}   |   {av:>2}/{m}{flag}")
        print("-" * 76)
        print("=> the MEAN can say adaptive is SAFER (it routes many runs wide), yet its")
        print("   WORST case breaches far more often: a mean clearance metric hides the cost.")
        print()
        print("  control: within the baseline (no boost), does healing itself cost clearance?")
        print("  R | base heal | mean gap[healed] | mean gap[cut]")
        print("-" * 56)
        for R in RADII:
            base = [_run(s, R) for s in seeds]
            bh_gap = [_min_gap(r, R) for r in base if r.connected]
            bc_gap = [_min_gap(r, R) for r in base if not r.connected]
            sh = f"{np.mean(bh_gap):.3f}" if bh_gap else "   -  "
            sc = f"{np.mean(bc_gap):.3f}" if bc_gap else "   -  "
            print(f" {R:>2} |   {sum(1 for r in base if r.connected):>2}/{m}   |      {sh}      |    {sc}")
        print("-" * 56)
        print("=> a baseline run that HEALS keeps ~full clearance (gap[healed] ~= gap[cut]):")
        print("   crossing per se is not the tax; the boosted reach pulling across the disk is.")

    else:  # fix
        R = 13
        print(f"Can the clearance cost be removed while keeping the heal? (R={R}, m={m})")
        print("  STRUCTURAL lever: widen obstacle influence margin obs_infl")
        print("  infl | adapt heal | min gap | mean gap")
        print("-" * 46)
        for infl in (4.0, 6.0, 8.0, 10.0):
            adpt = [_run(s, R, reach_boost=BOOST, reach_kmin=KMIN, obs_infl=infl) for s in seeds]
            gaps = [_min_gap(r, R) for r in adpt]
            print(f" {infl:>4} |   {sum(r.connected for r in adpt):>2}/{m}    |  {min(gaps):.2f}   |  {np.mean(gaps):.2f}")
        print("  MAGNITUDE lever: raise repulsion gain c_obs")
        print("  c_obs | adapt heal | min gap | mean gap")
        print("-" * 46)
        for c in (20.0, 40.0, 80.0, 160.0):
            adpt = [_run(s, R, reach_boost=BOOST, reach_kmin=KMIN, c_obs=c) for s in seeds]
            gaps = [_min_gap(r, R) for r in adpt]
            print(f" {c:>5} |   {sum(r.connected for r in adpt):>2}/{m}    |  {min(gaps):.2f}   |  {np.mean(gaps):.2f}")
        print("-" * 46)
        print("=> widening the margin is impotent (min gap flat); only stronger repulsion")
        print("   restores clearance (heal kept). The fix is magnitude, not structure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
