"""Render the clearance cost of the local-reach cure as a two-panel GIF.

Both panels: the SAME migrating flock (Algorithm 2 + β-agent obstacle avoidance)
flows left-to-right past a disk wide enough to sever the base-range flock.

Top    (baseline, base range): the disk SPLITS the flock; the trailing lobe is
       abandoned but every agent keeps its distance — it never enters the dashed
       safety ring (R + rho).
Bottom (adaptive reach): low-degree agents enlarge their reach and pull the lobe
       back across the disk — the flock heals into one component, but the
       re-cohering agents HUG the surface, breaching the dashed safety ring.

Same act, two ledgers: cohesion is bought with clearance.

  python scripts/render_reach_clearance_gif.py --out docs/images/swarm_reach_clearance.gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _flocking import simulate  # noqa: E402
from render_flocking_gif import _component_ids  # noqa: E402

OBS_X = 40.0
GVX = 5.0
RHO = 0.5      # drone radius / required safety margin (dashed ring at R + RHO)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Circle

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="docs/images/swarm_reach_clearance.gif")
    args = ap.parse_args()

    r = 1.2 * 7.0
    R = 13.0
    common = dict(algorithm=2, n=24, seed=args.seed, spread=14.0, c2a=8.0, grad_gain=1.0,
                  c1g=1.0, c2g=0.6, goal=(0.0, 0.0), goal_vel=(GVX, 0.0), goal_moves=True,
                  obs_infl=4.0, c_obs=20.0, steps=1500, record=True,
                  obstacles=((OBS_X, 0.0, R),))
    rb = simulate(**common)
    ra = simulate(reach_boost=3.0, reach_kmin=5, **common)
    T = min(len(rb.traj), len(ra.traj))
    B, A = np.array(rb.traj[:T]), np.array(ra.traj[:T])

    def _gap(P):
        v = P - np.array([OBS_X, 0.0])
        return float((np.sqrt((v * v).sum(-1)) - R).min())

    idsB, kB = _component_ids(B[-1], r)
    colB = plt.cm.tab10(idsB % 10)
    colA = "#2c7fb8"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.6))
    spec = ((ax1, B, colB, f"baseline — disk splits the flock: {kB} lobes, clearance kept"),
            (ax2, A, colA, "adaptive reach — heals to 1 flock, but hugs the disk"))
    arts = []
    xlo = min(B[..., 0].min(), A[..., 0].min()) - 6   # shared scale so both disks render equal
    xhi = max(B[..., 0].max(), A[..., 0].max()) + 6
    for ax, P, col, title in spec:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(-34, 34)
        ax.add_patch(Circle((OBS_X, 0.0), R, color="0.4", zorder=1))
        ax.add_patch(Circle((OBS_X, 0.0), R + RHO, fill=False, ls="--",
                            ec="crimson", lw=1.0, zorder=2))
        sc = ax.scatter(P[0, :, 0], P[0, :, 1], c=col, s=28,
                        edgecolors="k", linewidths=0.3, zorder=3)
        arts.append(sc)

    def update(f):
        arts[0].set_offsets(B[f])
        arts[1].set_offsets(A[f])
        return arts

    fig.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=T, interval=40, blit=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=animation.PillowWriter(fps=25), dpi=80)
    print(f"wrote {out}  (baseline min gap {_gap(B):.2f}, adaptive min gap {_gap(A):.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
