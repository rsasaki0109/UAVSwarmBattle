"""Render the placement effect as a two-panel GIF: same adoption budget, clumped vs spread.

Both panels: K flocks (distinct colours) converge on a shared hub, and the SAME
number of agents follow the right-of-way veer. Adopters wear a black ring;
free-riders are drawn pale.

Left   (CLUSTERED): the whole budget goes to a few flocks; the un-adopting flocks
       stay coherent walls and the hub JAMS.
Right  (MIXED): the same budget is spread thin across every flock; the cohesion
       lattice drags each flock's free-riders through and the hub becomes a ROUNDABOUT.

  python scripts/render_critmass_gif.py --out docs/images/swarm_critmass.gif
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
_spec = importlib.util.spec_from_file_location(
    "_flocking_hub", str(Path(__file__).resolve().parent / "_flocking_hub.py"))
_H = importlib.util.module_from_spec(_spec)
sys.modules["_flocking_hub"] = _H
_spec.loader.exec_module(_H)
simulate_hub = _H.simulate_hub

K, PER_FLOCK = 6, 10
N = K * PER_FLOCK


def mask_mixed(budget):
    m = np.zeros(N, bool)
    base, rem = divmod(budget, K)
    for k in range(K):
        c = base + (1 if k < rem else 0)
        m[k * PER_FLOCK: k * PER_FLOCK + c] = True
    return m


def mask_clustered(budget):
    m = np.zeros(N, bool)
    m[:budget] = True
    return m


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=15)
    ap.add_argument("--out", default="docs/images/swarm_critmass.gif")
    args = ap.parse_args()

    mc = mask_clustered(args.budget)
    mm = mask_mixed(args.budget)
    common = dict(K=K, seed=args.seed, steps=1600, record=True)
    rc = simulate_hub(bias=1.0, adopt=mc, **common)
    rm = simulate_hub(bias=1.0, adopt=mm, **common)
    T = min(len(rc.traj), len(rm.traj))
    C, M = np.array(rc.traj[:T]), np.array(rm.traj[:T])
    grp = rc.grp
    base = plt.cm.tab10(grp % 10)

    allc = np.concatenate([C.reshape(-1, 2), M.reshape(-1, 2)])
    lim = np.abs(allc).max() + 6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 5.4))
    spec = ((ax1, C, mc, f"CLUSTERED {args.budget}/{N} — free-rider flocks wall the hub"),
            (ax2, M, mm, f"MIXED {args.budget}/{N} — cohesion drags the rest through"))
    arts = []
    for ax, D, msk, title in spec:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.plot(0, 0, "+", color="0.7", ms=10, zorder=0)
        edge = np.where(msk, "k", "0.7")
        lw = np.where(msk, 1.1, 0.3)
        # pale fill for free-riders, full colour for adopters
        face = base.copy()
        face[~msk, 3] = 0.35
        sc = ax.scatter(D[0, :, 0], D[0, :, 1], c=face, s=28,
                        edgecolors=edge, linewidths=lw, zorder=3)
        arts.append(sc)

    def update(f):
        arts[0].set_offsets(C[f]); arts[1].set_offsets(M[f])
        return arts

    fig.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=T, interval=40, blit=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=animation.PillowWriter(fps=25), dpi=80)
    print(f"wrote {out}  (clustered passed={rc.n_passed}/{K}, mixed passed={rm.n_passed}/{K})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
