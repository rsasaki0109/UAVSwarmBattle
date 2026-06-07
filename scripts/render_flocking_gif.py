"""Render the flocking-fragmentation contrast as a two-panel GIF.

Left  (Algorithm 1, free flocking): cohesion + alignment only -> the group
      SPLINTERS into several flocks (coloured by final cluster).
Right (Algorithm 2, + navigational structure): the same start reunites into one
      α-lattice and MIGRATES cohesively after a moving goal (the white star).

  python scripts/render_flocking_gif.py --seed 2 --out docs/images/swarm_flocking.gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _flocking import simulate, _components  # noqa: E402


def _component_ids(q, radius):
    n = len(q)
    diff = q[:, None, :] - q[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))
    adj = (dist < radius) & ~np.eye(n, dtype=bool)
    ids = -np.ones(n, dtype=int)
    cid = 0
    for s in range(n):
        if ids[s] >= 0:
            continue
        stack = [s]
        ids[s] = cid
        while stack:
            u = stack.pop()
            for v in np.nonzero(adj[u] & (ids < 0))[0]:
                ids[v] = cid
                stack.append(v)
        cid += 1
    return ids, cid


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--steps", type=int, default=750)
    ap.add_argument("--out", default="docs/images/swarm_flocking.gif")
    args = ap.parse_args()

    r = 1.2 * 7.0
    gvx, gvy = 4.0, 1.5
    common = dict(n=args.n, seed=args.seed, spread=12.0, c2a=14.0, steps=args.steps,
                  grad_gain=1.0, record=True)
    r1 = simulate(algorithm=1, **common)
    r2 = simulate(algorithm=2, goal=(0.0, 0.0), goal_vel=(gvx, gvy),
                  goal_moves=True, **common)
    T = min(len(r1.traj), len(r2.traj))
    A, B = np.array(r1.traj[:T]), np.array(r2.traj[:T])

    ids1, k1 = _component_ids(A[-1], r)
    cmap = plt.cm.tab10
    colours1 = cmap(ids1 % 10)

    def _lims(P, pad=8.0):
        lo = P.reshape(-1, 2).min(0) - pad
        hi = P.reshape(-1, 2).max(0) + pad
        span = (hi - lo).max()
        cx, cy = (lo + hi) / 2
        return cx - span / 2, cx + span / 2, cy - span / 2, cy + span / 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, P in ((ax1, A), (ax2, B)):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        x0, x1, y0, y1 = _lims(P)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax1.set_title(f"Algorithm 1: free flocking → {k1} flocks", fontsize=12)
    ax2.set_title("Algorithm 2: + navigational structure → one flock", fontsize=12)

    sc1 = ax1.scatter(A[0, :, 0], A[0, :, 1], c=colours1, s=40, edgecolors="k", linewidths=0.4)
    sc2 = ax2.scatter(B[0, :, 0], B[0, :, 1], c="#2c7fb8", s=40, edgecolors="k", linewidths=0.4)
    star, = ax2.plot([0], [0], marker="*", color="goldenrod", markersize=20,
                     markeredgecolor="k", linestyle="none")
    dt_rec = 0.02 * 5

    def update(f):
        sc1.set_offsets(A[f])
        sc2.set_offsets(B[f])
        gx, gy = gvx * f * dt_rec, gvy * f * dt_rec
        star.set_data([gx], [gy])
        return sc1, sc2, star

    anim = animation.FuncAnimation(fig, update, frames=T, interval=40, blit=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=animation.PillowWriter(fps=25), dpi=80)
    print(f"wrote {out}  ({k1} flocks on the left, 1 on the right)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
