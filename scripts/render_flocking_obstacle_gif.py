"""Render the obstacle-split contrast as a two-panel GIF.

Both panels: the SAME migrating flock (Algorithm 2, navigational structure +
β-agent obstacle avoidance) flows left-to-right past a disk on its path.

Left  (small disk, below the critical radius): the flock threads it INTACT — one
      cohesive α-lattice the whole way.
Right (large disk, above the critical radius): the disk SPLITS the flock into two
      lobes (coloured by final cluster) that the navigational term migrates onward
      but never re-merges.

  python scripts/render_flocking_obstacle_gif.py --out docs/images/swarm_flocking_obstacle.gif
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


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Circle

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--out", default="docs/images/swarm_flocking_obstacle.gif")
    args = ap.parse_args()

    r = 1.2 * 7.0
    common = dict(algorithm=2, n=24, seed=args.seed, spread=14.0, c2a=8.0, grad_gain=1.0,
                  c1g=1.0, c2g=0.6, goal=(0.0, 0.0), goal_vel=(GVX, 0.0), goal_moves=True,
                  obs_infl=4.0, c_obs=20.0, steps=1200, record=True)
    R_small, R_big = 2.0, 9.0
    rs = simulate(obstacles=((OBS_X, 0.0, R_small),), **common)
    rb = simulate(obstacles=((OBS_X, 0.0, R_big),), **common)
    T = min(len(rs.traj), len(rb.traj))
    A, B = np.array(rs.traj[:T]), np.array(rb.traj[:T])

    idsB, kB = _component_ids(B[-1], r)
    colB = plt.cm.tab10(idsB % 10)
    idsA, kA = _component_ids(A[-1], r)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.4))
    spec = ((ax1, A, R_small, "#2c7fb8", f"small disk (R={R_small:.0f}) — threads intact: {kA} flock"),
            (ax2, B, R_big, colB, f"large disk (R={R_big:.0f}) — split: {kB} flocks"))
    arts = []
    for ax, P, R, col, title in spec:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11)
        xs = P[..., 0]; ax.set_xlim(xs.min() - 6, xs.max() + 6)
        ax.set_ylim(-34, 34)
        ax.add_patch(Circle((OBS_X, 0.0), R, color="0.4", zorder=1))
        sc = ax.scatter(P[0, :, 0], P[0, :, 1], c=col, s=28, edgecolors="k", linewidths=0.3, zorder=3)
        arts.append(sc)

    def update(f):
        arts[0].set_offsets(A[f])
        arts[1].set_offsets(B[f])
        return arts

    fig.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=T, interval=40, blit=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=animation.PillowWriter(fps=25), dpi=80)
    print(f"wrote {out}  (top {kA} flock, bottom {kB} flocks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
