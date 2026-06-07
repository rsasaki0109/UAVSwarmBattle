"""Render the cut-and-heal contrast as a two-panel GIF.

Both panels: the same migrating flock (Algorithm 2 + β-agent obstacle) passes a
large disk that severs it (above the critical radius).

Top    (no rendezvous): the halves stay split forever — a local flock cannot
       reconnect across the gap (the #143 result).
Bottom (gated global rendezvous): once each agent clears the disk, a pull toward
       the flock's global centroid REUNITES the lobes into one flock again.

  python scripts/render_flocking_rendezvous_gif.py --out docs/images/swarm_flocking_rendezvous.gif
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
R = 9.0


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Circle

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default="docs/images/swarm_flocking_rendezvous.gif")
    args = ap.parse_args()

    rng_r = 1.2 * 7.0
    common = dict(algorithm=2, n=24, seed=args.seed, spread=14.0, c2a=8.0, grad_gain=1.0,
                  c1g=1.0, c2g=0.6, goal=(0.0, 0.0), goal_vel=(5.0, 0.0), goal_moves=True,
                  obs_infl=4.0, c_obs=20.0, obstacles=((OBS_X, 0.0, R),), steps=1500, record=True)
    rb = simulate(c_rdv=0.0, **common)
    rg = simulate(c_rdv=3.0, rdv_gate_x=OBS_X + R + 5.0, **common)
    T = min(len(rb.traj), len(rg.traj))
    A, B = np.array(rb.traj[:T]), np.array(rg.traj[:T])

    idsA, kA = _component_ids(A[-1], rng_r)
    colA = plt.cm.tab10(idsA % 10)
    idsB, kB = _component_ids(B[-1], rng_r)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.4))
    spec = ((ax1, A, colA, f"no rendezvous — stays cut: {kA} flocks"),
            (ax2, B, "#2c7fb8", f"gated global rendezvous — re-merged: {kB} flock"))
    arts = []
    for ax, P, col, title in spec:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11)
        xs = P[..., 0]; ax.set_xlim(xs.min() - 6, xs.max() + 6); ax.set_ylim(-34, 34)
        ax.add_patch(Circle((OBS_X, 0.0), R, color="0.4", zorder=1))
        arts.append(ax.scatter(P[0, :, 0], P[0, :, 1], c=col, s=28,
                               edgecolors="k", linewidths=0.3, zorder=3))

    def update(f):
        arts[0].set_offsets(A[f]); arts[1].set_offsets(B[f])
        return arts

    fig.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=T, interval=40, blit=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=animation.PillowWriter(fps=25), dpi=80)
    print(f"wrote {out}  (top {kA} flocks, bottom {kB} flock)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
