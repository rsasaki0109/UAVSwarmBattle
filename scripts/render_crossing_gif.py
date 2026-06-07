"""Render two crossing flocks as a two-panel GIF: jam vs right-of-way pass.

Both panels: two Olfati-Saber flocks driven head-on (blue left→right, orange
right→left), identical start.

Top    (no convention): they JAM at the centre — a mutual wall of α-repulsion —
       and stall there while their goals run on ahead (they never collide).
Bottom (right-of-way bias): each agent veers right of its goal heading, so the
       flocks pass right-shoulder-to-right-shoulder and clear the crossing cleanly.

  python scripts/render_crossing_gif.py --out docs/images/swarm_crossing.gif
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


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="docs/images/swarm_crossing.gif")
    args = ap.parse_args()

    common = dict(seed=args.seed, n=24, steps=1400, record=True)
    rj = simulate_crossing(bias=0.0, **common)
    rp = simulate_crossing(bias=1.0, **common)
    T = min(len(rj.traj), len(rp.traj))
    J, P = np.array(rj.traj[:T]), np.array(rp.traj[:T])
    grp = rj.grp
    col = np.where(grp == 0, "#2c7fb8", "#e6550d")

    allx = np.concatenate([J[..., 0].ravel(), P[..., 0].ravel()])
    ally = np.concatenate([J[..., 1].ravel(), P[..., 1].ravel()])
    xlo, xhi = allx.min() - 6, allx.max() + 6
    ylo, yhi = ally.min() - 6, ally.max() + 6

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.6))
    spec = ((ax1, J, "no convention — the flocks JAM at the centre"),
            (ax2, P, "right-of-way — they pass on the right and clear it"))
    arts = []
    for ax, D, title in spec:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.axvline(0.0, color="0.85", lw=1, zorder=0)
        sc = ax.scatter(D[0, :, 0], D[0, :, 1], c=col, s=28,
                        edgecolors="k", linewidths=0.3, zorder=3)
        arts.append(sc)

    def update(f):
        arts[0].set_offsets(J[f]); arts[1].set_offsets(P[f])
        return arts

    fig.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=T, interval=40, blit=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=animation.PillowWriter(fps=25), dpi=80)
    print(f"wrote {out}  (jam pass_step={rj.pass_step}, row pass_step={rp.pass_step})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
