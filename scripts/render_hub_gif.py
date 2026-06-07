"""Render a K-way flocking hub as a two-panel GIF: jam vs roundabout.

Both panels: K flocks (distinct colours) start on a ring and migrate through the
centre to the antipodal side.

Left   (no convention): they JAM at the hub — a mutual wall of α-repulsion — and
       stall there.
Right  (right-of-way bias): each agent veers right of its goal heading, so the
       fan-in becomes a ROUNDABOUT and every flock clears the hub.

  python scripts/render_hub_gif.py --out docs/images/swarm_hub.gif
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


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--out", default="docs/images/swarm_hub.gif")
    args = ap.parse_args()

    common = dict(K=args.K, seed=args.seed, steps=1600, record=True)
    rj = simulate_hub(bias=0.0, **common)
    rr = simulate_hub(bias=1.0, **common)
    T = min(len(rj.traj), len(rr.traj))
    J, R = np.array(rj.traj[:T]), np.array(rr.traj[:T])
    grp = rj.grp
    col = plt.cm.tab10(grp % 10)

    allc = np.concatenate([J.reshape(-1, 2), R.reshape(-1, 2)])
    lim = np.abs(allc).max() + 6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 5.4))
    spec = ((ax1, J, f"{args.K}-way hub, no convention — JAM"),
            (ax2, R, f"{args.K}-way hub, right-of-way — ROUNDABOUT"))
    arts = []
    for ax, D, title in spec:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.plot(0, 0, "+", color="0.7", ms=10, zorder=0)
        sc = ax.scatter(D[0, :, 0], D[0, :, 1], c=col, s=24,
                        edgecolors="k", linewidths=0.3, zorder=3)
        arts.append(sc)

    def update(f):
        arts[0].set_offsets(J[f]); arts[1].set_offsets(R[f])
        return arts

    fig.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=T, interval=40, blit=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=animation.PillowWriter(fps=25), dpi=80)
    print(f"wrote {out}  (jam passed={rj.n_passed}/{args.K}, roundabout passed={rr.n_passed}/{args.K})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
