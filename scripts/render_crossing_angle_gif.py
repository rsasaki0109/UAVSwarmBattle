"""Render the angle-gating of the crossing jam: perpendicular slips past, head-on jams.

Both panels have NO convention (bias=0), same flock sizes and seed:

Left   (90°, perpendicular): the two flocks cross at right angles and slip past —
       briefly deflected, but both clear the crossing. No jam.
Right  (180°, head-on): the same two flocks driven straight at each other JAM at
       the centre and stall there. The jam is a head-on phenomenon.

  python scripts/render_crossing_angle_gif.py --out docs/images/swarm_crossing_angle.gif
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
    ap.add_argument("--out", default="docs/images/swarm_crossing_angle.gif")
    args = ap.parse_args()

    common = dict(seed=args.seed, n=24, steps=1400, bias=0.0, record=True)
    rp = simulate_crossing(encounter_angle=90.0, **common)
    rh = simulate_crossing(encounter_angle=180.0, **common)
    T = min(len(rp.traj), len(rh.traj))
    Pp, H = np.array(rp.traj[:T]), np.array(rh.traj[:T])
    grp = rp.grp
    col = np.where(grp == 0, "#2c7fb8", "#e6550d")

    allx = np.concatenate([Pp[..., 0].ravel(), H[..., 0].ravel()])
    ally = np.concatenate([Pp[..., 1].ravel(), H[..., 1].ravel()])
    lim = max(abs(allx).max(), abs(ally).max()) + 6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 5.4))
    spec = ((ax1, Pp, "90° perpendicular — they slip past (no jam)"),
            (ax2, H, "180° head-on — they JAM at the centre"))
    arts = []
    for ax, D, title in spec:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.axhline(0.0, color="0.9", lw=1, zorder=0)
        ax.axvline(0.0, color="0.9", lw=1, zorder=0)
        sc = ax.scatter(D[0, :, 0], D[0, :, 1], c=col, s=26,
                        edgecolors="k", linewidths=0.3, zorder=3)
        arts.append(sc)

    def update(f):
        arts[0].set_offsets(Pp[f]); arts[1].set_offsets(H[f])
        return arts

    fig.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=T, interval=40, blit=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=animation.PillowWriter(fps=25), dpi=80)
    print(f"wrote {out}  (90° passed={rp.passed}, 180° passed={rh.passed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
