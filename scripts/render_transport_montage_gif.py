"""Team-size montage: fixed-orientation carrying collapses as the team grows,
reorienting carrying does not — all through the SAME doorway.

A 2×4 grid: top row holds the beam fixed (perpendicular to travel), bottom row
lets it reorient; columns are N = 2, 4, 6, 8 drones (beam length = 2.5·(N−1) m).
At a fixed doorway, the fixed row passes only at N=2 and slams the wall for every
larger team; the adaptive row threads at every team size.

Visual companion to scripts/coop_transport_doorway_phase.py (the TeamHOI
"team-size-agnostic" cell).

  python scripts/render_transport_montage_gif.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _coop_transport import simulate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BG = "#0d1117"
PANEL = "#161b22"
WALL = "#30363d"
GRID = "#21262d"
GOAL = "#3fb950"
CRASH = "#f85149"


def _pad(res, n):
    C, T, D = list(res.centres), list(res.thetas), list(res.drones)
    while len(C) < n:
        C.append(C[-1]); T.append(T[-1]); D.append(D[-1])
    return C, T, D


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--gap", type=float, default=6.0)
    ap.add_argument("--spacing", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "images" / "swarm_transport_montage.gif"))
    ap.add_argument("--fps", type=int, default=18)
    args = ap.parse_args()

    GAPY, W = 25.0, 50.0
    arms = [("Fixed", False), ("Reorienting", True)]
    cells = {}
    for arm_name, ad in arms:
        for n in args.ns:
            bl = args.spacing * (n - 1)
            cells[(arm_name, n)] = (bl, simulate(n=n, bar_len=bl, gap_w=args.gap,
                                                 seed=args.seed, adaptive=ad,
                                                 jitter=0.0, record=True))
    nframes = min(max(len(v[1].centres) for v in cells.values()), 240)
    data = {k: _pad(v[1], nframes) for k, v in cells.items()}

    nrow, ncol = 2, len(args.ns)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol + 0.6, 2.5 * nrow + 0.8))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.02,
                        wspace=0.06, hspace=0.10)

    artists = {}
    for r, (arm_name, ad) in enumerate(arms):
        for cc, n in enumerate(args.ns):
            ax = axes[r][cc]
            bl, res = cells[(arm_name, n)]
            ax.set_facecolor(PANEL)
            ax.set_xlim(0, W); ax.set_ylim(0, W); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
            wx, wt = 25.0, 1.5
            ax.add_patch(plt.Rectangle((wx - wt / 2, 0), wt, GAPY - args.gap / 2,
                                       color=WALL, zorder=2))
            ax.add_patch(plt.Rectangle((wx - wt / 2, GAPY + args.gap / 2), wt,
                                       W - (GAPY + args.gap / 2), color=WALL, zorder=2))
            if r == 0:
                ax.set_title(f"N = {n}", color="#c9d1d9", fontsize=12, pad=6)
            if cc == 0:
                ax.set_ylabel(arm_name, color="#c9d1d9", fontsize=12)
            cmap = plt.cm.turbo(np.linspace(0.1, 0.9, n))
            beam, = ax.plot([], [], "-", color="#8b949e", lw=3.0, zorder=3,
                            solid_capstyle="round")
            scat = ax.scatter([], [], s=34, zorder=5, edgecolors="white", linewidths=0.4)
            flash, = ax.plot([], [], "-", color=CRASH, lw=4.5, alpha=0.0, zorder=6)
            artists[(arm_name, n)] = dict(beam=beam, scat=scat, flash=flash,
                                          res=res, cmap=cmap)

    fig.suptitle("Same doorway, growing team — fixed carrying collapses, reorienting does not",
                 color="#c9d1d9", fontsize=13, y=0.965)

    def update(f):
        out = []
        for key, a in artists.items():
            C, T, D = data[key]
            res = a["res"]
            dpos = D[f]
            a["beam"].set_data([dpos[0, 0], dpos[-1, 0]], [dpos[0, 1], dpos[-1, 1]])
            a["scat"].set_offsets(dpos)
            a["scat"].set_color(a["cmap"])
            if (not res.success) and res.collide_step >= 0 and f >= res.collide_step:
                a["flash"].set_data([dpos[0, 0], dpos[-1, 0]], [dpos[0, 1], dpos[-1, 1]])
                a["flash"].set_alpha(0.85 if (f - res.collide_step) % 6 < 3 else 0.2)
            out += [a["beam"], a["scat"], a["flash"]]
        return out

    anim = FuncAnimation(fig, update, frames=nframes, interval=1000 / args.fps, blit=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"[gif] {out}  ({out.stat().st_size // 1024} KB, {nframes} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
