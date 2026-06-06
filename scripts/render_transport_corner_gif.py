"""Render the L-corner "ladder around a corner" ceiling as a side-by-side GIF.

Left:  a short beam (small team) rounds the right-angle corridor corner cleanly.
Right: a longer beam (bigger team) jams at the critical ~45° configuration — no
       reorientation can fit it, because its length exceeds the classical ladder
       bound L_max = (a^(2/3)+b^(2/3))^(3/2) for the corridor widths.

Companion to scripts/coop_transport_corner_phase.py: unlike a straight doorway,
the corner imposes a hard geometric ceiling on team size, however the formation
reshapes.

  python scripts/render_transport_corner_gif.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _coop_transport import simulate_corner, corner_Lmax  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BG = "#0d1117"
WALL = "#30363d"
FREE = "#161b22"
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
    ap.add_argument("--width", type=float, default=4.0)
    ap.add_argument("--n-pass", type=int, default=4)
    ap.add_argument("--n-jam", type=int, default=6)
    ap.add_argument("--spacing", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "images" / "swarm_transport_corner.gif"))
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    W = 30.0
    w = args.width
    runs = {}
    for label, n in (("rounds the corner", args.n_pass), ("jams at the corner", args.n_jam)):
        bl = args.spacing * (n - 1)
        runs[label] = (n, bl, simulate_corner(n=n, bar_len=bl, corridor_w=w,
                                              seed=args.seed, jitter=0.0, W=W,
                                              record=True, n_theta=110))
    nframes = max(len(r[2].centres) for r in runs.values())
    data = {k: _pad(v[2], nframes) for k, v in runs.items()}

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.6))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.03, wspace=0.06)

    colors = {}
    artists = {}
    for ax, (title, (n, bl, res)) in zip(axes, runs.items()):
        ax.set_facecolor(WALL)  # walls = background; carve out the L free space
        ax.set_xlim(12, 33); ax.set_ylim(-3, 17); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        # L-shaped free space: horizontal corridor + vertical corridor
        ax.add_patch(plt.Rectangle((12, 0), W - 12, w, color=FREE, zorder=1))
        ax.add_patch(plt.Rectangle((W - w, 0), w, 17, color=FREE, zorder=1))
        col = GOAL if res.success else CRASH
        ax.set_title(f"{n} drones · {bl:.0f} m beam — {title}",
                     color=col, fontsize=12, fontweight="bold", pad=8)
        cmap = plt.cm.turbo(np.linspace(0.1, 0.9, n))
        colors[title] = cmap
        beam, = ax.plot([], [], "-", color="#8b949e", lw=4.0, zorder=3,
                        solid_capstyle="round")
        scat = ax.scatter([], [], s=70, zorder=5, edgecolors="white", linewidths=0.6)
        flash, = ax.plot([], [], "-", color=CRASH, lw=7.0, alpha=0.0, zorder=6)
        artists[title] = dict(beam=beam, scat=scat, flash=flash, res=res)

    fig.suptitle(f"Carrying a beam around an L-corner  ·  corridor {w:.0f} m  ·  "
                 f"ladder ceiling L_max = {corner_Lmax(w, w):.1f} m",
                 color="#c9d1d9", fontsize=13, y=0.975)

    def update(f):
        out = []
        for title in runs:
            C, T, D = data[title]
            a = artists[title]
            res = a["res"]
            dpos = D[f]
            a["beam"].set_data([dpos[0, 0], dpos[-1, 0]], [dpos[0, 1], dpos[-1, 1]])
            a["scat"].set_offsets(dpos)
            a["scat"].set_color(colors[title])
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
