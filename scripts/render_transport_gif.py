"""Render the cooperative-transport doorway demo as a side-by-side GIF.

Left:  FIXED orientation  -- the rigid beam stays perpendicular to travel and
       slams into the wall (the carried payload is wider than the gap).
Right: ADAPTIVE reorientation -- the same team rotates the beam to align with
       travel and threads the same gap.

Both panels run the SAME seed, so the only difference on screen is whether the
formation is allowed to reorient. This is the visual companion to
scripts/coop_transport_doorway_phase.py (TeamHOI-style "any team size" probe).

  python scripts/render_transport_gif.py
  python scripts/render_transport_gif.py --n 6 --bar-len 12 --gap 5
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
    """Pad a recorded run to n frames by repeating its last pose."""
    C = list(res.centres)
    T = list(res.thetas)
    D = list(res.drones)
    while len(C) < n:
        C.append(C[-1]); T.append(T[-1]); D.append(D[-1])
    return C, T, D


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--bar-len", type=float, default=10.0)
    ap.add_argument("--gap", type=float, default=5.0)
    ap.add_argument("--wall-x", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "images" / "swarm_transport_doorway.gif"))
    ap.add_argument("--fps", type=int, default=24)
    args = ap.parse_args()

    common = dict(n=args.n, bar_len=args.bar_len, gap_w=args.gap,
                  wall_x=args.wall_x, seed=args.seed, record=True, jitter=0.0)
    runs = {
        "Fixed orientation": simulate(adaptive=False, **common),
        "Adaptive reorientation": simulate(adaptive=True, **common),
    }
    nframes = max(len(r.centres) for r in runs.values())
    nframes = min(nframes, 600)
    data = {k: _pad(v, nframes) for k, v in runs.items()}

    gy = args.wall_x  # gap centred at world centre (25); wall vertical at wall_x
    GAPY = 25.0
    W = 50.0
    colors = plt.cm.turbo(np.linspace(0.1, 0.9, args.n))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.6))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.03, wspace=0.06)

    artists = {}
    for ax, (title, res) in zip(axes, runs.items()):
        ax.set_facecolor(PANEL)
        ax.set_xlim(0, W); ax.set_ylim(0, W); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        col = GOAL if res.success else CRASH
        ax.set_title(title, color=col, fontsize=13, fontweight="bold", pad=8)
        # wall + doorway
        wx, wt = args.wall_x, 1.5
        ax.add_patch(plt.Rectangle((wx - wt / 2, 0), wt, GAPY - args.gap / 2,
                                   color=WALL, zorder=2))
        ax.add_patch(plt.Rectangle((wx - wt / 2, GAPY + args.gap / 2), wt,
                                   W - (GAPY + args.gap / 2), color=WALL, zorder=2))
        # goal line
        ax.axvline(42.0, color=GOAL, lw=1.0, ls=":", alpha=0.5, zorder=1)
        beam, = ax.plot([], [], "-", color="#8b949e", lw=4.0, zorder=3,
                        solid_capstyle="round")
        trail, = ax.plot([], [], "-", color="#58a6ff", lw=1.2, alpha=0.5, zorder=2)
        scat = ax.scatter([], [], s=70, zorder=5, edgecolors="white", linewidths=0.6)
        rings = ax.scatter([], [], s=320, facecolors="none",
                           edgecolors=colors, linewidths=1.2, alpha=0.5, zorder=4)
        flash, = ax.plot([], [], "-", color=CRASH, lw=6.0, alpha=0.0, zorder=6)
        artists[title] = dict(beam=beam, trail=trail, scat=scat, rings=rings,
                              flash=flash, res=res)

    fig.suptitle("Cooperative transport through a doorway  ·  "
                 f"{args.n} drones carry a {args.bar_len:.0f} m beam, gap {args.gap:.0f} m",
                 color="#c9d1d9", fontsize=13, y=0.985)

    def update(f):
        out = []
        for title in runs:
            C, T, D = data[title]
            a = artists[title]
            res = a["res"]
            dpos = D[f]
            a["beam"].set_data([dpos[0, 0], dpos[-1, 0]], [dpos[0, 1], dpos[-1, 1]])
            a["scat"].set_offsets(dpos)
            a["scat"].set_color(colors)
            a["rings"].set_offsets(dpos)
            cs = np.array(C[: f + 1])
            a["trail"].set_data(cs[:, 0], cs[:, 1])
            # crash flash on the collision frame onward (fixed arm)
            if (not res.success) and res.collide_step >= 0 and f >= res.collide_step:
                a["flash"].set_data([dpos[0, 0], dpos[-1, 0]],
                                    [dpos[0, 1], dpos[-1, 1]])
                a["flash"].set_alpha(0.8 if (f - res.collide_step) % 6 < 3 else 0.2)
            out += [a["beam"], a["trail"], a["scat"], a["rings"], a["flash"]]
        return out

    anim = FuncAnimation(fig, update, frames=nframes, interval=1000 / args.fps,
                         blit=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    kb = out.stat().st_size // 1024
    print(f"[gif] {out}  ({kb} KB, {nframes} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
