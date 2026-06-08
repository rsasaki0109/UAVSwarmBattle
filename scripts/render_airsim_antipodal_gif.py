"""Top-down GIF of the AirSim N=4 antipodal hub: stock jam vs convention.

Reads the per-drone episode JSONs (steps[].true_pos) written by the multi
runner for two episodes — a STOCK (lateral_bias=0) run that collides at the
hub and a CONVENTION (lateral_bias>0) run that clears it — and animates the
x-y top-down view side by side. A red ring flags the frame a drone first
registers a collision.

  python scripts/render_airsim_antipodal_gif.py \
      <stock_episode_dir> <conv_episode_dir> --out docs/images/swarm_airsim_antipodal.gif
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_episode(ep_dir: Path):
    """Return (positions [n_drones, T, 2], collided [n_drones, T] bool, names)."""
    files = sorted(ep_dir.glob("episode_000_drone_*.json"))
    paths, cols, names = [], [], []
    for f in files:
        d = json.load(open(f))
        steps = d["steps"]
        paths.append(np.array([s["true_pos"][:2] for s in steps]))
        cols.append(np.array([bool(s["collision"]) for s in steps]))
        names.append(d["meta"]["drone_name"])
    T = min(len(p) for p in paths)
    pos = np.array([p[:T] for p in paths])
    col = np.array([c[:T] for c in cols])
    return pos, col, names


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    ap = argparse.ArgumentParser()
    ap.add_argument("stock_dir")
    ap.add_argument("conv_dir")
    ap.add_argument("--out", default="docs/images/swarm_airsim_antipodal.gif")
    ap.add_argument("--title-left", default="STOCK (no convention) — JAM at the hub")
    ap.add_argument("--title-right", default="RIGHT-OF-WAY convention — ROUNDABOUT")
    args = ap.parse_args()

    sp, sc, names = _load_episode(Path(args.stock_dir))
    cp, cc, _ = _load_episode(Path(args.conv_dir))
    # Don't truncate to the shorter (collision) run — that would clip the
    # convention's roundabout. Pad the shorter episode by freezing its last
    # frame so both play to completion.
    T = max(sp.shape[1], cp.shape[1])

    def _pad(P, C, T):
        t = P.shape[1]
        if t >= T:
            return P[:, :T], C[:, :T]
        P = np.concatenate([P, np.repeat(P[:, -1:], T - t, axis=1)], axis=1)
        C = np.concatenate([C, np.repeat(C[:, -1:], T - t, axis=1)], axis=1)
        return P, C

    sp, sc = _pad(sp, sc, T)
    cp, cc = _pad(cp, cc, T)
    n = sp.shape[0]
    col = plt.cm.tab10(np.arange(n) % 10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 5.4))
    spec = ((ax1, sp, sc, args.title_left),
            (ax2, cp, cc, args.title_right))
    trails, heads, rings = [], [], []
    for ax, P, C, title in spec:
        ax.set_aspect("equal"); ax.set_title(title, fontsize=11)
        ax.set_xlim(0, 60); ax.set_ylim(0, 60)
        ax.set_xticks([]); ax.set_yticks([])
        ax.plot(30, 30, "+", color="0.6", ms=12, zorder=0)
        for k in range(n):
            ax.plot(P[k, 0, 0], P[k, 0, 1], "o", color=col[k], ms=5, alpha=0.4, zorder=1)
            ax.plot(P[k, -1, 0], P[k, -1, 1], "*", color=col[k], ms=11, alpha=0.5, zorder=1)
        tr = [ax.plot([], [], "-", color=col[k], lw=1.4, alpha=0.7)[0] for k in range(n)]
        hd = ax.scatter(P[:, 0, 0], P[:, 0, 1], c=col, s=44, edgecolors="k",
                        linewidths=0.4, zorder=4)
        rg = ax.scatter([], [], s=180, facecolors="none", edgecolors="red",
                        linewidths=1.8, zorder=5)
        trails.append(tr); heads.append(hd); rings.append(rg)

    def update(f):
        arts = []
        for (P, C), tr, hd, rg in ((( sp, sc), trails[0], heads[0], rings[0]),
                                   ((cp, cc), trails[1], heads[1], rings[1])):
            for k in range(n):
                tr[k].set_data(P[k, :f + 1, 0], P[k, :f + 1, 1])
            hd.set_offsets(P[:, f, :])
            hit = C[:, f]
            rg.set_offsets(P[hit, f, :] if hit.any() else np.empty((0, 2)))
            arts += tr + [hd, rg]
        return arts

    fig.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=T, interval=60, blit=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=animation.PillowWriter(fps=18), dpi=80)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
