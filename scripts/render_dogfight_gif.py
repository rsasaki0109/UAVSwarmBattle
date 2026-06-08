"""Render two 1-v-1 dogfights side by side: matched stalemate vs an agility win.

Left  : a matched duel (equal speed & turn rate) — the symmetric "circle of death",
        neither UAV ever holds the other's six (a stalemate).
Right : one UAV given a turn-rate edge — it out-turns onto the opponent's six and wins.

Each UAV is drawn as an arrow (heading); a faint trail shows its recent path; the
winner's nose flashes a marker when it holds the six.

  python scripts/render_dogfight_gif.py --out docs/images/swarm_dogfight.gif
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
_spec = importlib.util.spec_from_file_location(
    "_dogfight", str(Path(__file__).resolve().parent / "_dogfight.py"))
_D = importlib.util.module_from_spec(_spec)
sys.modules["_dogfight"] = _D
_spec.loader.exec_module(_D)


def _run(seed, **kw):
    r = _D.duel(seed=seed, record=True, **kw)
    P = np.array([[s0, s1] for s0, s1 in r.traj])   # (T, 2 drones, 3)
    return P, r.winner


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="docs/images/swarm_dogfight.gif")
    args = ap.parse_args()

    Pp, _ = _run(args.seed)                                  # parity → stalemate
    Pa, wa = _run(args.seed, wmax0=2.55, wmax1=1.5)          # P0 1.7× turn rate → wins
    T = min(len(Pp), len(Pa))
    Pp, Pa = Pp[:T], Pa[:T]
    arena = 40.0
    col = ["#1f77b4", "#d62728"]   # P0 blue, P1 red

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 5.4))
    spec = ((ax1, Pp, "Matched (equal agility) — STALEMATE circle"),
            (ax2, Pa, "P0 out-turns (1.7× turn rate) — P0 takes the six"))
    arts = []
    for ax, P, title in spec:
        ax.set_aspect("equal"); ax.set_xlim(0, arena); ax.set_ylim(0, arena)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)
        trails = [ax.plot([], [], "-", color=col[k], lw=1.2, alpha=0.5)[0] for k in range(2)]
        quivs = ax.quiver(P[0, :, 0], P[0, :, 1], np.cos(P[0, :, 2]), np.sin(P[0, :, 2]),
                          color=col, scale=22, width=0.010, zorder=4)
        arts.append((trails, quivs, P))

    def update(f):
        out = []
        for trails, quivs, P in arts:
            for k in range(2):
                g = max(0, f - 60)
                trails[k].set_data(P[g:f + 1, k, 0], P[g:f + 1, k, 1])
            quivs.set_offsets(P[f, :, :2])
            quivs.set_UVC(np.cos(P[f, :, 2]), np.sin(P[f, :, 2]))
            out += trails + [quivs]
        return out

    fig.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=T, interval=40, blit=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=animation.PillowWriter(fps=25), dpi=80)
    print(f"wrote {out}  (right-panel winner = P{wa})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
