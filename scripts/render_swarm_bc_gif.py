"""Side-by-side GIF of the two distilled teammate-token policies on the antipodal hub.

Left:  student<-plain  — distilled from a symmetric avoider; reimports the deadlock
       (every agent mirror-swerves into the hub and collides).
Right: student<-conv   — distilled from a convention teacher; the SAME architecture
       learned the right-of-way and spirals into a clean roundabout.

Both are the same NumPy deep-set, both trained to bc_mse ~1e-4 on random scenes
only — the only difference is whether the teacher had a convention. Loads the models
cached by scripts/swarm_bc_symmetry_phase.py.

  python scripts/swarm_bc_symmetry_phase.py --episodes 1   # writes the cache
  python scripts/render_swarm_bc_gif.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _swarm_policy as sp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "results" / "swarm_bc_models.npz"
BG = "#0d1117"
PANEL = "#161b22"
GRID = "#21262d"
CRASH = "#f85149"
GOAL = "#3fb950"
PKEYS = ["phi1", "phi1b", "phi2", "phi2b", "ego1", "ego1b",
         "out1", "out1b", "out2", "out2b"]
SKEYS = ["em", "es", "pm", "ps"]


def _load(prefix, z):
    P = {k: z[f"{prefix}_{k}"] for k in PKEYS}
    stats = {k: z[f"{prefix}_s_{k}"] for k in SKEYS}
    return sp.make_student_controller(P, stats)


def _pad(traj, n):
    t = list(traj)
    while len(t) < n:
        t.append(t[-1])
    return t


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "images" / "swarm_bc_convention.gif"))
    ap.add_argument("--fps", type=int, default=24)
    args = ap.parse_args()

    if not CACHE.exists():
        print(f"missing {CACHE}; run swarm_bc_symmetry_phase.py first")
        return 1
    z = np.load(CACHE)
    ctrls = {"student ← symmetric teacher": _load("plain", z),
             "student ← convention teacher": _load("conv", z)}

    # find a seed where the plain student collides and the conv student clears
    seed = 0
    for s in range(200):
        rng = np.random.default_rng(50_000 + s)
        st, gl = sp.antipodal(args.n, rng)
        rp = sp.rollout(st, gl, ctrls["student ← symmetric teacher"], record=True)
        rc = sp.rollout(st, gl, ctrls["student ← convention teacher"], record=True)
        if (not rp.success) and rc.success:
            seed = 50_000 + s
            break
    rng = np.random.default_rng(seed)
    start, goal = sp.antipodal(args.n, rng)
    runs = {k: sp.rollout(start, goal, c, record=True) for k, c in ctrls.items()}
    nf = min(max(len(r.traj) for r in runs.values()), 400)
    data = {k: _pad(v.traj, nf) for k, v in runs.items()}
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, args.n))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.6))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.03, wspace=0.06)
    art = {}
    for ax, (title, res) in zip(axes, runs.items()):
        ax.set_facecolor(PANEL); ax.set_aspect("equal")
        ax.set_xlim(-11, 11); ax.set_ylim(-11, 11)
        ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_color(GRID)
        ax.set_title(title, color=(GOAL if res.success else CRASH),
                     fontsize=12, fontweight="bold", pad=8)
        ax.scatter(goal[:, 0], goal[:, 1], s=90, marker="*", c=colors,
                   alpha=0.45, zorder=2)
        trails = [ax.plot([], [], "-", color=colors[i], lw=1.3, alpha=0.55, zorder=3)[0]
                  for i in range(args.n)]
        scat = ax.scatter(start[:, 0], start[:, 1], s=80, c=colors,
                          edgecolors="white", linewidths=0.6, zorder=5)
        flash = ax.scatter(start[:, 0], start[:, 1], s=320, facecolors="none",
                           edgecolors=CRASH, linewidths=2.5, zorder=6, alpha=0.0)
        art[title] = dict(trails=trails, scat=scat, flash=flash, res=res)

    fig.suptitle(f"Same deep-set teammate-token policy, two teachers — antipodal swap, N={args.n}",
                 color="#c9d1d9", fontsize=13, y=0.975)

    def update(f):
        out = []
        for title in runs:
            a = art[title]; res = a["res"]; T = data[title]
            pos = T[f]
            a["scat"].set_offsets(pos)
            for i, tr in enumerate(a["trails"]):
                arr = np.array([T[k][i] for k in range(f + 1)])
                tr.set_data(arr[:, 0], arr[:, 1])
            if (not res.success) and f >= len(res.traj) - 1:
                a["flash"].set_offsets(pos)
                a["flash"].set_alpha(0.85 if (f - (len(res.traj) - 1)) % 6 < 3 else 0.25)
            out += a["trails"] + [a["scat"], a["flash"]]
        return out

    anim = FuncAnimation(fig, update, frames=nf, interval=1000 / args.fps, blit=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"[gif] {out}  ({out.stat().st_size // 1024} KB, seed {seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
