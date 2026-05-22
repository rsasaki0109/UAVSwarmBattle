"""L: 3-cell aggregator response curve at σ=0.5 — shape varies dramatically.

After G/J/H/I/J-σ/K established the U-shape mechanism on intersection
cells, L tests the boundary further by adding the 4-way cell (30 obs,
mid-density 3D escape volume). Three distinct aggregator response
shapes emerge:

- wave  (intersection, 0 static obs, σ=3 needed to read off U)
  U-shape:    vanilla is valley, both extremes recover
- 4-way (multi-drone, 30 static obs, 3D escape volume, σ=0.5)
  Monotonic:  argmin (t=0.1) is the WORST aggregator (55%), uniform
              (t=10) is BEST (85%); the more we trust cost, the worse
- peer  (multi-drone, 120 static obs, dense coordination, σ=0.5)
  Flat:       all temperatures collapse to 40% (cost landscape is
              coordination-chaos-flat)

Refutes the "t=0.3 is the universal default" prescription from the
phase diagram — different cell shapes produce different aggregator
response curves, including ones where t=0.3 is sub-optimal.

Mechanism interpretation:
- wave: cost signal informative; rollouts have informative
  disagreement; vanilla averages two divergent rollouts into phantom
- 4-way: 3D escape volume means most rollouts succeed; argmin picks
  ONE rollout that might be unlucky; uniform averages many successful
  rollouts and stays close to prior which is correct
- peer: 4-drone-cross coordination chaos floods the rollout cost
  landscape; rollouts agree on "everything is bad" so aggregator
  choice is moot
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_EPS = 20
TEMPS = [("t01", 0.1), ("t03", 0.3), ("t10", 1.0), ("t30", 3.0), ("t100", 10.0)]
CELLS = [
    ("wave_noisy30", "wave (σ=3, intersection)",
        "results/intersection_wave_noisy30_{}_mppi_n20",
        "results/intersection_wave_noisy30_mpc_n20",
        "#1f77b4"),
    ("4way_noisy05", "4-way (σ=0.5, 30 obs, 3D escape)",
        "results/multi_drone_3d_4_noisy05_{}_mppi_n20",
        "results/multi_drone_3d_4_noisy05_mpc_n20",
        "#2ca02c"),
    ("4way_noisy30", "4-way (σ=3, 30 obs, 3D escape)",
        "results/multi_drone_3d_4_noisy30_{}_mppi_n20",
        "results/multi_drone_3d_4_noisy30_mpc_n20",
        "#9467bd"),
    ("peer_noisy05", "peer (σ=0.5, 120 obs)",
        "results/multi_drone_peer_noisy05_{}_mppi_n20",
        "results/multi_drone_peer_noisy05_mpc_n20",
        "#d62728"),
]
OUT = Path("docs/images/aggregator_3cell_compare.png")


def rate(d):
    p = Path(d)
    if not p.exists():
        return None
    n_ok = 0
    n = 0
    for ep in range(N_EPS):
        f = p / f"episode_{ep:03d}_joint.json"
        if not f.exists():
            continue
        n += 1
        if json.load(open(f))["outcome"] == "success":
            n_ok += 1
    return n_ok / n if n else None


def main() -> int:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    print("| cell | MPC | t=0.1 | t=0.3 | t=1.0 | t=3.0 | t=10 |")
    print("|---|---|---|---|---|---|---|")
    for tag, label, mppi_tpl, mpc_dir, color in CELLS:
        xs, ys = [], []
        row = []
        for t_tag, t_val in TEMPS:
            r = rate(mppi_tpl.format(t_tag))
            if r is not None:
                xs.append(t_val)
                ys.append(r * 100)
                row.append(f"{r*100:.0f}%")
            else:
                row.append("—")
        ax.plot(xs, ys, "o-", color=color, lw=2.2, ms=10,
                markeredgecolor="white", markeredgewidth=1.2,
                label=f"MPPI {label}")
        mpc_r = rate(mpc_dir)
        if mpc_r is not None:
            ax.axhline(mpc_r * 100, color=color, ls=":", lw=1.3, alpha=0.85,
                       label=f"MPC ref {label} = {mpc_r*100:.0f}%")
            mpc_s = f"{mpc_r*100:.0f}%"
        else:
            mpc_s = "—"
        print(f"| {label} | {mpc_s} | {' | '.join(row)} |")

    ax.axvspan(0.7, 1.5, color="#ffcccc", alpha=0.20, zorder=0)
    ax.text(1.0, 5, "vanilla MPPI\n(default t=1.0)", ha="center", va="bottom",
            fontsize=9, color="#a02020", fontweight="bold")

    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.3, 1.0, 3.0, 10.0])
    ax.set_xticklabels(["0.1\n(argmin)", "0.3", "1.0\n(vanilla)", "3.0", "10.0\n(uniform)"])
    ax.set_xlabel("MPPI softmax temperature")
    ax.set_ylabel(f"joint success (n={N_EPS})")
    ax.set_ylim(-2, 105)
    ax.axhline(100, color="grey", ls=":", lw=0.5)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(
        "L/M: aggregator response curves across 4 (cell, σ) points — U-shape on intersection wave (σ=3), "
        "monotonic-increasing on 4-way (σ=0.5 AND σ=3, gap deepens),\n"
        "flat on peer (σ=0.5). Cell-shape, not noise, determines the response shape.",
        fontsize=10,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
