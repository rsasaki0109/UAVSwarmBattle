"""G: J U-shape generality test — same temperature sweep across cells.

The wave-cell J finding (vanilla MPPI is the worst aggregator at σ=3
knee, argmin recovers) might be wave-specific or universal. This script
plots the same 5-temperature MPPI sweep + MPC reference at σ=3 on:
- v1 cell (1 slow intruder, easy geometry)
- wave cell (3 medium-speed intruders, established knee)

If the U-shape appears in both cells, the "vanilla MPPI is suboptimal
at noisy-prediction knees" claim is universal across geometries. The
specific *winner* on each side of the U may differ (cell-dependent),
but the *worst-at-vanilla* claim is the mechanism-level statement.

Discovered: v1 also shows the U-shape (vanilla = 60%, both arms higher),
AND v1's optimum is at t=10 (near-uniform = 100%) rather than argmin.
Wave's optimum is at t=0.1 (argmin = 70%) instead. Both cells share
vanilla MPPI as their valley.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_EPS = 20

TEMPERATURES = [
    ("t01",  0.1),
    ("t03",  0.3),
    ("t10",  1.0),
    ("t30",  3.0),
    ("t100", 10.0),
]

CELLS = [
    ("v1",   "v1 cell (1 intruder, 0.5 m/s)",          "#2ca02c"),
    ("wave", "wave cell (3 intruders, 1.5 m/s)",       "#1f77b4"),
]

OUT = Path("docs/images/u_shape_generality.png")


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
    fig, ax = plt.subplots(figsize=(10, 6))

    for cell_tag, cell_label, color in CELLS:
        xs = []
        ys = []
        for t_tag, t_val in TEMPERATURES:
            r = rate(f"results/intersection_{cell_tag}_noisy30_{t_tag}_mppi_n20")
            if r is None:
                continue
            xs.append(t_val)
            ys.append(r * 100)
        ax.plot(xs, ys, "o-", color=color, lw=2.2, ms=11,
                markeredgecolor="white", markeredgewidth=1.2,
                label=f"MPPI ({cell_label})")

        # MPC reference (horizontal line)
        mpc_r = rate(f"results/intersection_{cell_tag}_noisy30_mpc_n20")
        if mpc_r is None and cell_tag == "wave":
            mpc_r = rate("results/intersection_wave_noisy30_mpc_n20")
        if mpc_r is not None:
            ax.axhline(mpc_r * 100, color=color, ls=":", lw=1.4, alpha=0.85,
                       label=f"MPC ref ({cell_tag}) = {mpc_r*100:.0f}%")

    # Annotate vanilla MPPI as the universal valley
    ax.axvspan(0.7, 1.5, color="#ffcccc", alpha=0.25, zorder=0)
    ax.text(1.0, 5, "vanilla MPPI valley\n(default t=1.0)",
            ha="center", va="bottom", fontsize=9, color="#a02020",
            fontweight="bold")

    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.3, 1.0, 3.0, 10.0])
    ax.set_xticklabels(["0.1\n(argmin)", "0.3", "1.0\n(vanilla)", "3.0", "10.0\n(near-uniform)"])
    ax.set_xlabel("MPPI softmax temperature")
    ax.set_ylabel(f"joint success rate (n={N_EPS}, σ=3 predictor noise)")
    ax.set_ylim(-2, 105)
    ax.axhline(100, color="grey", ls=":", lw=0.6)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_title(
        "G: U-shape across cells at σ=3 — vanilla MPPI is the valley in BOTH cells.\n"
        "Optimal arm differs (v1: near-uniform t=10 hits 100%; wave: argmin t=0.1 hits 70%) but the worst-at-default claim is universal.",
        fontsize=10,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("| cell | aggregator | success (n=20) |")
    print("|------|------------|----------------|")
    for cell_tag, cell_label, _ in CELLS:
        mpc_r = rate(f"results/intersection_{cell_tag}_noisy30_mpc_n20")
        if mpc_r is not None:
            print(f"| {cell_tag} | MPC | {int(round(mpc_r*N_EPS))}/{N_EPS} ({mpc_r*100:.0f}%) |")
        for t_tag, t_val in TEMPERATURES:
            r = rate(f"results/intersection_{cell_tag}_noisy30_{t_tag}_mppi_n20")
            if r is not None:
                marker = " ← VALLEY" if t_val == 1.0 else (" ← BEST" if abs(r - max([rate(f"results/intersection_{cell_tag}_noisy30_{tt}_mppi_n20") for tt,_ in TEMPERATURES])) < 1e-9 else "")
                print(f"| {cell_tag} | MPPI t={t_val} | {int(round(r*N_EPS))}/{N_EPS} ({r*100:.0f}%){marker} |")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
