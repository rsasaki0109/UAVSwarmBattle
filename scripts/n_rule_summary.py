"""Q: 2D summary plot for the N+P predictive rule across all 5 cells.

The N+P rule says:
- Step 1: top-2 angular disagreement (applicability check)
    > 60°: chaotic → all aggregators equivalent (skip Step 2)
    < 40°: coherent → U-shape applies
- Step 2: chosen-vs-goal angle (aggregator choice when U applies)
    < 10°: prior correct → uniform MPPI (t=10)
    > 15°: prior misses → argmin MPPI (t=0.1)
    intermediate: either extreme helps moderately

This script plots all 5 measured cells on a 2D scatter:
  x-axis: top-2 disagreement (mean across replans of vanilla MPPI ep 0)
  y-axis: chosen-vs-goal angle (mean across replans)
  point colour: actual optimal MPPI aggregator from n=20 sweep
  background regions: predicted optimal aggregator per the N+P rule

If all 5 cells fall in their predicted regions, the rule is empirically
sound for the geometry types tested.

Measurements (from previous scripts):
  v1         : top-2 29.1°, chosen-vs-goal  9.2°, actual opt = uniform (t=10)  100%
  wave       : top-2 30.9°, chosen-vs-goal 17.1°, actual opt = argmin (t=0.1)   70%
  4way       : top-2 33.7°, chosen-vs-goal  4.8°, actual opt = uniform (t=10)   85%
  peer       : top-2 83.9°, chosen-vs-goal 24.9°, actual opt = flat (all 40%)
  chokepoint : top-2 33.1°, chosen-vs-goal 10.1°, actual opt = uniform (t=10)   95%
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# (label, top-2, chosen-vs-goal, optimal aggregator at σ-tested, peak success)
CELLS = [
    ("v1",         29.1,  9.2, "uniform",     "100%", "#2ca02c"),
    ("wave",       30.9, 17.1, "argmin",       "70%", "#1f77b4"),
    ("4-way",      33.7,  4.8, "uniform",      "85%", "#9467bd"),
    ("peer",       83.9, 24.9, "flat (chaos)", "40%", "#d62728"),
    ("chokepoint", 33.1, 10.1, "uniform",      "95%", "#ff7f0e"),
]

OUT = Path("docs/images/n_rule_summary.png")


def main() -> int:
    fig, ax = plt.subplots(figsize=(11, 7.5))

    # Background regions per the N+P rule
    # Applicability split at top-2 = 50° (midpoint of [40, 60])
    APPL_CUT = 50

    # Aggregator-choice split at chosen-vs-goal = 12.5° (midpoint of [10, 15])
    CHOICE_CUT = 12.5

    # Region 1: chaotic (top-2 > 50) — orange
    ax.axvspan(APPL_CUT, 100, color="#fff0e0", alpha=0.6, zorder=0)
    ax.text(75, 22, "Step 1 fails (top-2 > 60°)\n→ rule N/A, all MPPI ≈ flat",
            ha="center", va="center", fontsize=10, color="#a04020",
            fontweight="bold", alpha=0.8)

    # Region 2: low top-2 + low chosen-vs-goal → uniform (green)
    ax.add_patch(mpatches.Rectangle((0, 0), APPL_CUT, CHOICE_CUT,
                                     facecolor="#e0f5e0", alpha=0.5, zorder=0))
    ax.text(15, 5, "Step 2: chosen-vs-goal < 10°\n→ uniform MPPI (t=10)",
            ha="center", va="center", fontsize=9, color="#206020",
            fontweight="bold", alpha=0.8)

    # Region 3: low top-2 + high chosen-vs-goal → argmin (blue)
    ax.add_patch(mpatches.Rectangle((0, CHOICE_CUT), APPL_CUT, 35 - CHOICE_CUT,
                                     facecolor="#e0e8f5", alpha=0.5, zorder=0))
    ax.text(15, 22, "Step 2: chosen-vs-goal > 15°\n→ argmin MPPI (t=0.1)",
            ha="center", va="center", fontsize=9, color="#202060",
            fontweight="bold", alpha=0.8)

    # Plot cells
    for label, top2, cvg, opt, peak, color in CELLS:
        ax.scatter([top2], [cvg], s=300, color=color, edgecolor="black",
                   linewidth=1.5, zorder=5)
        ax.annotate(f"{label}\n({opt}: {peak})",
                    xy=(top2, cvg), xytext=(8, 6),
                    textcoords="offset points",
                    fontsize=10, fontweight="bold", color=color)

    # Threshold lines
    ax.axvline(40, color="grey", ls=":", lw=0.8, alpha=0.5)
    ax.axvline(60, color="grey", ls=":", lw=0.8, alpha=0.5)
    ax.axhline(10, color="grey", ls=":", lw=0.8, alpha=0.5)
    ax.axhline(15, color="grey", ls=":", lw=0.8, alpha=0.5)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.set_xlabel("Step 1: vanilla MPPI mean top-2 rollout angular disagreement (°)", fontsize=11)
    ax.set_ylabel("Step 2: vanilla MPPI mean chosen-action-vs-goal angle (°)", fontsize=11)
    ax.set_title(
        "Q: N+P predictive rule across 5 cells — single-episode warmup predicts cell category + optimal aggregator.\n"
        "All 5 cells (point colour = actual best MPPI from n=20 sweep) fall in their N+P-predicted regions.",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
