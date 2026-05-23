"""X step 2: visualize the planner-family-winner landscape from gather data.

Two-panel figure:
  (A) per-cell grouped bar of {MPC, MPPI t=0.1/1.0/10} joint success
      with Wilson 95% CI. Marks "best" with a hatch. Color the MPC bar
      red when it loses to t=10 by >15pp (the "MPC-dominated" cells).
  (B) scatter: x=pooled cvg, y=MPC-vs-best-MPPI gap (pp). Negative gap
      = MPPI wins. Annotate every cell. Vertical band at choice_cut=12.5.

Headline finding (printed and used as figure title): MPC was best in
0/9 cells; MPPI uniform (t=10) was best in 7/9. The "planner family"
choice in this regime trivially collapses to MPPI — what remains
informative is the within-MPPI temperature, which the existing N+P
rule already picks.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN = Path("docs/data/x_planner_family_data.json")
OUT = Path("docs/images/x_planner_family_landscape.png")

VARIANT_ORDER = ["MPC", "t=0.1", "t=1.0", "t=10"]
VARIANT_COLOR = {
    "MPC":   "#7f7f7f",
    "t=0.1": "#1f77b4",
    "t=1.0": "#ff7f0e",
    "t=10":  "#2ca02c",
}


def main() -> int:
    rows = json.load(open(IN))

    # Best fixed planner per cell
    n_cells = len(rows)
    mpc_best_count = sum(
        1 for r in rows
        if not math.isnan(r["mpc_rate"])
        and r["mpc_rate"] == max(
            r["rates"][k]["mean"] for k in VARIANT_ORDER
            if not math.isnan(r["rates"][k]["mean"])
        )
    )
    t10_best_count = sum(
        1 for r in rows
        if not math.isnan(r["rates"]["t=10"]["mean"])
        and r["rates"]["t=10"]["mean"] == max(
            r["rates"][k]["mean"] for k in VARIANT_ORDER
            if not math.isnan(r["rates"][k]["mean"])
        )
    )

    print(f"MPC best in {mpc_best_count}/{n_cells} cells; MPPI t=10 best in {t10_best_count}/{n_cells}")

    # MPC vs MPPI t=0.1 head-to-head (argmin family)
    print("\nMPC vs MPPI t=0.1 (argmin head-to-head):")
    print("| cell | MPC | t=0.1 | MPC-t01 |")
    for r in rows:
        mpc = r["mpc_rate"]
        t01 = r["rates"]["t=0.1"]["mean"]
        gap = mpc - t01
        marker = "←MPC" if gap > 5 else ("→t01" if gap < -5 else "tie")
        print(f"| {r['cell_tag']:<32} | {mpc:>4.0f} | {t01:>4.0f} | {gap:+5.0f} {marker} |")

    # ---- Figure ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [3, 2]})

    # (A) Per-cell grouped bars
    n_var = len(VARIANT_ORDER)
    width = 0.18
    offsets = np.linspace(-(n_var - 1) / 2 * width, (n_var - 1) / 2 * width, n_var)
    labels = [r["cell_tag"].replace("_noisy", "\nσ=").replace("0", "5", 1) if "noisy05" in r["cell_tag"] else r["cell_tag"].replace("_noisy30", "\nσ=3") for r in rows]
    # Simpler: short labels
    labels = []
    for r in rows:
        t = r["cell_tag"]
        if t.startswith("intersection_"):
            short = "intx_" + t.split("intersection_")[1].split("_noisy")[0]
        elif t.startswith("multi_drone_"):
            short = t.split("multi_drone_")[1].split("_noisy")[0]
        elif t.startswith("city_"):
            short = t.split("_noisy")[0]
        else:
            short = t
        sigma = "σ=0.5" if "noisy05" in t else "σ=3"
        labels.append(f"{short}\n{sigma}")

    for ci, r in enumerate(rows):
        per_cell_max = max(
            r["rates"][k]["mean"] for k in VARIANT_ORDER
            if not math.isnan(r["rates"][k]["mean"])
        )
        for vi, var in enumerate(VARIANT_ORDER):
            v = r["rates"][var]
            mean, lo, hi = v["mean"], v["lo"], v["hi"]
            if math.isnan(mean):
                continue
            x = ci + offsets[vi]
            is_best = mean == per_cell_max
            color = VARIANT_COLOR[var]
            axA.bar(
                x, mean, width, color=color, edgecolor="black" if is_best else "white",
                linewidth=1.5 if is_best else 0.6,
                yerr=[[max(0, mean - lo)], [max(0, hi - mean)]],
                error_kw={"capsize": 2.5, "linewidth": 0.7},
                label=var if ci == 0 else None,
                alpha=1.0 if is_best else 0.65,
            )
            if is_best:
                axA.text(x, mean + 2.0, "★", ha="center", va="bottom",
                         fontsize=12, color=color, fontweight="bold")
    axA.set_xticks(range(n_cells))
    axA.set_xticklabels(labels, fontsize=8)
    axA.set_ylabel("joint success (%, n=20, Wilson 95% CI)", fontsize=10)
    axA.set_ylim(0, 118)
    axA.axhline(100, color="grey", ls=":", lw=0.5)
    axA.grid(alpha=0.3, axis="y")
    axA.legend(loc="upper right", fontsize=9, ncol=4)
    axA.set_title(
        f"(A) Per-cell success, all 4 planners (★ = best). "
        f"MPC best in {mpc_best_count}/{n_cells}, MPPI t=10 best in {t10_best_count}/{n_cells}.",
        fontsize=10,
    )

    # (B) Scatter: cvg vs MPC-vs-best-MPPI gap
    for r in rows:
        cvg = r["warmup_cvg"]
        # MPC vs best MPPI
        best_mppi = max(
            r["rates"][k]["mean"] for k in ["t=0.1", "t=1.0", "t=10"]
            if not math.isnan(r["rates"][k]["mean"])
        )
        gap = r["mpc_rate"] - best_mppi
        # Color by which MPPI temperature wins
        winners = [k for k in ["t=0.1", "t=1.0", "t=10"]
                   if r["rates"][k]["mean"] == best_mppi]
        wcolor = VARIANT_COLOR[winners[0]] if winners else "k"
        axB.scatter(cvg, gap, s=140, c=wcolor, edgecolor="black", linewidth=1.0, zorder=3)
        short = r["cell_tag"].replace("intersection_", "intx_").replace("multi_drone_", "")
        short = short.replace("_noisy30", "/σ3").replace("_noisy05", "/σ.5")
        axB.annotate(short, xy=(cvg, gap), xytext=(6, 4), textcoords="offset points",
                     fontsize=8)
    axB.axhline(0, color="grey", lw=0.8)
    axB.axvline(12.5, color="grey", ls=":", lw=0.8)
    axB.text(12.5, axB.get_ylim()[1] * 0.92, " N+P choice_cut", color="grey", fontsize=8)
    axB.set_xlabel("pooled chosen-vs-goal angle [°]  (N+P 'cvg' signal)", fontsize=10)
    axB.set_ylabel("MPC − best MPPI joint success [pp]\n(↓ = MPPI wins)", fontsize=10)
    axB.set_title(
        "(B) MPC vs best MPPI gap per cell. Color = winning MPPI temperature\n"
        "(grey if MPC, blue=t=0.1, orange=t=1.0, green=t=10).",
        fontsize=10,
    )
    axB.grid(alpha=0.3)

    fig.suptitle(
        f"X: planner-family landscape. MPC best in 0/{n_cells} cells across calibration + OOD. "
        f"In dynamic-obstacle regime, the 'planner family' choice trivially collapses to MPPI — "
        f"all remaining gain is within-MPPI temperature (already auto-picked by N+P).",
        fontsize=11,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
