"""T: full city-wide N+P stress test.

4 city cells (v1, wave, chokepoint, 3x3) x 5 planner variants
(MPC, MPPI t=0.1, t=1.0, t=10, warmup_select_mppi), n=20 each.

For each cell prints:
  - per-variant joint success with Wilson 95% CI
  - N+P warmup diagnostic (pooled top-2, pooled cvg, selected temp)
  - verdict: did warmup_select pick the empirical-best fixed temperature?

Plots a grouped bar chart with 4 cell clusters x 5 bars, with the
warmup_select bar marked with a ★ when it matches the empirical-best.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uav_nav_lab.analysis import diagnose_warmup, joint_success_rate

N_EPS = 20

# (cell_tag, label, warmup_yaml, replan_period, max_steps)
CELLS = [
    ("city_v1_noisy30",         "city_v1\n(4 buildings, 1 intruder)",
     "examples/exp_city_v1_noisy30_warmup_select_mppi_n20.yaml", 0.2, 1200),
    ("city_wave_noisy30",       "city_wave\n(4 buildings, 3 wave intruders)",
     "examples/exp_city_wave_noisy30_warmup_select_mppi_n20.yaml", 0.2, 1200),
    ("city_chokepoint_noisy30", "city_chokepoint\n(4 buildings + 4 cubes, 1 intruder)",
     "examples/exp_city_chokepoint_noisy30_warmup_select_mppi_n20.yaml", 0.2, 1200),
    ("city_3x3_noisy30",        "city_3x3\n(9 buildings, 4 drones, 1 intruder)",
     "examples/exp_city_3x3_noisy30_warmup_select_mppi_n20.yaml", 0.2, 1500),
]

# variant labels & colors (order matters for the bar plot)
VARIANTS = [
    ("MPC",      "mpc_n20",                    "#7f7f7f"),
    ("t=0.1",    "t01_mppi_n20",               "#1f77b4"),
    ("t=1.0",    "t10_mppi_n20",               "#ff7f0e"),
    ("t=10",     "t100_mppi_n20",              "#2ca02c"),
    ("warmup",   "warmup_select_mppi_n20",     "#d62728"),
]

OUT = Path("docs/images/city_full_validate.png")


def main() -> int:
    fig, ax = plt.subplots(figsize=(16, 7.5))
    n_cells = len(CELLS)
    n_var = len(VARIANTS)
    width = 0.16
    offsets = np.linspace(-(n_var - 1) / 2 * width, (n_var - 1) / 2 * width, n_var)

    print("| cell | MPC | t=0.1 | t=1.0 | t=10 | warmup | N+P pick | best fixed | match? |")
    print("|---|---|---|---|---|---|---|---|---|")

    matches = 0
    for ci, (cell_tag, cell_label, ws_yaml, rp, ms) in enumerate(CELLS):
        rates: dict[str, tuple[float, float, float]] = {}
        for var_label, var_suffix, _ in VARIANTS:
            dirpath = f"results/{cell_tag}_{var_suffix}"
            rates[var_label] = joint_success_rate(dirpath, n_eps=N_EPS)[:3]

        # N+P diagnostic + auto-pick (replan_period / max_steps come
        # from the YAML inside diagnose_warmup, so the per-CELLS rp/ms
        # tuple is informational only).
        try:
            diag = diagnose_warmup(ws_yaml, episodes=2)
            ptop2, pcvg, ptemp = diag.top2_mean, diag.cvg_mean, diag.selected_temperature
        except Exception as e:
            ptop2, pcvg, ptemp = float("nan"), float("nan"), float("nan")
            print(f"  warmup diag failed for {cell_tag}: {e}")

        # Empirical best among fixed variants
        fixed_vars = {k: v[0] for k, v in rates.items() if k != "warmup" and not math.isnan(v[0])}
        best_var = max(fixed_vars, key=fixed_vars.get) if fixed_vars else "—"
        best_rate = fixed_vars.get(best_var, float("nan"))
        ws_rate = rates["warmup"][0]
        # Translate N+P-picked temp to variant label
        pick_label = {0.1: "t=0.1", 1.0: "t=1.0", 10.0: "t=10"}.get(round(ptemp, 1), f"t={ptemp}")
        is_match = pick_label == best_var
        matches += 1 if is_match else 0

        # Print row
        cells = [cell_tag]
        for var_label, _, _ in VARIANTS:
            r, lo, hi = rates[var_label]
            cells.append(f"{r:.0f}% [{lo:.0f},{hi:.0f}]" if not math.isnan(r) else "—")
        cells.append(f"{pick_label} (cvg={pcvg:.1f}, top2={ptop2:.1f})")
        cells.append(f"{best_var}={best_rate:.0f}%")
        cells.append("✓" if is_match else "✗")
        print("| " + " | ".join(cells) + " |")

        # Bar group for this cell
        for vi, (var_label, _, color) in enumerate(VARIANTS):
            r, lo, hi = rates[var_label]
            x = ci + offsets[vi]
            if math.isnan(r):
                continue
            ax.bar(x, r, width, color=color, edgecolor="white", linewidth=0.7,
                   yerr=[[max(0, r - lo)], [max(0, hi - r)]],
                   error_kw={"capsize": 2.5, "linewidth": 0.8},
                   label=var_label if ci == 0 else None)
            # ★ marker when warmup_select matches best fixed
            if var_label == "warmup" and is_match:
                ax.text(x, r + 4, "★", ha="center", va="bottom",
                        fontsize=14, color="#d62728", fontweight="bold")
            ax.text(x, r + 1.5, f"{r:.0f}", ha="center", va="bottom",
                    fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(range(n_cells))
    ax.set_xticklabels([c[1] for c in CELLS], fontsize=9)
    ax.set_ylabel("joint success (%, n=20, Wilson 95% CI)", fontsize=11)
    ax.set_ylim(0, 118)
    ax.axhline(100, color="grey", ls=":", lw=0.5)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=10, ncol=5)
    ax.set_title(
        f"T: city N+P stress test — {matches}/{n_cells} cells, warmup_select_mppi auto-picks "
        f"the empirical-best fixed temperature (★) without recalibration. "
        f"All cells use σ=3 predictor noise, no per-cell tuning.",
        fontsize=10,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwarmup_select matched empirical-best in {matches}/{n_cells} cells")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
