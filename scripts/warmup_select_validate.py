"""R: validate warmup_select_mppi against the fixed-temperature baselines.

For each of the 5 calibration cells (v1, wave, 4-way, peer, chokepoint)
compare:

  - warmup_select_mppi (auto)  — 20 episodes, ep 0 warmup, ep 1-19 selected
  - MPPI t=0.1  (argmin)
  - MPPI t=1.0  (vanilla)
  - MPPI t=10   (uniform)
  - the N+P-rule-predicted best aggregator (per-cell)

Reports:
  - grouped bar chart, one cluster per cell, four bars + a horizontal
    marker for the predicted best
  - terminal table with Wilson 95% CIs

The auto-selected planner runs ep 0 at vanilla so its peak is bounded
by (1/20) * vanilla_rate + (19/20) * selected_rate when the selection
matches the cell-wide optimum. Bars above the bound = lucky warmup;
bars below = selection drift (per-drone calibration vs per-cell).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N_EPS = 20

# (cell label, predicted-best aggregator label, predicted-best result dir, warmup_select dir, baseline dirs)
CELLS = [
    ("v1 (σ=3, 1 intruder)",
     "uniform (t=10)",
     "results/intersection_v1_noisy30_t100_mppi_n20",
     "results/intersection_v1_noisy30_warmup_select_mppi_n20",
     {
        "t=0.1 (argmin)":  "results/intersection_v1_noisy30_t01_mppi_n20",
        "t=1.0 (vanilla)": "results/intersection_v1_noisy30_t10_mppi_n20",
        "t=10 (uniform)":  "results/intersection_v1_noisy30_t100_mppi_n20",
     }),
    ("wave (σ=3, 3 intruders)",
     "argmin (t=0.1)",
     "results/intersection_wave_noisy30_t01_mppi_n20",
     "results/intersection_wave_noisy30_warmup_select_mppi_n20",
     {
        "t=0.1 (argmin)":  "results/intersection_wave_noisy30_t01_mppi_n20",
        "t=1.0 (vanilla)": "results/intersection_wave_noisy30_t10_mppi_n20",
        "t=10 (uniform)":  "results/intersection_wave_noisy30_t100_mppi_n20",
     }),
    ("4-way (σ=0.5, 30 obs)",
     "uniform (t=10)",
     "results/multi_drone_3d_4_noisy05_t100_mppi_n20",
     "results/multi_drone_3d_4_noisy05_warmup_select_mppi_n20",
     {
        "t=0.1 (argmin)":  "results/multi_drone_3d_4_noisy05_t01_mppi_n20",
        "t=1.0 (vanilla)": "results/multi_drone_3d_4_noisy05_t10_mppi_n20",
        "t=10 (uniform)":  "results/multi_drone_3d_4_noisy05_t100_mppi_n20",
     }),
    ("peer (σ=0.5, 120 obs)",
     "chaotic → vanilla",
     "results/multi_drone_peer_noisy05_t10_mppi_n20",
     "results/multi_drone_peer_noisy05_warmup_select_mppi_n20",
     {
        "t=0.1 (argmin)":  "results/multi_drone_peer_noisy05_t01_mppi_n20",
        "t=1.0 (vanilla)": "results/multi_drone_peer_noisy05_t10_mppi_n20",
        "t=10 (uniform)":  "results/multi_drone_peer_noisy05_t100_mppi_n20",
     }),
    ("chokepoint (σ=3, 4 cubes)",
     "uniform (t=10)",
     "results/intersection_chokepoint_noisy30_t100_mppi_n20",
     "results/intersection_chokepoint_noisy30_warmup_select_mppi_n20",
     {
        "t=0.1 (argmin)":  "results/intersection_chokepoint_noisy30_t01_mppi_n20",
        "t=1.0 (vanilla)": "results/intersection_chokepoint_noisy30_t10_mppi_n20",
        "t=10 (uniform)":  "results/intersection_chokepoint_noisy30_t100_mppi_n20",
     }),
]

OUT = Path("docs/images/warmup_select_validate.png")


def joint_outcomes(d: str) -> list[str]:
    p = Path(d)
    if not p.exists():
        return []
    outs = []
    for ep in range(N_EPS):
        f = p / f"episode_{ep:03d}_joint.json"
        if f.exists():
            outs.append(json.load(open(f))["outcome"])
    return outs


def success_rate(outs: list[str]) -> tuple[float, float, float]:
    """Returns (rate, ci_lo, ci_hi) using Wilson 95% CI."""
    n = len(outs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    k = sum(1 for o in outs if o == "success")
    p = k / n
    z = 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p * 100, max(0, (centre - halfw)) * 100, min(1, (centre + halfw)) * 100


def main() -> int:
    rows = []
    for cell_label, pred_label, pred_dir, ws_dir, baselines in CELLS:
        row = {"cell": cell_label, "pred": pred_label}
        ws_outs = joint_outcomes(ws_dir)
        row["warmup_select"] = success_rate(ws_outs)
        row["pred_actual"] = success_rate(joint_outcomes(pred_dir))
        for k, d in baselines.items():
            row[k] = success_rate(joint_outcomes(d))
        rows.append(row)

    # Print table
    bars = ["t=0.1 (argmin)", "t=1.0 (vanilla)", "t=10 (uniform)", "warmup_select"]
    header = "| cell | predicted | " + " | ".join(bars) + " |"
    print(header)
    print("|" + "---|" * (len(bars) + 2))
    for r in rows:
        cells = [f"{r['cell']}", f"{r['pred']}"]
        for b in bars:
            rate, lo, hi = r[b]
            if math.isnan(rate):
                cells.append("—")
            else:
                cells.append(f"{rate:.0f}% [{lo:.0f},{hi:.0f}]")
        print("| " + " | ".join(cells) + " |")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(rows))
    width = 0.20

    colors = {
        "t=0.1 (argmin)":  "#1f77b4",
        "t=1.0 (vanilla)": "#7f7f7f",
        "t=10 (uniform)":  "#2ca02c",
        "warmup_select":   "#d62728",
    }
    offsets = {
        "t=0.1 (argmin)":  -1.5 * width,
        "t=1.0 (vanilla)": -0.5 * width,
        "t=10 (uniform)":   0.5 * width,
        "warmup_select":    1.5 * width,
    }

    for b in bars:
        rates  = [r[b][0] for r in rows]
        ci_lo  = [r[b][1] for r in rows]
        ci_hi  = [r[b][2] for r in rows]
        # Replace NaN with 0 for plotting, but track the bars
        valid = [not math.isnan(v) for v in rates]
        rates_p = [v if vd else 0 for v, vd in zip(rates, valid)]
        err_lo  = [max(0, v - lo) for v, lo in zip(rates, ci_lo)]
        err_hi  = [max(0, hi - v) for v, hi in zip(rates, ci_hi)]
        ax.bar(
            x + offsets[b], rates_p, width,
            color=colors[b],
            edgecolor="white",
            linewidth=0.8,
            yerr=[err_lo, err_hi] if any(valid) else None,
            error_kw={"linewidth": 1.0, "capsize": 3},
            label=b,
        )
        for xi, (rate, vd) in enumerate(zip(rates, valid)):
            if vd:
                ax.text(xi + offsets[b], rate + 2, f"{rate:.0f}%",
                        ha="center", va="bottom", fontsize=8, color=colors[b])

    # Predicted-best markers
    for i, r in enumerate(rows):
        pred_rate = r["pred_actual"][0]
        if not math.isnan(pred_rate):
            ax.axhline(0)  # no-op to keep ax in scope
            ax.scatter([i], [pred_rate], marker="v", color="black",
                       s=90, zorder=5, label=("N+P-predicted" if i == 0 else None))

    ax.set_xticks(x)
    ax.set_xticklabels([r["cell"] for r in rows], fontsize=9, rotation=0)
    ax.set_ylabel("joint success (%, n=20, Wilson 95% CI)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.axhline(100, color="grey", ls=":", lw=0.5)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.set_title(
        "R: warmup_select_mppi (1-episode warmup → N+P rule) vs fixed-temperature baselines, n=20.\n"
        "Black ▼ marks the N+P-predicted best aggregator's actual success on each cell. "
        "warmup_select runs ep 0 at vanilla then switches per-drone for ep 1-19.",
        fontsize=10,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
