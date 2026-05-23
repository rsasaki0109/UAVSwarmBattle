"""U step 1: per-step warmup time-series across 9 cells, with focus on
why city_chokepoint is the one N+P confident miss.

For each cell with a warmup_select_mppi YAML, run ep 0 via
:func:`uav_nav_lab.analysis.diagnose_warmup` and pull the pooled
(top2, cvg) series off the returned :class:`WarmupDiagnostic`. The
series are interleaved per-drone per-replan in the order ``plan()``
was called by the two-phase multi-drone runner.

Output:
  (A) per-cell time series of top2 + cvg (small-multiple grid)
  (B) per-cell aggregator table: mean (current rule), max, p75,
      latter-half-mean, std — to spot which (if any) signal would flag
      city_chokepoint without breaking the other low-cvg "hit" cells.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uav_nav_lab.analysis import diagnose_warmup

# (cell_tag, warmup_yaml, n_drones, hit_or_miss_or_uniform_pick)
# hit/miss refers to whether N+P's auto-pick matched empirical-best in T.
CELLS = [
    ("intersection_v1_noisy30",         "examples/exp_intersection_v1_noisy30_warmup_select_mppi_n20.yaml",         2, "hit"),
    ("intersection_wave_noisy30",       "examples/exp_intersection_wave_noisy30_warmup_select_mppi_n20.yaml",       2, "hit"),
    ("intersection_chokepoint_noisy30", "examples/exp_intersection_chokepoint_noisy30_warmup_select_mppi_n20.yaml", 2, "hit"),
    ("multi_drone_3d_4_noisy05",        "examples/exp_multi_drone_3d_4_noisy05_warmup_select_mppi_n20.yaml",        4, "hit"),
    ("multi_drone_peer_noisy05",        "examples/exp_multi_drone_peer_noisy05_warmup_select_mppi_n20.yaml",        4, "chaos"),
    ("city_v1_noisy30",                 "examples/exp_city_v1_noisy30_warmup_select_mppi_n20.yaml",                 2, "hit"),
    ("city_wave_noisy30",               "examples/exp_city_wave_noisy30_warmup_select_mppi_n20.yaml",               2, "hit"),
    ("city_chokepoint_noisy30",         "examples/exp_city_chokepoint_noisy30_warmup_select_mppi_n20.yaml",         2, "MISS"),
    ("city_3x3_noisy30",                "examples/exp_city_3x3_noisy30_warmup_select_mppi_n20.yaml",                4, "hit"),
]

CHOICE_CUT = 12.5
APPL_CUT = 50.0
OUT_TS = Path("docs/images/u_chokepoint_timeseries.png")
OUT_JSON = Path("docs/data/u_chokepoint_aggregators.json")


def aggregators(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs, float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {k: float("nan") for k in ["mean", "median", "max", "p75", "lhalf", "lq", "std"]}
    n = a.size
    lhalf = a[n // 2 :]
    lq = a[3 * n // 4 :]
    return {
        "mean":   float(a.mean()),
        "median": float(np.median(a)),
        "max":    float(a.max()),
        "p75":    float(np.percentile(a, 75)),
        "lhalf":  float(lhalf.mean()) if lhalf.size else float("nan"),
        "lq":     float(lq.mean()) if lq.size else float("nan"),
        "std":    float(a.std()),
    }


def main() -> int:
    rows = []
    for cell_tag, ws_yaml, n_drones, hm in CELLS:
        print(f"running ep 0 for {cell_tag} ...")
        diag = diagnose_warmup(ws_yaml, episodes=1)
        rows.append({
            "cell_tag": cell_tag,
            "n_drones": diag.n_drones,
            "n_samples": diag.n_samples,
            "hit_miss": hm,
            "top2": diag.top2_series,
            "cvg": diag.cvg_series,
            "top2_agg": aggregators(diag.top2_series),
            "cvg_agg": aggregators(diag.cvg_series),
        })

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {OUT_JSON}")

    # ---- Aggregator comparison table ----
    print("\nCVG aggregator comparison (lower → uniform picked, higher → argmin picked, cut=12.5):")
    print(f"{'cell':<32} {'mean':>6} {'med':>6} {'max':>6} {'p75':>6} {'lhalf':>6} {'lq':>6} {'std':>6}  hit?")
    for r in rows:
        a = r["cvg_agg"]
        flag = "MISS" if r["hit_miss"] == "MISS" else ""
        print(f"{r['cell_tag']:<32} "
              f"{a['mean']:>6.1f} {a['median']:>6.1f} {a['max']:>6.1f} "
              f"{a['p75']:>6.1f} {a['lhalf']:>6.1f} {a['lq']:>6.1f} {a['std']:>6.1f}  "
              f"{r['hit_miss']:<6} {flag}")

    print("\nTOP2 aggregator comparison (mean > 50 = chaos branch):")
    print(f"{'cell':<32} {'mean':>6} {'med':>6} {'max':>6} {'p75':>6} {'lhalf':>6} {'lq':>6} {'std':>6}")
    for r in rows:
        a = r["top2_agg"]
        print(f"{r['cell_tag']:<32} "
              f"{a['mean']:>6.1f} {a['median']:>6.1f} {a['max']:>6.1f} "
              f"{a['p75']:>6.1f} {a['lhalf']:>6.1f} {a['lq']:>6.1f} {a['std']:>6.1f}")

    # ---- Plot: 9-cell small multiple grid (3x3) ----
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharey=False)
    for idx, r in enumerate(rows):
        ax = axes[idx // 3, idx % 3]
        cvg = r["cvg"]
        top2 = r["top2"]
        x = np.arange(len(cvg))
        ax.plot(x, cvg, color="#1f77b4", lw=0.9, label="cvg")
        ax.plot(x, top2, color="#d62728", lw=0.9, alpha=0.7, label="top2")
        ax.axhline(CHOICE_CUT, color="#1f77b4", ls=":", lw=0.6)
        ax.axhline(APPL_CUT, color="#d62728", ls=":", lw=0.6)
        mean_cvg = r["cvg_agg"]["mean"]
        max_cvg = r["cvg_agg"]["max"]
        mean_top2 = r["top2_agg"]["mean"]
        title_color = "#d62728" if r["hit_miss"] == "MISS" else "black"
        ax.set_title(
            f"{r['cell_tag']}  [{r['hit_miss']}]\n"
            f"cvg mean={mean_cvg:.1f}  max={max_cvg:.1f}  |  top2 mean={mean_top2:.1f}",
            fontsize=8, color=title_color,
        )
        ax.set_xlabel("warmup replan index (interleaved by drone)", fontsize=7)
        ax.set_ylabel("angle [°]", fontsize=7)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=7)
    OUT_TS.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(
        "U step 1: per-step warmup signal across 9 cells. "
        "Dotted lines = choice_cut (12.5° blue) and appl_cut (50° red). "
        f"Only city_chokepoint is a confident N+P MISS.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_TS, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT_TS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
