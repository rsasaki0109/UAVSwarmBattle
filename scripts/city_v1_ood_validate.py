"""S: OOD validation — does warmup_select_mppi pick correctly on a cell
that wasn't in the N+P calibration set?

city_v1 is the first cell with structured urban geometry (4 buildings
bracketing a 12 m corridor) — calibration was done on toy intersections
(0-4 cubes) and random voxel multi-drone cells. If warmup_select picks
the same temperature as the empirical-best aggregator on city_v1
without recalibration, the N+P rule is *predictive* rather than just
*describing* the calibration set.

For city_v1 reports:
  - actual joint success for MPC, MPPI t=0.1, t=1.0, t=10
  - warmup_select_mppi success (single number — the auto rule)
  - what temperature warmup_select picked + the N+P diagnostic that
    drove the pick (pooled top-2, pooled cvg)
  - verdict: did the picked temperature match the empirical-best?
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uav_nav_lab.config import ExperimentConfig
from uav_nav_lab.runner.multi.builder import _build_multi
from uav_nav_lab.runner.multi.episode import run_episode_multi
from uav_nav_lab.planner.warmup_select_mppi import _SHARED_SESSIONS

N_EPS = 20

VARIANTS = [
    ("MPC (argmin)",       "results/city_v1_noisy30_mpc_n20",                  "#7f7f7f"),
    ("MPPI t=0.1 (argmin)", "results/city_v1_noisy30_t01_mppi_n20",            "#1f77b4"),
    ("MPPI t=1.0 (vanilla)","results/city_v1_noisy30_t10_mppi_n20",            "#ff7f0e"),
    ("MPPI t=10 (uniform)", "results/city_v1_noisy30_t100_mppi_n20",           "#2ca02c"),
    ("warmup_select_mppi",  "results/city_v1_noisy30_warmup_select_mppi_n20",  "#d62728"),
]

OUT = Path("docs/images/city_v1_ood.png")


def joint_outcomes(d: str) -> list[str]:
    p = Path(d)
    if not p.exists():
        return []
    return [
        json.load(open(p / f"episode_{ep:03d}_joint.json"))["outcome"]
        for ep in range(N_EPS)
        if (p / f"episode_{ep:03d}_joint.json").exists()
    ]


def success_rate(outs: list[str]) -> tuple[float, float, float]:
    n = len(outs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    k = sum(1 for o in outs if o == "success")
    p = k / n
    z = 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p * 100, max(0, centre - halfw) * 100, min(1, centre + halfw) * 100


def diagnose_warmup() -> tuple[float, float, float, str]:
    """Re-run one warmup episode and return (pooled_top2, pooled_cvg,
    selected_temperature, selected_reason)."""
    _SHARED_SESSIONS.clear()
    cfg = ExperimentConfig.from_yaml(
        Path("examples/exp_city_v1_noisy30_warmup_select_mppi_n20.yaml")
    )
    cfg.num_episodes = 2
    scenario, sims, planners, sensors = _build_multi(cfg)
    run_episode_multi(
        scenario, sims, planners, sensors,
        seed=42, replan_period=0.2, max_steps=1200,
        episode_index=0, frame_dirs=[None] * scenario.n_drones,
    )
    sess = list(_SHARED_SESSIONS.values())[0]
    pooled_top2 = float(np.nanmean(sess.top2))
    pooled_cvg = float(np.nanmean(sess.cvg))
    run_episode_multi(
        scenario, sims, planners, sensors,
        seed=43, replan_period=0.2, max_steps=1200,
        episode_index=1, frame_dirs=[None] * scenario.n_drones,
    )
    return pooled_top2, pooled_cvg, planners[0].temperature, planners[0]._selected_reason


def main() -> int:
    rates: dict[str, tuple[float, float, float]] = {}
    print(f"{'variant':<25} {'success%':>12}")
    for label, dirpath, _ in VARIANTS:
        outs = joint_outcomes(dirpath)
        r = success_rate(outs)
        rates[label] = r
        if math.isnan(r[0]):
            print(f"{label:<25} {'—':>12}")
        else:
            print(f"{label:<25} {r[0]:>7.0f}% [{r[1]:.0f},{r[2]:.0f}]")

    pooled_top2, pooled_cvg, selected_t, reason = diagnose_warmup()
    print()
    print("N+P diagnostic (warmup ep 0, pooled across 2 drones):")
    print(f"  pooled mean top-2 disagreement : {pooled_top2:.1f}° (appl_cut=50)")
    print(f"  pooled mean chosen-vs-goal     : {pooled_cvg:.1f}° (choice_cut=12.5)")
    print(f"  → selected temperature         : t={selected_t}")
    print(f"  reason                         : {reason}")

    # Determine the empirical best (excluding warmup_select)
    fixed_rates = {
        k: v[0] for k, v in rates.items()
        if not k.startswith("warmup") and not math.isnan(v[0])
    }
    best_fixed = max(fixed_rates, key=fixed_rates.get) if fixed_rates else None
    print()
    if best_fixed is not None:
        print(f"empirical best fixed planner : {best_fixed} = {fixed_rates[best_fixed]:.0f}%")
        ws_rate = rates["warmup_select_mppi"][0]
        if not math.isnan(ws_rate):
            print(f"warmup_select_mppi (auto)    : {ws_rate:.0f}%")
            gap = ws_rate - fixed_rates[best_fixed]
            print(f"gap vs best fixed            : {gap:+.0f}pp")

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 6.5))
    x = np.arange(len(VARIANTS))
    rs = [rates[v[0]][0] for v in VARIANTS]
    err_lo = [max(0, rates[v[0]][0] - rates[v[0]][1]) for v in VARIANTS]
    err_hi = [max(0, rates[v[0]][2] - rates[v[0]][0]) for v in VARIANTS]
    colors = [v[2] for v in VARIANTS]
    ax.bar(x, rs, color=colors, edgecolor="white", linewidth=1.0,
           yerr=[err_lo, err_hi], error_kw={"capsize": 3, "linewidth": 1.0})
    for xi, r in enumerate(rs):
        if not math.isnan(r):
            ax.text(xi, r + 1.5, f"{r:.0f}%", ha="center", va="bottom",
                    fontsize=10, color=colors[xi], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([v[0] for v in VARIANTS], fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("joint success (%, n=20, Wilson 95% CI)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.axhline(100, color="grey", ls=":", lw=0.5)
    ax.grid(alpha=0.3, axis="y")

    diag_txt = (
        "OOD city_v1 — 4 buildings (24x24x10m) + 12m corridor + 1 slow intruder\n"
        f"warmup pooled: top-2={pooled_top2:.1f}°, cvg={pooled_cvg:.1f}° "
        f"→ N+P picks t={selected_t} (uniform)\n"
        "Calibration set (toy intersections + random voxel) had cvg ranging 4.8-24.9°; "
        "city_v1's 5.7° is in-distribution by signal even though geometry is OOD."
    )
    ax.set_title(diag_txt, fontsize=9)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
