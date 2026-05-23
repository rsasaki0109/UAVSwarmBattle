"""X step 1: gather N+P signal vs planner-family-winner across 9 cells.

For each cell with all 4 {MPC, MPPI t=0.1, MPPI t=1.0, MPPI t=10} plus a
warmup_select_mppi YAML:

  - recompute pooled N+P signal (top-2 disagreement, chosen-vs-goal angle)
    by re-running ep 0 of the warmup
  - load joint success rates (with Wilson 95% CI) for all 4 fixed planners
  - identify the empirical winner family (MPC vs best-of-MPPI) and the
    within-MPPI best temperature
  - print a table + dump JSON for downstream rule-fitting / plotting

The downstream question (next script): does the existing N+P signal
already predict MPC-vs-MPPI winner, or is a new signal needed?
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from uav_nav_lab.analysis import diagnose_warmup, joint_success_rate

N_EPS = 20

# (cell_tag, label, warmup_yaml)
CELLS = [
    ("intersection_v1_noisy30",         "intx_v1\n(2 drones, 1 slow intruder)",
     "examples/exp_intersection_v1_noisy30_warmup_select_mppi_n20.yaml"),
    ("intersection_wave_noisy30",       "intx_wave\n(2 drones, 3 wave)",
     "examples/exp_intersection_wave_noisy30_warmup_select_mppi_n20.yaml"),
    ("intersection_chokepoint_noisy30", "intx_choke\n(2 drones + cubes)",
     "examples/exp_intersection_chokepoint_noisy30_warmup_select_mppi_n20.yaml"),
    ("multi_drone_3d_4_noisy05",        "4way_3d\n(4 drones, random vox σ=0.5)",
     "examples/exp_multi_drone_3d_4_noisy05_warmup_select_mppi_n20.yaml"),
    ("multi_drone_peer_noisy05",        "peer\n(4 drones, 120 vox σ=0.5)",
     "examples/exp_multi_drone_peer_noisy05_warmup_select_mppi_n20.yaml"),
    ("city_v1_noisy30",                 "city_v1\n(2 drones, 4 buildings)",
     "examples/exp_city_v1_noisy30_warmup_select_mppi_n20.yaml"),
    ("city_wave_noisy30",               "city_wave\n(2 drones, 4 bldg + 3 wave)",
     "examples/exp_city_wave_noisy30_warmup_select_mppi_n20.yaml"),
    ("city_chokepoint_noisy30",         "city_choke\n(2 drones, 4 bldg + cubes)",
     "examples/exp_city_chokepoint_noisy30_warmup_select_mppi_n20.yaml"),
    ("city_3x3_noisy30",                "city_3x3\n(4 drones, 9 bldg)",
     "examples/exp_city_3x3_noisy30_warmup_select_mppi_n20.yaml"),
]

VARIANT_DIRS = [
    ("MPC",   "mpc_n20"),
    ("t=0.1", "t01_mppi_n20"),
    ("t=1.0", "t10_mppi_n20"),
    ("t=10",  "t100_mppi_n20"),
]

OUT_JSON = Path("docs/data/x_planner_family_data.json")


def main() -> int:
    print("| cell | top2 | cvg | MPC | t=0.1 | t=1.0 | t=10 | best | family_winner |")
    print("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for cell_tag, cell_label, ws_yaml in CELLS:
        rates = {}
        for var_label, var_suffix in VARIANT_DIRS:
            d = f"results/{cell_tag}_{var_suffix}"
            rates[var_label] = joint_success_rate(d, n_eps=N_EPS)

        diag = diagnose_warmup(ws_yaml, episodes=1)
        top2, cvg = diag.top2_mean, diag.cvg_mean

        # Determine best within MPPI temperature group
        mppi = {k: v for k, v in rates.items() if k != "MPC" and not math.isnan(v[0])}
        if mppi:
            best_mppi_name = max(mppi, key=lambda k: mppi[k][0])
            best_mppi_rate = mppi[best_mppi_name][0]
        else:
            best_mppi_name = "—"
            best_mppi_rate = float("nan")
        mpc_rate = rates["MPC"][0]

        # Family winner: MPC vs MPPI (gap in pp, considering CIs)
        if math.isnan(mpc_rate) or math.isnan(best_mppi_rate):
            family = "—"
            gap = float("nan")
            ci_overlap = False
        else:
            gap = mpc_rate - best_mppi_rate  # positive = MPC wins
            # Wilson CI overlap test: do MPC's CI and best-MPPI's CI overlap?
            mpc_lo, mpc_hi = rates["MPC"][1], rates["MPC"][2]
            mp_lo, mp_hi = mppi[best_mppi_name][1], mppi[best_mppi_name][2]
            ci_overlap = not (mpc_hi < mp_lo or mp_hi < mpc_lo)
            if abs(gap) < 1e-6 or ci_overlap:
                family = "TIE"
            elif gap > 0:
                family = "MPC"
            else:
                family = "MPPI"

        # Print row
        row_cells = [cell_tag, f"{top2:.1f}", f"{cvg:.1f}"]
        for var_label, _ in VARIANT_DIRS:
            r = rates[var_label][0]
            row_cells.append(f"{r:.0f}" if not math.isnan(r) else "—")
        row_cells.append(f"{best_mppi_name}={best_mppi_rate:.0f}" if not math.isnan(best_mppi_rate) else "—")
        row_cells.append(f"{family} ({gap:+.0f}pp)" if not math.isnan(gap) else "—")
        print("| " + " | ".join(row_cells) + " |")

        rows.append({
            "cell_tag": cell_tag,
            "cell_label": cell_label,
            "warmup_top2": top2,
            "warmup_cvg": cvg,
            "rates": {
                k: {"mean": rates[k][0], "lo": rates[k][1], "hi": rates[k][2],
                    "k": rates[k][3], "n": rates[k][4]}
                for k in rates
            },
            "best_mppi_name": best_mppi_name,
            "best_mppi_rate": best_mppi_rate,
            "mpc_rate": mpc_rate,
            "family_winner": family,
            "family_gap_pp": gap,
            "ci_overlap": ci_overlap,
        })

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
