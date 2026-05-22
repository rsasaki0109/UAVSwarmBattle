"""Wave cell predictor-fidelity sweep, n=20, seeded predictor.

Re-run of the E5 wave panel after fixing the predictor.reset() bug
(see commit msg / docs/findings.md "predictor.reset seeding bug").

The corrected numbers reverse the n=5 claim:
- At σ=3 (the knee): MPC 45% > MPPI 35% (was MPC 20% vs MPPI 80% under
  unseeded noise — pure luck of the draw).
- At σ=10: both planners collapse (MPC 5%, MPPI 10%); not statistically
  distinguishable.
- Both planners saturate at σ≤1 and floor at σ≥10. Knee window narrows
  to σ ∈ {1, 3}.

The presence-switch claim (nopred → 0) is still safe — that yaml's
predictor is `use_prediction: false`, no RNG.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N_EPS = 20

CONDITIONS = [
    ("nopred",   "no predictor"),  # deterministic; reuses n=5 result
    ("noisy100", "noisy σ=10.0"),
    ("noisy30",  "noisy σ=3.0"),
    ("noisy10",  "noisy σ=1.0"),
    ("noisy05",  "noisy σ=0.5"),
    ("noisy02",  "noisy σ=0.2"),
    ("perfect",  "constant-vel"),  # deterministic; reuses n=5 result
]
PLANNER_COLORS = {"mpc": "#d62728", "mppi": "#1f77b4"}
OUT = Path("docs/images/intersection_wave_predictor_sweep_n20.png")


def wave_path(tag, planner, suffix="_n20"):
    if tag == "perfect":
        return f"results/intersection_wave_v1_{planner}", 5  # n=5 baseline
    if tag == "nopred":
        return f"results/intersection_nopred_{planner}", 5  # deterministic, reuse n=5
    return f"results/intersection_wave_{tag}_{planner}{suffix}", N_EPS


def joint_success_rate(path_and_n):
    run_dir, n_eps = path_and_n
    p = Path(run_dir)
    if not p.exists():
        return None
    n = 0
    n_ok = 0
    for ep in range(n_eps):
        f = p / f"episode_{ep:03d}_joint.json"
        if not f.exists():
            continue
        n += 1
        if json.load(open(f))["outcome"] == "success":
            n_ok += 1
    if n == 0:
        return None
    return n_ok, n


def main() -> int:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    rows = []
    for planner, dx, marker in [("mpc", -0.08, "o"), ("mppi", +0.08, "s")]:
        ys = []
        xs_ok = []
        for i, (tag, label) in enumerate(CONDITIONS):
            r = joint_success_rate(wave_path(tag, planner))
            if r is None:
                rows.append((planner, tag, label, None, None))
                continue
            n_ok, n_total = r
            rate = n_ok / n_total
            ys.append(rate)
            xs_ok.append(i + dx)
            rows.append((planner, tag, label, rate, (n_ok, n_total)))
        ax.plot(xs_ok, ys, "-", color=PLANNER_COLORS[planner], lw=1.5, alpha=0.7)
        ax.scatter(xs_ok, ys, marker=marker, color=PLANNER_COLORS[planner],
                   s=110, edgecolor="white", linewidth=1.0, zorder=5,
                   label=planner.upper())

    ax.set_xticks(np.arange(len(CONDITIONS)))
    ax.set_xticklabels([c[1] for c in CONDITIONS], rotation=20, ha="right",
                       fontsize=9)
    ax.set_ylim(-0.05, 1.10)
    ax.set_ylabel(f"joint success rate (n={N_EPS}; nopred/perfect n=5, deterministic)")
    ax.grid(alpha=0.3)
    ax.axhline(1.0, color="grey", ls=":", lw=0.8)
    ax.axhline(0.0, color="grey", ls=":", lw=0.8)
    ax.legend(loc="lower right")
    ax.set_title(
        "Wave cell predictor-fidelity sweep (n=20, seeded) — corrected from n=5 unseeded. "
        "Knee narrows to σ ∈ {1, 3}; MPC slightly more robust at the knee.",
        fontsize=10,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("| planner | predictor | label | joint success |")
    print("|---------|-----------|-------|---------------|")
    for planner, tag, label, rate, count in rows:
        rs = "—" if rate is None else f"{count[0]}/{count[1]} ({rate*100:.0f}%)"
        print(f"| {planner} | {tag} | {label} | {rs} |")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
