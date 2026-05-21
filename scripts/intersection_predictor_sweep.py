"""E5 predictor-quality sweep: joint success rate vs predictor across MPC/MPPI.

Reads the v1 and wave intersection cells across 6 predictor conditions
(nopred / noisy σ=1.0/0.5/0.2 / constant-vel / kalman) for both planners
and plots joint success rate per cell.

Hypothesis: the *predictor* sets the success axis (binary), while the
*planner aggregator* (argmin vs softmax) sets the fingerprint axis. The
v1 cell saturates at 1.0 except when predictor is absent, so the
predictor-fidelity gradient (σ-axis) is only visible on the harder wave
cell with 3 fast intruders.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N_EPS = 5
PLANNERS = ["mpc", "mppi"]

# Order: worst predictor → best.
CONDITIONS = [
    ("nopred",   "no predictor"),
    ("noisy100", "noisy σ=10.0"),
    ("noisy30",  "noisy σ=3.0"),
    ("noisy10",  "noisy σ=1.0"),
    ("noisy05",  "noisy σ=0.5"),
    ("noisy02",  "noisy σ=0.2"),
    ("perfect",  "constant-vel"),
    ("kalman",   "kalman"),
]
PLANNER_COLORS = {"mpc": "#d62728", "mppi": "#1f77b4"}

# Cell-specific result-dir templates: tag is filled with the predictor tag,
# planner with the planner. None means "skip this cell-condition pair".
def v1_path(tag, planner):
    if tag == "nopred":
        return f"results/intersection_nopred_{planner}"
    if tag == "perfect":
        return f"results/intersection_v1_{planner}"
    return f"results/intersection_v1_{tag}_{planner}"


def wave_path(tag, planner):
    if tag == "perfect":
        return f"results/intersection_wave_v1_{planner}"
    if tag == "kalman":
        return None  # kalman not run for wave
    return f"results/intersection_wave_{tag}_{planner}"


CELLS = [
    ("v1",   v1_path,   "v1 (1 intruder, 0.5 m/s)"),
    ("wave", wave_path, "wave (3 intruders, 1.5 m/s)"),
]
OUT = Path("docs/images/intersection_predictor_sweep.png")


def joint_success_rate(run_dir):
    if run_dir is None:
        return None
    p = Path(run_dir)
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
    if n == 0:
        return None
    return n_ok / n


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    rows = []
    for ax, (cell_tag, path_fn, cell_label) in zip(axes, CELLS):
        for planner, dx, marker in [("mpc", -0.08, "o"), ("mppi", +0.08, "s")]:
            ys = []
            xs_ok = []
            for i, (tag, label) in enumerate(CONDITIONS):
                r = joint_success_rate(path_fn(tag, planner))
                if r is None:
                    rows.append((cell_tag, planner, tag, label, None))
                    continue
                ys.append(r)
                xs_ok.append(i + dx)
                rows.append((cell_tag, planner, tag, label, r))
            ax.plot(xs_ok, ys, "-", color=PLANNER_COLORS[planner],
                    lw=1.5, alpha=0.7)
            ax.scatter(xs_ok, ys, marker=marker, color=PLANNER_COLORS[planner],
                       s=90, edgecolor="white", linewidth=1.0, zorder=5,
                       label=planner.upper())

        ax.set_xticks(np.arange(len(CONDITIONS)))
        ax.set_xticklabels([c[1] for c in CONDITIONS], rotation=20, ha="right",
                           fontsize=9)
        ax.set_ylim(-0.05, 1.10)
        ax.set_title(cell_label, fontsize=10)
        ax.grid(alpha=0.3)
        ax.axhline(1.0, color="grey", ls=":", lw=0.8)
        ax.axhline(0.0, color="grey", ls=":", lw=0.8)
        ax.legend(loc="lower right")
    axes[0].set_ylabel(f"joint success rate (n={N_EPS})")
    fig.suptitle(
        "E5: success vs predictor quality — v1 is binary (predictor presence is a switch); "
        "wave reveals a fidelity gradient at σ≥3 where MPPI's softmax aggregator "
        "is more robust to bad predictions, but breaks first at total chaos (σ=10)",
        fontsize=9, y=1.0,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("| cell | planner | predictor | label | joint success rate |")
    print("|------|---------|-----------|-------|--------------------|")
    for cell_tag, planner, tag, label, r in rows:
        rs = "—" if r is None else f"{int(round(r*N_EPS))}/{N_EPS} ({r*100:.0f}%)"
        print(f"| {cell_tag} | {planner} | {tag} | {label} | {rs} |")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
