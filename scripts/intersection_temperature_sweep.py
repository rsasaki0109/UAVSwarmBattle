"""J: aggregator-temperature sweep on the wave cell.

Tests the E5 mechanism claim that "softmax = confidence amplifier".

At σ=10 (predictor chaos), E5 showed MPPI dies (0/5) while MPC partially
survives (2/5). Hypothesis: MPPI's softmax-weighted aggregation averages
across many high-cost-but-similar-cost rollouts, committing with high
confidence to a phantom-averaged action. If this is the mechanism, then
forcing MPPI to behave like argmin (temperature → 0) should recover
MPC-like behaviour at σ=10. Conversely, very soft MPPI (temperature → ∞)
should collapse even at σ=3 where vanilla MPPI was strong.

Cells: wave (intersection_wave) × {σ=3 (noisy30), σ=10 (noisy100)} × 5
temperatures × 5 episodes. MPC reference is plotted as a horizontal
dotted line per σ (already in results/intersection_wave_{noisyXX}_mpc).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_EPS = 20
SUFFIX = "_n20"
TEMPERATURES = [
    ("t01",  0.1),
    ("t03",  0.3),
    ("t10",  1.0),
    ("t30",  3.0),
    ("t100", 10.0),
]
SIGMAS = [
    ("noisy30",  "#1f77b4", "σ=3 (knee, MPPI strong)"),
    ("noisy100", "#d62728", "σ=10 (chaos, MPPI dies)"),
]
OUT = Path("docs/images/intersection_temperature_sweep.png")


def joint_success_rate(run_dir):
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
    fig, ax = plt.subplots(figsize=(9, 5.5))
    rows = []
    for sigma_tag, color, sigma_label in SIGMAS:
        xs = []
        ys = []
        for t_tag, t_val in TEMPERATURES:
            r = joint_success_rate(
                f"results/intersection_wave_{sigma_tag}_{t_tag}_mppi{SUFFIX}"
            )
            rows.append((sigma_tag, t_tag, t_val, r))
            if r is None:
                continue
            xs.append(t_val)
            ys.append(r)
        ax.plot(xs, ys, "o-", color=color, lw=2.0, ms=10,
                markeredgecolor="white", markeredgewidth=1.2,
                label=f"MPPI {sigma_label}")
        # MPC reference
        mpc_r = joint_success_rate(f"results/intersection_wave_{sigma_tag}_mpc{SUFFIX}")
        if mpc_r is not None:
            ax.axhline(mpc_r, color=color, ls=":", lw=1.4, alpha=0.9,
                       label=f"MPC ref {sigma_label} = {mpc_r*N_EPS:.0f}/{N_EPS}")
            rows.append((sigma_tag, "mpc-ref", None, mpc_r))

    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.3, 1.0, 3.0, 10.0])
    ax.set_xticklabels(["0.1", "0.3", "1.0\n(default)", "3.0", "10.0"])
    ax.set_xlabel("MPPI softmax temperature (argmin ← → uniform)")
    ax.set_ylabel(f"joint success rate (n={N_EPS})")
    ax.set_ylim(-0.05, 1.10)
    ax.axhline(1.0, color="grey", ls=":", lw=0.6)
    ax.axhline(0.0, color="grey", ls=":", lw=0.6)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(
        "J: aggregator-temperature sweep on wave cell (n=20, seeded predictor) — "
        "vanilla MPPI (t=1.0) is the worst; argmin (t=0.1) recovers cleanly",
        fontsize=10,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("| sigma | temperature | joint success |")
    print("|-------|-------------|---------------|")
    for sigma_tag, t_tag, t_val, r in rows:
        rs = "—" if r is None else f"{int(round(r*N_EPS))}/{N_EPS} ({r*100:.0f}%)"
        tval_str = f"{t_val}" if t_val is not None else "ref"
        print(f"| {sigma_tag} | {t_tag} ({tval_str}) | {rs} |")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
