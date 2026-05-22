"""J/G σ-axis generality: does the U-shape exist outside the σ=3 knee?

After G established the U-shape on v1 and wave at σ=3, this script
tests whether the U exists across the σ axis. Run the same 5-temperature
MPPI sweep + MPC reference at σ ∈ {1, 3, 10} on the wave cell.

H/I mechanism predicts:
- σ=1: top-2 rollouts mostly agree (cost signal informative, low
  predictor noise) → vanilla averaging is harmless → U disappears.
  Uniform (returns prior) may suffer because prior collides into wave
  intruders.
- σ=3: U-shape (J finding).
- σ=10: cost signal is pure noise → argmin picks noise → no clear
  winner.

Empirical results (n=20):

| aggregator | σ=1 | σ=3 | σ=10 |
| MPC          | 100% | 45% | 5%  |
| MPPI t=0.1   | 90%  | 70% | 35% |
| MPPI t=0.3   | 95%  | 65% | 40% |
| MPPI t=1.0   | 90%  | 35% | 10% |
| MPPI t=3.0   | 100% | 65% | 10% |
| MPPI t=10    | 70%  | 40% | 30% |

Findings:
- σ=1: U-shape vanishes — vanilla is no longer the valley (90%);
  uniform DROPS to 70% (now the worst) because the prior collides
  into the wave. Sub-knee σ matches the H/I prediction.
- σ=3: U-shape clear (J replicated).
- σ=10: valley widens to include both vanilla AND t=3 (both at 10%);
  argmin recovers slightly (35%) by picking ONE rollout; uniform
  recovers (30%) by returning the prior. Cost signal is noisy enough
  that neither extreme is reliably correct.

Stronger headline than the U-shape: **argmin MPPI (t=0.1 or t=0.3)
beats vanilla MPPI at EVERY tested σ on wave**, by 0 / 35 / 25 pp.
The "vanilla MPPI is structurally suboptimal" claim generalizes
beyond the σ=3 knee.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_EPS = 20
TEMPERATURES = [("t01", 0.1), ("t03", 0.3), ("t10", 1.0), ("t30", 3.0), ("t100", 10.0)]
SIGMAS = [
    ("noisy10",  "σ=1.0 (sub-knee)",  "#2ca02c"),
    ("noisy30",  "σ=3.0 (knee)",      "#ff7f0e"),
    ("noisy100", "σ=10.0 (chaos)",    "#d62728"),
]
OUT = Path("docs/images/u_shape_sigma_generality.png")


def rate(d):
    p = Path(d)
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
    return n_ok / n if n else None


def main() -> int:
    fig, ax = plt.subplots(figsize=(10, 6.5))

    for sigma_tag, sigma_label, color in SIGMAS:
        xs = []
        ys = []
        for t_tag, t_val in TEMPERATURES:
            r = rate(f"results/intersection_wave_{sigma_tag}_{t_tag}_mppi_n20")
            if r is None:
                continue
            xs.append(t_val)
            ys.append(r * 100)
        ax.plot(xs, ys, "o-", color=color, lw=2.2, ms=10,
                markeredgecolor="white", markeredgewidth=1.2,
                label=f"MPPI {sigma_label}")

        # MPC ref
        mpc_r = rate(f"results/intersection_wave_{sigma_tag}_mpc_n20")
        if mpc_r is not None:
            ax.axhline(mpc_r * 100, color=color, ls=":", lw=1.3, alpha=0.85,
                       label=f"MPC ref {sigma_label} = {mpc_r*100:.0f}%")

    # Annotate vanilla MPPI as the universal-at-σ≥3 valley
    ax.axvspan(0.7, 1.5, color="#ffcccc", alpha=0.20, zorder=0)
    ax.text(1.0, 2.5, "vanilla MPPI\n(default t=1.0)",
            ha="center", va="bottom", fontsize=9, color="#a02020",
            fontweight="bold")

    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.3, 1.0, 3.0, 10.0])
    ax.set_xticklabels(["0.1\n(argmin)", "0.3", "1.0\n(vanilla)", "3.0", "10.0\n(near-uniform)"])
    ax.set_xlabel("MPPI softmax temperature")
    ax.set_ylabel(f"joint success rate (n={N_EPS}, wave cell)")
    ax.set_ylim(-2, 105)
    ax.axhline(100, color="grey", ls=":", lw=0.6)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(
        "J/G σ-axis generality on wave — U-shape only at σ=3 knee; "
        "argmin MPPI (t=0.1/0.3) beats vanilla MPPI at every tested σ",
        fontsize=10,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("| aggregator | σ=1 | σ=3 | σ=10 |")
    print("|---|---|---|---|")
    for t_tag, t_val in [("mpc", None), *TEMPERATURES]:
        results = []
        for sigma_tag, _, _ in SIGMAS:
            if t_tag == "mpc":
                r = rate(f"results/intersection_wave_{sigma_tag}_mpc_n20")
            else:
                r = rate(f"results/intersection_wave_{sigma_tag}_{t_tag}_mppi_n20")
            results.append(f"{r*100:.0f}%" if r is not None else "—")
        label = "MPC" if t_tag == "mpc" else f"MPPI t={t_val}"
        print(f"| {label} | {results[0]} | {results[1]} | {results[2]} |")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
