"""F peer cell predictor sweep, n=20 (MPC only, seeded).

Re-run of the F peer panel after fixing the predictor.reset() bug.
GPU MPPI not re-run here because PyTorch is not installed in the
current venv — the F gpu_mppi numbers in findings.md remain at the
original n=5 unseeded state and should be treated as illustrative.

Corrected n=20 MPC numbers compared to original n=5 unseeded:

| sigma     | n=5 unseeded | n=20 seeded | shift |
| nopred    | 0/5 (0%)     | 1/20 (5%)   | +5pp  |
| noisy0.2  | 0/5 (0%)     | 6/20 (30%)  | +30pp |
| noisy0.5  | 1/5 (20%)    | 6/20 (30%)  | +10pp |
| noisy1.0  | 1/5 (20%)    | 5/20 (25%)  | +5pp  |
| noisy3.0  | 1/5 (20%)    | 2/20 (10%)  | -10pp |
| noisy10   | 0/5 (0%)     | 1/20 (5%)   | +5pp  |

The original "peer floors at 0-2/5" claim was an artifact of unlucky
unseeded draws (esp. noisy0.2 = 0/5). At n=20 the actual range is
5-30%, with the SAME knee structure as wave (success drops at σ=3,
floors at σ=10). Peer is consistently ~30-50 pp harder than wave at
the same σ, confirming that peer-coordination complexity dominates
over predictor fidelity in absolute terms — but the relative effects
of σ are still visible.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = [
    ("nopred",   "no predictor"),
    ("noisy100", "noisy σ=10.0"),
    ("noisy30",  "noisy σ=3.0"),
    ("noisy10",  "noisy σ=1.0"),
    ("noisy05",  "noisy σ=0.5"),
    ("noisy02",  "noisy σ=0.2"),
]
PEER_COLOR  = "#d62728"
WAVE_COLOR  = "#1f77b4"
OUT = Path("docs/images/peer_predictor_sweep_n20.png")


def joint_success_rate(run_dir, n_max=20):
    p = Path(run_dir)
    if not p.exists():
        return None
    n_ok = 0
    n = 0
    for ep in range(n_max):
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

    # Peer MPC at n=20
    rows = []
    xs_p = []
    ys_p = []
    for i, (tag, label) in enumerate(CONDITIONS):
        r = joint_success_rate(f"results/multi_drone_peer_{tag}_mpc_n20", n_max=20)
        if r is None:
            rows.append((tag, "peer-mpc", None))
            continue
        n_ok, n_total = r
        rate = n_ok / n_total
        xs_p.append(i)
        ys_p.append(rate)
        rows.append((tag, "peer-mpc", (n_ok, n_total, rate)))
    ax.plot(xs_p, ys_p, "o-", color=PEER_COLOR, lw=2, ms=10,
            markeredgecolor="white", markeredgewidth=1.2,
            label="Peer cell (4-drone cross, 120 obs) — MPC, n=20")

    # Wave MPC at n=20 for context
    xs_w = []
    ys_w = []
    for i, (tag, label) in enumerate(CONDITIONS):
        if tag == "nopred":
            r = joint_success_rate("results/intersection_nopred_mpc", n_max=5)
        else:
            r = joint_success_rate(f"results/intersection_wave_{tag}_mpc_n20", n_max=20)
        # All peer dirs at n=20 — nopred peer is 1/20 (5%), use that
        if r is None:
            continue
        n_ok, n_total = r
        rate = n_ok / n_total
        xs_w.append(i)
        ys_w.append(rate)
        rows.append((tag, "wave-mpc", (n_ok, n_total, rate)))
    ax.plot(xs_w, ys_w, "s--", color=WAVE_COLOR, lw=1.5, ms=8, alpha=0.7,
            markeredgecolor="white", markeredgewidth=1.0,
            label="Wave cell (3 intruders) — MPC, n=20 (context)")

    ax.set_xticks(np.arange(len(CONDITIONS)))
    ax.set_xticklabels([c[1] for c in CONDITIONS], rotation=20, ha="right",
                       fontsize=9)
    ax.set_ylim(-0.05, 1.10)
    ax.set_ylabel("joint success rate (n=20; nopred n=5, deterministic)")
    ax.axhline(1.0, color="grey", ls=":", lw=0.6)
    ax.axhline(0.0, color="grey", ls=":", lw=0.6)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(
        "F peer cell predictor sweep (n=20, seeded predictor) — corrected from n=5 unseeded. "
        "Same knee structure as wave but uniformly ~30-50 pp harder.",
        fontsize=10,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("| sigma | cell | result |")
    print("|-------|------|--------|")
    for tag, cell, r in rows:
        if r is None:
            print(f"| {tag} | {cell} | — |")
        else:
            n_ok, n_total, rate = r
            print(f"| {tag} | {cell} | {n_ok}/{n_total} ({rate*100:.0f}%) |")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
