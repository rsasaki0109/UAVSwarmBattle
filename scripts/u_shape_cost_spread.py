"""H: cost-spread mechanism for the U-shape cell-dependence.

G showed that vanilla MPPI is the worst aggregator on both v1 and wave
at σ=3 (universality), but the optimal *arm* of the U differs by cell:
- v1 cell: near-uniform MPPI (t=10) → 100%   (prior-trust wins)
- wave cell: argmin MPPI (t=0.1) → 70%      (cost-trust wins)

Hypothesis: the cell-dependence is explained by the **shape of the
per-replan rollout cost distribution**:
- On v1 (one slow intruder), most rollouts succeed regardless of
  evasion direction → cost distribution is narrow → softmax weights
  are nearly uniform anyway → vanilla MPPI returns a phantom-averaged
  evasion, but uniform forces true uniform and returns the prior
  (straight-to-goal) which works because the problem is forgiving.
- On wave (three intruders), only one rollout's specific evasion
  direction succeeds → cost distribution is wide → softmax weights
  are already skewed toward one rollout → argmin (t=0.1) picks it
  natively, uniform (t=10) ignores the signal and returns the prior
  which collides.

This script instruments MPPIPlanner.plan() via monkey-patch (using the
_last_costs/_last_weights state added in the same commit) and runs ONE
episode of the wave_noisy30 and v1_noisy30 t=1.0 (vanilla) yamls. It
captures per-replan cost arrays + softmax weights and computes:
- per-replan softmax entropy:   H = -Σ w_i log w_i  (in nats; max = log(n_samples) ≈ 3.47 for n=32)
- per-replan cost spread:       (cost_max - cost_min) / max(|cost_min|, 1)
- effective # rollouts:         (Σ w_i)² / Σ w_i²   (Simpson's; ∈ [1, n_samples])

If the hypothesis holds, v1 should show consistently HIGHER entropy /
LOWER cost spread / HIGHER effective # rollouts than wave throughout
the episode.

Output: docs/images/u_shape_cost_spread.png (3-pane)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uav_nav_lab.config import ExperimentConfig
from uav_nav_lab.planner.mppi import MPPIPlanner
from uav_nav_lab.runner.multi.experiment import run_experiment_multi

# Tag-based cost dump.
COST_DUMPS = {"v1": [], "wave": []}
_current_label = ["v1"]

_orig_plan = MPPIPlanner.plan


def _wrapped_plan(self, *args, **kwargs):
    plan = _orig_plan(self, *args, **kwargs)
    if self._last_costs is not None and self._last_weights is not None:
        COST_DUMPS[_current_label[0]].append({
            "costs": self._last_costs.tolist(),
            "weights": self._last_weights.tolist(),
            "temperature": float(self.temperature),
        })
    return plan


MPPIPlanner.plan = _wrapped_plan


def _entropy(w):
    """Shannon entropy of softmax weights in nats."""
    w = np.asarray(w, dtype=float)
    eps = 1e-12
    return float(-np.sum(w * np.log(w + eps)))


def _cost_spread(c):
    """Relative cost spread: (max - min) / max(|min|, 1)."""
    c = np.asarray(c, dtype=float)
    cmin = float(np.min(c))
    cmax = float(np.max(c))
    return (cmax - cmin) / max(abs(cmin), 1.0)


def _effective_n(w):
    """Simpson's effective number of samples: (Σw)² / Σw²."""
    w = np.asarray(w, dtype=float)
    return float(np.sum(w) ** 2 / np.sum(w ** 2))


def _run_one_ep(yaml_path, label):
    """Run num_episodes=1 of the given yaml; tag costs with `label`."""
    _current_label[0] = label
    COST_DUMPS[label].clear()
    cfg = ExperimentConfig.from_yaml(Path(yaml_path))
    # Force 1 episode for speed.
    cfg.num_episodes = 1
    out_dir = Path(f"/tmp/u_shape_cost_{label}")
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    run_experiment_multi(cfg, out_dir)


def main() -> int:
    _run_one_ep("examples/exp_intersection_v1_noisy30_t10_mppi_n20.yaml",   "v1")
    _run_one_ep("examples/exp_intersection_wave_noisy30_t10_mppi_n20.yaml", "wave")

    # Compute per-replan metrics for each cell.
    metrics = {}
    for label in ("v1", "wave"):
        dumps = COST_DUMPS[label]
        ents = [_entropy(d["weights"]) for d in dumps]
        spreads = [_cost_spread(d["costs"]) for d in dumps]
        effns = [_effective_n(d["weights"]) for d in dumps]
        metrics[label] = {
            "entropy": ents,
            "spread": spreads,
            "effective_n": effns,
            "n_replan": len(dumps),
        }
        print(f"=== {label} (n_replan={len(dumps)}) ===")
        print(f"  entropy:     mean {np.mean(ents):.2f}  (max possible = log(32) = 3.47)")
        print(f"  cost spread: mean {np.mean(spreads):.2f}")
        print(f"  effective n: mean {np.mean(effns):.1f}  (max = 32)")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    cell_colors = {"v1": "#2ca02c", "wave": "#1f77b4"}
    cell_labels = {"v1": "v1 (1 intruder, easy)", "wave": "wave (3 intruders, hard)"}

    # Pane (a) — softmax entropy over time
    for label in ("v1", "wave"):
        ents = metrics[label]["entropy"]
        axes[0].plot(range(len(ents)), ents, "-o", color=cell_colors[label],
                     lw=1.4, ms=4, alpha=0.8, label=cell_labels[label])
    axes[0].axhline(np.log(32), color="grey", ls=":", lw=0.8, label="max (log 32)")
    axes[0].set_xlabel("replan index (ep 0)")
    axes[0].set_ylabel("softmax entropy (nats)")
    axes[0].set_title("(a) softmax weight entropy per replan\n"
                      "high = uniform weights, low = peaked weights",
                      fontsize=10)
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(alpha=0.3)

    # Pane (b) — relative cost spread
    for label in ("v1", "wave"):
        spreads = metrics[label]["spread"]
        axes[1].plot(range(len(spreads)), spreads, "-o", color=cell_colors[label],
                     lw=1.4, ms=4, alpha=0.8, label=cell_labels[label])
    axes[1].set_xlabel("replan index (ep 0)")
    axes[1].set_ylabel("(cost_max − cost_min) / max(|cost_min|, 1)")
    axes[1].set_title("(b) relative cost spread per replan\n"
                      "high = one rollout clearly best, low = all rollouts similar",
                      fontsize=10)
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3)

    # Pane (c) — effective number of rollouts
    for label in ("v1", "wave"):
        effns = metrics[label]["effective_n"]
        axes[2].plot(range(len(effns)), effns, "-o", color=cell_colors[label],
                     lw=1.4, ms=4, alpha=0.8, label=cell_labels[label])
    axes[2].axhline(32, color="grey", ls=":", lw=0.8, label="max (n_samples=32)")
    axes[2].axhline(1, color="grey", ls=":", lw=0.6, label="min (1 = pure argmin)")
    axes[2].set_xlabel("replan index (ep 0)")
    axes[2].set_ylabel("effective # rollouts (Simpson)")
    axes[2].set_title("(c) effective rollout count per replan\n"
                      "32 = all rollouts equally weighted, 1 = one rollout dominates",
                      fontsize=10)
    axes[2].legend(loc="lower right", fontsize=9)
    axes[2].grid(alpha=0.3)

    means = {label: (np.mean(metrics[label]["entropy"]),
                     np.mean(metrics[label]["spread"]),
                     np.mean(metrics[label]["effective_n"]))
             for label in ("v1", "wave")}
    fig.suptitle(
        f"H: vanilla MPPI cost-distribution shape on v1 vs wave at σ=3, ep 0.\n"
        f"v1 mean: entropy={means['v1'][0]:.2f}, cost spread={means['v1'][1]:.0f}, eff n={means['v1'][2]:.1f} ; "
        f"wave mean: entropy={means['wave'][0]:.2f}, cost spread={means['wave'][1]:.0f}, eff n={means['wave'][2]:.1f}.\n"
        "Both cells show eff n ≈ 1.8 (softmax averages the top ~2 rollouts) — the cost-shape itself does NOT explain the cell-dependence of the U.",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()

    out = Path("docs/images/u_shape_cost_spread.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

    # Also save raw data for downstream analysis.
    json_out = Path("docs/images/u_shape_cost_spread.json")
    with json_out.open("w") as f:
        json.dump({k: {kk: vv if not isinstance(vv, (list, tuple)) else list(vv) for kk, vv in v.items()} for k, v in metrics.items()}, f)
    print(f"wrote {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
