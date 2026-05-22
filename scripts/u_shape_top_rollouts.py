"""I: direct mechanism check for U-shape cell-dependence.

H refuted the cost-shape hypothesis (both cells have effective # rollouts
≈ 1.8). The refined hypothesis was: vanilla MPPI averages two *specific*
rollouts whose actions disagree (creating a phantom direction); recovery
depends on whether the prior (straight-to-goal) happens to coincide
with the truly correct plan.

This script directly measures, per replan, for vanilla MPPI (t=1.0) on
both cells at σ=3 ep 0:

(a) Top-2 rollout angular disagreement (degrees). If both cells show
    disagreement, vanilla averaging produces a phantom in both.

(b) Cosine alignment between vanilla chosen action and goal direction.
    Low alignment → vanilla deviated from prior; if v1 deviation is
    large but wave deviation is small, uniform (returns prior) helps
    v1 more.

(c) Cosine alignment between top-1 rollout action and goal direction.
    Low alignment on wave → the truly best rollout is a sharp evasion
    (so argmin finds it); high alignment on v1 → top-1 is near prior
    (so uniform finds it too).

Uses MPPI's _last_actions / _last_weights / _last_chosen_action /
_last_goal_dir storage (added to mppi.py for this analysis — no
behavior change).
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

DUMP = {"v1": [], "wave": [], "4way": []}
_label = ["v1"]
_orig_plan = MPPIPlanner.plan


def _wrapped_plan(self, *args, **kwargs):
    plan = _orig_plan(self, *args, **kwargs)
    if self._last_actions is not None and self._last_weights is not None:
        DUMP[_label[0]].append({
            "actions":      self._last_actions.tolist(),
            "weights":      self._last_weights.tolist(),
            "chosen":       self._last_chosen_action.tolist(),
            "goal_dir":     self._last_goal_dir.tolist() if self._last_goal_dir is not None else None,
            "temperature":  float(self.temperature),
        })
    return plan


MPPIPlanner.plan = _wrapped_plan


def _cos(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def _angle_between(a, b):
    c = _cos(a, b)
    if np.isnan(c):
        return np.nan
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def _top2_angle(actions, weights):
    """Angle between the two highest-weighted rollout actions."""
    w = np.asarray(weights, float)
    a = np.asarray(actions, float)
    if a.shape[0] < 2:
        return np.nan
    idx = np.argsort(-w)
    return _angle_between(a[idx[0]], a[idx[1]])


def _run_one(yaml_path, label):
    _label[0] = label
    DUMP[label].clear()
    cfg = ExperimentConfig.from_yaml(Path(yaml_path))
    cfg.num_episodes = 1
    out = Path(f"/tmp/u_shape_top_{label}")
    if out.exists():
        import shutil
        shutil.rmtree(out)
    run_experiment_multi(cfg, out)


def main() -> int:
    _run_one("examples/exp_intersection_v1_noisy30_t10_mppi_n20.yaml",   "v1")
    _run_one("examples/exp_intersection_wave_noisy30_t10_mppi_n20.yaml", "wave")
    _run_one("examples/exp_multi_drone_3d_4_noisy05_t10_mppi_n20.yaml",  "4way")

    # Per-replan metrics
    results = {}
    for label in ("v1", "wave", "4way"):
        d = DUMP[label]
        top2 = [_top2_angle(r["actions"], r["weights"]) for r in d]
        chosen_vs_goal = [_angle_between(r["chosen"], r["goal_dir"]) for r in d]
        top1_vs_goal = [
            _angle_between(np.asarray(r["actions"])[int(np.argmax(r["weights"]))], r["goal_dir"])
            for r in d
        ]
        results[label] = {
            "top2_angle":         top2,
            "chosen_vs_goal":     chosen_vs_goal,
            "top1_vs_goal":       top1_vs_goal,
        }
        print(f"=== {label}  (n_replan={len(d)}) ===")
        print(f"  top-2 angular disagreement (mean): {np.nanmean(top2):.1f}°")
        print(f"  chosen-vs-goal angle (mean):       {np.nanmean(chosen_vs_goal):.1f}°")
        print(f"  top-1-vs-goal angle (mean):        {np.nanmean(top1_vs_goal):.1f}°")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    cell_colors = {"v1": "#2ca02c", "wave": "#1f77b4", "4way": "#9467bd"}
    cell_labels = {"v1": "v1 (1 intruder, easy)",
                   "wave": "wave (3 intruders, hard)",
                   "4way": "4-way (3D escape, 30 obs)"}

    for label in ("v1", "wave", "4way"):
        r = results[label]
        axes[0].plot(range(len(r["top2_angle"])), r["top2_angle"],
                     "-o", ms=4, lw=1.3, alpha=0.8, color=cell_colors[label],
                     label=cell_labels[label])
        axes[1].plot(range(len(r["chosen_vs_goal"])), r["chosen_vs_goal"],
                     "-o", ms=4, lw=1.3, alpha=0.8, color=cell_colors[label],
                     label=cell_labels[label])
        axes[2].plot(range(len(r["top1_vs_goal"])), r["top1_vs_goal"],
                     "-o", ms=4, lw=1.3, alpha=0.8, color=cell_colors[label],
                     label=cell_labels[label])

    axes[0].axhline(0, color="grey", ls=":", lw=0.6)
    axes[0].axhline(90, color="grey", ls=":", lw=0.6)
    axes[0].set_xlabel("replan index")
    axes[0].set_ylabel("angle (°)")
    axes[0].set_title("(a) top-2 rollout disagreement (vanilla MPPI t=1.0)\n"
                      "high = top-2 disagree → vanilla phantom-averages them\n"
                      "low = top-2 agree → averaging is harmless",
                      fontsize=9)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(0, color="grey", ls=":", lw=0.6)
    axes[1].axhline(90, color="grey", ls=":", lw=0.6)
    axes[1].set_xlabel("replan index")
    axes[1].set_ylabel("angle (°)")
    axes[1].set_title("(b) vanilla MPPI chosen action vs goal direction\n"
                      "low = chosen ≈ goal_dir (prior is approximate target)\n"
                      "high = chosen deviates from goal (evasion direction)",
                      fontsize=9)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3)

    axes[2].axhline(0, color="grey", ls=":", lw=0.6)
    axes[2].axhline(90, color="grey", ls=":", lw=0.6)
    axes[2].set_xlabel("replan index")
    axes[2].set_ylabel("angle (°)")
    axes[2].set_title("(c) top-1 weighted rollout vs goal direction\n"
                      "low = best rollout ≈ straight-to-goal (prior is correct)\n"
                      "high = best rollout deviates from goal (specific evasion)",
                      fontsize=9)
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        "I: direct mechanism — top-rollout disagreement vs prior-alignment, vanilla MPPI σ=3 ep 0",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    out = Path("docs/images/u_shape_top_rollouts.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

    # Save raw data
    json_out = Path("docs/images/u_shape_top_rollouts.json")
    with json_out.open("w") as f:
        json.dump(results, f)
    print(f"wrote {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
