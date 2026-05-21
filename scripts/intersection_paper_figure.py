"""4-panel paper figure for the intersection-coordination finding.

(a) v1 trajectory: 2-drone open intersection, MPC vs MPPI overlay.
(b) wave trajectory: same 2 drones with 3 phase-controlled intruders,
    showing MPC's wide detour vs MPPI's narrow weave.
(c) speed-vs-time + |Δcmd|-vs-time for ep 0 drone-east, MPC vs MPPI.
(d) fingerprint summary: bar chart with 1.96·SEM error bars for
    max |Δcmd| and plan_ms across all 4 cells × 2 planners.

Output: docs/images/intersection_fingerprint_paper.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

OUT = Path("docs/images/intersection_fingerprint_paper.png")
N_EPS = 5

CELLS = [
    ("v1",         "results/intersection_v1_mpc",            "results/intersection_v1_mppi",            "examples/exp_intersection_v1_mpc.yaml",            2),
    ("4-way",      "results/intersection_4way_mpc",          "results/intersection_4way_mppi",          "examples/exp_intersection_4way_mpc.yaml",          4),
    ("chokepoint", "results/intersection_chokepoint_v1_mpc", "results/intersection_chokepoint_v1_mppi", "examples/exp_intersection_chokepoint_v1_mpc.yaml", 2),
    ("wave",       "results/intersection_wave_v1_mpc",       "results/intersection_wave_v1_mppi",       "examples/exp_intersection_wave_v1_mpc.yaml",       2),
]
DRONE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
INTRUDER_COLOR = "#cc1f1f"
PLANNER_COLORS = {"MPC": "#d62728", "MPPI": "#1f77b4"}


def _reflect(p, lim):
    v = p % (2.0 * lim)
    if v > lim:
        v = 2.0 * lim - v
    return v


def _intruder_pos(start, vel, t, size):
    p = np.asarray(start, dtype=float) + np.asarray(vel, dtype=float) * t
    return np.array([_reflect(p[k], size[k]) for k in range(3)])


def _load_ep(run_dir, ep, n_drones):
    return [json.load(open(Path(run_dir) / f"episode_{ep:03d}_drone_{i:02d}.json")) for i in range(n_drones)]


def _plot_trajectory_pane(ax, mpc_dir, mppi_dir, yaml_path, n_drones, title):
    cfg = yaml.safe_load(open(yaml_path))
    size = cfg["scenario"]["size"]
    dt = cfg["simulator"]["dt"]
    intruders = cfg["scenario"]["dynamic_obstacles"]

    # static box obstacles (chokepoint cell)
    boxes = cfg["scenario"].get("obstacles", {}).get("boxes", []) or []
    for box in boxes:
        c = box.get("center")
        s = box.get("size")
        if c is None or s is None:
            continue
        s = s if isinstance(s, (list, tuple)) else [s, s, s]
        rect = plt.Rectangle((c[0] - s[0]/2, c[1] - s[1]/2), s[0], s[1],
                             color="#666666", alpha=0.4, zorder=1)
        ax.add_patch(rect)

    # MPC + MPPI ep 0
    for planner_name, run_dir, ls, alpha in [("MPC", mpc_dir, "-", 0.9), ("MPPI", mppi_dir, "--", 0.9)]:
        drones = _load_ep(run_dir, 0, n_drones)
        for i, d in enumerate(drones):
            xs = [s["true_pos"][0] for s in d["steps"]]
            ys = [s["true_pos"][1] for s in d["steps"]]
            label = f"{planner_name}" if i == 0 else None
            ax.plot(xs, ys, ls, color=PLANNER_COLORS[planner_name],
                    lw=1.6, alpha=alpha, label=label, zorder=4)
            ax.scatter([xs[0]], [ys[0]], marker="o", color=PLANNER_COLORS[planner_name],
                       s=40, edgecolor="white", linewidth=0.8, zorder=5)
            ax.scatter([xs[-1]], [ys[-1]], marker="*", color=PLANNER_COLORS[planner_name],
                       s=90, edgecolor="white", linewidth=0.8, zorder=5)

    # intruder trajectories (ground truth, time range = max ep length)
    max_steps = max(len(d["steps"]) for d in _load_ep(mpc_dir, 0, n_drones))
    ts = np.arange(0, max_steps * dt, dt)
    for intr in intruders:
        pts = np.array([_intruder_pos(intr["start"], intr["velocity"], t, size) for t in ts])
        ax.plot(pts[:, 0], pts[:, 1], "-", color=INTRUDER_COLOR, lw=1.2, alpha=0.5, zorder=2)
        # mark current pos every 1 s
        for k in range(0, len(ts), int(1.0 / dt)):
            circ = plt.Circle((pts[k, 0], pts[k, 1]), intr.get("radius", 0.5),
                              color=INTRUDER_COLOR, alpha=0.10, zorder=2)
            ax.add_patch(circ)

    ax.set_aspect("equal")
    ax.set_xlim(0, size[0]); ax.set_ylim(0, size[1])
    ax.grid(alpha=0.3)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(title, fontsize=10)
    if not ax.has_data():
        return
    leg = ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    leg.set_zorder(10)


def _plot_speed_dcmd_pane(ax, mpc_dir, mppi_dir, n_drones):
    """Show |v|(t) and |Δcmd|(t) for ep 0 drone-east (idx 1)."""
    ax2 = ax.twinx()
    for planner_name, run_dir, ls in [("MPC", mpc_dir, "-"), ("MPPI", mppi_dir, "--")]:
        d = _load_ep(run_dir, 0, n_drones)[1]  # drone-east
        vels = np.array([s["true_vel"] for s in d["steps"]])
        cmds = np.array([s["cmd"] for s in d["steps"]])
        ts = np.array([s["t"] for s in d["steps"]])
        speeds = np.linalg.norm(vels, axis=1)
        dcmds = np.concatenate([[0], np.linalg.norm(np.diff(cmds, axis=0), axis=1)])
        ax.plot(ts, speeds, ls, color=PLANNER_COLORS[planner_name], lw=1.8,
                label=f"{planner_name} |v|", alpha=0.95)
        ax2.plot(ts, dcmds, ls, color=PLANNER_COLORS[planner_name], lw=1.0,
                 alpha=0.45)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("drone speed |v| (m/s)", color="black")
    ax2.set_ylabel("|Δcmd| step-to-step (m/s, faded)", color="#666666")
    ax.grid(alpha=0.3)
    ax.set_title("(c) v1 ep 0 drone-east: |v|(t) solid, |Δcmd|(t) faded", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)


def _fingerprint_bar_pane(ax, metric_key, ylabel, title):
    """Bar chart with 1.96·SEM error bars for the given metric across all 4 cells × 2 planners.
    Reuses the per-episode metric computation from intersection_fingerprint.py."""
    from intersection_fingerprint import cell_fingerprint, _stats

    width = 0.35
    x = np.arange(len(CELLS))
    mpc_means, mpc_errs, mppi_means, mppi_errs = [], [], [], []
    for tag, mpc_dir, mppi_dir, yaml_path, n_drones in CELLS:
        mpc = cell_fingerprint(mpc_dir, yaml_path, n_drones)
        mppi_yaml = yaml_path.replace("_mpc.yaml", "_mppi.yaml")
        mppi = cell_fingerprint(mppi_dir, mppi_yaml, n_drones)
        mm, mh = _stats(mpc[metric_key])
        pm, ph = _stats(mppi[metric_key])
        mpc_means.append(mm); mpc_errs.append(mh)
        mppi_means.append(pm); mppi_errs.append(ph)

    ax.bar(x - width/2, mpc_means, width, yerr=mpc_errs, color=PLANNER_COLORS["MPC"],
           label="MPC (argmin)", capsize=4, alpha=0.85)
    ax.bar(x + width/2, mppi_means, width, yerr=mppi_errs, color=PLANNER_COLORS["MPPI"],
           label="MPPI (softmax)", capsize=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([tag for tag, *_ in CELLS], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=8)


def main() -> int:
    import sys
    sys.path.insert(0, "scripts")

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    _plot_trajectory_pane(
        ax_a, *("results/intersection_v1_mpc", "results/intersection_v1_mppi",
                "examples/exp_intersection_v1_mpc.yaml", 2,
                "(a) v1 ep 0: open intersection, 1 intruder"))
    _plot_trajectory_pane(
        ax_b, *("results/intersection_wave_v1_mpc", "results/intersection_wave_v1_mppi",
                "examples/exp_intersection_wave_v1_mpc.yaml", 2,
                "(b) wave ep 0: 3 phase-controlled intruders — MPC detours wide, MPPI weaves"))
    _plot_speed_dcmd_pane(ax_c, "results/intersection_v1_mpc", "results/intersection_v1_mppi", 2)
    _fingerprint_bar_pane(ax_d, "max_dcmd", "max |Δcmd| (m/s)",
                          "(d) Behavioral fingerprint across 4 cells: max |Δcmd|")

    fig.suptitle(
        "Intersection coordination — behavioral fingerprint separates MPC vs MPPI"
        " even when binary success saturates",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
