"""Side-by-side GIF: MPC teacher vs swarm_transformer on hub-crossing obstacle.

Same 50×50 antipodal+obstacle geometry and seed; left = MPC + lateral_bias=4,
right = learned transformer (RL checkpoint). Uses the framework multi runner.

  python scripts/render_swarm_transformer_obstacle_compare_gif.py --seed 6006
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from uav_nav_lab.config import ExperimentConfig  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
YAML = REPO_ROOT / "examples/exp_multi_drone_antipodal_obstacle.yaml"
XF_YAML = REPO_ROOT / "examples/exp_multi_drone_antipodal_obstacle_swarm_transformer.yaml"
CKPT = REPO_ROOT / "results/swarm_transformer_framework_obstacle_rl_best.npz"
OUT = REPO_ROOT / "docs/images/swarm_transformer_obstacle_compare.gif"

import render_swarm_transformer_obstacle_gif as rs  # noqa: E402

MPC_PLANNER = {
    "type": "mpc",
    "max_speed": 5.0,
    "replan_period": 0.2,
    "horizon": 40,
    "dt_plan": 0.05,
    "n_samples": 32,
    "resolution": 1.0,
    "inflate": 1,
    "goal_radius": 1.5,
    "safety_margin": 0.5,
    "use_prediction": True,
    "w_goal": 1.0,
    "w_obs": 100.0,
    "w_smooth": 0.05,
    "lateral_bias": 4.0,
    "predictor": {"type": "game_theoretic"},
}
XF_PLANNER = {
    "type": "swarm_transformer",
    "max_speed": 5.0,
    "replan_period": 0.2,
    "neighbor_dist": 15.0,
    "interaction_radius": 4.0,
    "goal_radius": 1.5,
    "predictor": {"type": "game_theoretic"},
    "checkpoint": str(CKPT),
}


def _pad(traj: list[np.ndarray], n: int) -> list[np.ndarray]:
    if not traj:
        return traj
    out = list(traj)
    while len(out) < n:
        out.append(out[-1])
    return out


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=6003)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=220)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    if not CKPT.is_file():
        print(f"missing checkpoint: {CKPT}")
        return 1

    cfg = ExperimentConfig.from_yaml(YAML)
    runs = {
        "MPC + convention (teacher)": rs._rollout(cfg, args.seed, planner=MPC_PLANNER),
        "swarm_transformer (student)": rs._rollout(
            ExperimentConfig.from_yaml(XF_YAML), args.seed, planner=XF_PLANNER,
        ),
    }
    nf = min(max(len(r.drones) for r in runs.values()), args.max_frames)
    nf = min(nf, args.max_frames)
    stride = max(1, args.stride)
    frames = list(range(0, nf, stride))
    n = runs["MPC + convention (teacher)"].goals.shape[0]
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, n))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4))
    fig.patch.set_facecolor(rs.BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.04, wspace=0.06)
    art = {}

    for ax, (title, run) in zip(axes, runs.items()):
        ax.set_facecolor(rs.PANEL)
        ax.set_aspect("equal")
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(rs.GRID)
        ax.add_patch(Circle((25, 25), 3.0, fill=False, ec=rs.HUB, lw=1.2, ls="--", alpha=0.6))
        ax.scatter(run.goals[:, 0], run.goals[:, 1], s=90, marker="*", c=colors, alpha=0.45, zorder=2)
        drones = _pad(run.drones, nf)
        obs = _pad(run.obstacle, nf)
        trails = [ax.plot([], [], "-", color=colors[i], lw=1.2, alpha=0.55, zorder=3)[0]
                  for i in range(n)]
        scat = ax.scatter(drones[0][:, 0], drones[0][:, 1], s=65, c=colors,
                          edgecolors="white", linewidths=0.5, zorder=5)
        o0 = obs[0] if obs else np.array([25.0, 2.0])
        threat = Circle(tuple(o0), 1.5, fc=rs.THREAT, ec="white", lw=0.8, alpha=0.85, zorder=4)
        ax.add_patch(threat)
        ok = run.joint == "success"
        ax.set_title(
            f"{title}\njoint {run.joint}",
            color=rs.GOAL_C if ok else rs.THREAT,
            fontsize=10, fontweight="bold", pad=6,
        )
        art[title] = dict(trails=trails, scat=scat, threat=threat, drones=drones, obs=obs)

    fig.suptitle(
        f"Hub-crossing obstacle, antipodal N=6 — seed {args.seed}",
        color="#c9d1d9", fontsize=12, y=0.96,
    )

    def update(k):
        f = frames[k]
        out = []
        for title in runs:
            a = art[title]
            pos = a["drones"][f]
            a["scat"].set_offsets(pos)
            for i, tr in enumerate(a["trails"]):
                hist = np.array([a["drones"][j][i] for j in range(f + 1)])
                tr.set_data(hist[:, 0], hist[:, 1])
            if f < len(a["obs"]):
                a["threat"].center = tuple(a["obs"][f])
            out += a["trails"] + [a["scat"], a["threat"]]
        return out

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / args.fps, blit=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(
        f"[gif] {out}  ({out.stat().st_size // 1024} KB, seed {args.seed}, "
        f"mpc={runs['MPC + convention (teacher)'].joint}, "
        f"xf={runs['swarm_transformer (student)'].joint})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
