"""Top-down GIF: swarm_transformer crossing a hub with a moving obstacle.

Uses the same multi-drone runner as ``uav-nav run`` so the animation matches
eval. Rolls out the framework 50×50 antipodal+obstacle geometry with the RL
checkpoint and renders drone trails plus the bouncing intruder.

  python scripts/render_swarm_transformer_obstacle_gif.py
  python scripts/render_swarm_transformer_obstacle_gif.py --seed 6000
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from uav_nav_lab.config import ExperimentConfig  # noqa: E402
from uav_nav_lab.runner.multi.builder import _build_multi  # noqa: E402
from uav_nav_lab.runner.multi.episode import run_episode_multi  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
YAML = REPO_ROOT / "examples/exp_multi_drone_antipodal_obstacle_swarm_transformer.yaml"
CKPT = REPO_ROOT / "results/swarm_transformer_framework_obstacle_rl_best.npz"
BG = "#0d1117"
PANEL = "#161b22"
GRID = "#21262d"
THREAT = "#f85149"
GOAL_C = "#3fb950"
HUB = "#484f58"


@dataclass
class _Rollout:
    drones: list[np.ndarray]
    obstacle: list[np.ndarray]
    goals: np.ndarray
    joint: str


def _rollout(cfg: ExperimentConfig, seed: int, *, planner: dict | None = None) -> _Rollout:
    if planner is not None:
        cfg = ExperimentConfig.from_dict({
            **cfg.raw,
            "planner": {**dict(cfg.planner), **planner},
        })
    scenario, sims, planners, sensors = _build_multi(cfg)
    replan = float(cfg.planner.get("replan_period", 0.2))
    max_steps = int(cfg.simulator.get("max_steps", 1000))
    dt = float(cfg.simulator.get("dt", 0.05))

    recs = run_episode_multi(
        scenario, sims, planners, sensors,
        seed=seed, replan_period=replan, max_steps=max_steps,
        episode_index=0,
    )
    n = len(recs)
    goals = np.stack([np.asarray(d.goal, dtype=float)[:2] for d in scenario.drones])

    per_drone = [np.array([s["true_pos"][:2] for s in r.steps], dtype=float) for r in recs]
    nf = max(len(t) for t in per_drone) if per_drone else 0
    drone_hist: list[np.ndarray] = []
    for k in range(nf):
        drone_hist.append(np.stack([
            per_drone[i][min(k, len(per_drone[i]) - 1)] for i in range(n)
        ]))

    # Replay obstacle motion with the same set_targets / advance cadence.
    scenario.reseed(seed)
    obs_hist: list[np.ndarray] = []
    for k in range(nf):
        scenario.set_targets([drone_hist[k][i] for i in range(n)])
        dyn = scenario.dynamic_obstacles
        if dyn:
            obs_hist.append(np.asarray(dyn[0]["position"], dtype=float)[:2])
        if k + 1 < nf:
            scenario.advance(dt)

    outcomes = [r.outcome for r in recs]
    if all(o == "success" for o in outcomes):
        joint = "success"
    elif any(o == "collision" for o in outcomes):
        joint = "collision"
    else:
        joint = "timeout"

    return _Rollout(drones=drone_hist, obstacle=obs_hist, goals=goals, joint=joint)


def _pick_seed(cfg: ExperimentConfig, prefer: int) -> int:
    run = _rollout(cfg, prefer)
    if run.joint == "success":
        return prefer
    for s in range(6000, 6020):
        if s == prefer:
            continue
        if _rollout(cfg, s).joint == "success":
            return s
    return prefer


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=6000)
    ap.add_argument("--pick-seed", action="store_true", help="scan 6000-6019 for joint success")
    ap.add_argument("--yaml", default=str(YAML))
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--stride", type=int, default=2, help="frame subsample")
    ap.add_argument("--max-frames", type=int, default=220)
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "docs/images/swarm_transformer_obstacle.gif"),
    )
    args = ap.parse_args()

    if not CKPT.is_file():
        print(f"missing checkpoint: {CKPT}")
        return 1

    cfg = ExperimentConfig.from_yaml(args.yaml)
    pcfg = dict(cfg.planner)
    pcfg["checkpoint"] = str(CKPT)
    cfg.planner = pcfg

    seed = _pick_seed(cfg, args.seed) if args.pick_seed else args.seed
    run = _rollout(cfg, seed)
    frames = list(range(0, len(run.drones), max(1, args.stride)))
    if len(frames) > args.max_frames:
        frames = frames[: args.max_frames]
    nf = len(frames)
    n = run.goals.shape[0]
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, n))

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    ax.set_aspect("equal")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.add_patch(Circle((25, 25), 3.0, fill=False, ec=HUB, lw=1.2, ls="--", alpha=0.6))
    ax.scatter(run.goals[:, 0], run.goals[:, 1], s=120, marker="*", c=colors,
               alpha=0.5, zorder=2)
    trails = [ax.plot([], [], "-", color=colors[i], lw=1.4, alpha=0.6, zorder=3)[0]
              for i in range(n)]
    scat = ax.scatter(run.drones[0][:, 0], run.drones[0][:, 1], s=70, c=colors,
                      edgecolors="white", linewidths=0.5, zorder=5)
    o0 = run.obstacle[0] if run.obstacle else np.array([25.0, 2.0])
    threat = Circle(tuple(o0), 1.5, fc=THREAT, ec="white", lw=0.8, alpha=0.85, zorder=4)
    ax.add_patch(threat)

    title_c = GOAL_C if run.joint == "success" else THREAT
    ax.set_title(
        f"swarm_transformer — antipodal hub + moving obstacle  (joint {run.joint}, seed {seed})",
        color=title_c, fontsize=11, fontweight="bold", pad=10,
    )

    def update(k):
        f = frames[k]
        pos = run.drones[f]
        scat.set_offsets(pos)
        for i, tr in enumerate(trails):
            hist = np.array([run.drones[j][i] for j in range(f + 1)])
            tr.set_data(hist[:, 0], hist[:, 1])
        if f < len(run.obstacle):
            threat.center = tuple(run.obstacle[f])
        return trails + [scat, threat]

    anim = FuncAnimation(fig, update, frames=nf, interval=1000 / args.fps, blit=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"[gif] {out}  ({out.stat().st_size // 1024} KB, joint {run.joint}, seed {seed}, {nf} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
