#!/usr/bin/env python3
"""3-way battle montage from swarm_policy_battle results.

Picks a seed where the champion succeeds and challengers fail (from
phase.json if available, else defaults), then renders a side-by-side GIF.

  python scripts/render_swarm_policy_battle_gif.py
  python scripts/render_swarm_policy_battle_gif.py --seed 6003 --arms mpc_gt swarm_transformer
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from uav_nav_lab.config import ExperimentConfig  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE = REPO_ROOT / "results/swarm_policy_battle/phase.json"
OUT = REPO_ROOT / "docs/images/swarm_policy_battle_obstacle.gif"

import render_swarm_transformer_obstacle_gif as rs  # noqa: E402
import swarm_policy_battle_phase as battle  # noqa: E402


def _pick_seed(
    phase_path: Path,
    scenario: str,
    ref: str,
    challengers: list[str],
) -> int | None:
    if not phase_path.is_file():
        return None
    data = json.loads(phase_path.read_text())
    cell = data.get("cells", {}).get(scenario, {})
    by_arm = cell.get("outcomes_by_seed", {})
    if ref not in by_arm:
        return None
    seeds = sorted(int(s) for s in by_arm[ref].keys())
    for s in seeds:
        if by_arm[ref].get(str(s)) != "success":
            continue
        if all(by_arm.get(c, {}).get(str(s)) != "success" for c in challengers):
            return s
    return None


def _cfg(scenario: str) -> ExperimentConfig:
    raw = battle._cfg(scenario, "orca", seed=6000, n_eps=1)
    return ExperimentConfig.from_dict(raw)


def _pad(traj: list[np.ndarray], n: int) -> list[np.ndarray]:
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
    ap.add_argument("--scenario", default="obstacle", choices=("peers", "obstacle"))
    ap.add_argument("--ref", default="swarm_transformer")
    ap.add_argument("--arms", nargs="+", default=["navrl", "mpc_gt", "swarm_transformer"])
    ap.add_argument("--seed", type=int, default=0, help="0 = auto-pick from phase.json")
    ap.add_argument("--phase", default=str(PHASE))
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    challengers = [a for a in args.arms if a != args.ref]
    seed = args.seed
    if seed == 0:
        seed = _pick_seed(Path(args.phase), args.scenario, args.ref, challengers) or 6003

    cfg = _cfg(args.scenario)
    labels = {
        "orca": "ORCA (stock)",
        "orca_conv": "ORCA + convention",
        "hrvo": "HRVO",
        "mgr": "MGR",
        "mpc_gt": "MPC + GT (teacher)",
        "swarm_transformer": "swarm_transformer",
        "navrl": "NavRL (upstream ckpt)",
    }
    runs = {}
    for arm in args.arms:
        runs[labels.get(arm, arm)] = rs._rollout(
            cfg, seed, planner=battle._planner(arm, args.scenario),
        )

    nf = min(max(len(r.drones) for r in runs.values()), args.max_frames)
    frames = list(range(0, nf, max(1, args.stride)))
    n = next(iter(runs.values())).goals.shape[0]
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, n))

    fig, axes = plt.subplots(1, len(args.arms), figsize=(4.2 * len(args.arms), 4.8))
    if len(args.arms) == 1:
        axes = [axes]
    fig.patch.set_facecolor(rs.BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.05, wspace=0.08)
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
        if args.scenario == "obstacle":
            ax.add_patch(Circle((25, 25), 3.0, fill=False, ec=rs.HUB, lw=1.0, ls="--", alpha=0.5))
        ax.scatter(run.goals[:, 0], run.goals[:, 1], s=70, marker="*", c=colors, alpha=0.45, zorder=2)
        drones = _pad(run.drones, nf)
        obs = _pad(run.obstacle, nf) if run.obstacle else []
        trails = [ax.plot([], [], "-", color=colors[i], lw=1.1, alpha=0.55, zorder=3)[0] for i in range(n)]
        scat = ax.scatter(drones[0][:, 0], drones[0][:, 1], s=55, c=colors,
                          edgecolors="white", linewidths=0.4, zorder=5)
        threat = None
        if obs:
            threat = Circle(tuple(obs[0]), 1.5, fc=rs.THREAT, ec="white", lw=0.6, alpha=0.85, zorder=4)
            ax.add_patch(threat)
        ok = run.joint == "success"
        ax.set_title(f"{title}\njoint {run.joint}", color=rs.GOAL_C if ok else rs.THREAT,
                     fontsize=9, fontweight="bold", pad=4)
        art[title] = dict(trails=trails, scat=scat, threat=threat, drones=drones, obs=obs)

    fig.suptitle(
        f"Swarm policy battle — {args.scenario}, seed {seed}",
        color="#c9d1d9", fontsize=11, y=0.96,
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
            if a["threat"] is not None and f < len(a["obs"]):
                a["threat"].center = tuple(a["obs"][f])
            out += a["trails"] + [a["scat"]] + ([a["threat"]] if a["threat"] else [])
        return out

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / args.fps, blit=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"[gif] {out}  ({out.stat().st_size // 1024} KB, seed {seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
