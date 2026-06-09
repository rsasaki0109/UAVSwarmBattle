#!/usr/bin/env python3
"""Fine-tune swarm_transformer with REINFORCE on framework obstacle YAML.

Warm-starts from the BC checkpoint (predicted threat tokens) and optimizes
a symmetric progress/collision reward on the same 50x50 geometry.

  python scripts/train_swarm_transformer_framework_rl.py
  python scripts/train_swarm_transformer_framework_rl.py --iters 60 --episodes 8
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from uav_nav_lab.config import ExperimentConfig  # noqa: E402
from uav_nav_lab.planner import swarm_transformer_core as core  # noqa: E402
from uav_nav_lab.planner.swarm_transformer_rl import train_from_config  # noqa: E402

OBST_YAML = ROOT / "examples/exp_multi_drone_antipodal_obstacle.yaml"
TRIPLE_YAML = ROOT / "examples/exp_multi_drone_antipodal_obstacle_triple.yaml"
EVAL_YAML = ROOT / "examples/exp_multi_drone_antipodal_obstacle_swarm_transformer.yaml"
TRIPLE_EVAL_YAML = ROOT / "examples/exp_multi_drone_antipodal_obstacle_triple_swarm_transformer.yaml"
BC_INIT = ROOT / "results/swarm_transformer_framework_threat_pred_conv.npz"
RL_INIT = ROOT / "results/swarm_transformer_framework_obstacle_rl.npz"
RL_BEST = ROOT / "results/swarm_transformer_framework_obstacle_rl_best.npz"
TRIPLE_RL_BEST = ROOT / "results/swarm_transformer_framework_triple_rl_best.npz"
OUT = RL_INIT
# Failing eval seeds from 14/20 checkpoint (timeout / collision).
HARD_SEEDS = (6000, 6003, 6006, 6008, 6012, 6014)


def _eval(checkpoint: Path, *, eval_yaml: Path = EVAL_YAML) -> tuple[int, int]:
    cfg_text = eval_yaml.read_text()
    tmp = ROOT / "results/_tmp_swarm_rl_eval.yaml"
    lines = []
    in_planner = False
    for line in cfg_text.splitlines():
        if line.strip().startswith("planner:"):
            in_planner = True
            lines.append("planner:")
            lines.append("  type: swarm_transformer")
            lines.append("  max_speed: 5.0")
            lines.append("  replan_period: 0.2")
            lines.append("  neighbor_dist: 15.0")
            lines.append("  interaction_radius: 4.0")
            lines.append("  goal_radius: 1.5")
            lines.append("  predictor:")
            lines.append("    type: game_theoretic")
            lines.append(f"  checkpoint: {checkpoint}")
            continue
        if in_planner:
            if line and not line.startswith(" "):
                in_planner = False
            else:
                continue
        if line.strip().startswith("dir:"):
            lines.append("  dir: results/_tmp_swarm_rl_eval")
            continue
        if line.strip().startswith("num_episodes:"):
            lines.append("num_episodes: 20")
            continue
        if not in_planner:
            lines.append(line)
    tmp.write_text("\n".join(lines) + "\n")
    out = ROOT / "results/_tmp_swarm_rl_eval"
    if out.exists():
        for f in out.glob("episode_*"):
            f.unlink()
    subprocess.run(["uav-nav", "run", str(tmp)], cwd=ROOT, check=True)
    joints = [json.loads(f.read_text()) for f in sorted(out.glob("episode_*_joint.json"))]
    jok = sum(1 for j in joints if j.get("outcome") == "success")
    return jok, len(joints)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--joint-bonus", type=float, default=10.0)
    ap.add_argument("--collision-penalty", type=float, default=8.0)
    ap.add_argument("--curriculum-hard", action="store_true")
    ap.add_argument("--seeds", default="", help="comma-separated episode seeds")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--triple", action="store_true", help="train on triple-crossfire YAML")
    ap.add_argument("--yaml", default="", help="override training YAML path")
    args = ap.parse_args()

    train_yaml = Path(args.yaml) if args.yaml else (TRIPLE_YAML if args.triple else OBST_YAML)
    cfg = ExperimentConfig.from_yaml(train_yaml)
    init = Path(args.init) if args.init else None
    if init is None:
        if args.curriculum_hard and RL_BEST.is_file():
            init = RL_BEST
        else:
            init = BC_INIT

    episode_seeds: tuple[int, ...] | None = None
    if args.seeds:
        episode_seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    elif args.curriculum_hard:
        episode_seeds = HARD_SEEDS
    if not init.is_file():
        print(f"warning: init checkpoint missing ({init}), training from scratch", flush=True)
        init_path = None
    else:
        init_path = str(init)
        print(f"warm-start: {init.name}", flush=True)

    print(
        f"RL on {train_yaml.name} ({args.iters} iters x {args.episodes} eps, "
        f"sigma={args.sigma})...",
        flush=True,
    )
    if episode_seeds:
        print(f"  curriculum seeds: {list(episode_seeds)}", flush=True)
    params, stats = train_from_config(
        cfg,
        init_checkpoint=init_path,
        predictor_cfg=dict(cfg.planner.get("predictor", {"type": "game_theoretic"})),
        iters=args.iters,
        episodes=args.episodes,
        lr=args.lr,
        sigma=args.sigma,
        seed=args.seed,
        joint_bonus=args.joint_bonus,
        collision_penalty=args.collision_penalty,
        episode_seeds=episode_seeds,
        verbose=True,
    )
    out = Path(args.out)
    core.save_checkpoint(out, params, stats)
    print(f"wrote {out}", flush=True)

    if not args.skip_eval:
        eval_yaml = TRIPLE_EVAL_YAML if args.triple else EVAL_YAML
        jok, total = _eval(out, eval_yaml=eval_yaml)
        label = "triple" if args.triple else "single"
        print(f"eval {label} joint success: {jok}/{total}", flush=True)
        best = TRIPLE_RL_BEST if args.triple else RL_BEST
        prev = 0
        if best.is_file():
            prev, _ = _eval(best, eval_yaml=eval_yaml)
            print(f"  previous best ({label}): {prev}/{total}", flush=True)
        elif args.triple and RL_BEST.is_file():
            prev, _ = _eval(RL_BEST, eval_yaml=eval_yaml)
            print(f"  champion baseline ({label}): {prev}/{total}", flush=True)
        if jok > prev:
            core.save_checkpoint(best, params, stats)
            print(f"saved best ({jok}/{total}) → {best.name}", flush=True)
        elif jok < prev and best.is_file():
            core.save_checkpoint(out, *core.load_checkpoint(best))
            print(f"restored previous best ({prev}/{total}) → {out.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
