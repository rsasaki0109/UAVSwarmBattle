#!/usr/bin/env python3
"""Train swarm_transformer on framework-scale BC (50x50 antipodal geometry).

Peers: ORCA + lateral_bias=0.2 teacher (exp_multi_drone_antipodal_orca.yaml layout).
Obstacle: MPC + lateral_bias=4 + game_theoretic predictor teacher
(exp_multi_drone_antipodal_obstacle.yaml layout).

  python scripts/train_swarm_transformer_framework_bc.py
  python scripts/train_swarm_transformer_framework_bc.py --obstacle
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _swarm_transformer as st  # noqa: E402
from uav_nav_lab.config import ExperimentConfig  # noqa: E402
from uav_nav_lab.planner.swarm_transformer_bc import collect_from_config  # noqa: E402

PEER_YAML = ROOT / "examples/exp_multi_drone_antipodal_orca.yaml"
OBST_YAML = ROOT / "examples/exp_multi_drone_antipodal_obstacle.yaml"
OUT_PEER = ROOT / "results/swarm_transformer_framework_conv.npz"
OUT_OBST = ROOT / "results/swarm_transformer_framework_threat_pred_conv.npz"


def _eval_yaml(
    yaml_path: Path,
    checkpoint: Path,
    *,
    max_speed: float = 5.0,
    replan_period: float | None = None,
) -> tuple[int, int]:
    import json
    import subprocess

    cfg = yaml_path.read_text()
    tmp = ROOT / "results" / "_tmp_swarm_xf_eval.yaml"
    lines = []
    in_planner = False
    for line in cfg.splitlines():
        if line.strip().startswith("planner:"):
            in_planner = True
            lines.append("planner:")
            lines.append("  type: swarm_transformer")
            lines.append(f"  max_speed: {max_speed}")
            if replan_period is not None:
                lines.append(f"  replan_period: {replan_period}")
            lines.append("  neighbor_dist: 15.0")
            lines.append("  interaction_radius: 4.0")
            lines.append("  goal_radius: 1.5")
            lines.append(f"  checkpoint: {checkpoint}")
            if replan_period is not None:
                lines.append("  predictor:")
                lines.append("    type: game_theoretic")
            continue
        if in_planner:
            if line and not line.startswith(" "):
                in_planner = False
            else:
                continue
        if line.strip().startswith("output:"):
            in_planner = False
        if line.strip().startswith("dir:"):
            lines.append(f"  dir: results/_tmp_swarm_xf_eval")
            continue
        if line.strip().startswith("num_episodes:"):
            lines.append("num_episodes: 20")
            continue
        if not in_planner:
            lines.append(line)
    tmp.write_text("\n".join(lines) + "\n")
    out = ROOT / "results/_tmp_swarm_xf_eval"
    if out.exists():
        for f in out.glob("episode_*"):
            f.unlink()
    subprocess.run(["uav-nav", "run", str(tmp)], cwd=ROOT, check=True)
    ok = sum(
        1 for f in out.glob("episode_*_joint.json")
        if json.loads(f.read_text()).get("outcome") == "success"
    )
    total = len(list(out.glob("episode_*_joint.json")))
    return ok, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obstacle", action="store_true")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--yaml", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    yaml_path = Path(args.yaml) if args.yaml else (OBST_YAML if args.obstacle else PEER_YAML)
    out = Path(args.out) if args.out else (OUT_OBST if args.obstacle else OUT_PEER)
    cfg = ExperimentConfig.from_yaml(yaml_path)

    predictor_cfg = None
    if args.obstacle:
        teacher_cfg = dict(cfg.planner)
        predictor_cfg = dict(cfg.planner.get("predictor", {"type": "game_theoretic"}))
        print(
            f"teacher: MPC (lateral_bias={teacher_cfg.get('lateral_bias', 0)}), "
            f"token predictor: {predictor_cfg.get('type')}",
            flush=True,
        )
    else:
        teacher_cfg = None
        print("teacher: ORCA lateral_bias=0.2", flush=True)

    print(f"collecting framework BC from {yaml_path.name} ({args.episodes} episodes)...",
          flush=True)
    data = collect_from_config(
        cfg,
        teacher_cfg=teacher_cfg,
        predictor_cfg=predictor_cfg,
        n_episodes=args.episodes,
        seed0=args.seed,
    )
    print(f"  {len(data[0])} samples", flush=True)

    print(f"training ({args.epochs} epochs)...", flush=True)
    P, stats = st.train_bc(data, epochs=args.epochs, seed=args.seed, verbose=True)
    st.save_checkpoint(out, P, stats)
    print(f"wrote {out}", flush=True)

    if not args.skip_eval:
        eval_yaml = (
            ROOT / "examples/exp_multi_drone_antipodal_obstacle_swarm_transformer.yaml"
            if args.obstacle
            else ROOT / "examples/exp_multi_drone_antipodal_swarm_transformer.yaml"
        )
        replan = float(cfg.planner.get("replan_period", 0.5)) if args.obstacle else None
        ok, total = _eval_yaml(eval_yaml, out, replan_period=replan)
        print(f"uav-nav eval {eval_yaml.name}: {ok}/{total} joint success", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
