#!/usr/bin/env python3
"""Triple-crossfire curriculum for swarm_transformer.

Warm-starts from the single-missile champion, then:
  1. BC on triple geometry (MPC + game_theoretic teacher)
  2. Staged REINFORCE: single → dual_west → triple (retain vertical skill)
  3. Hard-seed REINFORCE on triple (eval seeds 6000–6019)

  python scripts/train_swarm_transformer_triple_curriculum.py
  python scripts/train_swarm_transformer_triple_curriculum.py --skip-bc --quick
"""
from __future__ import annotations

import argparse
import copy
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

import _swarm_transformer as st  # noqa: E402
import _triple_crossfire as tc  # noqa: E402
from uav_nav_lab.config import ExperimentConfig  # noqa: E402
from uav_nav_lab.planner import swarm_transformer_core as core  # noqa: E402
from uav_nav_lab.planner.swarm_transformer_bc import collect_from_config  # noqa: E402
from uav_nav_lab.planner.swarm_transformer_rl import train_from_config  # noqa: E402

BASE_YAML = ROOT / "examples/exp_multi_drone_antipodal_obstacle.yaml"
TRIPLE_YAML = ROOT / "examples/exp_multi_drone_antipodal_obstacle_triple.yaml"
EVAL_YAML = ROOT / "examples/exp_multi_drone_antipodal_obstacle_triple_swarm_transformer.yaml"
SINGLE_EVAL_YAML = ROOT / "examples/exp_multi_drone_antipodal_obstacle_swarm_transformer.yaml"

SINGLE_BEST = ROOT / "results/swarm_transformer_framework_obstacle_rl_best.npz"
OUT_BC = ROOT / "results/swarm_transformer_framework_triple_bc.npz"
OUT_RL = ROOT / "results/swarm_transformer_framework_triple_rl.npz"
OUT_BEST = ROOT / "results/swarm_transformer_framework_triple_rl_best.npz"

HARD_SEEDS = tuple(range(6000, 6020))


def _cfg_with_dyn_obs(dyn_obs: list[dict]) -> ExperimentConfig:
    cfg = ExperimentConfig.from_yaml(TRIPLE_YAML)
    sc = copy.deepcopy(dict(cfg.scenario))
    sc["dynamic_obstacles"] = [dict(d) for d in dyn_obs]
    cfg.scenario = sc
    return cfg


def _eval_yaml(yaml_path: Path, checkpoint: Path, *, num_episodes: int = 20) -> tuple[int, int]:
    cfg_text = yaml_path.read_text()
    tmp = ROOT / "results/_tmp_triple_curriculum_eval.yaml"
    lines: list[str] = []
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
            lines.append("  dir: results/_tmp_triple_curriculum_eval")
            continue
        if line.strip().startswith("num_episodes:"):
            lines.append(f"num_episodes: {num_episodes}")
            continue
        if not in_planner:
            lines.append(line)
    tmp.write_text("\n".join(lines) + "\n")
    out = ROOT / "results/_tmp_triple_curriculum_eval"
    if out.exists():
        for f in out.glob("episode_*"):
            f.unlink()
    subprocess.run(["uav-nav", "run", str(tmp)], cwd=ROOT, check=True)
    joints = [json.loads(f.read_text()) for f in sorted(out.glob("episode_*_joint.json"))]
    jok = sum(1 for j in joints if j.get("outcome") == "success")
    return jok, len(joints)


def _run_bc(*, episodes: int, epochs: int) -> tuple[dict, dict]:
    cfg = ExperimentConfig.from_yaml(TRIPLE_YAML)
    teacher_cfg = dict(cfg.planner)
    predictor_cfg = dict(cfg.planner.get("predictor", {"type": "game_theoretic"}))
    print(f"BC on triple ({episodes} episodes, teacher=MPC)...", flush=True)
    data = collect_from_config(
        cfg,
        teacher_cfg=teacher_cfg,
        predictor_cfg=predictor_cfg,
        n_episodes=episodes,
        seed0=0,
    )
    print(f"  {len(data[0])} samples", flush=True)
    return st.train_bc(data, epochs=epochs, seed=0, verbose=True)


def _run_rl_stage(
    name: str,
    dyn_obs: list[dict],
    *,
    init: Path,
    iters: int,
    episodes: int,
    sigma: float,
    episode_seeds: tuple[int, ...] | None,
    out: Path,
) -> tuple[dict, dict]:
    cfg = _cfg_with_dyn_obs(dyn_obs)
    predictor_cfg = dict(cfg.planner.get("predictor", {"type": "game_theoretic"}))
    print(
        f"RL stage {name!r}: {len(dyn_obs)} missile(s), "
        f"{iters} iters x {episodes} eps, sigma={sigma}",
        flush=True,
    )
    if episode_seeds:
        print(f"  seeds: {list(episode_seeds)}", flush=True)
    params, stats = train_from_config(
        cfg,
        init_checkpoint=str(init),
        predictor_cfg=predictor_cfg,
        iters=iters,
        episodes=episodes,
        lr=1e-3,
        sigma=sigma,
        seed=0,
        joint_bonus=10.0,
        collision_penalty=10.0,
        episode_seeds=episode_seeds,
        verbose=True,
    )
    core.save_checkpoint(out, params, stats)
    print(f"  wrote {out.name}", flush=True)
    return params, stats


def _maybe_save_best(
    checkpoint: Path,
    *,
    min_single: int = 16,
    baseline_triple: int = 0,
) -> tuple[int, int]:
    triple_ok, triple_n = _eval_yaml(EVAL_YAML, checkpoint)
    single_ok, single_n = _eval_yaml(SINGLE_EVAL_YAML, checkpoint)
    print(f"  eval triple: {triple_ok}/{triple_n}  single: {single_ok}/{single_n}", flush=True)
    prev_triple = baseline_triple
    if OUT_BEST.is_file():
        prev_triple, _ = _eval_yaml(EVAL_YAML, OUT_BEST)
        print(f"  previous best triple: {prev_triple}/{triple_n}", flush=True)
    if triple_ok > prev_triple and single_ok >= min_single:
        data = core.load_checkpoint(checkpoint)
        core.save_checkpoint(OUT_BEST, *data)
        print(f"  saved best triple ({triple_ok}/{triple_n}) → {OUT_BEST.name}", flush=True)
    elif triple_ok > prev_triple:
        print(f"  skip save: triple improved but single {single_ok}/{single_n} < {min_single}", flush=True)
    return triple_ok, single_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bc", action="store_true", help="skip MPC BC (RL-only curriculum)")
    ap.add_argument("--quick", action="store_true", help="shorter iters for smoke test")
    ap.add_argument("--medium", action="store_true", help="between quick and full (default for CI)")
    ap.add_argument("--bc-episodes", type=int, default=120)
    ap.add_argument("--bc-epochs", type=int, default=120)
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    if args.quick:
        stage_iters = (15, 20, 30)
        hard_iters = 40
        rl_episodes = 4
        sigma = 0.10
        if args.bc_episodes == 120:
            args.bc_episodes = 40
        if args.bc_epochs == 120:
            args.bc_epochs = 60
    elif args.medium:
        stage_iters = (25, 35, 50)
        hard_iters = 80
        rl_episodes = 5
        sigma = 0.09
        if args.bc_episodes == 120:
            args.bc_episodes = 60
        if args.bc_epochs == 120:
            args.bc_epochs = 80
    else:
        stage_iters = (30, 40, 60)
        hard_iters = 120
        rl_episodes = 6
        sigma = 0.08

    baseline_triple = 0
    print("=== baseline (single-missile champion on triple) ===", flush=True)
    if not args.skip_eval and SINGLE_BEST.is_file():
        base_triple, base_single = _eval_yaml(EVAL_YAML, SINGLE_BEST), _eval_yaml(
            SINGLE_EVAL_YAML, SINGLE_BEST
        )
        baseline_triple = base_triple[0]
        print(
            f"  {SINGLE_BEST.name}: triple {base_triple[0]}/{base_triple[1]}, "
            f"single {base_single[0]}/{base_single[1]}",
            flush=True,
        )

    if not args.skip_bc:
        P, stats = _run_bc(episodes=args.bc_episodes, epochs=args.bc_epochs)
        core.save_checkpoint(OUT_BC, P, stats)
        print(f"wrote {OUT_BC}", flush=True)

    # REINFORCE always warm-starts from the single-missile champion so vertical
    # skill is not washed out by triple-only BC stats.
    rl_init = SINGLE_BEST if SINGLE_BEST.is_file() else OUT_BC
    if not Path(rl_init).is_file():
        raise SystemExit(f"missing RL init checkpoint: {rl_init}")
    print(f"RL warm-start: {Path(rl_init).name}", flush=True)

    work = OUT_RL
    for (stage_name, dyn_obs), iters in zip(tc.CURRICULUM_STAGES, stage_iters):
        _run_rl_stage(
            stage_name,
            dyn_obs,
            init=Path(rl_init),
            iters=iters,
            episodes=rl_episodes,
            sigma=sigma,
            episode_seeds=None,
            out=work,
        )
        rl_init = work
        if not args.skip_eval:
            _maybe_save_best(work, baseline_triple=baseline_triple)

    _run_rl_stage(
        "triple_hard",
        tc.TRIPLE,
        init=Path(rl_init),
        iters=hard_iters,
        episodes=rl_episodes,
        sigma=sigma * 0.9,
        episode_seeds=HARD_SEEDS,
        out=work,
    )

    if not args.skip_eval:
        triple_ok, single_ok = _maybe_save_best(work, baseline_triple=baseline_triple)
        print(
            f"\n=== done ===\ntriple: {triple_ok}/20  single: {single_ok}/20  "
            f"best: {OUT_BEST}",
            flush=True,
        )
    else:
        print(f"\n=== done (skip-eval) === checkpoint: {work}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
