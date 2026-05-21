"""Multi-drone experiment driver — N episodes × per-drone logs + joint summary."""

from __future__ import annotations

import json as _json
from pathlib import Path

import yaml

from ...config import ExperimentConfig
from .builder import _build_multi
from .episode import run_episode_multi


def run_experiment_multi(cfg: ExperimentConfig, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)

    scenario, sims, planners, sensors = _build_multi(cfg)
    n = scenario.n_drones
    replan_period = float(cfg.planner.get("replan_period", 0.5))
    max_steps = int(cfg.simulator.get("max_steps", 2000))

    save_frames = bool((cfg.output or {}).get("save_camera_frames", False))

    print(
        f"[run] {cfg.name}: {cfg.num_episodes} episode(s), {n} drone(s) → {output_dir}"
    )
    for ep in range(cfg.num_episodes):
        seed = cfg.seed + ep
        # Per-drone frame directory; only created when save_camera_frames
        # is on, otherwise the runner skips PNG writes entirely.
        frame_dirs: list[Path | None] = []
        for i in range(n):
            if save_frames:
                fd = output_dir / f"frames_{ep:03d}_drone_{i:02d}"
                fd.mkdir(parents=True, exist_ok=True)
                frame_dirs.append(fd)
            else:
                frame_dirs.append(None)
        recs = run_episode_multi(
            scenario, sims, planners, sensors,
            seed=seed,
            replan_period=replan_period,
            max_steps=max_steps,
            episode_index=ep,
            frame_dirs=frame_dirs,
        )
        outcomes = [r.outcome for r in recs]
        for i, rec in enumerate(recs):
            rec.save(output_dir / f"episode_{ep:03d}_drone_{i:02d}.json")
        joint_outcome = (
            "success" if all(o == "success" for o in outcomes)
            else "collision" if any(o == "collision" for o in outcomes)
            else "timeout"
        )
        joint = {
            "meta": {"episode": ep, "seed": seed, "n_drones": n},
            "outcome": joint_outcome,
            "per_drone_outcomes": outcomes,
            "drone_names": [d.name for d in scenario.drones],
            "final_t": max(float(r.summary.get("final_t", 0.0)) for r in recs),
        }
        with (output_dir / f"episode_{ep:03d}_joint.json").open("w", encoding="utf-8") as f:
            _json.dump(joint, f, indent=2)
        print(f"  ep {ep:03d} seed={seed} per-drone={outcomes} joint={joint_outcome}")

    return output_dir
