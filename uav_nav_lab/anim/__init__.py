"""Episode replay → animated GIF.

Re-walks a recorded episode at simulator time and renders one GIF per
episode (sub-sampled to a target FPS). Dynamic obstacles, lidar memory
state, and the drone trajectory build up frame by frame so the
animation matches what the planner actually saw at each replan.

The animation re-reads the saved config so the scenario seed and
dynamic obstacle setup match the original run exactly. Both 2D
(matplotlib axes) and 3D (Axes3D with rotating camera) renderings are
supported; the dispatcher picks based on ``scenario.ndim`` and the
scenario type (``multi_drone_*`` → multi-drone variants).

Split from a 652-line single-file module into:

- :mod:`._common`  — palette, mpl shim, frame-index down-sampler,
  dynamic-obstacle replay, and the "most recent replan with rollouts"
  lookup (used by both 3D animators).
- :mod:`.single_2d` / :mod:`.single_3d` — one-drone animators.
- :mod:`.multi_2d`  / :mod:`.multi_3d`  — multi-drone animators.

:func:`viz_anim` is the only public entry point — every external
caller already imports it from :mod:`uav_nav_lab.anim`.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from ..config import ExperimentConfig
from ..scenario import SCENARIO_REGISTRY
from ._common import need_mpl_anim
from .multi_2d import animate_episode_multi_2d
from .multi_3d import animate_episode_multi_3d
from .single_2d import animate_episode_2d
from .single_3d import animate_episode_3d

__all__ = [
    "viz_anim",
    "animate_episode_2d",
    "animate_episode_3d",
    "animate_episode_multi_2d",
    "animate_episode_multi_3d",
]


_MULTI_SCENARIO_TYPES = ("multi_drone_grid", "multi_drone_voxel")


def viz_anim(run_dir: Path, fps: int = 20) -> list[Path]:
    plt, animation = need_mpl_anim()
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found — anim needs the saved config")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = ExperimentConfig.from_dict(yaml.safe_load(f))
    scenario_cls = SCENARIO_REGISTRY.get(cfg.scenario.get("type", "grid_world"))
    scenario = scenario_cls.from_config(cfg.scenario)
    if scenario.ndim not in (2, 3):
        raise NotImplementedError(f"anim supports 2D / 3D scenarios (got ndim={scenario.ndim}).")

    is_multi = str(cfg.scenario.get("type", "")) in _MULTI_SCENARIO_TYPES
    saved: list[Path] = []

    if is_multi:
        # Group per-drone JSONs by episode index, render one GIF per episode.
        episodes: list[dict] = []
        for ef in sorted(run_dir.glob("episode_*.json")):
            if ef.stem.endswith("_joint"):
                continue
            with ef.open("r", encoding="utf-8") as f:
                episodes.append(json.load(f))
        by_ep: dict[int, list[dict]] = {}
        for ep in episodes:
            by_ep.setdefault(int(ep["meta"]["episode"]), []).append(ep)
        animator = animate_episode_multi_3d if scenario.ndim == 3 else animate_episode_multi_2d
        for ep_idx in sorted(by_ep):
            result = animator(
                plt, animation, cfg, by_ep[ep_idx], scenario, fps=fps,
            )
            if result is None:
                continue
            fig, anim = result
            out = run_dir / f"episode_{ep_idx:03d}.gif"
            anim.save(out, writer="pillow", fps=fps)
            plt.close(fig)
            saved.append(out)
        return saved

    for ef in sorted(run_dir.glob("episode_*.json")):
        if "_drone_" in ef.stem or ef.stem.endswith("_joint"):
            continue
        with ef.open("r", encoding="utf-8") as f:
            ep = json.load(f)
        if scenario.ndim == 3:
            result = animate_episode_3d(plt, animation, cfg, ep, scenario, fps=fps)
        else:
            result = animate_episode_2d(plt, animation, cfg, ep, scenario, fps=fps)
        if result is None:
            continue
        fig, anim = result
        out = run_dir / f"episode_{ep['meta']['episode']:03d}.gif"
        anim.save(out, writer="pillow", fps=fps)
        plt.close(fig)
        saved.append(out)
    return saved
