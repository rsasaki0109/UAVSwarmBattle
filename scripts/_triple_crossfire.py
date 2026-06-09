"""Shared triple-crossfire dynamic-obstacle presets (battle / curriculum).

Single-missile geometry matches examples/exp_multi_drone_antipodal_obstacle*.yaml.
Triple crossfire matches scripts/swarm_policy_battle_phase.py DYN_OBS.
"""
from __future__ import annotations

# South-edge vertical only — champion training / eval geometry.
SINGLE: list[dict] = [
    {"start": [25.0, 2.0], "velocity": [0.0, 4.5], "radius": 1.5, "reflect": True},
]

# Vertical + west horizontal (curriculum stage 2).
DUAL_WEST: list[dict] = [
    {"start": [25.0, 2.0], "velocity": [0.0, 4.5], "radius": 1.5, "reflect": True},
    {"start": [2.0, 25.0], "velocity": [4.2, 0.0], "radius": 1.5, "reflect": True},
]

# Full triple crossfire (battle / README GIF).
TRIPLE: list[dict] = [
    {"start": [25.0, 2.0], "velocity": [0.0, 4.5], "radius": 1.5, "reflect": True},
    {"start": [2.0, 25.0], "velocity": [4.2, 0.0], "radius": 1.5, "reflect": True},
    {"start": [48.0, 25.0], "velocity": [-4.2, 0.0], "radius": 1.5, "reflect": True},
]

CURRICULUM_STAGES: tuple[tuple[str, list[dict]], ...] = (
    ("single", SINGLE),
    ("dual_west", DUAL_WEST),
    ("triple", TRIPLE),
)
