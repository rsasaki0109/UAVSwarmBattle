"""Shared helpers for ``scripts/record_airsim_*.py``.

All five recording scripts used to each carry their own AirSim camera
setup, their own subprocess-driven experiment runner, and their own
two-pass ffmpeg GIF builder (with the same decimation formula
re-derived four times). Pulling these out keeps each driver script
focused on its own configuration (YAML path, output path, camera
specifics) — the boilerplate only lives here.

The helpers are organised by external coupling so unit tests can
exercise them with fake clients / mocked subprocess:

* :mod:`.airsim_camera`     — pitch front_center, set top-down camera
* :mod:`.experiment_runner` — ``uav-nav run`` via subprocess
* :mod:`.ffmpeg_gif`        — PNG sequence → GIF, optional decimation
"""

from __future__ import annotations

from .airsim_camera import pitch_front_center, set_topdown_camera
from .experiment_runner import run_uav_nav_experiment
from .ffmpeg_gif import build_ffmpeg_vf, count_frames, frames_to_gif

__all__ = [
    "build_ffmpeg_vf",
    "count_frames",
    "frames_to_gif",
    "pitch_front_center",
    "run_uav_nav_experiment",
    "set_topdown_camera",
]
