"""Per-episode spawn jitter for multi_drone_grid (`start_jitter`).

The crossing predictor study needs the symmetric 2-drone crossing to vary
per seed; `start_jitter` provides that without touching any fixed-layout
scenario (zero jitter must stay byte-identical to the old fixed starts).
"""

from __future__ import annotations

import numpy as np

from uav_nav_lab.scenario import SCENARIO_REGISTRY


def _cfg(jitter_a: float = 0.0, jitter_b: float = 0.0) -> dict:
    return {
        "size": [50, 50],
        "obstacles": {"type": "none"},
        "drones": [
            {"name": "east", "start": [4.0, 25.0], "goal": [46.0, 25.0], "start_jitter": jitter_a},
            {"name": "north", "start": [25.0, 4.0], "goal": [25.0, 46.0], "start_jitter": jitter_b},
        ],
    }


def _build(cfg: dict):
    return SCENARIO_REGISTRY.get("multi_drone_grid").from_config(cfg)


def test_from_config_parses_start_jitter() -> None:
    sc = _build(_cfg(jitter_a=2.0, jitter_b=3.0))
    assert sc.drones[0].start_jitter == 2.0
    assert sc.drones[1].start_jitter == 3.0


def test_zero_jitter_returns_nominal_starts_for_any_seed() -> None:
    sc = _build(_cfg())  # both jitters default 0.0
    for seed in (0, 1, 42, 12345):
        starts = sc.episode_drone_starts(seed)
        assert np.array_equal(starts[0], [4.0, 25.0])
        assert np.array_equal(starts[1], [25.0, 4.0])


def test_jitter_perturbs_and_is_seeded_reproducible() -> None:
    sc = _build(_cfg(jitter_a=2.0, jitter_b=2.0))
    a = sc.episode_drone_starts(42)
    b = sc.episode_drone_starts(42)
    # same seed → identical realization
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    # actually moved off the nominal spawn
    assert not np.array_equal(a[0], [4.0, 25.0])
    assert not np.array_equal(a[1], [25.0, 4.0])
    # different seed → different realization
    c = sc.episode_drone_starts(43)
    assert not np.array_equal(a[0], c[0])


def test_jitter_breaks_mirror_symmetry() -> None:
    # The whole point: the two drones must NOT receive mirror-image offsets,
    # or the crossing stays symmetric. Offsets are drawn independently.
    sc = _build(_cfg(jitter_a=2.0, jitter_b=2.0))
    starts = sc.episode_drone_starts(7)
    off_a = np.asarray(starts[0]) - np.array([4.0, 25.0])
    off_b = np.asarray(starts[1]) - np.array([25.0, 4.0])
    assert not np.allclose(off_a, off_b)


def test_jitter_only_affects_enabled_drones() -> None:
    sc = _build(_cfg(jitter_a=0.0, jitter_b=2.0))
    starts = sc.episode_drone_starts(99)
    assert np.array_equal(starts[0], [4.0, 25.0])     # disabled → untouched
    assert not np.array_equal(starts[1], [25.0, 4.0])  # enabled → moved
