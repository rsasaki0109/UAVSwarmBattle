"""Characterization tests for uav_nav_lab.viz.episode_gif.

The renderers extract drones' (true, ref) trajectories with subtle
mode/pad semantics that used to drift between the aerobatic and race
scripts. These tests pin both modes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uav_nav_lab.viz.episode_gif import (
    DRONE_COLORS,
    load_drones,
    trajectory_arrays,
)


def _drone(steps: list[dict], outcome: str = "success") -> dict:
    return {"steps": steps, "outcome": outcome}


def _step(p, r=None, *, collision: bool = False) -> dict:
    s: dict = {"true_pos": list(p)}
    if r is not None:
        s["reference_pos"] = list(r)
    if collision:
        s["collision"] = True
    return s


def test_drone_colors_has_four_distinct_hex_strings():
    assert len(DRONE_COLORS) == 4
    assert len(set(DRONE_COLORS)) == 4
    for c in DRONE_COLORS:
        assert c.startswith("#") and len(c) == 7


def test_load_drones_reads_per_drone_json(tmp_path: Path):
    for i in range(3):
        payload = {"drone_index": i, "outcome": "success"}
        (tmp_path / f"episode_007_drone_{i:02d}.json").write_text(json.dumps(payload))
    out = load_drones(tmp_path, ep=7, n_drones=3)
    assert [d["drone_index"] for d in out] == [0, 1, 2]


def test_trajectory_arrays_fit_min_truncates_to_shortest_drone():
    drones = [
        _drone([_step([0.0, 0.0, 0.0]), _step([1.0, 0.0, 0.0]), _step([2.0, 0.0, 0.0])]),
        _drone([_step([0.0, 1.0, 0.0]), _step([0.0, 2.0, 0.0])]),  # shorter
    ]
    true_p, ref_p, coll = trajectory_arrays(drones, fit="min")
    assert true_p.shape == (2, 2, 3)
    assert ref_p.shape == (2, 2, 3)
    np.testing.assert_array_equal(true_p[0, 1], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(true_p[1, 1], [0.0, 2.0, 0.0])
    # No collision flags → all entries should equal T
    assert (coll == 2).all()


def test_trajectory_arrays_fit_max_pads_shorter_drone_with_last_position():
    drones = [
        _drone([_step([0.0, 0.0, 0.0]), _step([1.0, 0.0, 0.0]), _step([2.0, 0.0, 0.0])]),
        _drone([_step([5.0, 5.0, 5.0])]),  # length 1 → padded to 3
    ]
    true_p, _, _ = trajectory_arrays(drones)  # default fit="max"
    assert true_p.shape == (2, 3, 3)
    np.testing.assert_array_equal(true_p[1, 0], [5.0, 5.0, 5.0])
    np.testing.assert_array_equal(true_p[1, 1], [5.0, 5.0, 5.0])  # held
    np.testing.assert_array_equal(true_p[1, 2], [5.0, 5.0, 5.0])  # held


def test_trajectory_arrays_T_pad_overrides_when_longer_than_longest_drone():
    drones = [_drone([_step([0.0, 0.0, 0.0]), _step([1.0, 0.0, 0.0])])]
    true_p, _, coll = trajectory_arrays(drones, T_pad=5)
    assert true_p.shape == (1, 5, 3)
    # The last 3 steps are padded with the final position
    np.testing.assert_array_equal(true_p[0, 1], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(true_p[0, 4], [1.0, 0.0, 0.0])
    # No collision → collision_step == T_pad
    assert coll[0] == 5


def test_trajectory_arrays_falls_back_to_true_pos_when_reference_missing():
    drones = [_drone([_step([1.0, 2.0, 3.0])])]  # no reference_pos
    true_p, ref_p, _ = trajectory_arrays(drones)
    np.testing.assert_array_equal(ref_p[0, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(true_p[0, 0], ref_p[0, 0])


def test_trajectory_arrays_records_first_collision_step():
    drones = [
        _drone([
            _step([0.0, 0.0, 0.0]),
            _step([1.0, 0.0, 0.0], collision=True),  # first collision at k=1
            _step([2.0, 0.0, 0.0], collision=True),
        ])
    ]
    _, _, coll = trajectory_arrays(drones)
    assert coll[0] == 1


def test_trajectory_arrays_falls_back_to_outcome_when_no_step_collision_flag():
    # Steps don't carry collision flags, but the drone's overall outcome
    # says "collision" — fall back to the final step's index.
    drones = [_drone([_step([0.0, 0.0, 0.0]), _step([1.0, 0.0, 0.0])],
                     outcome="collision")]
    _, _, coll = trajectory_arrays(drones)
    assert coll[0] == 1  # last step index
