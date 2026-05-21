"""Characterization tests for uav_nav_lab.anim._common.

The four animators used to each carry their own frame-down-sampling,
dynamic-obstacle replay, and replan-lookup logic. These tests pin the
shared helpers so a regression in any of them surfaces immediately,
without needing to actually rasterise a GIF.
"""

from __future__ import annotations

import pytest

from uav_nav_lab.anim._common import (
    PALETTE,
    dynamic_obstacle_positions_at,
    frame_indices_for_episode,
    replan_at_or_before,
)


def test_palette_has_eight_distinct_tab_colors():
    # Multi-drone animators rely on PALETTE having at least 8 entries —
    # any fewer and 8-drone scenarios silently alias colours.
    assert len(PALETTE) == 8
    assert len(set(PALETTE)) == 8
    for c in PALETTE:
        assert c.startswith("tab:")


def test_frame_indices_empty_episode_returns_empty_list():
    assert frame_indices_for_episode(0, 0.05, 20) == []


def test_frame_indices_round_sim_fps_to_render_fps():
    # dt=0.05 → sim_fps=20, render fps=20 → stride=1, every step kept.
    out = frame_indices_for_episode(5, 0.05, 20)
    assert out == [0, 1, 2, 3, 4]


def test_frame_indices_downsamples_with_correct_stride():
    # dt=0.05 → sim_fps=20, render fps=10 → stride=2
    out = frame_indices_for_episode(10, 0.05, 10)
    assert out == [0, 2, 4, 6, 8, 9]  # 9 appended so animation ends on final step


def test_frame_indices_always_includes_final_step():
    # n_steps=11, stride=2 → range gives 0,2,4,6,8,10 (10 IS final → no append)
    assert frame_indices_for_episode(11, 0.05, 10) == [0, 2, 4, 6, 8, 10]
    # n_steps=12, stride=2 → range gives 0,2,4,6,8,10 → final=11 missing → append
    assert frame_indices_for_episode(12, 0.05, 10) == [0, 2, 4, 6, 8, 10, 11]


def test_frame_indices_clamps_stride_to_at_least_one():
    # render fps higher than sim fps → stride would round to 0 — clamped to 1
    out = frame_indices_for_episode(3, 0.5, 100)  # sim_fps=2, render=100
    assert out == [0, 1, 2]


def test_dynamic_obstacle_positions_empty_specs_returns_empty_tuples():
    xs, ys, zs = dynamic_obstacle_positions_at(10, [], 0.05, (10.0, 10.0, 10.0))
    assert xs == [] and ys == [] and zs == []


def test_dynamic_obstacle_positions_no_motion_when_j_is_zero():
    spec = {"start": [1.0, 2.0, 3.0], "velocity": [5.0, 5.0, 5.0]}
    xs, ys, zs = dynamic_obstacle_positions_at(0, [spec], 0.05, (10.0, 10.0, 10.0))
    assert xs == [1.0]
    assert ys == [2.0]
    assert zs == [3.0]


def test_dynamic_obstacle_positions_advances_with_constant_velocity():
    # vel=[1,0,0], dt=0.5, j=4 → pos.x grows by 1*0.5*4 = 2 → 5.0 + 2.0 = 7.0
    spec = {"start": [5.0, 5.0, 5.0], "velocity": [1.0, 0.0, 0.0], "reflect": False}
    xs, _, _ = dynamic_obstacle_positions_at(4, [spec], 0.5, (100.0, 100.0, 100.0))
    assert xs[0] == pytest.approx(7.0)


def test_dynamic_obstacle_positions_reflects_off_upper_bound():
    # vel=[1,0,0], bound=10. After step1: x=10.5 → reflects to 9.5 with vel=-1.
    spec = {"start": [9.5, 5.0, 5.0], "velocity": [1.0, 0.0, 0.0], "reflect": True}
    xs, _, _ = dynamic_obstacle_positions_at(2, [spec], 1.0, (10.0, 100.0, 100.0))
    # step 1: 9.5 + 1 = 10.5 → > 10 → reflect to 2*10 - 10.5 = 9.5, vel = -1
    # step 2: 9.5 + (-1) = 8.5 (no further reflection)
    assert xs[0] == pytest.approx(8.5)


def test_dynamic_obstacle_positions_reflects_off_lower_bound():
    spec = {"start": [0.5, 5.0, 5.0], "velocity": [-1.0, 0.0, 0.0], "reflect": True}
    xs, _, _ = dynamic_obstacle_positions_at(2, [spec], 1.0, (10.0, 100.0, 100.0))
    # step 1: 0.5 - 1 = -0.5 → < 0 → reflect to 0.5, vel = 1
    # step 2: 0.5 + 1 = 1.5
    assert xs[0] == pytest.approx(1.5)


def test_dynamic_obstacle_positions_handles_multiple_specs_independently():
    s1 = {"start": [0.0, 0.0, 0.0], "velocity": [1.0, 0.0, 0.0], "reflect": False}
    s2 = {"start": [5.0, 5.0, 5.0], "velocity": [0.0, 2.0, 0.0], "reflect": False}
    xs, ys, _ = dynamic_obstacle_positions_at(3, [s1, s2], 1.0, (100.0, 100.0, 100.0))
    assert xs == pytest.approx([3.0, 5.0])
    assert ys == pytest.approx([0.0, 11.0])


def test_replan_at_or_before_returns_none_when_no_replan_qualifies():
    replans = [{"t": 1.0, "rollouts": [[[0, 0, 0]]]}, {"t": 2.0, "rollouts": [[[1, 1, 1]]]}]
    assert replan_at_or_before(replans, 0.5) is None


def test_replan_at_or_before_returns_most_recent_replan_within_epsilon():
    r1 = {"t": 1.0, "rollouts": [[[0, 0, 0]]]}
    r2 = {"t": 2.0, "rollouts": [[[1, 1, 1]]]}
    # At t = 2.0 + 1e-10 → still inside epsilon → r2 is chosen
    assert replan_at_or_before([r1, r2], 2.0 + 1e-10) is r2
    # At t = 2.0 + 1e-3 → past epsilon but past r2 → still r2 (the loop break
    # only fires when r.t > cur_t + eps, so r2 is processed normally first)
    assert replan_at_or_before([r1, r2], 2.0 + 1e-3) is r2


def test_replan_at_or_before_skips_replans_without_rollouts():
    # An MPC replan logs no rollouts → should be skipped even though it is
    # the most recent before cur_t.
    r1 = {"t": 1.0, "rollouts": [[[0, 0, 0]]]}
    r2 = {"t": 2.0, "rollouts": None}
    assert replan_at_or_before([r1, r2], 3.0) is r1


def test_replan_at_or_before_breaks_when_replan_strictly_in_future():
    # Verify the early-break by passing an unsorted-on-purpose tail with t > cur_t;
    # the function expects replans to be time-ordered, so the loop must break.
    r1 = {"t": 1.0, "rollouts": [[[0, 0, 0]]]}
    r2 = {"t": 5.0, "rollouts": [[[1, 1, 1]]]}
    # cur_t = 2.0 → r2 should not be reached as chosen
    assert replan_at_or_before([r1, r2], 2.0) is r1
