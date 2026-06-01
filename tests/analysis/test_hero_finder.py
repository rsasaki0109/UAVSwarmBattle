"""Auto hero-finder tests.

Layers:
  1. score_cell()  — the pure drama/significance math on outcome pairs.
  2. pair_by_seed / find_heroes — I/O wrappers over run dirs, exercised with
     tiny synthetic episode_*.json files written to tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

from uav_nav_lab.analysis.hero_finder import (
    find_heroes,
    pair_by_seed,
    score_cell,
)


# --- score_cell ----------------------------------------------------------

def test_dramatic_flip_is_significant_and_high_drama():
    # 10 seeds: baseline always fails, proposed always succeeds → 0% → 100%
    pairs = [(False, True)] * 10
    s = score_cell("flip", pairs)
    assert s.baseline_rate == 0.0
    assert s.proposed_rate == 1.0
    assert s.margin == 1.0
    assert s.c == 10 and s.b == 0
    assert s.significant
    # headroom bonus: margin 1.0 * (1 + (1-0)) = 2.0
    assert s.drama == 2.0


def test_no_difference_scores_zero_drama():
    pairs = [(True, True)] * 6 + [(False, False)] * 4
    s = score_cell("nodiff", pairs)
    assert s.margin == 0.0
    assert not s.significant
    assert s.drama == 0.0


def test_wrong_direction_not_significant():
    # proposed is WORSE than baseline → must not be flagged as a hero
    pairs = [(True, False)] * 8 + [(True, True)] * 2
    s = score_cell("regression", pairs)
    assert s.margin < 0
    assert s.b == 8 and s.c == 0
    assert not s.significant
    assert s.drama == 0.0


def test_small_sample_not_significant():
    # 2 discordant pairs both favouring proposed: McNemar p = 2*0.25 = 0.5
    pairs = [(False, True), (False, True)]
    s = score_cell("tiny", pairs)
    assert s.c == 2 and s.b == 0
    assert s.mcnemar_p > 0.05
    assert not s.significant
    assert s.drama == 0.0


def test_low_baseline_outranks_high_baseline_at_equal_margin():
    # both +50 pp and significant (8 discordant pairs → McNemar p≈0.008),
    # but the lower baseline should score higher via the headroom bonus.
    low = [(False, True)] * 8 + [(False, False)] * 8   # 0% → 50%
    high = [(True, True)] * 8 + [(False, True)] * 8     # 50% → 100%
    s_low = score_cell("low", low)
    s_high = score_cell("high", high)
    assert abs(s_low.margin - 0.5) < 1e-9
    assert abs(s_high.margin - 0.5) < 1e-9
    assert s_low.significant and s_high.significant
    assert s_low.drama > s_high.drama


# --- pair_by_seed / find_heroes ------------------------------------------

def _write_run(dir_: Path, seed_to_success: dict[int, bool]) -> None:
    """Write minimal single-drone episode_*.json files for the given seeds.

    Mirrors the real recorder layout: the seed lives under `meta.seed`.
    """
    dir_.mkdir(parents=True, exist_ok=True)
    for i, (seed, ok) in enumerate(sorted(seed_to_success.items())):
        ep = {
            "meta": {"episode": i, "seed": seed, "drone_id": 0},
            "outcome": "success" if ok else "collision",
            "steps": [],
            "replans": [],
        }
        (dir_ / f"episode_{i:03d}.json").write_text(json.dumps(ep))


def test_pair_by_seed_only_pairs_common_seeds(tmp_path: Path):
    base = tmp_path / "base"
    prop = tmp_path / "prop"
    _write_run(base, {1: False, 2: False, 3: True})
    _write_run(prop, {2: True, 3: True, 4: True})  # seed 4 not in base; 1 not in prop
    pairs = pair_by_seed(base, prop)
    # common seeds = {2, 3}
    assert len(pairs) == 2
    # seed 2: base fail, prop success ; seed 3: base success, prop success
    assert pairs == [(False, True), (True, True)]


def test_find_heroes_ranks_most_dramatic_first(tmp_path: Path):
    # cell A: 0% → 100% (dramatic) ; cell B: 80% → 100% (mild)
    a_base = tmp_path / "a_base"
    a_prop = tmp_path / "a_prop"
    _write_run(a_base, {s: False for s in range(10)})
    _write_run(a_prop, {s: True for s in range(10)})

    b_base = tmp_path / "b_base"
    b_prop = tmp_path / "b_prop"
    _write_run(b_base, {s: (s < 8) for s in range(10)})   # 80%
    _write_run(b_prop, {s: True for s in range(10)})        # 100%

    heroes = find_heroes(
        [("cellA", a_base, a_prop), ("cellB", b_base, b_prop)]
    )
    assert heroes[0].label == "cellA"
    assert heroes[0].drama > heroes[1].drama


def test_find_heroes_skips_cells_with_no_common_seeds(tmp_path: Path):
    base = tmp_path / "x_base"
    prop = tmp_path / "x_prop"
    _write_run(base, {1: False})
    _write_run(prop, {99: True})  # disjoint seeds
    heroes = find_heroes([("x", base, prop)])
    assert heroes == []


# --- _outcomes_by_seed layout handling (review follow-up) ---------------

def _write_joint_per_drone(dir_, seed, drone_outcomes):
    """Write per-drone episode_NNN_drone_MM.json files that SHARE a seed,
    mirroring the real multi-drone runner output (no joint file)."""
    import json as _json
    dir_.mkdir(parents=True, exist_ok=True)
    for m, ok in enumerate(drone_outcomes):
        ep = {
            "meta": {"episode": 0, "seed": seed, "drone_id": m},
            "outcome": "success" if ok else "collision",
            "steps": [], "replans": [],
        }
        (dir_ / f"episode_000_drone_{m:02d}.json").write_text(_json.dumps(ep))


def test_per_drone_shared_seed_and_merges_without_warning(tmp_path, recwarn):
    """Per-drone files (drone_id present) legitimately share a seed; they must
    AND-merge to a joint outcome with NO warning."""
    from uav_nav_lab.analysis.hero_finder import _outcomes_by_seed
    d = tmp_path / "md"
    _write_joint_per_drone(d, seed=42, drone_outcomes=[True, False])  # one fails
    out = _outcomes_by_seed(d, "joint")
    assert out == {42: False}  # joint fails because a drone failed
    assert len(recwarn) == 0   # no spurious warning for legitimate siblings


def test_non_drone_seed_collision_warns(tmp_path):
    """Two non-per-drone files sharing a seed means unrelated episodes were
    mixed in; that must warn (not silently AND-merge)."""
    import json as _json
    import pytest as _pytest
    from uav_nav_lab.analysis.hero_finder import _outcomes_by_seed
    d = tmp_path / "mixed"
    d.mkdir()
    for i, ok in enumerate([True, False]):
        ep = {"meta": {"episode": i, "seed": 5}, "outcome": "success" if ok else "collision",
              "steps": [], "replans": []}
        (d / f"episode_{i:03d}.json").write_text(_json.dumps(ep))
    with _pytest.warns(UserWarning, match="multiple non-"):
        out = _outcomes_by_seed(d, "joint")
    assert out == {5: False}
