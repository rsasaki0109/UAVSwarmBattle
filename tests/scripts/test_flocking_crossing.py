"""Characterization tests for scripts/_flocking_crossing.py.

Pins the headline of the crossing-flocks finding: two cohesive flocks driven
head-on JAM without a convention (and never collide), the right-of-way bias
clears the jam, and too strong a bias throws them off their lane (the band).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "_flocking_crossing.py"


def _load():
    spec = importlib.util.spec_from_file_location("_flocking_crossing", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_C = _load()


def test_no_convention_jams():
    # without a convention the two flocks do not clear the crossing in time.
    r = _C.simulate_crossing(seed=0, bias=0.0)
    assert not r.passed


def test_right_of_way_clears_the_jam():
    # the right-of-way veer lets both flocks pass within the budget, on their lane.
    r = _C.simulate_crossing(seed=0, bias=1.0)
    assert r.on_time            # passed AND stayed on lane


def test_flocks_jam_but_do_not_collide():
    # the universal alpha-repulsion holds inter-flock spacing: a jam, not a crash.
    r = _C.simulate_crossing(seed=0, bias=0.0)
    assert r.min_pair > 0.5 * 7.0     # closest inter-flock approach stays above contact


def test_too_strong_a_bias_leaves_the_lane():
    # past the band a strong veer slips them past but flings them off their lane.
    r = _C.simulate_crossing(seed=0, bias=3.0)
    assert r.passed and not r.on_lane


def test_bias_zero_is_symmetric_baseline():
    # bias=0 must leave the dynamics free of any lateral push: centroids stay near
    # the axis (the jam is on-axis, not a drift).
    r = _C.simulate_crossing(seed=0, bias=0.0)
    assert r.max_lateral < 15.0
