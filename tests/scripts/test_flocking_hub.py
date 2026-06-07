"""Characterization tests for scripts/_flocking_hub.py (K-way flocking hub).

Pins the headline: K flocks converging on a hub jam without a convention (and
stay cohesive while jammed), the right-of-way veer clears the jam at K>2 too,
and the roundabout self-spaces (no inter-flock collision).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "_flocking_hub.py"


def _load():
    spec = importlib.util.spec_from_file_location("_flocking_hub", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_H = _load()


def test_four_way_hub_jams_without_convention():
    r = _H.simulate_hub(K=4, bias=0.0, seed=0)
    assert not r.all_passed


def test_roundabout_clears_a_four_way_hub():
    r = _H.simulate_hub(K=4, bias=1.0, seed=0)
    assert r.all_passed


def test_roundabout_clears_a_six_way_hub():
    # the cure scales past the two-flock crossing.
    r = _H.simulate_hub(K=6, bias=1.0, seed=0)
    assert r.all_passed


def test_hub_jams_but_does_not_collide():
    # the universal alpha-repulsion holds inter-flock spacing even jammed.
    r = _H.simulate_hub(K=6, bias=0.0, seed=0)
    assert r.min_inter > 0.5 * 7.0


def test_jammed_flocks_stay_cohesive():
    # a jammed hub keeps each flock intact (the cost lands only when it moves).
    r = _H.simulate_hub(K=6, bias=0.0, seed=0)
    assert r.cohesion > 0.9
