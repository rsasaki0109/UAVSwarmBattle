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


def _within_mask(K, per_flock, frac):
    import numpy as np
    m = np.zeros(K * per_flock, bool)
    k = int(round(frac * per_flock))
    for i in range(K):
        m[i * per_flock: i * per_flock + k] = True
    return m


def test_adopt_mask_none_equals_full_bias():
    # an all-applied bias is the same as no mask at all (default path).
    import numpy as np
    full = _within_mask(6, 10, 1.0)
    r_mask = _H.simulate_hub(K=6, bias=1.0, adopt=full, seed=0)
    r_none = _H.simulate_hub(K=6, bias=1.0, seed=0)
    assert r_mask.all_passed == r_none.all_passed
    assert np.isclose(r_mask.cohesion, r_none.cohesion)


def test_partial_within_flock_adoption_still_clears():
    # cohesion drag: a minority per flock pulls the rest through the roundabout.
    r = _H.simulate_hub(K=6, bias=1.0, adopt=_within_mask(6, 10, 0.5), seed=0)
    assert r.all_passed


def test_clustered_free_rider_flocks_can_rejam():
    # the SAME budget clumped into whole flocks leaves coherent walls that jam.
    import numpy as np
    budget = 12
    clustered = np.zeros(60, bool); clustered[:budget] = True
    r = _H.simulate_hub(K=6, bias=1.0, adopt=clustered, seed=0)
    assert not r.all_passed
