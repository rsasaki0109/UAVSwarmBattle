"""Characterization tests for scripts/_flocking.py (Olfati-Saber flocking sim).

Pins the headline invariant of the fragmentation finding: free flocking
(Algorithm 1) fragments while the navigational structure (Algorithm 2) keeps a
single, spaced flock — and the σ-norm helpers stay well-formed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "_flocking.py"


def _load():
    spec = importlib.util.spec_from_file_location("_flocking", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass introspection needs the module registered
    spec.loader.exec_module(mod)
    return mod


_F = _load()


def test_bump_endpoints():
    z = np.array([0.0, 0.1, 0.5, 1.0, 1.5])
    out = _F._bump(z)
    assert out[0] == 1.0          # below h -> 1
    assert out[-1] == 0.0         # past 1 -> 0 (finite support)
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_sigma_norm_zero_and_monotone():
    assert _F._sigma_norm_scalar(0.0) == 0.0
    vals = _F._sigma_norm(np.array([0.0, 1.0, 4.0, 9.0]))
    assert np.all(np.diff(vals) > 0)


def test_algorithm2_makes_one_spaced_flock():
    r = _F.simulate(algorithm=2, n=20, seed=0, spread=12.0, c2a=14.0,
                    steps=800, record=True)
    assert r.connected and r.n_components == 1
    q = r.traj[-1]
    d = np.sqrt(((q[:, None] - q[None]) ** 2).sum(-1))
    np.fill_diagonal(d, 1e9)
    # spaced lattice near the desired spacing, not a collapsed pile
    assert d.min(1).mean() > 3.0


def test_algorithm1_fragments_at_matched_gain():
    r = _F.simulate(algorithm=1, n=20, seed=0, spread=12.0, c2a=14.0, steps=800)
    assert r.n_components > 1 and not r.connected


def test_more_cohesion_gain_does_not_reconnect():
    # the counterintuitive core: cranking the gain does not reduce fragmentation
    low = _F.simulate(algorithm=1, n=20, seed=1, spread=12.0, c2a=14.0,
                      grad_gain=1.0, steps=800).n_components
    high = _F.simulate(algorithm=1, n=20, seed=1, spread=12.0, c2a=14.0,
                       grad_gain=8.0, steps=800).n_components
    assert high >= low


_MIGRATE = dict(algorithm=2, n=24, spread=14.0, c2a=8.0, grad_gain=1.0,
                c1g=1.0, c2g=0.6, goal=(0.0, 0.0), goal_vel=(5.0, 0.0),
                goal_moves=True, obs_infl=4.0, c_obs=20.0, steps=1200)


def test_migrating_flock_threads_small_obstacle():
    r = _F.simulate(seed=4, obstacles=((40.0, 0.0, 2.0),), **_MIGRATE)
    assert r.connected and r.n_components == 1


def test_large_obstacle_splits_the_flock():
    r = _F.simulate(seed=4, obstacles=((40.0, 0.0, 12.0),), **_MIGRATE)
    assert r.n_components > 1 and not r.connected


def test_no_obstacle_control_stays_one_flock():
    r = _F.simulate(seed=4, obstacles=(), **_MIGRATE)
    assert r.connected


def test_gated_rendezvous_heals_a_cut_that_baseline_cannot():
    cut = dict(_MIGRATE, steps=1500, obstacles=((40.0, 0.0, 9.0),))
    base = _F.simulate(seed=7, c_rdv=0.0, **cut)
    gated = _F.simulate(seed=7, c_rdv=3.0, rdv_gate_x=40.0 + 9.0 + 5.0, **cut)
    assert base.n_components > 1            # local flock stays cut
    assert gated.n_components < base.n_components   # rendezvous re-merges


def test_rendezvous_gate_only_acts_past_the_gate():
    # with the gate far downstream of the whole run, the term never fires:
    # the result must match the no-rendezvous baseline exactly.
    cut = dict(_MIGRATE, steps=1500, obstacles=((40.0, 0.0, 9.0),))
    base = _F.simulate(seed=3, c_rdv=0.0, **cut)
    gated_off = _F.simulate(seed=3, c_rdv=3.0, rdv_gate_x=1e9, **cut)
    assert gated_off.n_components == base.n_components


def test_adaptive_local_reach_heals_a_cut():
    # a comms-free local reach boost (low-degree agents enlarge their range)
    # re-merges a cut the base-range flock cannot.
    cut = dict(_MIGRATE, steps=1500, obstacles=((40.0, 0.0, 13.0),))
    base = _F.simulate(seed=2, reach_boost=1.0, **cut)
    adapt = _F.simulate(seed=2, reach_boost=3.0, reach_kmin=5, **cut)
    assert base.n_components > 1
    assert adapt.connected and adapt.n_components == 1


def test_reach_boost_one_is_a_noop():
    # reach_boost == 1.0 must leave the base dynamics untouched.
    kw = dict(_MIGRATE, steps=600, obstacles=((40.0, 0.0, 9.0),))
    a = _F.simulate(seed=1, **kw)
    b = _F.simulate(seed=1, reach_boost=1.0, reach_kmin=5, **kw)
    assert a.n_components == b.n_components


def _min_surface_gap(res, cx, cy, R):
    """Worst (smallest) centre-to-surface gap to a disk over a recorded run."""
    import numpy as np
    g = np.inf
    for q in res.traj:
        v = q - np.array([cx, cy])
        g = min(g, float((np.sqrt((v * v).sum(-1)) - R).min()))
    return g


def test_adaptive_reach_heals_but_hugs_the_obstacle():
    # the clearance cost: the boosted reach heals the cut but pulls the
    # re-cohering agents markedly closer to the disk than the base range does.
    cut = dict(_MIGRATE, steps=1500, obstacles=((40.0, 0.0, 13.0),), record=True)
    base = _F.simulate(seed=0, reach_boost=1.0, **cut)
    adapt = _F.simulate(seed=0, reach_boost=3.0, reach_kmin=5, **cut)
    assert base.n_components > 1 and adapt.connected          # adaptive heals, base does not
    assert _min_surface_gap(adapt, 40.0, 0.0, 13.0) < _min_surface_gap(base, 40.0, 0.0, 13.0)


def test_stronger_repulsion_restores_clearance_and_keeps_heal():
    # the cost is a force-balance equilibrium: raising the repulsion gain (a
    # magnitude lever) widens the worst-case gap while the flock still heals.
    cut = dict(_MIGRATE, steps=1500, obstacles=((40.0, 0.0, 13.0),), record=True,
               reach_boost=3.0, reach_kmin=5)
    weak = _F.simulate(seed=0, **dict(cut, c_obs=20.0))
    strong = _F.simulate(seed=0, **dict(cut, c_obs=160.0))
    assert weak.connected and strong.connected               # heal preserved either way
    assert _min_surface_gap(strong, 40.0, 0.0, 13.0) > _min_surface_gap(weak, 40.0, 0.0, 13.0)
