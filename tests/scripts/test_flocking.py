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
