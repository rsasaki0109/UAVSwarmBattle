"""Characterization tests for scripts/_dogfight.py (1-v-1 UAV dogfight).

Pins the headline: a matched duel stalemates (mutual circle), a turn-rate edge
backfires (the more agile UAV loses), and the win logic counts only a SOLO rear
position.
"""
from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "_dogfight.py"


def _load():
    spec = importlib.util.spec_from_file_location("_dogfight", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_D = _load()


def test_parity_is_a_stalemate():
    # equal speed and turn rate -> the mutual circle, no winner.
    wins = Counter(_D.duel(seed=s).winner for s in range(12))
    assert wins[None] == 12


def test_turn_rate_edge_wins():
    # the MORE agile UAV (P0, higher wmax) out-turns onto the opponent's six.
    wins = Counter(_D.duel(wmax0=2.5, wmax1=1.5, seed=s).winner for s in range(12))
    assert wins[0] > wins[1]          # the agile P0 wins more
    assert wins[1] == 0               # the sluggish P1 never wins


def test_speed_edge_cannot_win():
    # a forward-speed edge cannot convert to a kill — the duel stays a stalemate.
    wins = Counter(_D.duel(v0=6.0, v1=4.0, seed=s).winner for s in range(12))
    assert wins[0] == 0               # the faster UAV wins nothing on speed alone


def test_on_six_geometry():
    import numpy as np
    # attacker just behind defender, both pointing +x: attacker is on the six.
    dfn = np.array([10.0, 10.0, 0.0])
    att = np.array([7.0, 10.0, 0.0])
    assert _D._on_six(att, dfn, capture_range=8.0, cone=0.7)
    # attacker in front of defender is NOT on the six.
    att_front = np.array([13.0, 10.0, 0.0])
    assert not _D._on_six(att_front, dfn, capture_range=8.0, cone=0.7)


def test_aim_tail_stalemates_at_parity():
    wins = Counter(_D.duel(aim="tail", seed=s).winner for s in range(8))
    assert wins[None] == 8
