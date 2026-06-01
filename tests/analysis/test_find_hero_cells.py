"""Regression tests for scripts/find_hero_cells.py sweep pairing.

Focus: the baseline/proposed group split must stay disjoint and pair by
scenario-cell key even when one match token is a substring of the other
(the real case in this repo: 'mppi' ⊂ 'cvar_mppi').
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "find_hero_cells.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("find_hero_cells", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mk_run(sweep: Path, dirname: str, name: str) -> None:
    d = sweep / dirname
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(f"name: {name}\n")


def test_overlapping_match_tokens_split_disjoint(tmp_path):
    """baseline_match='mppi' is a substring of proposed 'cvar_mppi'; a proposed
    dir must NOT also land in the baseline group, and cells must pair by key."""
    fhc = _load_module()
    sweep = tmp_path / "sweep"
    _mk_run(sweep, "run_000", "cell_d2_mppi")
    _mk_run(sweep, "run_001", "cell_d2_cvar_mppi")
    _mk_run(sweep, "run_002", "cell_d4_mppi")
    _mk_run(sweep, "run_003", "cell_d4_cvar_mppi")

    cells = fhc._pairs_from_sweep(sweep, "mppi", "cvar_mppi")
    # exactly two cells (d2, d4), each pairing the plain mppi vs cvar_mppi twin
    assert len(cells) == 2
    labels = {label for label, _, _ in cells}
    assert "cell_d2_mppi vs cell_d2_cvar_mppi" in labels
    assert "cell_d4_mppi vs cell_d4_cvar_mppi" in labels
    # the proposed dirs must never appear as a baseline of any pair
    for _label, base_dir, prop_dir in cells:
        assert "cvar_mppi" not in fhc._config_name(base_dir)
        assert "cvar_mppi" in fhc._config_name(prop_dir)


def test_keyed_pairing_survives_missing_cell(tmp_path):
    """A missing proposed cell must skip (warn), not misalign the rest."""
    fhc = _load_module()
    sweep = tmp_path / "sweep"
    _mk_run(sweep, "run_000", "cell_d2_mppi")
    _mk_run(sweep, "run_001", "cell_d2_cvar_mppi")
    _mk_run(sweep, "run_002", "cell_d4_mppi")  # no cvar twin for d4

    cells = fhc._pairs_from_sweep(sweep, "mppi", "cvar_mppi")
    assert len(cells) == 1
    assert cells[0][0] == "cell_d2_mppi vs cell_d2_cvar_mppi"
