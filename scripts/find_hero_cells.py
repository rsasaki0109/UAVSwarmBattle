#!/usr/bin/env python3
"""Auto hero-cell finder — rank paired baseline-vs-proposed runs by drama.

A "hero cell" is the scenario where a proposed method most decisively beats a
baseline on identical (seed-aligned) episodes — the result worth turning into a
GIF. This wraps :func:`uav_nav_lab.analysis.find_heroes` with two input modes:

1. Explicit pairs (any number):

     python scripts/find_hero_cells.py \
       --pair crossing results/mdx_const_vel results/mdx_game_theoretic \
       --pair gates results/gates_mppi results/gates_cvar_mppi

2. A sweep whose run dirs split into a baseline group and a proposed group by a
   substring in their config name (read from each run's config.yaml):

     python scripts/find_hero_cells.py --sweep results/my_sweep \
       --baseline-match mppi --proposed-match cvar_mppi

The top cell is printed with its full McNemar/Wilson stats and a ready-to-run
`uav-nav anim` hint. Use --level per_drone to rank on per-drone success instead
of joint mission success.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running from a checkout without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uav_nav_lab.analysis import find_heroes  # noqa: E402


def _config_name(run_dir: Path) -> str:
    """Best-effort read of the experiment name from a run's config.yaml."""
    cfg = run_dir / "config.yaml"
    if not cfg.exists():
        return run_dir.name
    for line in cfg.read_text(encoding="utf-8").splitlines():
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip()
    return run_dir.name


def _cell_key(name: str, match: str) -> str:
    """The scenario-cell identity of a config name, with the group's match
    token removed so a baseline and its proposed twin map to the same key.

    e.g. name='cell_d4_cvar_mppi', match='cvar_mppi' → 'cell_d4_'. Pairing on
    this key (instead of sorted position) is robust to missing cells and to
    naming/sort-order differences between the two groups.
    """
    return name.replace(match, "")


def _pairs_from_sweep(
    sweep_dir: Path, baseline_match: str, proposed_match: str
) -> list[tuple[str, Path, Path]]:
    """Split a sweep's run dirs into baseline/proposed by config-name substring,
    then pair them by SCENARIO-CELL KEY (the config name with the group token
    stripped), not by sorted position — so a missing cell or divergent naming
    can't silently misalign baseline cell i with an unrelated proposed cell i."""
    run_dirs = sorted(d for d in sweep_dir.iterdir() if d.is_dir())
    # Classify proposed first, then baseline = matches baseline_match but is NOT
    # already proposed. This keeps the two groups disjoint even when one match
    # token is a substring of the other (the real case here: 'mppi' ⊂
    # 'cvar_mppi'), where a naive `baseline_match in name` would also sweep the
    # proposed dirs into the baseline group.
    prop = [d for d in run_dirs if proposed_match in _config_name(d)]
    prop_set = set(prop)
    base = [
        d for d in run_dirs
        if baseline_match in _config_name(d) and d not in prop_set
    ]
    if not base or not prop:
        raise SystemExit(
            f"sweep split empty: {len(base)} baseline (match '{baseline_match}'), "
            f"{len(prop)} proposed (match '{proposed_match}')"
        )
    prop_by_key: dict[str, Path] = {}
    for d in prop:
        prop_by_key.setdefault(_cell_key(_config_name(d), proposed_match), d)
    cells = []
    unmatched = []
    for b in base:
        key = _cell_key(_config_name(b), baseline_match)
        p = prop_by_key.get(key)
        if p is None:
            unmatched.append(_config_name(b))
            continue
        cells.append((f"{_config_name(b)} vs {_config_name(p)}", b, p))
    if unmatched:
        print(
            f"[warn] {len(unmatched)} baseline cell(s) had no proposed twin by "
            f"key and were skipped: {', '.join(unmatched)}",
            file=sys.stderr,
        )
    if not cells:
        raise SystemExit(
            "no baseline/proposed cells paired by key — check that the two "
            "groups share scenario-cell names apart from the match tokens"
        )
    return cells


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--pair", nargs=3, action="append", metavar=("LABEL", "BASELINE", "PROPOSED"),
        default=[], help="a labelled baseline/proposed run-dir pair (repeatable)",
    )
    ap.add_argument("--sweep", type=Path, help="a sweep output root to auto-split")
    ap.add_argument("--baseline-match", default="baseline",
                    help="config-name substring identifying baseline runs")
    ap.add_argument("--proposed-match", default="proposed",
                    help="config-name substring identifying proposed runs")
    ap.add_argument("--level", choices=["joint", "per_drone"], default="joint")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--top", type=int, default=10, help="how many cells to print")
    args = ap.parse_args(argv)

    cells: list[tuple[str, Path, Path]] = [
        (label, Path(b), Path(p)) for label, b, p in args.pair
    ]
    if args.sweep:
        cells += _pairs_from_sweep(
            args.sweep, args.baseline_match, args.proposed_match
        )
    if not cells:
        ap.error("provide at least one --pair or a --sweep")

    heroes = find_heroes(cells, level=args.level, alpha=args.alpha)
    if not heroes:
        print("no comparable cells (no common seeds in any pair)")
        return 1

    print(f"\n=== hero ranking ({args.level} success, {len(heroes)} cells) ===")
    for h in heroes[: args.top]:
        print(h.headline())

    top = heroes[0]
    print("\n--- top hero ---")
    print(top.headline())
    if top.significant:
        print(
            "  → render it:  uav-nav anim <proposed run_dir>   "
            "# baseline loses, proposed wins on the same seeds"
        )
    else:
        print("  (top cell is not statistically significant; gather more seeds "
              "or pick a harder scenario)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
