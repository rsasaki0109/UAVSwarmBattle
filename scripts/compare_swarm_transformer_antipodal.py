#!/usr/bin/env python3
"""Quick antipodal comparison: ORCA vs swarm_transformer (peers / peers+obstacle).

Runs each YAML once (num_episodes from the file) and prints a one-line table.
Requires checkpoints from train_swarm_transformer_checkpoint.py.

  python scripts/compare_swarm_transformer_antipodal.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YAMLS = (
    ("orca/peers", "examples/exp_multi_drone_antipodal_orca.yaml"),
    ("xf/peers", "examples/exp_multi_drone_antipodal_swarm_transformer.yaml"),
    ("xf/obstacle", "examples/exp_multi_drone_antipodal_obstacle_swarm_transformer.yaml"),
)


def _joint_success(result_dir: Path) -> tuple[int, int]:
    ok = total = 0
    for p in sorted(result_dir.glob("episode_*_joint.json")):
        total += 1
        data = json.loads(p.read_text())
        if data.get("outcome") == "success":
            ok += 1
    return ok, total


def _run(yaml_rel: str) -> Path:
    yaml = ROOT / yaml_rel
    cfg = yaml.read_text()
    out_dir = None
    for line in cfg.splitlines():
        if line.strip().startswith("dir:"):
            out_dir = ROOT / line.split(":", 1)[1].strip()
            break
    if out_dir is None:
        raise RuntimeError(f"no output.dir in {yaml}")
    if out_dir.exists():
        for f in out_dir.glob("episode_*"):
            f.unlink()
    subprocess.run(
        ["uav-nav", "run", str(yaml)],
        cwd=ROOT,
        check=True,
    )
    return out_dir


def main() -> int:
    print(f"{'cell':>14} | joint success")
    print("-" * 32)
    for label, yaml_rel in YAMLS:
        out = _run(yaml_rel)
        ok, total = _joint_success(out)
        print(f"{label:>14} | {ok}/{total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
