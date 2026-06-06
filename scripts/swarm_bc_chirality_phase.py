"""A learned convention needs a representation that can REPRESENT handedness
(neta A, follow-up to swarm_bc_symmetry_phase.py).

swarm_bc_symmetry_phase.py showed a teammate-token policy is only as
symmetry-breaking as its TEACHER: distilling a right-of-way convention clears the
antipodal hub, distilling a symmetric avoider reimports the deadlock. But that
used the ego-goal frame, a pure rotation that PRESERVES chirality — "right" is a
globally-consistent direction every agent shares. Is that shared handedness
reference NECESSARY, or could the convention be carried by a chirality-free
representation?

Isolate it by changing ONLY the representation, not the teacher. Distill the SAME
convention teacher into the SAME deep set under two featurizations:

  conv_std    ego-goal frame (rotation only; chirality preserved)   [the #131 winner]
  conv_refl   ego-goal frame + reflection canonicalization (the y-axis is flipped
              so the peers' lateral sum is non-negative -> the frame is invariant
              to a left/right mirror, so "right" has no globally-consistent meaning)

plus a floor reference:

  plain_std   the symmetric teacher in the standard frame (deadlocks)

If conv_refl collapses to the plain_std floor while conv_std clears, the convention
requires a representation that can represent handedness — the teacher carrying it is
necessary but not sufficient. Same architecture, same teacher, same BC objective
(all reach bc_mse ~1e-4); only the representation's symmetry differs.

Arms seed-paired McNemar-exact, models trained once (fixed seed).

  python scripts/swarm_bc_chirality_phase.py --episodes 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _swarm_policy as sp  # noqa: E402

from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "swarm_bc_chirality_phase.json"


def _bc_mse(P, stats, data):
    egos = (data[0] - stats["em"]) / stats["es"]
    peers = np.where(data[2][:, :, None] > 0, (data[1] - stats["pm"]) / stats["ps"], 0.0)
    return float(np.mean((sp.forward(P, egos, peers, data[2]) - data[3]) ** 2))


def _train(teacher, refl, n_scenes, epochs, seed):
    data = sp.make_dataset(teacher, n_list=[3, 4, 5, 6], n_scenes=n_scenes,
                           seed0=seed, reflect_canonical=refl)
    P, stats = sp.train_bc(data, epochs=epochs, seed=seed)
    ctrl = sp.make_student_controller(P, stats, reflect_canonical=refl)
    return ctrl, _bc_mse(P, stats, data)


def _eval_bits(ctrl, n, seeds):
    out = []
    for s in seeds:
        st, gl = sp.antipodal(n, np.random.default_rng(10_000 + s))  # disjoint from train
        out.append(sp.rollout(st, gl, ctrl).success)
    return out


def _mc(a, b):
    bb = sum(1 for x, y in zip(a, b) if x and not y)
    cc = sum(1 for x, y in zip(a, b) if y and not x)
    return bb, cc, mcnemar_exact_p(bb, cc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--n-scenes", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    seeds = list(range(args.episodes))

    print("training students (BC on random scenes only; no antipodal seen)...", flush=True)
    arms = {}
    mses = {}
    for name, teacher, refl in (("conv_std", sp.teacher_conv, False),
                                ("conv_refl", sp.teacher_conv, True),
                                ("plain_std", sp.teacher_plain, False)):
        arms[name], mses[name] = _train(teacher, refl, args.n_scenes, args.epochs, args.seed)
        print(f"  {name:10} bc_mse={mses[name]:.4f}", flush=True)

    print(f"\nclosed-loop antipodal success ({args.episodes} eval seeds, unseen in training)")
    print(f"{'N':>3} | {'conv_std':>9} {'conv_refl':>10} {'plain_std':>10}")
    bits = {}
    for n in (4, 6, 8):
        bits[n] = {name: _eval_bits(ctrl, n, seeds) for name, ctrl in arms.items()}
        print(f"{n:>3} | {sum(bits[n]['conv_std']):>9} {sum(bits[n]['conv_refl']):>10} "
              f"{sum(bits[n]['plain_std']):>10}", flush=True)

    print("\nMcNemar exact (paired by eval seed):")
    print(f"{'N':>3}  {'comparison':>26}  {'b':>3} {'c':>3} {'p':>10}")
    report = {"episodes": args.episodes, "bc_mse": mses, "cells": []}
    for n in (4, 6, 8):
        r = bits[n]
        row = {"n": n}
        for a in arms:
            row[a] = sum(r[a])
        for label, x, y in (
            ("conv_std  vs  conv_refl", r["conv_std"], r["conv_refl"]),
            ("conv_refl vs  plain_std", r["conv_refl"], r["plain_std"]),
        ):
            bb, cc, p = _mc(x, y)
            print(f"{n:>3}  {label:>26}  {bb:>3} {cc:>3} {p:>10.2e}")
            row[label.split(" vs")[0].strip() + "_vs_" + label.split("vs")[1].strip()] = \
                {"b": bb, "c": cc, "p": p}
        report["cells"].append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
