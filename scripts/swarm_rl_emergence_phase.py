"""Does a policy DISCOVER the right-of-way convention from a symmetric reward?
(neta A, RL counterpart to swarm_bc_symmetry_phase.py / swarm_bc_chirality_phase.py)

BC can only TRANSPORT a convention its teacher already demonstrated, and only in a
chirality-capable representation. The open question: is the symmetry-breaking
*learnable* from a reward with no built-in handedness — does symmetric optimization
spontaneously break the symmetry, or fall into the symmetric (deadlocking) solution?

Train the SAME NumPy teammate-token deep set by REINFORCE (scripts/_swarm_rl.py) on
a reflection-SYMMETRIC reward (progress − collision + goal), across several training
seeds, in two representations:

  standard   ego-goal frame (chirality preserved)
  reflect    ego-goal frame + reflection canonicalization (chirality-free)

Reference floor/ceiling from BC (same architecture):
  bc_plain   distilled from the symmetric teacher  (the deadlock floor)
  bc_conv    distilled from the convention teacher  (the cure ceiling)

Reported: closed-loop antipodal success (mean over training seeds × eval seeds) and,
for RL, the handedness consistency |mean lateral| / mean|lateral| (1.0 = a single
self-generated convention; 0 = symmetric / no preferred side).

  python scripts/swarm_rl_emergence_phase.py --train-seeds 6 --iters 250
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _swarm_policy as sp  # noqa: E402
import _swarm_rl as rl  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "swarm_rl_emergence_phase.json"
EVAL_SEEDS = 30
NS = (4, 6, 8)


def _eval_success(ctrl):
    return {n: sum(sp.rollout(*sp.antipodal(n, np.random.default_rng(10_000 + s)), ctrl).success
                   for s in range(EVAL_SEEDS)) for n in NS}


def _train_eval(job, iters, episodes, sigma):
    frame, refl, seed = job
    P, stats = rl.train_reinforce(reflect=refl, iters=iters, episodes=episodes,
                                  sigma=sigma, seed=seed)
    ctrl = rl.policy_controller(P, stats, refl)
    succ = _eval_success(ctrl)
    hmean, hlat = rl.handedness(P, stats, refl)
    cons = abs(hmean) / hlat if hlat > 1e-6 else 0.0
    return (frame, seed, succ, cons)


def _bc_ref(teacher_name):
    teacher = sp.teacher_conv if teacher_name == "conv" else sp.teacher_plain
    data = sp.make_dataset(teacher, n_list=[3, 4, 5, 6], n_scenes=150, seed0=0)
    P, stats = sp.train_bc(data, epochs=150, seed=0)
    ctrl = sp.make_student_controller(P, stats)
    return _eval_success(ctrl)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seeds", type=int, default=6)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    jobs = [(frame, refl, s)
            for frame, refl in (("standard", False), ("reflect", True))
            for s in range(args.train_seeds)]
    fn = partial(_train_eval, iters=args.iters, episodes=args.episodes, sigma=args.sigma)
    print(f"training {len(jobs)} RL policies ({args.train_seeds} seeds × 2 frames, "
          f"{args.iters} iters)...", flush=True)
    with Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(fn, jobs)

    print("BC reference floor/ceiling...", flush=True)
    bc = {"bc_plain": _bc_ref("plain"), "bc_conv": _bc_ref("conv")}

    by_frame = {"standard": [], "reflect": []}
    for frame, seed, succ, cons in res:
        by_frame[frame].append((seed, succ, cons))

    report = {"train_seeds": args.train_seeds, "iters": args.iters, "eval_seeds": EVAL_SEEDS,
              "bc_reference": bc, "rl": {}}
    print(f"\nRL discovery from a symmetric reward (success out of {EVAL_SEEDS}, "
          f"mean[min..max] over {args.train_seeds} train seeds)")
    print(f"{'arm':>10} | " + " | ".join(f"N={n:>2}" for n in NS) + " | handed-consistency")
    print("-" * 70)
    for frame in ("standard", "reflect"):
        rows = by_frame[frame]
        line = []
        cell = {}
        for n in NS:
            vals = [s[1][n] for s in rows]
            line.append(f"{np.mean(vals):4.1f}[{min(vals)}..{max(vals)}]")
            cell[n] = {"mean": float(np.mean(vals)), "min": int(min(vals)), "max": int(max(vals)),
                       "per_seed": [int(s[1][n]) for s in rows]}
        cons = [s[2] for s in rows]
        print(f"{frame:>10} | " + " | ".join(f"{x:>10}" for x in line) +
              f" | {np.mean(cons):.2f}[{min(cons):.2f}..{max(cons):.2f}]")
        report["rl"][frame] = {"cells": cell, "consistency": [float(c) for c in cons]}
    for name, succ in bc.items():
        print(f"{name:>10} | " + " | ".join(f"{succ[n]:>3}/{EVAL_SEEDS:<6}" for n in NS) + " |  (BC reference)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    main()
