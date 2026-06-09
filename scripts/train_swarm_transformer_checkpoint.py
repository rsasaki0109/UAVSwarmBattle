#!/usr/bin/env python3
"""Train swarm_transformer checkpoints for YAML planner use.

Peers-only (antipodal convention):
  python scripts/train_swarm_transformer_checkpoint.py --epochs 200

Peers + moving threats (hub crossing):
  python scripts/train_swarm_transformer_checkpoint.py --threat --epochs 200
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _swarm_policy as sp  # noqa: E402
import _swarm_threat as thr  # noqa: E402
import _swarm_transformer as st  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PEER = ROOT / "results" / "swarm_transformer_conv.npz"
DEFAULT_THREAT = ROOT / "results" / "swarm_transformer_threat_conv.npz"


def _eval_antipodal(sc):
    for n in (4, 6, 8):
        ok = sum(
            sp.rollout(*sp.antipodal(n, np.random.default_rng(s)), sc).success
            for s in range(30)
        )
        print(f"  antipodal N={n}: {ok}/30", flush=True)


def _eval_threat(ctrl):
    for n in (4, 6):
        ok = sum(
            thr.rollout(*thr.hub_scene(n, 2, np.random.default_rng(s)), ctrl).success
            for s in range(25)
        )
        print(f"  hub+2-threat N={n}: {ok}/25", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", choices=("plain", "conv"), default="conv")
    ap.add_argument("--threat", action="store_true", help="train on hub+threat scenes")
    ap.add_argument("--n-scenes", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.out:
        out = Path(args.out)
    else:
        out = DEFAULT_THREAT if args.threat else DEFAULT_PEER

    if args.threat:
        teacher = thr.teacher_conv if args.teacher == "conv" else thr.teacher_plain
        print(
            f"BC xf<-{args.teacher} on hub+threat "
            f"({args.n_scenes} scenes, {args.epochs} epochs)...",
            flush=True,
        )
        data = thr.make_xf_dataset(
            teacher, n_list=[3, 4, 5, 6], k_list=[1, 2, 3],
            n_scenes=args.n_scenes, seed0=args.seed,
        )
    else:
        teacher = sp.teacher_conv if args.teacher == "conv" else sp.teacher_plain
        print(
            f"BC xf<-{args.teacher} on peers-only "
            f"({args.n_scenes} scenes, {args.epochs} epochs)...",
            flush=True,
        )
        data = st.make_dataset(
            teacher, n_list=[3, 4, 5, 6], n_scenes=args.n_scenes, seed0=args.seed,
        )

    P, stats = st.train_bc(data, epochs=args.epochs, seed=args.seed, verbose=True)
    st.save_checkpoint(out, P, stats)
    print(f"wrote {out}", flush=True)

    if args.threat:
        _eval_threat(thr.make_xf_controller(P, stats))
    else:
        _eval_antipodal(st.make_student_controller(P, stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
