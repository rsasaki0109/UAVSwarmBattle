"""Threat-token BC: ally-only deep set vs transformer with ROLE_THREAT tokens.

Same threat-aware teacher (convention + peer/threat repulsion), but:
  ds-blind  deep set trained on ally-only features (cannot see threats at eval)
  xf-threat transformer with ally + threat tokens

Eval on unseen hub crossing with K moving threats through the centre.

  python scripts/swarm_transformer_threat_phase.py --episodes 30
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _swarm_policy as sp  # noqa: E402
import _swarm_threat as thr  # noqa: E402
import _swarm_transformer as st  # noqa: E402

from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p  # noqa: E402


def _eval(ctrl, n, k, seeds):
    out = []
    for s in seeds:
        rng = np.random.default_rng(20_000 + s)
        scene = thr.hub_scene(n, k, rng)
        out.append(thr.rollout(*scene, ctrl).success)
    return out


def _mc(a, b):
    b_ = sum(1 for x, y in zip(a, b) if x and not y)
    c_ = sum(1 for x, y in zip(a, b) if y and not x)
    return b_, c_, mcnemar_exact_p(b_, c_)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--n-scenes", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-k", type=int, default=2)
    args = ap.parse_args()
    seeds = list(range(args.episodes))
    n_list = [3, 4, 5, 6]
    k_list = [1, 2, 3]

    print("BC from threat-aware convention teacher...", flush=True)
    ds_data = thr.make_ds_dataset(
        thr.teacher_conv, n_list=n_list, k_list=k_list,
        n_scenes=args.n_scenes, seed0=args.seed,
    )
    xf_data = thr.make_xf_dataset(
        thr.teacher_conv, n_list=n_list, k_list=k_list,
        n_scenes=args.n_scenes, seed0=args.seed,
    )
    Pd, stats_d = sp.train_bc(ds_data, epochs=args.epochs, seed=args.seed)
    Px, stats_x = st.train_bc(xf_data, epochs=args.epochs, seed=args.seed)
    print(f"  ds-blind: {len(ds_data[0])} samples", flush=True)
    print(f"  xf-threat: {len(xf_data[0])} samples", flush=True)

    arms = {
        "teacher": lambda: thr.teacher_conv,
        "ds-blind": lambda: thr.make_ds_controller(Pd, stats_d),
        "xf-threat": lambda: thr.make_xf_controller(Px, stats_x),
    }

    print(f"\nhub crossing success ({args.episodes} eval seeds, K={args.eval_k} threats)")
    print(f"{'N':>3} | {'teacher':>8} {'ds-blind':>9} {'xf-threat':>10}")
    bits = {}
    for n in (4, 6):
        row = {}
        for name, mk in arms.items():
            row[name] = _eval(mk(), n, args.eval_k, seeds)
        bits[n] = row
        print(
            f"{n:>3} | {sum(row['teacher']):>8} "
            f"{sum(row['ds-blind']):>9} {sum(row['xf-threat']):>10}",
            flush=True,
        )

    print("\nMcNemar exact (paired by eval seed):")
    for n in (4, 6):
        r = bits[n]
        for label, a, b in (
            ("xf-threat vs ds-blind", "xf-threat", "ds-blind"),
            ("xf-threat vs teacher", "xf-threat", "teacher"),
        ):
            bb, cc, p = _mc(r[a], r[b])
            print(f"  N={n}  {label:>24}  b={bb} c={cc}  p={p:.2e}")

    st0, gl, tpos, tvel = thr.hub_scene(4, args.eval_k, np.random.default_rng(0))
    vel0 = np.zeros_like(st0)
    ego, pad, m, _ = st.featurize(
        st0, vel0, gl, 0, threats=thr._threat_list(tpos, tvel),
    )
    mass = st.attention_mass(Px, stats_x, ego, pad, m)
    roles = pad[: int(m.sum()), 5]
    if len(roles):
        ally_mass = float(mass[: int((roles == 0).sum())].sum())
        thr_mass = float(mass[int((roles == 0).sum()): int(m.sum())].sum())
        print(
            f"\nattention split (xf-threat, N=4 agent 0): "
            f"ally={ally_mass:.2f} threat={thr_mass:.2f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
