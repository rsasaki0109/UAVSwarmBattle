"""RL emergence: deep-set vs transformer on antipodal + threat-mixed training.

Trains both policies by REINFORCE on a symmetric reward, then evaluates:
  - antipodal hub (convention discovery)
  - hub + K moving threats (threat-aware generalization)

  python scripts/swarm_transformer_rl_phase.py --iters 180 --episodes 24
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
import _swarm_rl as ds_rl  # noqa: E402
import _swarm_threat as thr  # noqa: E402
import _swarm_transformer_rl as xf_rl  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "swarm_transformer_rl_phase.json"
EVAL = 25
NS = (4, 6)
THREAT_K = 2


def _antipodal_succ(ctrl_fn):
    return {
        n: sum(
            sp.rollout(*sp.antipodal(n, np.random.default_rng(30_000 + s)), ctrl_fn()).success
            for s in range(EVAL)
        )
        for n in NS
    }


def _threat_succ(ctrl_fn):
    return {
        n: sum(
            thr.rollout(*thr.hub_scene(n, THREAT_K, np.random.default_rng(40_000 + s)), ctrl_fn()).success
            for s in range(EVAL)
        )
        for n in NS
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=180)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    print("training deep-set RL...", flush=True)
    Pd, sd = ds_rl.train_reinforce(
        iters=args.iters, episodes=args.episodes, sigma=args.sigma,
        seed=args.seed, verbose=True,
    )
    ds_ctrl = ds_rl.policy_controller(Pd, sd)

    print("\ntraining transformer RL (antipodal + threat mix)...", flush=True)
    Px, sx = xf_rl.train_reinforce(
        iters=args.iters, episodes=args.episodes, sigma=args.sigma,
        seed=args.seed, threat_frac=0.5, verbose=True,
    )
    xf_ant_ctrl = xf_rl.policy_controller(Px, sx)
    xf_thr_ctrl = xf_rl.threat_controller(Px, sx)

    report = {
        "iters": args.iters,
        "episodes": args.episodes,
        "eval_seeds": EVAL,
        "threat_k": THREAT_K,
        "antipodal": {},
        "threat_hub": {},
    }

    print(f"\nantipodal success (/{EVAL})")
    print(f"{'arm':>12} | " + " | ".join(f"N={n}" for n in NS))
    for name, fn in (("ds-rl", lambda: ds_ctrl), ("xf-rl", lambda: xf_ant_ctrl)):
        succ = _antipodal_succ(fn)
        report["antipodal"][name] = succ
        print(f"{name:>12} | " + " | ".join(f"{succ[n]:>3}" for n in NS))

    print(f"\nhub + {THREAT_K} threats success (/{EVAL})")
    print(f"{'arm':>12} | " + " | ".join(f"N={n}" for n in NS))
    ds_thr = thr.make_ds_controller(Pd, sd)

    def _ds_threat():
        return ds_thr

    for name, fn in (("ds-blind", _ds_threat), ("xf-threat", lambda: xf_thr_ctrl)):
        succ = _threat_succ(fn)
        report["threat_hub"][name] = succ
        print(f"{name:>12} | " + " | ".join(f"{succ[n]:>3}" for n in NS))

    _, ds_h = ds_rl.handedness(Pd, sd, False)
    _, xf_h = xf_rl.handedness(Px, sx, False)
    report["handedness"] = {"ds": ds_h, "xf": xf_h}
    print(f"\nhandedness |lat| mean: ds={ds_h:.3f}  xf={xf_h:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
