"""A teammate-token policy is only as symmetry-breaking as its teacher (neta A).

TeamHOI (CVPR 2026) learns one decentralized teammate-token policy that scales to
any team size. The lab's recurring result is that the symmetric antipodal hub
deadlocks every symmetric reactive avoider, and the cure is an explicit right-of-way
CONVENTION that breaks the left/right symmetry. Does the permutation-equivariant
teammate-token ARCHITECTURE create or cure the deadlock — or just transport whatever
its teacher had?

We distill two closed-form teachers into the SAME NumPy deep-set policy (per-peer
encoder -> permutation-invariant mean-pool over teammate tokens -> readout, in the
ego-goal frame), trained by behavioral cloning on RANDOM scenes only (no antipodal),
then evaluate closed-loop on the unseen antipodal hub:

  teacher_plain  goal + symmetric peer repulsion (deadlocks on antipodal)
  teacher_conv   plain + right-of-way convention   (clears the hub)

Arms (seed-paired McNemar-exact, models trained once with a fixed seed):
  student<-plain   distilled from the symmetric teacher
  student<-conv    distilled from the convention teacher

Headline cell: student<-conv vs student<-plain at each N. Same architecture, same
BC objective (both reach bc_mse ~1e-4) — only the teacher's convention differs.

  python scripts/swarm_bc_symmetry_phase.py --episodes 60
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

from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p  # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "results" / "swarm_bc_models.npz"


def _bc_mse(P, stats, data):
    egos = (data[0] - stats["em"]) / stats["es"]
    peers = np.where(data[2][:, :, None] > 0, (data[1] - stats["pm"]) / stats["ps"], 0.0)
    pr = sp.forward(P, egos, peers, data[2])
    return float(np.mean((pr - data[3]) ** 2))


def train_students(n_scenes, epochs, seed):
    students = {}
    for name, teacher in (("plain", sp.teacher_plain), ("conv", sp.teacher_conv)):
        data = sp.make_dataset(teacher, n_list=[3, 4, 5, 6], n_scenes=n_scenes, seed0=seed)
        P, stats = sp.train_bc(data, epochs=epochs, seed=seed)
        students[name] = (P, stats, _bc_mse(P, stats, data))
        print(f"  trained student<-{name}: {len(data[0])} samples, bc_mse={students[name][2]:.4f}",
              flush=True)
    return students


def eval_bits(ctrl, n, seeds):
    out = []
    for s in seeds:
        rng = np.random.default_rng(10_000 + s)  # eval seeds disjoint from train
        st, gl = sp.antipodal(n, rng)
        out.append(sp.rollout(st, gl, ctrl).success)
    return out


def _mc(a, b):
    """b_ = a-only success, c_ = b-only success."""
    b_ = sum(1 for x, y in zip(a, b) if x and not y)
    c_ = sum(1 for x, y in zip(a, b) if y and not x)
    return b_, c_, mcnemar_exact_p(b_, c_)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--n-scenes", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seeds = list(range(args.episodes))

    print("training the two students (BC on random scenes only; no antipodal seen)...",
          flush=True)
    students = train_students(args.n_scenes, args.epochs, args.seed)
    Pp, sp_p, _ = students["plain"]
    Pc, sp_c, _ = students["conv"]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **{f"plain_{k}": v for k, v in Pp.items()},
             **{f"plain_s_{k}": v for k, v in sp_p.items()},
             **{f"conv_{k}": v for k, v in Pc.items()},
             **{f"conv_s_{k}": v for k, v in sp_c.items()})

    ctrls = {
        "teacher_plain": sp.teacher_plain,
        "student<-plain": sp.make_student_controller(Pp, sp_p),
        "teacher_conv": sp.teacher_conv,
        "student<-conv": sp.make_student_controller(Pc, sp_c),
    }

    print(f"\nclosed-loop antipodal success ({args.episodes} eval seeds, unseen in training)")
    print(f"{'N':>3} | {'t_plain':>8} {'s<-plain':>9} | {'t_conv':>7} {'s<-conv':>8}")
    bits = {}
    for n in (4, 6, 8):
        row = {}
        for name, ctrl in ctrls.items():
            row[name] = eval_bits(ctrl, n, seeds)
        bits[n] = row
        print(f"{n:>3} | {sum(row['teacher_plain']):>8} {sum(row['student<-plain']):>9} "
              f"| {sum(row['teacher_conv']):>7} {sum(row['student<-conv']):>8}", flush=True)

    print("\nMcNemar exact (paired by eval seed):")
    print(f"{'N':>3}  {'comparison':>28}  {'b':>3} {'c':>3} {'p':>10}")
    for n in (4, 6, 8):
        r = bits[n]
        for label, a, b in (
            ("s<-conv  vs  s<-plain", r["student<-conv"], r["student<-plain"]),
            ("s<-plain vs  teacher_plain", r["student<-plain"], r["teacher_plain"]),
            ("s<-conv  vs  teacher_conv", r["student<-conv"], r["teacher_conv"]),
        ):
            bb, cc, p = _mc(a, b)
            print(f"{n:>3}  {label:>28}  {bb:>3} {cc:>3} {p:>10.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
