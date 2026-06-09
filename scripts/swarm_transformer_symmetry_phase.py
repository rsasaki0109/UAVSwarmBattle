"""Cross-attention teammate-token policy vs mean-pool deep set (TeamHOI probe).

Same teachers / antipodal eval as swarm_bc_symmetry_phase.py, but adds a
TeamHOI-style transformer arm (ego query x variable peer keys/values + role
channel). Headline: does selective attention change convention transport, or
does the architecture still just mirror its teacher?

  python scripts/swarm_transformer_symmetry_phase.py --episodes 40
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
import _swarm_transformer as st  # noqa: E402

from uav_nav_lab.analysis.joint_stats import mcnemar_exact_p  # noqa: E402


def _train_arm(module, n_scenes, epochs, seed):
    out = {}
    for name, teacher in (("plain", sp.teacher_plain), ("conv", sp.teacher_conv)):
        data = module.make_dataset(
            teacher, n_list=[3, 4, 5, 6], n_scenes=n_scenes, seed0=seed,
        )
        P, stats = module.train_bc(data, epochs=epochs, seed=seed)
        egos = (data[0] - stats["em"]) / stats["es"]
        peers = np.where(
            data[2][:, :, None] > 0,
            (data[1] - stats["pm"]) / stats["ps"],
            0.0,
        )
        mse = float(np.mean((module.forward(P, egos, peers, data[2]) - data[3]) ** 2))
        out[name] = (P, stats, mse)
        print(
            f"  {module.__name__:<22} student<-{name}: "
            f"{len(data[0])} samples, bc_mse={mse:.4f}",
            flush=True,
        )
    return out


def _eval_bits(make_ctrl, n, seeds):
    ctrl = make_ctrl()
    out = []
    for s in seeds:
        rng = np.random.default_rng(10_000 + s)
        st_gl = sp.antipodal(n, rng)
        out.append(sp.rollout(st_gl[0], st_gl[1], ctrl).success)
    return out


def _mc(a, b):
    b_ = sum(1 for x, y in zip(a, b) if x and not y)
    c_ = sum(1 for x, y in zip(a, b) if y and not x)
    return b_, c_, mcnemar_exact_p(b_, c_)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--n-scenes", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seeds = list(range(args.episodes))

    print("training deep-set + transformer students (BC on random scenes)...", flush=True)
    ds = _train_arm(sp, args.n_scenes, args.epochs, args.seed)
    xf = _train_arm(st, args.n_scenes, args.epochs, args.seed)

    arms = {
        "ds<-plain": lambda: sp.make_student_controller(*ds["plain"][:2]),
        "ds<-conv": lambda: sp.make_student_controller(*ds["conv"][:2]),
        "xf<-plain": lambda: st.make_student_controller(*xf["plain"][:2]),
        "xf<-conv": lambda: st.make_student_controller(*xf["conv"][:2]),
        "teacher_plain": lambda: sp.teacher_plain,
        "teacher_conv": lambda: sp.teacher_conv,
    }

    print(f"\nclosed-loop antipodal success ({args.episodes} eval seeds)")
    hdr = (
        f"{'N':>3} | {'t_plain':>7} {'ds_pl':>6} {'xf_pl':>6} | "
        f"{'t_conv':>7} {'ds_cv':>6} {'xf_cv':>6}"
    )
    print(hdr)
    bits = {}
    for n in (4, 6, 8):
        row = {name: _eval_bits(fn, n, seeds) for name, fn in arms.items()}
        bits[n] = row
        print(
            f"{n:>3} | {sum(row['teacher_plain']):>7} "
            f"{sum(row['ds<-plain']):>6} {sum(row['xf<-plain']):>6} | "
            f"{sum(row['teacher_conv']):>7} "
            f"{sum(row['ds<-conv']):>6} {sum(row['xf<-conv']):>6}",
            flush=True,
        )

    print("\nMcNemar exact (paired by eval seed):")
    print(f"{'N':>3}  {'comparison':>32}  {'b':>3} {'c':>3} {'p':>10}")
    pairs = (
        ("xf<-conv  vs  ds<-conv", "xf<-conv", "ds<-conv"),
        ("xf<-plain vs  ds<-plain", "xf<-plain", "ds<-plain"),
        ("xf<-conv  vs  xf<-plain", "xf<-conv", "xf<-plain"),
        ("ds<-conv  vs  ds<-plain", "ds<-conv", "ds<-plain"),
    )
    for n in (4, 6, 8):
        r = bits[n]
        for label, a, b in pairs:
            bb, cc, p = _mc(r[a], r[b])
            print(f"{n:>3}  {label:>32}  {bb:>3} {cc:>3} {p:>10.2e}")

    Pc, stats_c, _ = xf["conv"]
    st0, gl0 = sp.antipodal(6, np.random.default_rng(0))
    vel0 = np.zeros_like(st0)
    ego, pad, m, _ = st.featurize(st0, vel0, gl0, 0)
    mass = st.attention_mass(Pc, stats_c, ego, pad, m)
    active = int(m.sum())
    if active:
        top = int(np.argmax(mass[:active]))
        print(
            f"\nattention diagnostic (xf<-conv, N=6 antipodal, agent 0): "
            f"peer {top} mass={mass[top]:.3f} / {active} peers",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
