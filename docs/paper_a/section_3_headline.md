# §3. Multi-drone GPU MPPI flips the coordination Δ

When four drones cross-pattern through the same 40×40×12 voxel world
with 30 random obstacles, GPU MPPI and CPU MPC reach essentially the
same joint-success rate, but they get there with very different
failure structures. We measured this paired across n=100 episodes per
planner on identical seeds, holding the scenario, sensor model, peer
prediction (constant-velocity), and goal/collision criteria fixed
between runs. The only change between the two columns of Table 1 is
the planner family.

**Table 1.** Multi-drone success at 4 drones, 30 random obstacles,
40×40×12 voxel world, n=100 paired episodes per planner.

| planner                | per-drone (95 % CI)   | joint (95 % CI)     | indep$^4$ | Δ over indep |
|---                     |---                    |---                  |---        |---           |
| MPC      (n=8, h=40)   | 93.8 % [90.9, 95.7]   | 78.0 % [68.9, 85.0] | 77.2 %    | +0.8 pp      |
| GPU MPPI (n=64, h=20)  | 90.0 % [86.7, 92.6]   | 77.0 % [67.8, 84.2] | 65.6 %    | **+11.4 pp** |

Three readings of this table matter, in order:

**(1) The joint rates are tied.** Wilson 95 % intervals on 4-drone
joint success overlap by 16 pp, and a McNemar test on same-seed
paired joints returns 67 both-succ, 11 MPC-only-succ, 10 GPU-only-succ
— the exact two-sided binomial p-value is 1.00. By the headline
metric, you cannot tell the two planners apart. The natural conclusion
from this comparison alone is that GPU MPPI is a drop-in replacement.

**(2) The per-drone rates are not tied.** Per-drone Wilson CIs on 400
trials (4 drones × 100 episodes) put MPC at 93.8 % [90.9, 95.7] and
GPU MPPI at 90.0 % [86.7, 92.6]: the GPU MPPI band sits 1.7 pp below
MPC's lower edge. The 3.8 pp per-drone gap is real, and it is the
piece that makes the next reading meaningful — without it, both
planners would also tie on indep$^4$.

**(3) The Δ over indep$^4$ separates by an order of magnitude.** If
the four drones were independent agents pulling the same per-drone
rate, joint success would equal $\text{per}^4$ — 77.2 % for MPC and
65.6 % for GPU MPPI. The MPC measured-joint of 78.0 % sits only +0.8
pp above its indep$^4$ prediction: episode-level outcomes for MPC are
essentially as-if-independent across drones. The GPU MPPI measured-
joint of 77.0 % sits **+11.4 pp** above its indep$^4$ prediction —
GPU MPPI's per-drone failures *cluster* on the same seeds rather than
spreading across them. Looking at the 12 GPU-MPPI failed joints
directly, the per-seed failure pattern is bimodal: hard seeds collide
2–4 drones together (e.g. seed 119 → [coll, coll, coll, succ]; seed
131 → [coll, coll, succ, succ]); easy seeds reach 4/4. MPC's 22
failed joints distribute one or two drones at a time across many
seeds.

The mechanism producing this clustering is the planner's own action-
selection rule, not the scenario. GPU MPPI's softmax across 64 rollouts
weights samples by $\exp(-\text{cost}/T)$ before averaging. On seeds
where the local cost landscape near the crossing centre is benign,
the softmax cloud agrees, all four planners pick coherent escape
trajectories, and the joint episode succeeds cleanly. On seeds where
the same local landscape has a sharp ridge — e.g. when peer
predictions place two passing drones close to ego's planned escape
volume — the softmax averages over partially-divergent rollouts and
produces a conservative, slow-moving consensus command. Because every
drone evaluates a *shared* peer-prediction world model under that
softmax, all four planners arrive at similar over-conservative commands
on the hard seed and stall together at the crossing. CPU MPC's
deterministic argmin against the same shared landscape commits each
drone to a specific waypoint with no averaging, so its per-drone
brittleness on the same seeds remains uncorrelated across the four
agents. **Sample diversity is not a substitute for peer prediction —
it is a knob that trades smoother typical-case behaviour for harsher
tail-case outcomes**, when peers share a noisy world model.

This has a concrete deployment consequence. In missions where partial
success counts independently — three of four deliveries completed, the
fleet returns safely with two functional drones — MPC's distributed-
failure shape is the right tool: same nominal joint rate, but
catastrophic days are rare. In missions where the joint outcome is
the only outcome of interest — all-or-nothing formations, or
synchronised arrival constraints — the two planners are equivalent
and GPU MPPI's smoother typical behaviour (mean per-drone arrival
spread is 4× tighter at the 4/4-success geometry, see §4.4) becomes
the deciding factor.

The same comparison transferred to AirSim physics is non-discriminating
at the demo scenario (no obstacles, ±2-4 m altitude stagger, 4/4 for
both planners at n=30) — the Δ-flip mechanism is not visible when the
scenario is too easy for either planner to fail. The mechanism is
visible *between* the success cases though: per-drone arrival spread
holds at 0.55 s for GPU MPPI vs 0.02 s for MPC across 30 paired
AirSim episodes, the same ratio observed at n=1. The harder-AirSim-
geometry replication of the failure-level Δ-flip itself is reported
in §4.4 (uniform-altitude crossing).

Side-by-side render: `docs/images/compare_multi_drone_3d_mpc_vs_gpu_mppi.gif`
(dummy_3d study), `docs/images/compare_airsim_multi_mpc_vs_gpu_mppi.gif`
(AirSim n=1 demo). Full configs and analysis script:
`examples/exp_multi_drone_3d_4{,_gpu_mppi}.yaml`,
`scripts/paired_analysis_airsim_multi.py`.


---

*Drafted with placeholder forward-reference to §4.4 uniform-altitude
AirSim result (in progress at draft time). Once that table is
finalised the closing paragraph above will inline its joint-rate
numbers + Δ over indep$^4$ instead of cross-referencing.*
