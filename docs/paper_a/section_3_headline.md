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

This has a concrete deployment consequence at the N=4 baseline cell.
In missions where partial success counts independently — three of
four deliveries completed, the fleet returns safely with two
functional drones — MPC's distributed-failure shape is the right
tool: same nominal joint rate, but catastrophic days are rare. In
missions where the joint outcome is the only outcome of interest —
all-or-nothing formations, or synchronised arrival constraints — the
two planners are equivalent and GPU MPPI's smoother typical behaviour
(mean per-drone arrival spread is 4× tighter at the 4/4-success
geometry, see §4.4) becomes the deciding factor.

**Scope of the headline claim.** Table 1 is one cell of an
$(N, \text{density})$ grid we map in §6. Across
$N \in \{2, 3, 4, 6, 8, 10, 12\}$ and obstacle counts
$\{30, 120, 240\}$ on the same circular crossing geometry, the
structural claim — per-drone rates can stay tied while the
coordination $\Delta$ separates between planners — survives, but the
*sign* of the separation does not. At $N=4$ baseline (above) GPU MPPI
is the cluster source ($\Delta$ +11.4 vs +0.8 pp). At $N=4$ dense
(240 obstacles) MPC becomes the cluster source ($\Delta$ +6.7 vs
−1.2 pp). At $N=6$ no flip occurs across the density sweep; at $N=8$
baseline GPU MPPI's per-drone uniquely collapses under the
8-fold-symmetric crossing (per-drone 69 % vs MPC 92 %, McNemar
$p \approx 0.0001$); at $N=12$ GPU's $\Delta$ falls back below MPC's.
The mechanism — softmax-vs-argmin against a shared peer-prediction
world model — therefore predicts a **planner-dependent failure-
clustering signature**, not a planner-dependent winner. §6 documents
the full grid; the N=4 baseline cell described here is the cleanest
demonstration of the mechanism but not a universal one.

The same comparison transferred to AirSim physics produced four
complementary regimes. At the easier *staggered-altitude* crossing,
both planners hit 100 % joint success across n=30 paired seeds; the
failure-level Δ-flip cannot register, though the trajectory-level
signal is preserved (mean per-drone arrival spread 0.02 s for MPC vs
0.55 s for GPU MPPI). At the harder *uniform-altitude* crossing,
GPU MPPI's softmax-conservative ~30 %-slower commands keep drones at
the conflict centre long enough that per-drone success collapses to
28.3 % and joint success to 0/30, while MPC holds 46.7 % joint
(McNemar p ≈ 0.00012). A *static-cube staggered* cell with spawned
Blocks meshes lands MPC in the target band: MPC reaches 87.5 %
per-drone and 22/30 joint, while GPU MPPI reaches 120/120 per-drone
and 30/30 joint (GPU-only success 8, MPC-only 0, McNemar p ≈ 0.008).
Finally, a *static-cube density-swept* cell (§4.4.4, `base_ew06`)
closes the dummy_3d analogue: five widened pillars concentrate four
drones into one central crossing, GPU MPPI drops off the success
ceiling, and the $\Delta$-flip mechanism reproduces under AirSim
collision geometry — **but with the sign reversed**, consistent with
the N=4 dense corner of the dummy_3d grid. MPC, not GPU MPPI,
exhibits the multi-drone cluster failure mode (collision-object trace
confirms drone-drone collisions at the central crossing in MPC
episodes). Combined reading: **GPU MPPI's softmax conservatism is a
smoothing operator on the action space, and it can both help (remove
paired AirSim collision seeds, suppress the dense-corner cluster
mode) and hurt (catastrophic failure at speed-through-bottleneck
geometries) depending on whether the dominant coupling is behavioural
or geometric.** The robust paper-grade claim is the *qualitative*
mechanism: per-drone tie, $\Delta$ separates by planner, direction
set by $(N, \text{density}, \text{geometry})$. A dynamic-obstacle
extension (§6, findings.md "dummy_3d N=4 + moving obstacle speed
sweep") shows the same softmax-averaging operator also produces a
*bidirectional avoidance cancellation* under moving obstacles —
collapsing GPU MPPI's joint success from 86.7 % (§3 N=4 baseline) to
3.3 % at $v=2$ m/s, with the failed drone always the one whose
corridor the obstacle traverses. Under that regime, MPC's argmin
commit-to-one-side is the more robust policy.

Side-by-side render: `docs/images/compare_multi_drone_3d_mpc_vs_gpu_mppi.gif`
(dummy_3d study), `docs/images/compare_airsim_multi_mpc_vs_gpu_mppi.gif`
(AirSim n=1 demo). Full configs and analysis scripts:
`examples/exp_multi_drone_3d_4{,_gpu_mppi}.yaml`,
`examples/exp_airsim_multi_{n30,uniform_n30}{,_gpu_mppi}.yaml`,
`examples/exp_airsim_multi_discriminating_n30{,_gpu_mppi}.yaml`,
`examples/exp_airsim_multi_discriminating_central_n30{,_gpu_mppi}.yaml`
(base_ew06, §4.4.4),
`scripts/paired_analysis_airsim_multi.py`,
`scripts/run_airsim_multi_chunked.sh`.
