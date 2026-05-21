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

## Dynamic-obstacle extension: retracted, first re-tuned cell is qualitative

The previous draft used a dynamic-obstacle "Table 2" and the race /
gates / chaos / dyn4 follow-up cells as the second mode in a 4-mode
framework. Those claims remain retracted. Commit `1646e11`
(2026-05-21) fixed a multi-runner bug where dynamic obstacles could
remain frozen after a total-wipeout episode. Post-fix reruns of the
affected dynamic-obstacle scenarios collapse to 100 % collision for
the tested planners, so the earlier "MPC 51.7 % vs GPU MPPI 3.3 %"
and Smart MPPI v4-v5 improvements were pre-fix artifacts, not
paper-grade planner mechanisms.

The first re-tuned cell that *visibly* shows planner-level avoidance
under a dynamic obstacle is the **2-drone intersection coordination**
scenario (`examples/exp_intersection_v1_{mpc,mppi}.yaml`, n=5):
N-bound and E-bound drones cross at the centre, a slow dynamic
intruder (0.5 m/s) sits in the intersection, both planners reach 5/5
joint success / 0 collisions in 10 drone-episodes, and the same cost
produces qualitatively different avoidance — **MPC stops & waits the
N drone**, **MPPI swerves both drones around**. This is the current
README hero (`docs/images/compare_intersection_avoid.gif`); see
[findings.md → Intersection coordination](../findings.md#intersection-coordination-visible-mpc-stop-vs-mppi-swerve-under-a-dynamic-intruder)
for the trajectory analysis and reproduce commands.

This is intentionally a **qualitative** result, not a quantitative
Table 2 replacement: n=5, single geometric configuration, no
intruder-speed or drone-count sweep. It re-establishes that the
softmax-vs-argmin mechanism extends past static peers and choreography
into dynamic-obstacle avoidance, but a paper-grade table would still
need a sweep over intruder velocity / drone-arrival timing / drone
count, plus the GPU MPPI counterpart. The race / gates / chaos / dyn4
cells remain retired pending further re-tuning of their geometry.

## Sim transferability

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
episodes).

**Combined reading.** GPU MPPI's softmax conservatism is a single
**smoothing operator on the action space** with three currently valid
mode expressions across the regimes of this paper:

1. **Static-obstacle multi-drone clustering** (§3 N=4 baseline,
   Table 1): the operator amplifies seed-correlated peer-prediction
   noise into +11.4 pp $\Delta$ over indep$^4$, while MPC's argmin
   stays near zero. **MPC wins on Δ.**
2. **Sim-physics density-corner sign-reversal** (§4.4.4): at the
   N=4-dense corner of the (N, density) grid the planner roles swap
   — MPC's argmin lock-step concentrates failures across drones and
   GPU MPPI's averaging now suppresses the cluster mode.
   **GPU MPPI wins.**
3. **Aerobatic choreography precision** (`multi_drone_aerobatic`
   scenario, findings.md "Aerobatic synchronized loop"): with the
   mission goal shifted from "avoid failures" to "tight reference
   tracking + synchronized formation", the operator's smooth
   weighted-average command becomes precisely the right thing —
   GPU MPPI tracks a 4-drone synchronized loop with phase-offset
   RMSE 1.67° vs MPC's 10.7° (-84 % wobble), and per-drone tracking
   RMSE 1.04 m vs 1.31 m (-21 %), winning on 20/20 drone-episodes.
   **GPU MPPI wins decisively.**
4. **Dynamic-obstacle: two-axis separation** (intersection cell,
   `exp_intersection_v1_{mpc,mppi}.yaml`, n=5 / 10 drone-episodes /
   0 collisions; scales to 4-way 4-drone ablation at 20/20). The
   dynamic-obstacle axis splits cleanly into **two independent
   coordinates**:
   - *Predictor quality moves binary success.* With CV prediction
     on, both planners stay at 100 % joint success. Switching off
     the predictor (`use_prediction: false`,
     `exp_intersection_nopred_*.yaml`) drops both planners
     identically to 0/5 joint success, with the same 5/5 collisions
     on the drone whose path is collinear with the intruder. The
     planner aggregator does not move this number.
   - *Planner aggregator moves behavioral fingerprint.* Across the
     v1, 4-way, chokepoint, and wave cells, MPC's argmin produces
     **max |Δcmd| ~6 m/s** step-jumps and wide spatial detours
     under scheduling stress (lateral dev up to 4.31 m on the wave
     cell), while MPPI's softmax produces **~2.5–3 m/s smooth
     commands** at 4× the plan-time compute (~38 vs ~9 ms / replan),
     trading lateral detour for command-jump under the same stress.
   Reporting both axes is the honest framing — collapsing dynamic-
   obstacle results into "success rate gap" hides the predictor
   confound, while collapsing into "behavioral fingerprint only"
   ignores the predictor's role in keeping the cell tractable.

The shared structural mechanism — softmax averaging vs argmin
commit, against a shared world model — is one operator with four
quantitatively validated mode expressions. The deployment story is
therefore not "GPU MPPI is better" or "MPC is better" but a
**mode-dependent question**:
- *Coordination-Δ minimisation under static peers* → MPC.
- *Multi-drone safety under dense static obstacles* → GPU MPPI.
- *Choreography precision / formation flight* → GPU MPPI.
- *Dynamic-obstacle command-jump vs plan-time tradeoff* → MPC
  for fast/jerky commands (~9 ms/replan, max |Δcmd| ~6 m/s), MPPI
  for smooth/expensive commands (~38 ms/replan, max |Δcmd| ~2.5 m/s).
  Binary success ties; the choice is on smoothness vs compute.

The robust paper-grade claim is this **shared mechanism, four
quantitative mode expressions** — and the mission's metric (Δ,
joint success, tracking RMSE, phase sync, or command-jump vs
plan-time fingerprint) is what selects the right planner.

### Dynamic-obstacle race cells: retired from §3

The bouncing-intruder race, moving-gates race, chaos race, and dyn4
path-intersecting-intruders sections were previously used to argue
mode superposition and Smart MPPI repairs. They are no longer part of
the paper-grade §3 result set. The YAMLs and renders remain useful as
failure-mode regression tests and as starting points for a re-tuned
dynamic-obstacle cell, but their pre-`1646e11` quantitative tables
must not be cited as planner evidence. The intersection coordination
cell above is the first re-tuned scenario in this scenario family;
the original race / gates / chaos / dyn4 geometries still need
further widening (gaps, gate timing) before being promoted back.

### Reproduce maps

Side-by-side render: `docs/images/compare_multi_drone_3d_mpc_vs_gpu_mppi.gif`
(dummy_3d study), `docs/images/compare_airsim_multi_mpc_vs_gpu_mppi.gif`
(AirSim n=1 demo). Full configs and analysis scripts:
`examples/exp_multi_drone_3d_4{,_gpu_mppi}.yaml`,
`examples/exp_airsim_multi_{n30,uniform_n30}{,_gpu_mppi}.yaml`,
`examples/exp_airsim_multi_discriminating_n30{,_gpu_mppi}.yaml`,
`examples/exp_airsim_multi_discriminating_central_n30{,_gpu_mppi}.yaml`
(base_ew06, §4.4.4),
`scripts/paired_analysis_airsim_multi.py`,
`scripts/paired_analysis_dummy_3d_multi.py`,
`scripts/paired_analysis_aerobatic.py`,
`scripts/run_airsim_multi_chunked.sh`.
