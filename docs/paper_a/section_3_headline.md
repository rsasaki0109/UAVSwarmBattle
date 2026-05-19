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

## Dynamic-obstacle extension: the softmax-averaging operator under moving obstacles

Adding a *moving* obstacle to the N=4 baseline cell makes the
softmax-averaging mechanism visible in a second failure mode that
does not require the $\Delta$ statistic to see. We add one sphere
obstacle at $(20, 5, 6)$, radius 0.8, moving north along $x=20$ at
$v \in \{2, 4, 8\}$ m/s — directly along the north drone's corridor
$(20, 3, 6) \to (20, 37, 6)$. Same paired-seed protocol (n=30 per
cell, seeds 42-71), same static obstacles, same planner Pareto cells.

**Table 2.** N=4 baseline + one moving obstacle on the north corridor.
$v=0$ row is the §3 static baseline; $v \in \{2, 4, 8\}$ rows are
the dynamic-obstacle cells.

| $v$ (m/s) | MPC joint | GPU MPPI joint | McNemar | GPU MPPI failure attribution |
|---|---|---|---|---|
| 0 (static) | 78.0 % | 77.0 % | $p = 1.00$ | clustered across seeds (§3 mechanism) |
| 2 | 73.3 % | **3.3 %** (1/30) | $p \approx 0$ | north drone, $t \approx 5.0$ s, every failed seed |
| 4 | 80.0 % | **3.3 %** (1/30) | $p \approx 0$ | north drone, $t \approx 5.0$ s, every failed seed |
| 8 | 3.3 % (1/30) | 3.3 % (1/30) | $p = 1.00$ | both planners floor under fast obstacle |

At $v=2$ m/s GPU MPPI's joint success collapses by 74 pp against
the §3 static baseline; at $v=4$ the collapse persists with the same
$\sim 3.3$ % joint floor. MPC holds at 73-80 % joint over this
range. Reading the failure attribution per-seed makes the mechanism
transparent: in every GPU MPPI failed episode at $v \in \{2, 4\}$,
**only the north drone collides**, and the collision time is
$t = 0.15$ s (step 4) across all 29 of 30 failed seeds. The other
three drones (east, west, south, all on perpendicular corridors)
succeed in every failed episode. Episode-level `final_t` in the
analysis tables reads as $\sim 5$ s only because the other three
drones complete normally while drone 2 has already crashed; the
*drone-2* failure is immediate, not late-episode.

The dyn_v2 cell's immediacy is partly fortuitous: the static
obstacle seed (`seed: 7`, count=30) happens to place a voxel
obstacle at $(20, 5, 6)$ — the *exact* initial position of the
moving sphere. North drone at $(20, 3, 6)$ therefore faces a
stacked static + dynamic obstacle just 2 m ahead at $t = 0$, and
the §3-mechanism near-zero lateral command lets it crash into the
voxel cell within 4 steps. The off-corridor probe at $x = 18$
(below) is the cleaner manifestation of the same mechanism without
this incidental confound — there the cliff persists with no
static-obstacle overlap and the role swap (MPC failing instead)
emerges.

The mechanism is the *same* softmax-averaging operator that produced
the §3 static $\Delta$-flip, now expressed as **bidirectional
avoidance cancellation**: when the dynamic obstacle is dead ahead
on the north drone's corridor, half of GPU MPPI's 64 rollouts find
escape volumes by detouring east-of-obstacle and half by detouring
west-of-obstacle, with comparable cost. The softmax-weighted mean
lateral command averages the two sides to near zero, the north drone
slows in the central corridor, and the slow-moving obstacle catches
up from behind. MPC's argmin selects the single lowest-cost rollout
each replan, commits to one side, and clears.

This is the cleanest single-drone demonstration of the §3 mechanism
in the paper. The §3 static case manifests as a +11.4 pp $\Delta$
over indep$^4$ — an aggregate effect across 100 paired episodes that
requires the indep$^4$ baseline to read. The dynamic case manifests
as **a 74-pp joint-success cliff against a single nominal change to
the scenario** (adding one obstacle that moves at 25 % of drone
max-speed), with a deterministic failed-drone identity tied to the
obstacle's spatial alignment with the drone's corridor. The
deployment consequence inverts: under static peers GPU MPPI's
softmax conservatism is a coordination liability ($\Delta$ flip);
under dynamic obstacles it is a single-drone catastrophic failure
mode that MPC's argmin avoids.

At $v=8$ m/s (parity with drone `max_speed`), both planners drop to
the joint floor (3.3 %, 1/30). MPC's per-drone collapses to 50.8 %
and the failures distribute across all four drones, with three
seeds timing out at $t = 75$ s — the obstacle is fast enough that
the 1-2 s lookahead in both planner cells no longer supports a
stable detour. This $v=8$ floor establishes the dynamic regime's
upper bound but does not change the qualitative claim: at $v=2$ m/s
already, the $\Delta$-flip mechanism's softmax-averaging root cause
produces a deployment-relevant failure mode.

Three probes localise the mechanism. An **off-corridor probe**
($x=15$, 5 m offset, $v=4$) restores the §3 static baseline numbers
exactly (per-drone 95.8/95.0 %, joint 83.3/86.7 %, $\Delta$
$-1.0$/$+5.2$ pp, $p=1.00$). GPU MPPI is not generically bad at
moving obstacles — it fails specifically when the obstacle aligns
with a drone's corridor and presents bidirectional escape symmetry.
A **2-obstacle compound probe** (one obstacle each on the north and
east corridors) drops both planners to the joint floor (MPC 13.3 %,
GPU 3.3 %). The cancellation mechanism applies *per corridor
alignment*. Finally, an **off-corridor gradient probe** at $x \in
\{17, 18, 19\}$ (3, 2, 1 m offsets) traverses a regime where the
planner roles *swap*: at offset 2 m (i.e. $x=18$) **MPC collapses**
(per-drone 69 %, joint 6.7 %) while GPU MPPI holds (per-drone 87.5 %,
joint 70 %, McNemar $p \approx 0$, GPU-only successes on 20 seeds).
The MPC failure mode at offset 2 is the §3 mechanism with the planner
roles reversed: MPC's argmin commits to one detour side (east or
west of the obstacle), but with the obstacle offset 2 m the two
sides have *asymmetric* static-obstacle clearance, and on hard
seeds the argmin oscillates between sides as the obstacle moves —
freezing the drone (mean MPC episode time 23.6 s vs GPU 4.5 s).
GPU MPPI's softmax averages the two sides into a smooth lateral
command and clears. The regime map is therefore non-monotonic in
offset:

| offset (m) | who fails             | mechanism                                |
|---|---|---|
| 0          | GPU MPPI catastrophic | softmax cancels bidirectional escape     |
| 1          | tied, GPU slight edge | both planners stressed                   |
| 2          | MPC catastrophic      | argmin commits to wrong side under asymmetric static clutter |
| 3-5        | tied, GPU $\Delta$ edge | §3 static mechanism restored             |

**The smoothing operator helps when the argmin would commit to a
wrong side, and hurts when there is no right side to commit to.**
The two regimes are adjacent in scenario space; small geometry
changes (1-2 m corridor offset) flip the winner.

Full table and per-seed attribution in findings.md "dummy_3d N=4 +
moving obstacle speed sweep" (including all three probes). Repro
configs:
`examples/exp_multi_drone_3d_4_dyn_v{2,4,8}{,_gpu_mppi}.yaml`,
`examples/exp_multi_drone_3d_4_dyn_{off_v4,off{1,2,3}_v4,2x_v4}{,_gpu_mppi}.yaml`,
analysis script `scripts/paired_analysis_dummy_3d_multi.py`.

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
**smoothing operator on the action space** with four distinct
modes across the regimes of this paper:

1. **Static-obstacle multi-drone clustering** (§3 N=4 baseline,
   Table 1): the operator amplifies seed-correlated peer-prediction
   noise into +11.4 pp $\Delta$ over indep$^4$, while MPC's argmin
   stays near zero. **MPC wins on Δ.**
2. **Dynamic-obstacle bidirectional cancellation** (Table 2): the
   operator averages out left/right avoidance commits when a moving
   obstacle is dead ahead, collapsing the affected single drone's
   success from ~95 % to ~3 % at $v=2$ m/s. **MPC wins.**
3. **Sim-physics density-corner sign-reversal** (§4.4.4): at the
   N=4-dense corner of the (N, density) grid the planner roles swap
   — MPC's argmin lock-step concentrates failures across drones and
   GPU MPPI's averaging now suppresses the cluster mode.
   **GPU MPPI wins.**
4. **Aerobatic choreography precision** (`multi_drone_aerobatic`
   scenario, findings.md "Aerobatic synchronized loop"): with the
   mission goal shifted from "avoid failures" to "tight reference
   tracking + synchronized formation", the operator's smooth
   weighted-average command becomes precisely the right thing —
   GPU MPPI tracks a 4-drone synchronized loop with phase-offset
   RMSE 1.67° vs MPC's 10.7° (-84 % wobble), and per-drone tracking
   RMSE 1.04 m vs 1.31 m (-21 %), winning on 20/20 drone-episodes.
   **GPU MPPI wins decisively.**

The shared structural mechanism — softmax averaging vs argmin
commit, against a shared world model — is one operator with four
mode expressions. The deployment story is therefore not "GPU MPPI
is better" or "MPC is better" but a **mode-dependent question**:
- *Coordination-Δ minimisation under static peers* → MPC.
- *Avoidance commit under dynamic obstacles on corridor* → MPC.
- *Multi-drone safety under dense static obstacles* → GPU MPPI.
- *Choreography precision / formation flight* → GPU MPPI.

The robust paper-grade claim is this **shared mechanism, four mode
expressions** — and the mission's metric (Δ, joint success,
tracking RMSE, phase sync) is what selects the right planner.

### Mode superposition in a single scenario: drone race + bouncing intruder

The four modes are not mutually exclusive — a single mission can
straddle two or more simultaneously. The cleanest live demonstration
is `multi_drone_race` (findings.md "Drone race + bouncing intruder"):
4 drones tracking a horizontal-oval reference (mode 4 active for all
24 s) while a single bouncing intruder crosses the track every ~4 s
(mode 2 active whenever the intruder enters a drone's corridor).
Paired $n = 5$ at first cut (the scenario is seed-deterministic except
for planner-internal RNG, so the failure pattern is essentially
seed-stable; a paper-grade $n = 30$ extension is queued but the
per-replan Dijkstra recompute on a moving lookahead goal makes each
episode wallclock-expensive — $\sim 9$ min for MPC). Same
hyperparameters across planners, only the rollout aggregation differs:

- **MPC** — argmin commit. Survives the intruder ($50\,\%$ collision
  rate, ceiling-limited by the geometric collision at $t \approx 10$ s
  that no planner can dodge) but pays a tracking-precision tax
  (RMSE 1.76 m).
- **Vanilla GPU MPPI** — global softmax. Tracks tighter than MPC on
  *every* drone-episode (RMSE 1.66 m, $-6\,\%$, phase RMSE $-0.66°$),
  but the same softmax operator drives the bidirectional-cancellation
  regime when the intruder enters a corridor, costing $+5$ collisions
  ($75\,\%$ vs $50\,\%$). One operator, two opposite valences in the
  same episode.
- **Smart MPPI v4** (mode-aware cluster softmax) — keeps the
  softmax-precision edge (RMSE 1.72 m, $-2.5\,\%$ vs MPC on every
  drone-episode) and recovers MPC's $50\,\%$ collision rate by
  committing to one lateral cluster during cancellation events.
- **Smart MPPI v2** (asymmetric perturbation) — also tested here as
  a parallel comparator; numbers in findings.md.

The cell `base_ew06` on AirSim (§4.4.4) shows two-mode interaction
across episodes (cluster mode at the central crossing × peer-prediction
tail); the race scenario shows two-mode interaction *within a single
24 s episode* and proves the deployment recommendation cannot be a
static planner cell.

Live render: `docs/images/compare_race_oval4.gif` (hero, 3-pane
side-by-side with overlaid rollout cloud).

### Reproduce maps

Side-by-side render: `docs/images/compare_multi_drone_3d_mpc_vs_gpu_mppi.gif`
(dummy_3d study), `docs/images/compare_airsim_multi_mpc_vs_gpu_mppi.gif`
(AirSim n=1 demo). Full configs and analysis scripts:
`examples/exp_multi_drone_3d_4{,_gpu_mppi}.yaml`,
`examples/exp_multi_drone_3d_4_dyn_v{2,4,8}{,_gpu_mppi}.yaml`
(dynamic-obstacle Table 2),
`examples/exp_airsim_multi_{n30,uniform_n30}{,_gpu_mppi}.yaml`,
`examples/exp_airsim_multi_discriminating_n30{,_gpu_mppi}.yaml`,
`examples/exp_airsim_multi_discriminating_central_n30{,_gpu_mppi}.yaml`
(base_ew06, §4.4.4),
`scripts/paired_analysis_airsim_multi.py`,
`scripts/paired_analysis_dummy_3d_multi.py`,
`scripts/run_airsim_multi_chunked.sh`.
