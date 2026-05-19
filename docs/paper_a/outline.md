# Paper A — Outline (draft)

Status: full draft spine. Sections §1-§7 now have standalone prose
files or appendix material; this file remains the argument map. Source
of truth for individual findings remains `docs/findings.md`.

## Working title (pick one)

1. **"Pareto cells, not planners: how config sweeps and implementation
   correctness reshape planner comparisons for UAV navigation"** —
   methodology framing; multi-drone result sits in §4 as a payoff
   example.
2. **"Sample-cloud planners decorrelate multi-drone failures: a
   coordination-Δ flip with no peer prediction"** — headline-result
   framing; Pareto/correctness sit in §2 as the prerequisite that made
   the comparison fair.
3. **"What a fair single→multi UAV planner benchmark actually requires"** —
   methods-paper framing; Pareto-cell selection, correctness, multi-drone
   coordination Δ, sim transferability are all symptoms of the same
   benchmarking problem.

Lean: **option 2** if the venue is a planning/robotics conference, **option
1** for a methods venue, **option 3** for a workshop. Decide before
manuscript assembly.

## 1. Motivation (½ page)

- UAV planner comparisons routinely report headline numbers from a *single*
  config; the result depends strongly on which Pareto cell was picked.
- Multi-drone coordination Δ (joint succ minus `per_drone^N`) is reported
  inconsistently — sometimes absolute joint succ, sometimes Δ, rarely
  with the indep baseline computed from the same per-drone rate.
- Implementation bugs that affect only one planner family (e.g. a
  goal-mask term that the CPU path naturally short-circuits but the
  batched GPU path must apply explicitly) can invalidate every cell of
  a Pareto sweep without changing any *measured* numbers.
- AirSim ↔ dummy_3d transferability and ROS 2-bridge spatial invariance
  determine which sim is acceptable for which finding.

Claim: **none of these are research questions individually — but
treated together, they explain why two careful teams can publish
contradictory planner comparisons.**

## 2. Setup (1 page)

`uav-nav-lab` harness — single repo, single CLI, all results in this paper
are reproducible from one `examples/exp_*.yaml` each.

- **Simulators**: `dummy_2d`, `dummy_3d`, `airsim_bridge`, `ros2_bridge`.
  Same scenario/sensor/planner interface; sim swap is the only knob.
- **Scenarios**: `grid_world` (2D, 50×50, random obstacles), `voxel_world`
  (3D, 40×40×12, static + bouncing dynamic obstacles), `multi_drone_voxel`
  (N-drone cross / dense / random).
- **Planners**: CPU MPC (deterministic sample batch, `n_samples` × `horizon`),
  CPU MPPI (softmax-weighted), GPU MPPI (PyTorch batched, Fibonacci-sphere
  direction set in 3D, autograd-graph CUDA backend), A* (grid baseline),
  SAC (RL baseline scaffold).
- **Metrics**: success/collision with Wilson 95 % CIs, ATE, mean ± 1.96·SEM
  on continuous metrics, `plan_dt` reported with *first-call dropped*
  (CUDA warmup is 10× steady-state and not representative of a warmed-up
  deployment).
- **Compare CLI**: `uav-nav compare A B` produces side-by-side tables
  with the indep^N baseline computed from the per-drone rates of run A
  vs run B at the same N.

## 3. Headline result (1½ page) — multi-drone GPU MPPI flips coordination Δ

`exp_multi_drone_3d_4.yaml` (MPC) vs `exp_multi_drone_3d_4_gpu_mppi.yaml`
(GPU MPPI). 4-drone cross, 40×40×12 voxel world, 30 random obstacles,
n=30 episodes per planner, same seeds.

| planner               | per-drone (CI)       | joint (CI)          | indep `per^4` | Δ over indep |
|---|---|---|---|---|
| MPC      (n=8,  h=40) | 93.8 % [90.9, 95.7]  | 78.0 % [68.9, 85.0] | 77.2 %        | +0.8 pp      |
| GPU MPPI (n=64, h=20) | 90.0 % [86.7, 92.6]  | 77.0 % [67.8, 84.2] | 65.6 %        | **+11.4 pp** |

Three observations at n=100 paired (n=30 pilot reported the same
direction but biased numbers — per-drone estimates were ~4 pp too
high; the Δ flip widens from +5.2 → +11.4 pp; see findings.md
§"Multi-drone: GPU MPPI's rollout cloud flips the coordination Δ"):

1. Per-drone differs by ~4 pp (CIs still overlap by 1.7 pp).
2. Joint succ is TIED (78.0 vs 77.0; McNemar same-seed pairing
   gives both-succ 67, MPC-only 11, GPU-only 10 — not significant).
3. Δ over `indep^4` separates +0.8 vs **+11.4 pp**. GPU MPPI's
   failures *cluster within seeds* (the n=30 commit's "decorrelate"
   read was the wrong sign; same joint, but very different failure
   shape).

Mechanistic claim (revised after n=100): GPU MPPI's softmax across
64 rollouts **amplifies seed sensitivity** — on easy seeds the
rollout cloud agrees on a clean escape volume and all 4 drones make
it; on hard seeds the same averaging produces overly conservative
commands and 2–4 drones collide together. MPC's argmin is more
individually brittle (lower joint, lower per-drone) but each drone's
brittleness is uncorrelated. Sample diversity is **not** a substitute
for peer prediction — it's a knob that **trades smoother typical
behaviour for harsher tail outcomes**. In deployments where partial
success counts (3 of 4 packages delivered), MPC's distributed-failure
shape wins; where the goal is "all or none", they're equivalent.

Side-by-side: `docs/images/compare_multi_drone_3d_mpc_vs_gpu_mppi.gif`.

## 4. Prerequisites that made the comparison fair (3 pages)

### 4.1 Pareto cells, not planners

- 2D MPC Pareto: sole 100 %/min-cost cell at (n=16, h=20). Longer
  horizons *hurt* success because the goal bonus fires less.
- 3D MPC Pareto: preference flips to small n (n=8, h=20). Section
  "3D Pareto: the n_samples preference flips".
- 2D GPU MPPI Pareto (post-fix): (n=128, h=40) → 100 % / 3.0 ms.
- 3D GPU MPPI Pareto (post-fix): (n=64–256, h=20) → 100 % / 3.5 ms.
  **Beats CPU MPC 3D by +12 pp at 20× lower plan time.**

Lesson: any single-cell comparison is suspect. Report the cell that won
its Pareto sweep, *and* re-run the comparison at the opposing planner's
optimum.

### 4.2 Implementation correctness gates every cell

Goal-mask bug in GPU MPPI: pre-fix, the batched rollout summed collision
penalties over *all* horizon steps including post-goal drift. Pre-fix
table showed 0 % at every h ≥ 40 cell and a 90 %/30 ms optimum.
Post-fix:

- The "speed collapse at h ≥ 40" finding was a *bug*, not an MPPI
  property.
- The 30 ms plan_dt was CUDA-warmup-inflated — actual steady-state is
  3 ms.
- The Pareto optimum shifted to **(n=128, h=40)** in 2D.

This is the load-bearing methods point: a single line of correctness
flipped the conclusion of every cell in a 12-cell sweep, without
changing any of the *measurements*.

### 4.3 Plan-time reporting on CUDA backends

Steady-state vs first-call: drop the first replan of every episode.
Multi-drone × CUDA = each per-drone planner pays its own warmup *every
episode*; a 4-drone single-episode benchmark cannot reach steady-state
and reports 73 ms / replan when single-drone reports 3.5 ms. This is
a benchmark artifact, not a planner property. The paper must either
warm up once at startup or amortize over many episodes per drone.

### 4.4 Sim transferability and ROS 2 invariance

- AirSim vs dummy_3d: same plan, different physics — quantify what
  carries (success rate, plan structure) and what does not (velocity
  profile, action-jump magnitude).
- Static-cube discriminating cell (§4.4.3): GPU MPPI removes every
  paired AirSim seed that MPC loses (McNemar $p \approx 0.008$), but
  GPU sits at the 100 % ceiling so its $\Delta$ is degenerate.
- Density-sweep cell `base_ew06` (§4.4.4): widening the EW pillars
  drops GPU MPPI off ceiling and re-measures $\Delta$ at n=50 paired.
  The $\Delta$-flip mechanism transfers from dummy_3d, but **the sign
  reverses** — on AirSim, MPC is the planner that clusters failures
  ($\Delta_\text{MPC} = +3.8$ pp vs $\Delta_\text{GPU} = -1.2$ pp at
  tied per-drone rates), driven by 3 multi-drone cluster seeds out of
  50 (6 % cluster rate).
- ROS 2 bridge: spatial equivalence verified within 0.2 m on a single
  episode. The bridge hop is invariant; the 2× wall-clock is the
  real-time clock constraint, not loss of fidelity.
- AirSim-over-ROS 2 parity harness: full chain AirSim → ROS 2 →
  bridge produces equivalent spatial behaviour to AirSim → direct,
  modulo real-time clock.

These together justify the §3 result: it was measured on
`dummy_3d` but the §4.4 evidence is what makes the finding portable —
with the §4.4.4 caveat that the *direction* of the $\Delta$ flip is
backend-dependent.

## 5. Secondary findings

Status: `section_5_secondaries.md` written. It keeps the §3 narrative
tight by using two companion ablations:

1. **3D escape volume / density ablation** — coordination Δ disappears
   in open 3D, then returns at intermediate density.
2. **3D peer-prediction ablation** — removing CV peer prediction at
   that intermediate density is as damaging as an 8× static-obstacle
   increase.

## 6. Limitations

- ~~n=30 episodes per multi-drone cell — joint succ CIs are wide
  ([66, 93] vs [70, 95]); the Δ flip is robust under per-seed paired
  comparison but the absolute joint succ difference is not significant
  at n=30 alone.~~ **Done — n=100 paired re-run** (`results/_multi_drone_n100_*`).
  Per-drone bias-corrected (95 % → 93.8 % MPC, 95 % → 90.0 % GPU MPPI);
  Δ flip widens **+5.2 → +11.4 pp**; joint succ rates tie at 77–78 %
  under McNemar same-seed pairing. The n=30 commit's "decorrelate"
  mechanism interpretation was the wrong sign — the corrected
  read ("seed-sensitivity amplification → failures cluster") is the
  current §3 reading.
- ~~4 drones only. The N-scaling curve (`exp_multi_drone_3d_N{4,6,8}`) is
  collected for MPC but not yet for GPU MPPI. **Future-work TODO: full
  N-scaling sweep on GPU MPPI.**~~ **Done — n=30 paired sweep at
  $N \in \{2, 3, 4, 6, 8, 10, 12\}$** (`exp_multi_drone_3d_{2,3,4,6,8,10,12}{,_gpu_mppi}.yaml`):
  GPU MPPI's higher-$\Delta$ advantage is **non-monotonic in N** —
  holds at N=4 / N=6 / **N=10 (sweep max $\Delta = +24.3$ pp)**,
  reverses at N=2 (MPC argmin beats softmax on head-on, p ≈ 0.008),
  at N=8 (GPU per-drone uniquely collapses, p ≈ 0.0001), and at N=12
  (GPU $\Delta$ drops to +7.8). The §3 N=4 result is one point on a
  multi-regime curve, not a clean monotonic law.
- **Density sweep at $N \in \{4, 6\}$**
  (`exp_multi_drone_3d_{4,6}{,_dense,_packed}{,_gpu_mppi}.yaml`): at
  N=4 the $\Delta$ sign flips with obstacle count (GPU clusters at
  baseline → MPC clusters at packed), reproducing the AirSim
  base_ew06 sign-reversal in dummy_3d. At N=6 the same density sweep
  does *not* flip — GPU MPPI's per-drone advantage opens up across
  density (74 % vs 42 % at packed) so GPU wins joint through
  per-drone alone, not through the cluster mechanism. The sign-flip
  is conditional on per-drone rates staying close as density rises,
  which holds at N=4 but not at N=6. AirSim base_ew06 ≈ the N=4
  dense regime; it is one paired cell, not an AirSim-wide claim.
- ~~Temperature ablation: 3D GPU MPPI T=0.1 vs T=1.0 has overlapping
  CIs. The §3 result uses T=1.0 only.~~ **Done — see
  `exp_gpu_mppi_temp_ablation_3d.yaml` and findings.md §"Temperature
  ablation at the 3D Pareto cell".** Result: T=1.0 default is robust
  (T=0.1 is significantly worse, Fisher exact p ≈ 0.02; T=0.3 ties
  T=1.0). The multi-drone Δ-flip finding is unchanged.
- RL baseline (SAC) requires 100k–500k timesteps to reach a comparable
  number; the scaffold is functional but the comparison is not yet
  paper-worthy.
- ~~AirSim + GPU MPPI stack test exists in isolation (`demo_airsim` flow
  with the new planner)~~ **Done — single-drone (see
  `exp_airsim_demo_gpu_mppi.yaml`) and multi-drone (`exp_airsim_multi_demo_gpu_mppi.yaml`)
  both run end-to-end on AirSim, all drones reach goal at 4/4.**
  Single-drone parity: GPU MPPI ~30 % slower wall-clock (1.87 vs
  2.40 m/s), plan_dt dominated by sim overhead so the dummy_3d
  speed edge is not portable. Multi-drone parity at n=1: same shape +
  qualitative signal that per-drone final_t spread (0.55 s GPU vs
  0.05 s MPC) is preserved on AirSim. **n=30 paired done across
  altitude-only and static-cube geometries:**
  - Staggered altitude (±2-4 m): both planners hit 100 % joint
    success across 30 paired seeds — scenario ceiling-limited, the
    failure-level Δ-flip mechanism cannot register. Trajectory-level
    signal IS preserved (0.02 s vs 0.55 s arrival spread).
  - Uniform altitude (all z=30): MPC holds **46.7 %** [30.2, 63.9]
    joint, GPU MPPI **collapses to 0/30 = 0.0 %** joint
    (28.3 % per-drone vs 65.0 % MPC). McNemar paired exact
    p ≈ 0.00012 — the only AirSim cell where the planner
    comparison rejects the null. GPU MPPI's softmax-conservative
    1.87 m/s through a 4-way crossing leaves drones at the centre
    long enough for most of them to collide.
  - Mid-stagger ±1 m (z=29/31/30.5/29.5): both planners 100 %
    again. Tightest pair has 0.5 m vertical separation, clearing
    AirSim mesh radii by 0.1 m — picked deliberately to land in
    the discriminating per-drone 60-90 % band, but the response
    is essentially bimodal at the AirSim multi-drone task:
    every non-zero z-spread stays at the 4/4 ceiling; only
    z-coincidence drops it. Trajectory-spread ratio is preserved
    (GPU/MPC = 4× here vs 27× at ±2-4 m), so the softmax
    *trajectory-level* mechanism is robust across cells; only
    the *failure-level* Δ-flip cannot be exercised on the
    no-obstacle scenario.
  - Static-cube discriminating cell
    (`exp_airsim_multi_discriminating_n30*.yaml`): bridge-spawned
    Blocks cubes plus matching planner occupancy put MPC in the
    target band: per-drone **87.5 %** [80.4, 92.3], joint
    **22/30 = 73.3 %** [55.6, 85.8], Δ +14.7 pp. GPU MPPI clears
    **30/30 = 100 %** joint; McNemar paired exact p ≈ 0.008
    (GPU-only success 8, MPC-only 0).
  - **Combined reading**: the AirSim failure-level gap is no longer
    open — static cubes produce a significant paired planner
    separation. What remains open is narrower: the exact dummy_3d
    joint-tie / larger-GPU-Δ signature is still not directly measured
    on AirSim because the static-cube cell sends GPU MPPI to the
    100 % ceiling. Future-work TODO: static-cube density / placement
    sweep that also drops GPU MPPI into the 60-90 % per-drone band.

## 7. Reproducibility map (appendix)

Status: `section_7_repro_map.md` written.

The appendix maps each paper section to its exact `examples/exp_*.yaml`
artifact, any required runner script, and the matching `docs/findings.md`
anchor. It also corrects the stale outline placeholders:

- 3D MPC Pareto uses `examples/exp_3d_predictive.yaml`.
- dummy_3d ↔ AirSim transfer uses `examples/exp_transfer_dummy.yaml`
  and `examples/exp_transfer_airsim.yaml`.
- §5 density ablation uses `examples/exp_multi_drone_3d_4_dense.yaml`
  and `examples/exp_multi_drone_3d_4_packed.yaml`.
- §5 peer-prediction ablation uses
  `examples/exp_multi_drone_3d_4_dense_indep.yaml` and
  `examples/exp_multi_drone_3d_4_packed_indep.yaml`.

## 8. Open decisions before submission

1. **Title framing** (option 1 / 2 / 3 above).
2. **Venue target** — methods workshop vs robotics conference. Drives
   page budget and how much of §4 stays in main text vs appendix.
3. **Main-text budget** — whether §5 stays in the main paper or moves
   partly to appendix once the target venue page limit is known.
4. **AirSim caveat strength** — current text reports a significant
   static-cube planner separation but leaves the exact dummy_3d
   joint-tie / larger-GPU-Δ transfer as future work. A density sweep
   could either strengthen or simplify that caveat.

## 9. Next concrete steps

In approximate dependency order:

1. Run the AirSim static-cube density / placement sweep from
   `plan.md` §2.1 if the paper needs a direct AirSim test of the
   dummy_3d Δ mechanism.
2. Pick title and venue target; then trim §4-§5 into main text vs
   appendix according to page budget.
3. Convert the section drafts into a single manuscript file and add
   figure/table references.
4. Decide whether the SAC scaffold remains a limitation note or gets
   promoted to a short baseline appendix after a longer training run.
