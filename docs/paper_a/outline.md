# Paper A — Outline (draft)

Status: skeleton only. Goal of this file is to lock the spine before we
expand sections. Source of truth for individual findings remains
`docs/findings.md`; this file decides *which* findings go in, in what
order, and under what argument.

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
1** for a methods venue, **option 3** for a workshop. Decide before §1 is
written.

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
- ROS 2 bridge: spatial equivalence verified within 0.2 m on a single
  episode. The bridge hop is invariant; the 2× wall-clock is the
  real-time clock constraint, not loss of fidelity.
- AirSim-over-ROS 2 parity harness: full chain AirSim → ROS 2 →
  bridge produces equivalent spatial behaviour to AirSim → direct,
  modulo real-time clock.

These three together justify the §3 result: it was measured on
`dummy_3d` but the §4.4 evidence is what makes the finding portable.

## 5. Secondary findings (1 page each, candidates — pick 2)

Candidates, ranked by how much they reinforce the §3 narrative:

1. **3D escape volume erases coordination Δ** — companion to §3. Same
   scenario at lower density: Δ vanishes, regardless of planner. Sets
   the boundary of when §3's claim applies.
2. **3D peer-prediction ablation** — removing CV prediction is worse
   than 8× obstacle density. Establishes that *some* peer information
   is load-bearing even in escape volume; §3 then shows MPPI substitutes
   sample diversity for it.
3. **Action-jump cost tuning beats every smoothing layer** — orthogonal
   methods finding. Strong on its own but not aligned with §3's
   coordination story.
4. **Wind miscalibration: planner belief must match sim reality** —
   sim-side robustness finding. Pairs with §4.4 if we want sim
   transferability to be a recurring sub-theme.

Recommend: **(1) + (2)** for option-2 framing, **(2) + (3)** for option-1
framing.

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
- 4 drones only. The N-scaling curve (`exp_multi_drone_3d_N{4,6,8}`) is
  collected for MPC but not yet for GPU MPPI. **Future-work TODO: full
  N-scaling sweep on GPU MPPI.**
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
  speed edge is not portable. Multi-drone parity: same shape, +
  qualitative signal that trajectory **decorrelation** is visible
  even at 4/4 (GPU MPPI per-drone final_t spread 0.55 s vs MPC's
  0.05 s) — mechanism consistent with the dummy_3d Δ flip. **Remaining
  future-work TODO: n ≥ 30 paired AirSim multi-drone run to put a
  number on whether the +5.2 pp Δ flip survives the AirSim speed
  penalty (~100 min wall-clock, overnight study).**

## 7. Reproducibility map (appendix)

| paper section | YAML(s) | findings.md anchor |
|---|---|---|
| §3 multi-drone Δ flip | `exp_multi_drone_3d_4.yaml`, `exp_multi_drone_3d_4_gpu_mppi.yaml` | "Multi-drone: GPU MPPI's rollout cloud flips the coordination Δ" |
| §4.1 2D MPC Pareto | `exp_predictive.yaml` | "MPC compute Pareto" |
| §4.1 3D MPC Pareto | `exp_predictive_3d.yaml` | "3D Pareto: the n_samples preference flips" |
| §4.1 2D GPU MPPI Pareto | `exp_gpu_mppi_pareto.yaml` | "2D Pareto (post-fix)" |
| §4.1 3D GPU MPPI Pareto | `exp_gpu_mppi_pareto_3d.yaml` | "3D Pareto (post-fix)" |
| §4.1 3D GPU MPPI T-ablation | `exp_gpu_mppi_temp_ablation_3d.yaml` | "Temperature ablation at the 3D Pareto cell" |
| §4.2 goal-mask fix | commit `2a9d196` + `uav_nav_lab/planner/gpu_mppi.py` | "The goal-mask bug fix that changed every cell" |
| §4.4 AirSim vs dummy_3d | `exp_airsim_transfer.yaml` (TBD) | "AirSim vs dummy_3d transferability" |
| §4.4 AirSim + GPU MPPI parity (single) | `exp_airsim_demo_gpu_mppi.yaml` | "AirSim + GPU MPPI parity" |
| §4.4 AirSim + GPU MPPI parity (multi) | `exp_airsim_multi_demo_gpu_mppi.yaml` | "AirSim multi-drone parity" |
| §4.4 ROS 2 bridge | `scripts/ros2_dummy_sim.py` + `exp_basic.yaml` | "ROS 2 bridge: spatial equivalence verified" |
| §4.4 AirSim + ROS 2 | `exp_airsim_ros2.yaml`, `exp_airsim_ros2_direct.yaml` | "AirSim over ROS 2 parity harness" |
| §5 escape volume | `exp_multi_drone_3d_4.yaml` (sparse), `_density_8x` variant | "3D density ablation" |
| §5 peer-prediction | `exp_multi_drone_3d_4_no_peer.yaml` | "3D peer-prediction ablation" |

## 8. Open decisions to resolve before §1 is written

1. **Title framing** (option 1 / 2 / 3 above).
2. **Venue target** — methods workshop vs robotics conference. Drives
   page budget and how much of §4 stays in main text vs appendix.
3. **Which §5 pair** — (1)+(2) coordination-focused vs (2)+(3) methods-
   focused.
4. **§6 follow-ups** — which TODOs to land before submission vs leave
   for future work. n=100 paired multi-drone re-run is the most
   defensible to add; T-sweep and AirSim-multi-drone are nice-to-haves.

## 9. Next concrete steps

In approximate dependency order:

1. Decide §8.1–§8.3 (15 min, user call).
2. Run n=100 paired multi-drone MPC vs GPU MPPI (overnight; new seeds
   42–141). Decide if Δ flip survives at narrower CI.
3. Run 3D GPU MPPI T-ablation at the Pareto cell (`temperature ∈
   {0.1, 0.3, 1.0, 3.0}` × seeds). Decide T choice for §3.
4. Run AirSim multi-drone GPU MPPI (the missing portability check for
   §4.4 / §3).
5. Draft §1 (motivation) + §3 (headline) in full prose. The rest can
   stay outline-form until those two are agreed.
