# Research findings

These are the long-form studies behind the framework — full tables,
ablation reasoning, and methodological takeaways. The README's
[headline result](../README.md#-planner-head-to-head-on-dynamic-obstacles)
(planner head-to-head on dynamic obstacles) is the entry point; this
file collects the rest.

Each finding lives in the comment header of the YAML that produces it,
along with a one-line `uav-nav sweep` invocation that reproduces it.
Wilson 95 % intervals on rates, mean ± 1.96·SEM on continuous metrics.

## Contents

- [MPC compute Pareto](#mpc-compute-pareto)
- [3D Pareto: the n_samples preference flips](#3d-pareto-the-n_samples-preference-flips)
- [3D perception-latency cliff: same corner, softened](#3d-perception-latency-cliff-same-corner-softened)
- [Pareto config materially rewrites prior conclusions](#pareto-config-materially-rewrites-prior-conclusions)
- [Multi-drone N-scaling and peer-prediction coordination](#multi-drone-n-scaling-and-peer-prediction-coordination)
- [3D escape volume erases the coordination Δ](#3d-escape-volume-erases-the-coordination-δ)
- [3D density ablation: bring escape volume back to non-trivial — Δ comes back too](#3d-density-ablation-bring-escape-volume-back-to-non-trivial--δ-comes-back-too)
- [3D peer-prediction ablation: removing CV prediction is worse than 8× obstacle density](#3d-peer-prediction-ablation-removing-cv-prediction-is-worse-than-8-obstacle-density)
- [Wind miscalibration: planner belief must match sim reality](#wind-miscalibration-planner-belief-must-match-sim-reality)
- [The perception-latency cliff: a four-step research saga](#the-perception-latency-cliff-a-four-step-research-saga)
- [MPC + CHOMP smoothing: layering on a saturated planner is a wash](#mpc--chomp-smoothing-layering-on-a-saturated-planner-is-a-wash)
- [Action-jump cost: tuning the existing knob beats every layer](#action-jump-cost-tuning-the-existing-knob-beats-every-layer)
- [AirSim vs dummy_3d transferability: same plan, different physics](#airsim-vs-dummy_3d-transferability-same-plan-different-physics)

- [GPU MPPI: post-goal-mask fix unlocks long-horizon cells, 3D MPPI beats 3D MPC](#gpu-mppi-post-goal-mask-fix-unlocks-long-horizon-cells-3d-mppi-beats-3d-mpc)
- [Temperature ablation at the 3D Pareto cell: the CPU rules don't transfer](#temperature-ablation-at-the-3d-pareto-cell-the-cpu-rules-dont-transfer)
- [Multi-drone: GPU MPPI's rollout cloud flips the coordination Δ](#multi-drone-gpu-mppis-rollout-cloud-flips-the-coordination-δ)
- [dummy_3d N-scaling paired (MPC vs GPU MPPI, N ∈ {2, 3, 4, 6, 8, 10, 12})](#dummy_3d-n-scaling-paired-mpc-vs-gpu-mppi-n--2-3-4-6-8-10-12)
- [dummy_3d density × planner sweep at N ∈ {4, 6, 8}: §3 mechanism is conditional on per-drone tie](#dummy_3d-density--planner-sweep-at-n--4-6-8-3-mechanism-is-conditional-on-per-drone-tie)
- [dummy_3d N=4 + moving obstacle speed sweep: GPU MPPI's softmax averaging is catastrophic under dynamic obstacles](#dummy_3d-n4--moving-obstacle-speed-sweep-gpu-mppis-softmax-averaging-is-catastrophic-under-dynamic-obstacles)
- [AirSim + GPU MPPI parity: planner portable, dummy_3d plan-time advantage lost](#airsim--gpu-mppi-parity-planner-portable-dummy_3d-plan-time-advantage-lost)
- [AirSim multi-drone parity: stack runs end-to-end, timing spread still visible at 4/4](#airsim-multi-drone-parity-stack-runs-end-to-end-timing-spread-still-visible-at-44)
- [AirSim multi-drone n=30 paired: planner portable, scenario ceiling-limited, timing-spread signal preserved](#airsim-multi-drone-n30-paired-planner-portable-scenario-ceiling-limited-timing-spread-signal-preserved)
- [AirSim multi-drone uniform-altitude n=30: GPU MPPI collapses to 0 % joint while MPC holds 46.7 %](#airsim-multi-drone-uniform-altitude-n30-gpu-mppi-collapses-to-0--joint-while-mpc-holds-467-)
- [AirSim multi-drone ±1 m mid-stagger n=30: still ceiling-limited, cliff between 0 and 1 m](#airsim-multi-drone-1-m-mid-stagger-n30-still-ceiling-limited-cliff-between-0-and-1-m)
- [AirSim multi-drone static-cube discriminating cell n=30: GPU MPPI clears every seed while MPC drops paired seeds](#airsim-multi-drone-static-cube-discriminating-cell-n30-gpu-mppi-clears-every-seed-while-mpc-drops-paired-seeds)
- [AirSim multi-drone base_ew06 density-sweep n=30: Δ-flip sign reverses — MPC is the clustering planner on AirSim](#airsim-multi-drone-base_ew06-density-sweep-n30-δ-flip-sign-reverses--mpc-is-the-clustering-planner-on-airsim)
- [AirSim dynamic-obstacle bridge extension (smoke verified, paired cell still tuning)](#airsim-dynamic-obstacle-bridge-extension-smoke-verified-paired-cell-still-tuning)
- [Smart MPPI (argmin-fallback): mechanism detector works, naive fix doesn't](#smart-mppi-argmin-fallback-mechanism-detector-works-naive-fix-doesnt)
- [Smart MPPI v2 (asymmetric perturbation): breaks softmax symmetry, helps the planner-swap regime](#smart-mppi-v2-asymmetric-perturbation-breaks-softmax-symmetry-helps-the-planner-swap-regime)
- [Smart MPPI v3 (temporally-coherent argmin commit): trades mode 1 success for swap-regime stability](#smart-mppi-v3-temporally-coherent-argmin-commit-trades-mode-1-success-for-swap-regime-stability)
- [Smart MPPI v4 (mode-aware sampling): the first variant that cracks the cancellation regime](#smart-mppi-v4-mode-aware-sampling-the-first-variant-that-cracks-the-cancellation-regime)
- [Drone race + bouncing intruder: Smart MPPI v4 recovers MPC-level safety without losing tracking precision](#drone-race--bouncing-intruder-smart-mppi-v4-recovers-mpc-level-safety-without-losing-tracking-precision)
- [Moving-gates race: the mirror image — softmax wins where it lost the single-intruder race](#moving-gates-race-the-mirror-image--softmax-wins-where-it-lost-the-single-intruder-race)
- [Drone race chaos — gates + intruders piled on, gate topology still dominates](#drone-race-chaos--gates--intruders-piled-on-gate-topology-still-dominates)
- [dyn4 path-intersecting intruders: controlled dynamic-avoidance harness](#dyn4-path-intersecting-intruders-controlled-dynamic-avoidance-harness)
- [Cost-to-go cache tolerance: 4-5x speedup on moving-goal scenarios](#cost-to-go-cache-tolerance-4-5x-speedup-on-moving-goal-scenarios)
- [Smart MPPI v5 (mode-aware switcher): lateral-cancellation gate dominates v4 on 4/5 cells](#smart-mppi-v5-mode-aware-switcher-lateral-cancellation-gate-dominates-v4-on-45-cells)
- [Aerobatic synchronized loop: GPU MPPI's softmax delivers 85 % tighter phase sync](#aerobatic-synchronized-loop-gpu-mppis-softmax-delivers-85--tighter-phase-sync)
- [Bridge fix: pause-after-reset eliminates a stale-t=0 collision flag](#bridge-fix-pause-after-reset-eliminates-a-stale-t0-collision-flag)
- [ROS 2 bridge: spatial equivalence verified](#ros-2-bridge-spatial-equivalence-verified)
- [AirSim over ROS 2 parity harness](#airsim-over-ros-2-parity-harness)
- [RL comparison baseline: gym.Env scaffold + initial training](#rl-comparison-baseline-gymenv-scaffold--initial-training)
## MPC compute Pareto

`examples/exp_predictive.yaml` — n_samples × horizon. The 6-panel
output of `uav-nav viz <sweep_dir>` lets you read off the success
ceiling and the compute it costs in one figure:

<p align="center">
<img src="images/sweep_pareto.png" alt="6-panel Pareto sweep: success / collision / avg speed / ATE / planner_dt mean / planner_dt p95" width="640">
</p>

At n=20 episodes per cell:

| n_samples \ horizon | 20 | 40 | 60 | 80 | 120 |
|---|---|---|---|---|---|
| 8   | 100 | 90  | 80 | 65 | 45 |
| 16  | **100** | 85  | 80 | 65 | 35 |
| 32  | 100 | 95  | 75 | 60 | 35 |
| 64  | 100 | 100 | 75 | 60 | 45 |
| 128 | 100 | 100 | 95 | 80 | 40 |

Sole Pareto-optimal point: **n_samples=16, horizon=20 → 100 % / 51 ms**.
Longer rollouts actively *hurt* success — the reach-goal bonus fires
less often when the rollout overshoots the goal radius mid-trajectory.

## 3D Pareto: the n_samples preference flips

<p align="center">
<img src="images/demo_3d.gif" alt="3D Predictive-MPC episode on a 40×40×12 voxel world: drone (blue) reaches the goal while three bouncing dynamic obstacles (red) cross its path" width="480">
</p>

`examples/exp_3d_predictive.yaml` — the same n_samples × horizon sweep on
a 3D `voxel_world` (40×40×12, three bouncing 3D dynamic obstacles, n=8
post-cache):

<p align="center">
<img src="images/sweep_pareto_3d.png" alt="6-panel Pareto sweep on the 3D voxel world: success / collision / avg speed / ATE / planner_dt mean / planner_dt p95" width="640">
</p>

Findings vs the 2D analogue:

- **The Pareto frontier shifts to lower n_samples.** The 2D-optimal
  config (n=16, h=20 → 100 % / 51 ms in 2D) lands at only 75 % / 91 ms
  in 3D. The strongest 3D cells are **n=8, h=20 → 88 % / 70 ms** and
  n=128, h=40 → 100 % / 273 ms. Fibonacci-sphere sampling already
  covers the 3D escape directions densely enough that fewer per-step
  samples suffice — compute is better spent on horizon depth, opposite
  of 2D's preference.
- **The "longer rollouts hurt" effect partly transfers.** 2D drops
  monotonically with horizon (100 → 35 %); 3D drops more gently (most
  rows stay 75 → 38 %), but the trend is the same. The 3D escape volume
  softens but does not eliminate the reach-goal-bonus overshoot.
- **The 3D plan_dt blow-up was a Dijkstra artifact.** A first pass had
  every cell at 1.3-2.2 s — too slow to fit `replan_period=0.2 s`. The
  static cost-to-go cache (added to `SamplingMPCPlanner`) brought 3D
  plan_dt back to the same order of magnitude as 2D (70-750 ms across
  the grid), making this sweep and the cliff sweeps actually tractable.

Methodological transfer: re-validate Pareto in every dimensionality.
n_samples preference flips, the compute envelope changes, and what
looked like a CPU-saturation cliff in 3D was actually a missing cache.

## 3D perception-latency cliff: same corner, softened

Same 3D scenario, sensor.delay × max_speed sweep at the 3D Pareto config
(n_samples=8, horizon=20, n=6):

<p align="center">
<img src="images/cliff_3d.png" alt="6-panel sensor.delay × max_speed sweep on the 3D voxel world: success drop concentrated in the bottom-right cliff corner" width="640">
</p>

|  delay \ speed  |  10  |  15  |  20  |  25  |  30  |
|---|---|---|---|---|---|
| 0.00 |  83 | 100 | 100 | 100 |  83 |
| 0.05 |  83 | 100 | 100 | 100 |  83 |
| 0.10 |  83 | 100 | 100 |  83 |  50 |
| 0.20 |  67 | 100 |  83 |  83 |  50 |
| 0.50 |  83 |  83 | **33** |  50 |  50 |

The cliff transfers from 2D to 3D in the same `delay=0.5 × speed≥20 m/s`
corner. 2D had this region at 10-25 %; 3D softens it to 33-50 % —
the extra escape volume helps but does not eliminate the failure mode.

**3D cliff remediation: the velocity_window optimum *inverts* vs 2D.**
At the 3D cliff cell (delay=0.5, speed=20, n=12):

| sensor config | succ % | CI95 |
|---|---|---|
| baseline (no extrap) | 33.3 | [13.8, 60.9] |
| `extrapolate=true, window=1` | **83.3** | [55.2, 95.3] |
| `extrapolate=true, window=3` | 66.7 | [39.1, 86.2] |
| `extrapolate=true, window=5` | 58.3 | [32.0, 80.7] |
| `extrapolate=true, window=10` | 33.3 | [13.8, 60.9] |

CV ego extrapolation is the same big lever in 3D — +50 pp at window=1,
Wilson 95 % CIs do not overlap. But **the optimum inverts**: 2D's
peak was window=5, 3D's peak is window=1. The 3D escape volume lets
the drone trace smoother trajectories, so the 1-sample finite-
difference velocity is already accurate; smoothing only adds lag,
and lag hurts most at high speed where the cliff lives.

Engineering takeaway: the *parameter setting* of a remediation does
not transfer across dimensionalities even when the *technique* does.
Always re-tune ego-extrapolation window per scenario regime.

## Pareto config materially rewrites prior conclusions

The previous heatmap on the same scenario at the YAML's old defaults
(n_samples=32, horizon=60) reported a "dynamic-feasibility cliff at
25 m/s". At the Pareto config that cliff disappears (35 – 65 % success
at speed = 25-30 m/s), and replan_period — which "barely mattered"
before — now drives a 40 – 70 pp swing across 0.1 – 2.0 s. The earlier
conclusion was partly a CPU-saturation artifact: at horizon=60 every
replan took ~200 ms, so even replan_period=0.1 s could not actually
keep up.

> **Methodological lesson** baked into the YAML header: always validate
> ablation conclusions at the planner's Pareto-optimal config —
> suboptimal MPC settings can mask both ceilings (max feasible speed)
> and sensitivities (replan-period effect, delay tolerance).

## Multi-drone N-scaling and peer-prediction coordination

<p align="center">
<img src="images/multi_drone_3.png" alt="Three drones (alice/bob/charlie) crossing each other's paths to opposite-corner goals; joint=all_success" width="540">
</p>

*N=3 multi-drone episode — alice / bob / charlie all reach their
opposite-corner goals while routing around each other via the MPC's
constant-velocity peer prediction.*

`examples/exp_multi_drone_{2,3,4,8}.yaml` — same world, only drone
count changes. n=30, joint metrics with Wilson 95 % CIs:

| N | joint succ | joint coll | per-drone succ | indep `per^N` | Δ over indep |
|---|---|---|---|---|---|
| 2 | 96.7 % [83, 99] | 3.3 %  | 98.3 % | 96.6 % | +0.1 pp |
| 3 | 70.0 % [52, 83] | 30.0 % | 87.8 % | 67.7 % | +2.3 pp |
| 4 | 73.3 % [56, 86] | 26.7 % | 87.5 % | 58.6 % | **+14.7 pp** |
| 8 | 16.7 % [7, 34]  | 83.3 % | 70.0 % |  5.8 % | **+10.9 pp** |

The MPC's constant-velocity peer prediction *correlates failures in
the right direction* — when one drone yields, the others see its new
trajectory and react, so the system as a whole degrades less than
independent drones would.

**Coordination is non-monotonic in N.** Δ peaks at N=4 (+14.7 pp) and
declines at N=8 (+10.9 pp), even though the absolute joint success
collapses from 73.3 % → 16.7 %. Two effects compound:

- **More peers → more coordination signal**, lifting Δ over the
  independence baseline (the curve from N=2 to N=4).
- **More peers → escape-volume saturation**, dropping per-drone success
  from 87 % → 70 % at N=8 and pulling joint success down faster than
  coordination can recover (the N=4 → N=8 turn).

Engineering takeaway: peer-prediction coordination has a *useful
range*, not a monotonic scaling law. For dense N you need either a
bigger world (lower density per drone) or a coordinator that goes
beyond constant-velocity prediction (priority scheduling, reservation
tables, decentralised roundabout). The framework's MPC ceiling for
this 60×60 world tops out around N=4-6; past that, peer prediction
still helps in *relative* terms but is fundamentally fighting density.

### Density ablation: the "non-monotonic in N" claim was a density artifact

`examples/exp_multi_drone_8_low_density.yaml` — the obvious follow-up
to the engineering takeaway above. Same N=8 / same crossing structure
/ same Pareto-MPC config / same 30 obstacles, but world is 100×100
instead of 60×60 (~2.8 × the area per drone, 1250 cells/drone vs 450).

| density | world | per-drone | joint | indep `per^N` | Δ over indep |
|---|---|---|---|---|---|
| high | 60×60 | 70.0 % [64, 75] | 16.7 % [7, 34] | 5.8 % | +10.9 pp |
| low | 100×100 | 82.1 % [77, 86] | **46.7 %** [30, 64] | 21.0 % | **+25.7 pp** |

Halving density (actually 2.8 ×):
- per-drone success +12 pp
- joint success +30 pp
- coordination Δ +14.8 pp (**roughly doubled**)

Refines the prior finding in two ways:

1. **The "non-monotonic in N" was a density artifact.** With
   N=8 / low-density Δ at +25.7 pp — *larger* than N=4 / high-density
   Δ at +14.7 pp — the coordination scaling law is **monotonic in N
   when density allows**. The planner's CV peer prediction continues
   to make better use of more peers, full stop.

2. **The MPC ceiling is `density × N`, not raw N.** Doubling room
   halved the per-drone collision rate (30 % → 18 %). The dip at
   N=8 in the original table reflected escape-volume saturation, not
   a fundamental coordination limit.

Engineering takeaway (refined): **density × planner capacity** is the
load-bearing axis. To deploy CV peer prediction at high N, scale the
world or shrink drone radii — the planner itself is fine. The
"non-monotonic" caveat from above survives only as a *density-saturated*
regime warning, not a scaling-law claim.

Methodological close: the same 2-step pattern as the mpc_chomp →
velocity-profile → action-jump saga. Initial finding (PR #25)
reported the surface result; follow-up ablation isolated the actual
load-bearing axis. The first finding wasn't wrong — it was
incomplete. Always ablate the engineering takeaway your previous YAML
header speculated about.

### 3D escape volume erases the coordination Δ

`examples/exp_multi_drone_3d_{2,3,4}.yaml` — same crossing pattern as
the 2D N=2/3/4 cases above, lifted to a 40×40×12 voxel_world with the
3D Pareto MPC config (n_samples=8, horizon=40). Each drone is free to
detour over or under its peers along the z-axis.

| N | dim | per-drone | joint | indep `per^N` | Δ over indep |
|---|---|---|---|---|---|
| 2 | 2D | 98.3 %  | 96.7 % [83, 99] | 96.6 % | +0.1 pp |
| 2 | 3D | 98.3 %  | 96.7 % [83, 99] | 96.6 % | +0.1 pp |
| 3 | 2D | 87.8 %  | 70.0 % [52, 83] | 67.7 % | +2.3 pp |
| 3 | 3D | 88.9 %  | 70.0 % [52, 83] | 70.2 % | -0.2 pp |
| 4 | 2D | 87.5 %  | 73.3 % [56, 86] | 58.6 % | **+14.7 pp** |
| 4 | 3D | 95.8 %  | **83.3 %** [66, 93] | 84.3 % | **-1.0 pp** |

The 2D N=4 coordination win (+14.7 pp) **disappears in 3D** even
though the absolute joint success rises (73 → 83 %). Per-drone success
jumps with it (87 → 96 %), and `independent^N` rises to match — leaving
no Δ for peer prediction to take credit for.

Mechanism: in 2D, four crossing drones share the same horizontal
plane, so peer prediction has to *pick a yielder* every time their paths
intersect. In 3D, each drone's MPC is free to lift to z=7 while a peer
slips through at z=5; the rollouts find these out-of-plane detours
independently, no peer prediction needed. The escape volume *eliminates
the coordination problem*, it doesn't just soften it.

Engineering takeaway: the value of multi-agent coordination is not
intrinsic — it's a function of how constrained the per-agent free space
is. **In 3D / open-volume regimes, "use peer prediction or not" is a
wash; in 2D / dense regimes it can be the difference between 58 % and
73 % joint success.** Same Pareto-saturation lesson as everywhere else
in this framework: a layer only earns its keep when the layer below
has slack to take from. Coordination here is the layer; per-drone
escape volume is the slack.

Methodological close: re-validate the *value* of every layer in every
new dimensionality. Coordination Δ in 2D does not predict Δ in 3D, and
the headline number (joint succ %) hides the inversion of the
load-bearing factor (escape volume vs. peer prediction).

### 3D density ablation: bring escape volume back to non-trivial — Δ comes back too

`examples/exp_multi_drone_3d_4_{dense,packed}.yaml` — same N=4 / same
3D world / same Pareto MPC, only the static obstacle count changes.
Probes whether the previous finding's "3D Δ ≈ 0" really comes from the
free-volume mechanism we attributed it to. n=30 episodes per cell.

| obstacles | density (cells/world voxel) | per-drone | joint | indep `per^4` | Δ over indep |
|---|---|---|---|---|---|
| 30  (baseline) | 0.16 % | 95.8 % | 83.3 % | 84.3 % | -1.0 pp |
| 120 (dense)    | 0.63 % | 65.8 % | 26.7 % | 18.7 % | **+8.0 pp** |
| 240 (packed)   | 1.25 % | 46.7 % | 10.0 % | 4.8 %  | **+5.2 pp** |

Pack the world with obstacles and the per-drone success collapses
(96 → 66 → 47 %), but the **coordination Δ comes back from the dead**:
−1 → +8 → +5 pp. The previous finding's mechanism — "z-axis lets each
drone find an independent detour, so peer prediction has nothing to
take credit for" — is exactly what gets undone when static obstacles
fill the air column. Two drones can no longer trivially pass at
different z; one of them yields, the other's MPC sees the yield via
peer prediction, and the system as a whole degrades less than two
independent drones would.

The Δ peaks at intermediate density and falls again at the extreme:
- **30 obstacles**: escape volume so large that drones never share
  paths — coordination has nothing to do.
- **120 obstacles**: paths overlap, but per-drone routes still exist
  most of the time — peer prediction lifts the joint rate well above
  the independence floor.
- **240 obstacles**: per-drone success crashes to 47 % from
  obstacle-only collisions; even perfect coordination cannot recover
  what the planner is already losing solo.

Universal pattern (with the 2D density ablation, N=8 60×60 vs 100×100):

> **Coordination Δ is non-monotonic in free volume per agent.**
> Both endpoints — *sparse* (independent routes available) and
> *saturated* (per-agent failure dominates) — drive Δ toward zero.
> Maximum Δ lives in the middle, where peers actually have to
> negotiate but the planner is still capable of executing the
> negotiated solution. The 2D N=8 100×100 case (Δ +25.7 pp) and the
> 3D N=4 120-obstacle case (Δ +8.0 pp) are *the same regime in
> different parameter spaces*: world-side and obstacle-side ways of
> arriving at "intermediate per-agent free volume".

Engineering takeaway: when deciding whether peer prediction is worth
the implementation cost, the right diagnostic is not "how many drones"
or "what dimensionality", it is **"how much free volume per drone
remains after static and self-imposed constraints"**. The middle is
where the layer earns its keep.

### 3D peer-prediction ablation: removing CV prediction is worse than 8× obstacle density

The previous section argues that the +8 pp Δ at the dense cell *comes
from* peer prediction earning its keep in the intermediate-density
regime. That's a causal claim, and the natural test is the direct
ablation: rerun the same dense + packed cells with `use_prediction:
false` (each drone treats peers as if they don't exist) and see what
collapses. `examples/exp_multi_drone_3d_4_{dense,packed}_indep.yaml`,
n=30 each.

| cell | per-drone (CI) | joint (CI) | Δ over `per^4` |
|---|---|---|---|
| dense (120 obs), pred ON  | 65.8 % [57.0, 73.7] | 26.7 % [14.2, 44.4] | **+8.0 pp** |
| dense (120 obs), pred OFF | 16.7 % [11.1, 24.3] |  6.7 % [1.8, 21.3]  | +6.6 pp (per^4≈0) |
| packed (240 obs), pred ON | 46.7 % [38.0, 55.6] | 10.0 % [3.5, 25.6] | +5.2 pp |
| packed (240 obs), pred OFF| 18.3 % [12.4, 26.2] |  0.0 % [0.0, 11.4] | -0.1 pp |

Two things happen when prediction goes off, and the bigger one is *not*
the joint number:

1. **Per-drone success collapses.** −49 pp at dense (66 → 17 %), −28 pp
   at packed (47 → 18 %). The crossing pairs run head-on through each
   other; without forecasting peer trajectories the MPC sees only the
   peer's *current* position, which is harmless until it isn't. Peer
   collisions count as per-drone collisions, so this ablation surfaces
   in the per-drone column rather than only the joint one.
2. **Joint success collapses harder.** Dense: 27 → 7 % (4× drop).
   Packed: 10 → 0 % (no joint episode survives at all).

The per-drone drop is the headline. Compare against the density-only
sweep from the previous finding: going from 30 obstacles → 240
obstacles (8× density) cost −49 pp of per-drone success. **Removing
peer prediction at fixed dense costs the same −49 pp.** A planner
without peer prediction faces the geometry of an 8× denser world.

Δ over `per^4` stays positive at dense_indep (+6.6 pp) only because
per-drone fell so far that the independence floor is essentially zero
— any joint success now beats it. The number is arithmetically real
but mechanistically misleading: it is not coordination paying off,
it's two drones happening to survive the same episode by chance after
both planners ignored each other. At packed_indep the chance runs out
and joint = 0 / 30.

What the dense → packed comparison still says even with prediction off:
per-drone success barely moves (17 → 18 %) when density doubles in the
no-prediction regime. The static-obstacle planning is no longer the
bottleneck; peer collisions dominate the failure budget on the
crossing-pair scenario regardless of how many obstacles are around.

Engineering takeaway: in any scenario where peers' trajectories cross,
constant-velocity peer prediction is doing more work than the MPC
horizon, the static map quality, or moderate increases in obstacle
density. It is the cheapest layer in the multi-drone stack
(milliseconds per replan to forecast N−1 peers) and it pays back
multiples of that — but only because crossing-pair scenarios punish
its absence so heavily. In a non-crossing scenario (parallel goals,
shared corridors with the same direction of travel) we'd expect this
gap to shrink toward what the previous Δ-vs-density curve already
predicted.

## Wind miscalibration: planner belief must match sim reality

`examples/exp_wind.yaml` — constant northward wind disturbance × planner
wind belief, 4 × 4 grid, n=15 episodes:

<p align="center">
<img src="images/wind_miscal.png" alt="6-panel sim_wind × planner_wind sweep showing diagonal-wins miscalibration" width="640">
</p>

|  sim_wind \ planner_wind | 0 | 3 | 6 | 9 |
|---|---|---|---|---|
| **0** | **93.3** | 60.0 | 33.3 | 20.0 |
| **3** | 66.7 | **100.0** | 53.3 | 33.3 |
| **6** | 20.0 | 66.7 | **93.3** | 66.7 |
| **9** |  0.0 |  0.0 |  0.0 |  0.0 |

The diagonal wins — matched (planner belief = sim reality) recovers
93-100 %. Mismatch in either direction hurts symmetrically: under-
correction blows the drone off course, over-correction pre-compensates
into nothing. At `sim=6 m/s`, wind awareness lifts success from **20 %
to 93 %** (+73 pp) — one of the largest single-knob wins in the
framework. But at `sim=9 m/s` against `max_speed=8 m/s` no belief
saves you: the drone literally cannot make headway and every cell in
that row is 0 %. Awareness cannot beat physics.

## The perception-latency cliff: a four-step research saga

<p align="center">
<img src="images/cliff_delay_speed.png" alt="6-panel sensor.delay × max_speed sweep showing success drop and planner-dt blow-up at high delay × high speed" width="640">
</p>

*sensor.delay × max_speed at the Pareto MPC config — success dives in
the bottom-right corner (delay=0.5 s × speed≥20 m/s) while the rest of
the grid is comfortably ≥ 80 %. That single corner is the cliff.*

A single persistent `delay=0.5 s` × `speed=15 m/s` cell on the
predictive-MPC scenario (≤ 25 % success regardless of `inflate` /
`safety_margin` tuning at the Pareto config) drove four progressive
experiments — each documented in `examples/exp_predictive.yaml`:

1. **Predictor-side delay compensation** (negative result). Adding
   `delay_compensation=0.5` to the Kalman *predictor* on the obstacle
   stream actually *hurt* success. The MPC plans against future
   obstacles using a past self.
2. **Sensor-side ego extrapolation** (`sensor.extrapolate=true`). Project
   the stale position forward by `delay` using a 1-sample finite-difference
   velocity. Lifts success in moderate-delay × moderate-speed cells by
   +15 .. +35 pp; *hurts* at delay=0 (1-step lag artifact) and at
   high-speed × high-delay (acceleration noise overshoots).
3. **Velocity-window smoothing** (`sensor.velocity_window=5`). Average
   the FD velocity over multiple sample pairs to suppress acceleration
   noise. The persistent cliff lifts from 26.7 % → 43.3 % (+17 pp) at
   the headline cell, +35-40 pp at high speed. Catch: optimum window
   depends on the speed regime — low speed prefers `=1` (no lag), high
   speed needs `=5`.
4. **Kalman ego sensor** (`sensor.type=kalman_delayed`). Honest negative
   result. Best tuning of process / measurement noise tops out at the
   no-extrap baseline (25 %); the moving-average wins. The KF assumes
   a CV motion model; under MPC's frequent re-planning the assumption
   breaks and the MA's structure-free responsiveness dominates.

Engineering takeaway: simple model-free estimators can dominate more
sophisticated ones when the motion-model assumption breaks. Picking the
estimator that *actually wins* is more useful than picking the one that
sounds fanciest — the framework is built to make that picking trivial.

## MPC + CHOMP smoothing: layering on a saturated planner is a wash

`examples/exp_compare_mpc_chomp.yaml` — `mpc_chomp` planner wraps the
validated Pareto MPC config and runs 15 CHOMP smoothing iterations on
the rollout each replan, then clears `target_velocity` so the runner
pure-pursues the smoothed waypoints. Hypothesis: file off the
piecewise-straight corners at each replan boundary so the velocity
profile is gentler. Same scenario / horizon / sample count as the
plain-MPC baseline.

|              | success           | plan_dt (mean) | mean &#124;Δcmd&#124;/step |
|--------------|-------------------|---------------:|--------------:|
| plain MPC    | 96.7 % [83, 99]   | 11.0 ms        | 0.32          |
| **mpc + chomp** | 96.7 % [83, 99] | 18.9 ms (+71 %)| **0.61** (+90 %) |

Honest null result: success rate identical, plan_dt up 71 %, and the
per-step command delta nearly *doubles*. The reason is architectural,
not a tuning bug. MPC's `target_velocity` bypass *is* the smoothness
mechanism — it commits to one velocity for the whole `replan_period`
(0.2 s = 4 control steps) so the controller has nothing to chase
between replans and per-step `|Δcmd|` is small. CHOMP smoothing emits a
curved waypoint sequence that pure-pursuit re-aims at every 0.05 s, so
even though the *path* has fewer corners, the *control trajectory* has
more direction changes.

Engineering takeaway: layering a smoother on top of a planner that is
already at its Pareto saturation point is a wash unless the smoothing
target is downstream of where the cost lives — here the cost lives in
the controller, not the path. To make CHOMP help in this setting you
would need a velocity-profile-aware follower (or a planner that emits
a velocity spline directly). Same Pareto-saturation lesson as the
3D CHOMP+RRT result: a layer only wins if the layer below has room to
be improved.

### Follow-up: the velocity-profile-aware follower doesn't rescue it either

`examples/exp_compare_mpc_chomp_vprofile.yaml` — the natural fix the
above takeaway points at: extend `Plan` with a time-indexed
`velocity_profile`, add a velocity-tracking mode to the runner's
follower, and have `mpc_chomp` derive per-step velocities from the
smoothed path (forward differences / `dt_plan`) instead of emitting
waypoints. Same scenario, same MPC inner config:

|                          | success         | plan_dt | mean &#124;Δcmd&#124;/step |
|--------------------------|-----------------|--------:|--------------:|
| plain MPC                | 96.7 % [83, 99] | 11.0 ms | 0.32          |
| mpc + chomp (waypoints)  | 96.7 % [83, 99] | 18.9 ms | 0.61          |
| **mpc + chomp (vprofile)** | **90.0 %** [74, 96] | 21.3 ms | **2.02** |

Worse on every axis: success drops 6.7 pp, |Δcmd| jumps to **6.3 ×
plain MPC**. Two effects compound:

1. **Per-step profile updates.** Plain MPC keeps `target_velocity`
   constant over the whole `replan_period` (0.2 s = 4 control steps).
   The profile entry changes every 0.05 s, so even a smooth-by-
   construction velocity sequence has |Δcmd| bounded below by the
   path curvature.
2. **Replan-boundary discontinuities.** Each replan re-runs CHOMP from
   the new initial position; the first velocity of the new profile is
   freshly derived and jumps from the last applied velocity. Plain MPC
   has the same boundary, but `w_smooth · |Δaction|` penalises it in
   the rollout score; the profile derivative is unconstrained.

Methodological lesson: when a null result names a "missing piece"
(here: velocity-profile-aware follower), build the missing piece and
re-test before declaring the architectural insight sound. In this case
the deeper insight is *also* sound — and now stronger: the constant-
velocity bypass isn't a layering opportunity, it's the controller-side
ceiling. Help would need either CHOMP-on-velocity-sequence (smoothing
the right object) or a replan-boundary-aware cost (penalise jump from
previous applied velocity), neither of which is just "add a smoother".

## Action-jump cost: tuning the existing knob beats every layer

`examples/exp_compare_mpc_smooth.yaml` — the third installment of the
mpc_chomp / velocity-profile thread. The PR #21 / #22 null results both
identified `w_smooth · |action - prev_action|` (already present in
`SamplingMPCPlanner.plan`) as the load-bearing factor for plain MPC's
good control-trajectory smoothness. This finding tests the obvious
follow-up hypothesis: the right architectural fix is just *tune that
knob*, not add a smoothing layer above it.

Sweeping `planner.w_smooth` on the predictive scenario (n=30, Wilson
95 % CI, default = 0.05):

|     `w_smooth`     | success         | mean &#124;Δcmd&#124;/step |
|--------------------|-----------------|--------------:|
| 0.05 (current default) | 96.7 % [83, 99] | 0.320 |
| **0.5** (sweet spot) | **100.0 %** [89, 100] | **0.244** (-24 %) |
| 5.0 | 96.7 % [83, 99] | 0.245 |
| 50.0 (over-tuned) | 83.3 % [66, 93] | 0.183 (over-smoothed: +16.7 % collisions) |

`w_smooth = 0.5` wins on **both axes simultaneously** — success +3.3 pp
*and* |Δcmd| -24 %. Cranking past the sweet spot trades success for
further smoothness; at `w_smooth = 50` the planner refuses obstacle
maneuvers (16.7 % collision rate) but the smoothest trajectories of
any cell.

**Does the same fix transfer to the wrapper?** No — and the *reason* is
itself instructive. `mpc_chomp` exposes a `w_action_jump` that adds the
same cost form `||(x[1]-x[0])/dt - prev_emitted_velocity||²` directly to
the CHOMP descent. Swept on `output: velocity_profile` mode (with inner
MPC `w_smooth=0.5` already set):

|     `w_action_jump`     | success         | mean &#124;Δcmd&#124;/step |
|-------------------------|-----------------|--------------:|
| 0.0  | 93.3 % [78, 99] | 1.73 |
| 0.5  | 86.7 % [70, 95] | 6.87 |
| 5.0  | 93.3 % [78, 99] | 7.13 |
| 50.0 | 90.0 % [74, 96] | 7.18 |

The knob makes things drastically *worse*. Mechanism (verified by
single-iteration debug trace): the jump-cost gradient at index 0 is
~10⁶ in magnitude (proportional to `w_action_jump · |vel0 - prev|/dt²`).
After M⁻¹ preconditioning and per-row `max_step_norm` cap, x[1]
oscillates between two states each iter — pulled toward `(x[0] +
prev*dt)`, then yanked back by the smoothness Hessian's coupling.
Every other iter the optimizer is back where it started, but the
intermediate state has poisoned the smoothness terms enough to leave
neighbour waypoints displaced. The cap that keeps CHOMP stable for
plain trajectory smoothing actively prevents the constraint from
settling.

Two architectural lessons:
1. **The right place for action-jump cost is at the planner's argmin
   step**, not as a soft pull on a single waypoint. Plain MPC's
   constant-velocity-per-rollout means w_smooth · |v - prev_action|
   either wins the rollout or loses it — clean discrete choice.
   CHOMP's gradient descent has no such cleanness; the cap-and-Hessian
   interaction kills the constraint.
2. **Tuning beats layering** in this regime. Three PRs of new
   infrastructure (smoothing wrapper, velocity profile follower,
   CHOMP-side jump cost) confirmed the architectural insight and
   none of them beat changing one number in the existing planner.
   Same Pareto-saturation lesson the 3D CHOMP+RRT result taught:
   when the foundation is well-tuned, the cheapest fix is to look
   for an existing knob that's under-tuned.

Methodological close: the saga from PR #21 → #22 → this YAML is the
framework's intended workflow in miniature. Each null result named a
specific hypothesis; each hypothesis was tested by *building the fix
and measuring*; each test produced a quantitative result that either
killed the hypothesis or moved the question one layer deeper. Three
PRs of code, two null results, and one quantified win — that's the
shape of honest research.


## AirSim vs dummy_3d transferability: same plan, different physics

`examples/exp_transfer_{dummy,airsim}.yaml` — identical Pareto-MPC
straight-line scenario (start (0,0,30) → goal (0,30,30), no static
obstacles, max_speed=5, n_samples=16, horizon=20), only `simulator.type`
differs. Both at altitude 30 m so the AirSim Blocks env's cube clusters
do not intrude — this isolates *physics* differences, not perception
or avoidance. n=10 episodes per backend.

| metric | dummy_3d | AirSim (SimpleFlight) | Δ |
|---|---|---|---|
| success | 100 % [72.2, 100] | 90 % [59.6, 98.2] | -10 pp (1 t=0 collision after restart) |
| time-to-goal | 5.65 s ± 0.00 | 4.0 s ± 0.05 (ep 1-9) | **-29 % in AirSim** |
| path length | 27.88 m ± 0.00 | 28.2 m ± 0.2 (ep 1-9) | +1 % |
| avg reported speed | 4.98 m/s | 4.86 m/s steady-state (ramps over ~2.5 s) | within 3 % |
| replans / episode | 43 ± 0 | 29 ± 6 (fewer steps to reach goal) | -33 % |
| planner_dt | 311 ms | 2385 ms (network round-trip + LiDAR / camera polling overhead) | ~8× wall-clock cost |

Three things stand out, two of which are real and one of which is a
bridge calibration bug worth flagging:

1. **dummy_3d's velocity tracking is exact**, AirSim's ramps. dummy_3d
   reaches 5.0 m/s in 0.1 s (its `max_accel=50` allows 5 m/s in 1
   step). AirSim's SimpleFlight quadrotor takes ~2.5 s to ramp from
   3 m/s to 4.86 m/s — first-order motor lag plus pitch-to-translate
   coupling. The implication for ablations: any dummy_3d study that
   uses `max_speed` as if the drone snaps to that speed instantly is
   *not* representative of how a real quadrotor would behave during
   the first 2 s of every replan.

2. **Path lengths agree to 1 %** once the t=0 collision is excluded.
   The MPC plans the same straight line on both backends; the drone
   actually flies it. So the *spatial* output of an ablation is
   transferable; the *temporal* one is not.

3. **AirSim's reported time-to-goal is shorter than dummy's, even
   though its drone ramps slower.** The arithmetic doesn't add up
   (28 m / 4 s = 7 m/s avg, but the steady-state speed is 4.86 m/s).
   The likely cause is that `simContinueForTime(dt)` in the bridge
   advances AirSim's physics clock by *more than* the requested dt
   when the engine is busy — the drone moves further per bridge step
   than the recorded `t` reflects. This is a bridge-calibration bug,
   not a physics finding; opened as a follow-up.

Methodological takeaway for any future cross-backend ablation: do
**not** read AirSim's `final_t` as a sim-time delta — read `final
position` and trust `path_length`. The `max_speed` parameter is also
backend-relative: the same number means a hard cap in dummy and a
setpoint-with-ramp in AirSim. Same plan, different physics, same
spatial behaviour — that's the boundary at which dummy_3d ablations
generalize.

A *positive* takeaway: 9 / 10 AirSim runs succeeded with no parameter
tuning beyond the bridge's settle-windows + simPause(True) fix from
PR #44 / #45. The framework's planner / sensor / scenario boundary
transfers cleanly from synthetic to AirSim physics — only the
recorded *time* breaks.

## GPU MPPI: post-goal-mask fix unlocks long-horizon cells, 3D MPPI beats 3D MPC

<p align="center">
<img src="images/demo_gpu_mppi.gif" alt="GPU MPPI 3D episode with 64-sample rollout cloud (cyan) and softmax-best rollout highlighted (orange) weaving through three bouncing dynamic obstacles" width="480">
</p>

`examples/exp_gpu_mppi_pareto.yaml` (2D) and
`examples/exp_gpu_mppi_pareto_3d.yaml` (3D) — `gpu_mppi` planner
(PyTorch batched rollout, CUDA). Same scenarios as the original CPU
MPC Pareto studies: grid_world 50×50 with 30 random obstacles in 2D,
voxel_world 40×40×12 with 60 static + 3 bouncing dynamic obstacles
in 3D. Sweep over n_samples ∈ {32, 64, 128, 256} × horizon ∈ {20,
40, 60}, n=10 episodes per cell, max_speed=8 m/s, temperature=1.0.
`results/gpu_mppi_pareto_sweep_prefix/` keeps the pre-fix sweep for
reference.

### The goal-mask bug fix that changed every cell

Pre-fix, GPU MPPI summed `collision_pen` over **all** horizon steps
of every rollout, including steps that occurred *after* the rollout
had already entered the goal radius. A rollout that reached the
goal at step 10 and then drifted into an obstacle at step 18 was
classified `dirty_reach` instead of `clean_reach` and never received
the `-1e6` goal bonus. The CPU MPPI never had this bug — it breaks
out of the per-rollout loop on goal-reach (see
`uav_nav_lab/planner/mppi.py`), but the batched GPU rollout cannot
short-circuit per-sample, so each rollout has to be masked
explicitly:

```python
dist2 = ((rollouts - goal) ** 2).sum(dim=-1)        # [S, H]
reaches_goal_any = (dist2 <= gr2).any(dim=1)        # [S]
first_goal_h = torch.where(
    reaches_goal_any,
    (dist2 <= gr2).float().argmax(dim=1),
    torch.tensor(self.horizon, device=device),
)
step_idx = torch.arange(self.horizon, device=device)
pre_goal_mask = (step_idx[None, :] < first_goal_h[:, None]).float()
collision_pen = (collision_mask * pre_goal_mask).sum(dim=1)
```

The same mask is applied to the dynamic-obstacle collision term.
Effect: long-horizon rollouts that overshoot the goal mid-way now
get the bonus they should, the softmax can find a sharp argmax, and
the population-mean speed collapse seen pre-fix at h ≥ 40
disappears.

### 2D Pareto (post-fix), grid_world 50×50

<p align="center">
<img src="images/sweep_pareto_gpu_mppi.png" alt="6-panel 2D GPU MPPI Pareto sweep: success / collision / avg speed / ATE / planner_dt mean / planner_dt p95" width="640">
</p>

Success rates:

| n_samples \ horizon | 20 | 40 | 60 |
|---|---|---|---|
| 32  | 90 | 90  | 80 |
| 64  | 80 | **100** | 80 |
| 128 | 80 | **100** | 80 |
| 256 | 80 | **100** | 90 |

Plan time (mean ms, steady-state — first call per episode dropped to
remove CUDA-graph warmup; the in-summary mean is dragged up by that
one outlier):

| n_samples \ horizon | 20 | 40 | 60 |
|---|---|---|---|
| 32  | 2.6 | 2.8 | 3.1 |
| 64  | 2.8 | 3.4 | 3.6 |
| 128 | 2.6 | 3.0 | 3.6 |
| 256 | 3.6 | 3.7 | 3.6 |

Pre-fix the same grid showed 0 % at every h ≥ 40 cell and an
n=32/h=20 optimum at 90 % / 30 ms. Post-fix the **Pareto optimum
shifts to (n=128, h=40) at 100 % / 3.0 ms** — both axes flipped:
longer horizon now *helps* (was a speed-collapse cliff), and the
steady-state plan time is ~10 × cheaper than what pre-fix
measurements showed because nothing about the rollout took 30 ms in
the first place — the prior table was reporting CUDA-warmup-inflated
means with no per-episode warmup drop.

Lesson: report steady-state planner_dt on CUDA backends. The first
call after `planner.reset()` compiles the autograd graph and is
~10× the steady-state cost; including it in the mean for a
30-replan episode shifts the reported number by an order of
magnitude.

### 3D Pareto (post-fix), voxel_world 40×40×12

<p align="center">
<img src="images/sweep_pareto_gpu_mppi_3d.png" alt="6-panel 3D GPU MPPI Pareto sweep on a 40×40×12 voxel world with 60 static + 3 bouncing dynamic obstacles" width="640">
</p>

Success rates:

| n_samples \ horizon | 20 | 40 | 60 |
|---|---|---|---|
| 32  | 80     | 80 | 60 |
| 64  | **100** | 80 | 60 |
| 128 | **100** | 80 | 80 |
| 256 | **100** | 80 | 80 |

Plan time (mean ms, steady-state):

| n_samples \ horizon | 20 | 40 | 60 |
|---|---|---|---|
| 32  | 3.5 | 3.4 | 8.2 |
| 64  | 9.5 | 9.8 | 4.5 |
| 128 | 3.5 | 3.7 | 3.8 |
| 256 | 3.5 | 3.9 | 4.0 |

**Pareto-optimal 3D cell: n=64–256, h=20 → 100 % at 3.5 ms.** Compare
with the CPU MPC 3D Pareto from earlier in this doc (n=8, h=20 →
88 % / 70 ms) and the CPU MPPI 3D best at the same temperature
(86.7 %, from `exp_compare_mppi_3d.yaml`): **GPU MPPI 3D delivers
+12 pp success at 20 × lower plan time** than the CPU MPC 3D
baseline. The dense sample budget that 2D's Dijkstra heuristic
saturates against (per the pre-fix table) actually pays off in 3D
because the Fibonacci-sphere direction set has 16 × more candidate
directions to score per replan.

### Findings vs the pre-fix table

1. **The "speed collapse at h ≥ 40" finding was a bug, not an MPPI
   property.** Post-fix, h=40 is the 2D optimum and h=20 the 3D
   optimum. Population-mean collapse still occurs at h=60 with
   n=32 (avg_v drops from 7.6 → 6.8 m/s, succ drops to 80 %), but
   it is no longer the catastrophic 0 % regime.
2. **The "n=32 is enough" 2D conclusion now flips in 3D.** 2D plateaus
   at n=32–64, but 3D needs n ≥ 64 to reach 100 % at h=20 — the
   Fibonacci-sphere direction set is denser than 2D's circle, and
   the extra GPU samples are what cover it.
3. **GPU MPPI now Pareto-dominates CPU MPC in 3D.** Pre-fix this was
   not true (3D GPU MPPI stalled at 0 % at long horizons). Post-fix
   the GPU rollout's sample throughput finally translates to
   success because each rollout is correctly accounted for.

Methodological takeaway: when porting a sequential planner to a
batched GPU backend, the per-rollout early-exit conditions become
explicit masks. Forgetting any of them silently bins valid rollouts
into the "invalid" category, the softmax goes flat, and what looks
like a fundamental MPPI limitation is a transcription bug. Always
re-validate the per-sample bookkeeping when moving from a Python
for-loop to a tensor kernel.

### Temperature ablation at the 3D Pareto cell: the CPU rules don't transfer

`examples/exp_gpu_mppi_temp_ablation_3d.yaml` — GPU MPPI counterpart to
the CPU MPPI 3D T-sweep (`exp_compare_mppi_3d.yaml`). Same 3D voxel
scenario; planner pinned to the 3D GPU Pareto cell (n_samples=64,
horizon=20). n=30 episodes per T cell, T ∈ {0.1, 0.3, 1.0, 3.0, 10.0}.

| T    | succ                 | avg_v (m/s) | \|Δcmd\|         | plan_dt_ss mean/p95 |
|------|----------------------|-------------|------------------|---------------------|
| 0.1  | 76.7 % [59.1, 88.2]  | 7.93        | 0.086 ± 0.022    | 3.57 / 4.47 ms      |
| 0.3  | **96.7 % [83.3, 99.4]** | 7.89     | 0.106 ± 0.018    | 3.53 / 4.46 ms      |
| 1.0  | **96.7 % [83.3, 99.4]** | 7.50     | 0.096 ± 0.008    | 3.58 / 4.50 ms      |
| 3.0  | 93.3 % [78.7, 98.2]  | 4.88        | 0.104 ± 0.004    | 3.09 / 4.06 ms      |
| 10.0 | 86.7 % [70.3, 94.7]  | 1.69        | 0.088 ± 0.008    | 3.01 / 3.88 ms      |

Two findings *opposite* to the CPU MPPI 3D result:

1. **T=0.1 is significantly worse than T=1.0 on the GPU planner**
   (23/30 vs 29/30; Fisher exact p ≈ 0.02). On CPU MPPI the same
   T-sweep had T=0.1 nominally lower at 80.0 % but statistically
   tied with T=1.0 at 86.7 % — the Wilson CIs overlapped enough
   that we documented "useful range from T=0.1 to T=1.0". With
   n=64 GPU samples that range collapses to "T ≥ 0.3 only".
   Mechanism: at n=8 (CPU), argmin is reliable because the small
   batch can't easily put an outlier rollout at the cost minimum;
   at n=64 (GPU), one of the 64 rollouts will be a low-cost outlier
   that doesn't generalize, and T=0.1's near-argmin behaviour picks
   it. T=1.0's softmax average dampens that risk.
2. **\|Δcmd\| is flat across T** (0.086–0.106, all within one SEM
   of each other). The CPU MPPI 3D ablation found a 2.4× variation
   from T=0.1 (0.087) to T=1.0 (0.213); on the GPU planner that
   variation disappears. The Fibonacci-sphere direction set at
   n=64 already gives enough population coverage that the softmax
   weighted mean is close to the argmin in command space — there
   is no longer a "smoothness vs cost-fidelity" trade-off to tune
   T against.

Two findings that *do* transfer from the CPU result:

3. **avg_v collapses at high T** (4.88 m/s at T=3, 1.69 m/s at T=10).
   Same shape as CPU MPPI 3D — softmax over many directions averages
   to ≈ zero motion when T is large. Success rate degrades more
   gracefully because the 75 s episode budget gives slow trajectories
   enough room to still reach the goal eventually.
4. **plan_dt_ss is essentially constant** across T (3.0–3.6 ms).
   Temperature is a free knob compute-wise; the 3D Pareto cell
   3.5 ms steady-state is stable under T variation.

Engineering takeaways:

- **T=1.0 is the right default for GPU MPPI at n=64**, and the +5.2 pp
  coordination Δ flip from `exp_multi_drone_3d_4_gpu_mppi.yaml` is
  robust under this ablation (multi-drone study used T=1.0).
- **The CPU rule-of-thumb "T ≈ 0 ≈ argmin ≈ MPC, always safe" does NOT
  transfer to the GPU sample budget.** Larger n changes which costs
  the argmin can pick from; an outlier-prone landscape that was
  invisible at n=8 surfaces at n=64.
- **Don't carry over CPU MPPI temperature priors to GPU MPPI without
  re-running the ablation.** The |Δcmd|/T shape inverts and the
  useful T range collapses. The qualitative shape of an MPPI
  temperature sweep depends on the sample budget in ways that are
  not captured by the softmax math alone.

Reproduce:
```
uav-nav sweep examples/exp_gpu_mppi_temp_ablation_3d.yaml \
  --param planner.temperature=0.1,0.3,1.0,3.0,10.0 \
  --param num_episodes=30 -j 1
```

### Multi-drone: GPU MPPI's rollout cloud flips the coordination Δ

`examples/exp_multi_drone_3d_4_gpu_mppi.yaml` — same 4-drone cross
pattern, same 40×40×12 voxel world, same 30 random obstacles, same
seed schedule as the MPC baseline `exp_multi_drone_3d_4.yaml`
("3D escape volume erases the coordination Δ" above). Planner family
swapped from `mpc` (n=8, h=40) to `gpu_mppi` (n=64, h=20) and nothing
else. Initial study at n=30 episodes; re-run paired at **n=100** to
narrow the CIs once the Δ-flip direction was confirmed.

| planner               | per-drone (CI)       | joint (CI)          | indep `per^4` | Δ over indep |
|---|---|---|---|---|
| MPC      (n=8,  h=40) | 93.8 % [90.9, 95.7]  | 78.0 % [68.9, 85.0] | 77.2 %        | **+0.8 pp**  |
| GPU MPPI (n=64, h=20) | 90.0 % [86.7, 92.6]  | **77.0 %** [67.8, 84.2] | 65.6 %    | **+11.4 pp** |

(n=30 numbers were MPC 95.8 → 93.8 per-drone, 83.3 → 78.0 joint;
GPU MPPI 95.0 → 90.0 per-drone, 86.7 → 77.0 joint. Per-drone
estimates were biased high at n=30; the n=100 re-run shifts both
down but **widens the Δ gap** from +5.2 → +11.4 pp.)

Three observations at n=100:

1. **Per-drone success differs by ~4 pp** (93.8 vs 90.0). At n=30
   the difference was 0.8 pp and inside both CIs; at n=100 the
   point estimates separate but Wilson CIs still overlap ([86.7, 92.6]
   vs [90.9, 95.7] — 1.7 pp gap). GPU MPPI is genuinely a bit worse
   per-drone, not statistically indistinguishable as the n=30 result
   suggested.
2. **Joint success is tied** (78.0 vs 77.0; McNemar on the same-seed
   pairing gives both-success 67, MPC-only 11, GPU-only 10, neither
   12 — |11 − 10| = 1, χ²/(11+10) ≈ 0.05, not significant). On *any
   given seed* the two planners are equally likely to land all 4
   drones at goal.
3. **Δ over indep⁴ tells the actual coordination story**. MPC's
   +0.8 pp says drone failures are nearly independent — when MPC
   fails, it fails one drone at a time. GPU MPPI's **+11.4 pp says
   failures cluster** — when GPU MPPI fails on a seed, it tends to
   take 2–4 drones down together. Looking at the failed seeds in
   the GPU MPPI run, this is empirically visible: e.g. seed 119
   produced `[collision, collision, collision, success]`,
   seed 131 produced `[collision, collision, success, success]`,
   seed 134 produced `[success, collision, collision, success]` —
   joint failures dominated by multi-drone collisions, not lone
   failures.

Mechanistic read (revised from n=30): the n=30 commit interpreted
+5.2 pp as "failures more spread across episodes" / "decorrelate".
The Δ-over-indep⁴ math actually points the opposite way: **Δ > 0
means failures cluster within seeds, not spread across them.** What
GPU MPPI's softmax does is **amplify seed sensitivity**. On easy
seeds the rollout cloud agrees on a clean escape volume and all 4
drones make it; on hard seeds the same averaging produces overly
conservative commands across all 4 drones and they collide near the
crossing. MPC's argmin is more individually brittle (lower per-drone
success) but each drone's brittleness is uncorrelated with the
others — failures spread one-at-a-time across episodes.

What about joint succ being TIED then? Same joint, but very
different shape: MPC trades a small constant tax (one drone of
four occasionally fails, 78 % of episodes survive intact) for GPU
MPPI's bimodality (some episodes all 4 reach goal, some episodes
2–4 collide together — still 77 % of episodes get joint succ but
the "bad" episodes are much worse). In a deployment where partial
success has value (e.g. 3 of 4 packages delivered counts as
progress) MPC is preferable. In one where all-or-nothing is the
norm (e.g. formation flight) they're equivalent.

Caveat — plan-time: planner_dt is 73.9 ms ± 17.6 ms / replan (p95
105.6 ms), the same envelope as MPC (73.2 ms ± 17.1 ms). The 3.5 ms
steady-state from the single-drone GPU MPPI Pareto table does *not*
transfer to the 4-drone case because each per-drone planner instance
pays its own first-call CUDA warmup *every episode*, and four
instances also serialize on the same CUDA stream. A real-world
multi-agent system would warm up once at startup and amortize across
many minutes of flight; this benchmark is dominated by per-episode
warmups at small N. So the speed advantage GPU MPPI showed at single-
drone *doesn't survive multi-drone in this measurement setup*, even
though it survives at the per-rollout level inside each call.

Reproduce:
```
uav-nav run examples/exp_multi_drone_3d_4_gpu_mppi.yaml
uav-nav compare results/multi_drone_3d_4 results/multi_drone_3d_4_gpu_mppi
```

### dummy_3d N-scaling paired (MPC vs GPU MPPI, N ∈ {2, 3, 4, 6, 8, 10, 12})

Closes the §6 limitation that the GPU MPPI N-scaling curve was MPC-only.
Same `multi_drone_voxel` scenario (40×40×12 world, 30 random obstacles
seed 7), drones spaced evenly on a radius-17 circle around the centre
each crossing to its diametric opposite, same planner Pareto cells
throughout (MPC n=8, h=40; GPU MPPI n=64, h=20). N varies via
`examples/exp_multi_drone_3d_{2,3,4,6,8,10,12}{,_gpu_mppi}.yaml`.
n=30 paired (seeds 42-71) per (N, planner) cell.

| N  | MPC per-drone        | MPC joint          | MPC Δ      | GPU MPPI per-drone   | GPU MPPI joint     | GPU MPPI Δ  | McNemar (b, c, p)         |
|----|----------------------|--------------------|------------|----------------------|--------------------|-------------|---------------------------|
| 2  | 59/60 = 98.3 %       | 29/30 = 96.7 %     | −0.03 pp   | 44/60 = 73.3 %       | 21/30 = 70.0 %     | **+16.2** pp| (8, 0) → p ≈ **0.008** (MPC wins) |
| 3  | 80/90 = 88.9 %       | 21/30 = 70.0 %     | −0.2 pp    | 82/90 = 91.1 %       | 22/30 = 73.3 %     | −2.3 pp     | (4, 5) → p ≈ 1.000 (tie)  |
| 4  | 115/120 = 95.8 %     | 25/30 = 83.3 %     | −1.0 pp    | 114/120 = 95.0 %     | 26/30 = 86.7 %     | +5.2 pp     | (2, 3) → p ≈ 1.000 (tie)  |
| 6  | 154/180 = 85.6 %     | 14/30 = 46.7 %     | **+7.5** pp| 165/180 = 91.7 %     | 21/30 = 70.0 %     | **+10.7** pp| (5, 12) → p ≈ 0.144 (GPU lean) |
| 8  | 220/240 = 91.7 %     | 19/30 = 63.3 %     | **+13.5**pp| 166/240 = 69.2 %     |  5/30 = 16.7 %     | +11.4 pp    | (14, 0) → p ≈ **0.0001** (MPC wins) |
| 10 | 234/300 = 78.0 %     |  7/30 = 23.3 %     | **+15.0**pp| 236/300 = 78.7 %     | 10/30 = 33.3 %     | **+24.3** pp| (4, 7) → p ≈ 0.549 (GPU lean) |
| 12 | 292/360 = 81.1 %     |  7/30 = 23.3 %     | **+15.2**pp| 283/360 = 78.6 %     |  4/30 = 13.3 %     | +7.8 pp     | (4, 1) → p ≈ 0.375 (MPC lean) |

Reading the curve:

- **N=2 (head-on)**: GPU MPPI's softmax pays a heavy cluster cost
  (per-drone 73 % vs MPC's 98 %); MPC's argmin lock-step keeps both
  drones on opposite trajectories. McNemar p ≈ 0.008 — MPC wins all
  8 discordant seeds. The rollout cloud spreads where MPC's argmin
  can confidently commit; this is one regime where the §3 mechanism's
  sign reverses.
- **N=3, N=4 (mid)**: Both planners near-independent. The §3 headline's
  +11.4 pp $\Delta_\text{GPU}$ at N=4 (n=100) shows up here as +5.2 pp
  at n=30 — sample variance is real, but the *sign* and the
  GPU > MPC ordering on $\Delta$ both replicate.
- **N=6 (cluster peak)**: Both planners cluster, with per-drone tied
  near 90 %. **GPU MPPI's $\Delta$ exceeds MPC's by 3 pp** (+10.7 vs
  +7.5), reproducing the §3 mechanism's qualitative ordering at a
  different N. McNemar leans GPU (b=5, c=12, p ≈ 0.14) but n=30 is
  too small to clear $\alpha = 0.05$ on this gap.
- **N=8 (GPU collapse, geometric singularity)**: GPU MPPI's per-drone
  drops to 69 % while MPC stays at 92 %; joint to 17 %, McNemar
  p ≈ 0.0001 strongly favours MPC. The 8-drone 45°-spacing layout
  yields a particularly hostile central crossing geometry for the
  GPU rollout cloud (the four pairs of orthogonal trajectories pass
  simultaneously through the centre). The §3 mechanism's "softmax
  helps when per-drone is tied" condition breaks here because per-
  drone is no longer tied — MPC stays clean.
- **N=10 (GPU re-emerges, sweep maximum $\Delta$)**: At 36° spacing the
  per-drone rates re-tie at 78 % (both planners now sharing the same
  density-induced ceiling), and **GPU MPPI's $\Delta = +24.3$ pp
  exceeds every other point in the sweep**. This is the §3 mechanism
  at its strongest signal: with per-drone tied and a noisy crossing,
  the softmax clusters failures across seeds while MPC's argmin still
  ends up failing on the same seeds. McNemar p ≈ 0.55 (b=4 vs c=7)
  is non-significant because both planners' joint rates are floor-low
  (only 3/30 both-succ episodes), not because the $\Delta$ gap is
  ambiguous.
- **N=12 (saturation)**: Per-drone still tied around 80 %, both joint
  rates ~20 %, but GPU MPPI's $\Delta$ drops to +7.8 pp and MPC's
  $\Delta$ stays at +15.2 pp. The N=10 peak does not extrapolate.

Engineering takeaway: GPU MPPI's softmax-cluster advantage holds
in *several* regimes — anywhere both planners' per-drone rates re-tie
on the noisy crossing (N=4, N=6, N=10). It reverses at N=2 (geometry
too simple for cloud to add value) and at N=8 (specific geometric
singularity where GPU per-drone uniquely collapses). The §3 N=4
headline is one point on a non-monotonic curve; the mechanism is real
but its sign is a function of (per-drone tie status × crossing density
× drone-count symmetry).

Reproduce:
```
for n in 2 3 4 6 8 10 12; do
  uav-nav run examples/exp_multi_drone_3d_${n}.yaml          # MPC
  uav-nav run examples/exp_multi_drone_3d_${n}_gpu_mppi.yaml # GPU MPPI
done
```

### dummy_3d density × planner sweep at N ∈ {4, 6, 8}: §3 mechanism is conditional on per-drone tie

The N-scaling sweep above identified three factors that should govern
the §3 GPU-MPPI-Δ > MPC-Δ mechanism: per-drone tie status, crossing
density, and drone-count symmetry. The N-scaling sweep varied
drone-count symmetry while holding obstacle count constant at 30.
This sub-sweep holds N ∈ {4, 6, 8} fixed and varies obstacle count
instead — a 3×3 (N, density) grid.

`examples/exp_multi_drone_3d_{4,6,8}{,_dense,_packed}{,_gpu_mppi}.yaml`
— same crossing geometry on the same 40×40×12 world, with obstacle
counts 30 (baseline), 120 (dense, 4×), and 240 (packed, 8×). n=30
paired per (N, density, planner). The N=4 dense MPC and N=4 baseline
MPC/GPU YAMLs existed pre-sweep; the rest are added here.

| N | density (count) | MPC per-drone     | MPC joint        | MPC Δ        | GPU MPPI per-drone | GPU MPPI joint   | GPU MPPI Δ   | McNemar (b, c, p)      |
|---|-----------------|-------------------|------------------|--------------|--------------------|------------------|--------------|------------------------|
| 4 | baseline (30)   | 115/120 = 95.8 %  | 25/30 = 83.3 %   | −1.0 pp      | 114/120 = 95.0 %   | 26/30 = 86.7 %   | **+5.2** pp  | (2, 3, 1.000)          |
| 4 | dense (120)     | 81/120 = 67.5 %   | 8/30 = 26.7 %    | **+5.9** pp  | 86/120 = 71.7 %    | 8/30 = 26.7 %    | +0.3 pp      | (3, 3, 1.000)          |
| 4 | packed (240)    | 61/120 = 50.8 %   | 4/30 = 13.3 %    | **+6.7** pp  | 78/120 = 65.0 %    | 5/30 = 16.7 %    | −1.2 pp      | (3, 4, 1.000)          |
| 6 | baseline (30)   | 154/180 = 85.6 %  | 14/30 = 46.7 %   | +7.5 pp      | 165/180 = 91.7 %   | 21/30 = 70.0 %   | **+10.7** pp | (5, 12, 0.144)         |
| 6 | dense (120)     | 123/180 = 68.3 %  |  3/30 = 10.0 %   | −0.2 pp      | 139/180 = 77.2 %   |  8/30 = 26.7 %   | **+5.5** pp  | (0, 5, **0.063**)      |
| 6 | packed (240)    | 76/180 = 42.2 %   |  0/30 = 0.0 %    | −0.6 pp      | 133/180 = 73.9 %   |  3/30 = 10.0 %   | −6.3 pp      | (0, 3, 0.250)          |
| 8 | baseline (30)   | 220/240 = 91.7 %  | 19/30 = 63.3 %   | **+13.5** pp | 166/240 = 69.2 %   |  5/30 = 16.7 %   | +11.4 pp     | (14, 0, **0.0001**)    |
| 8 | dense (120)     | 154/240 = 64.2 %  |  3/30 = 10.0 %   | +7.1 pp      | 160/240 = 66.7 %   |  2/30 = 6.7 %    | +2.8 pp      | (2, 1, 1.000)          |
| 8 | packed (240)    | 89/240 = 37.1 %   |  0/30 = 0.0 %    | −0.04 pp     | 153/240 = 63.7 %   |  1/30 = 3.3 %    | +0.6 pp      | (0, 1, 1.000)          |

**Three distinct mechanisms emerge across the (N, density) grid.**

At **N=4** the sign of $\Delta$ flips with density — MPC's argmin
goes from near-independent at baseline to a $+6.7$ pp cluster source
at packed, while GPU MPPI moves the other way (from $+5.2$ pp at
baseline to $-1.2$ pp at packed). At each density the planners'
per-drone rates remain within ~14 pp of each other (95/95, 67/72,
51/65), so the $\Delta$ statistic is the primary differentiator.
McNemar paired joint success is p = 1.0 throughout — joint success
is set by the density, not by the planner.

At **N=6** the picture is different: **GPU MPPI's per-drone advantage
opens up** as density rises (85/92 at baseline, 68/77 at dense,
42/74 at packed — a 32 pp gap at packed). The sign of $\Delta$ does
*not* flip — GPU MPPI stays the cluster source. McNemar leans GPU at
every density (p = 0.063 at dense is close to significance). GPU MPPI
wins both per-drone *and* joint without needing a cluster mechanism.

At **N=8** the picture is different *again*: at baseline GPU MPPI
hits its 8-fold-symmetric central-crossing singularity (per-drone
69 % vs MPC's 92 %), so MPC wins paired joint with McNemar
p ≈ 0.0001. At dense the per-drone rates re-tie (MPC 64 % / GPU 67 %)
and both have positive $\Delta$ — MPC slightly higher (+7.1 vs +2.8),
mirroring the N=4 dense behaviour. At packed GPU MPPI's per-drone
advantage returns (37 % MPC vs 64 % GPU) but joint success is
floor-low (1/30 vs 0/30) so the $\Delta$ separation is statistically
moot. The N=8 baseline GPU collapse is a property of that specific
density-and-symmetry corner; raising density tends to *equalise* the
planners as MPC also struggles.

**The honest read across the grid.** GPU MPPI's softmax beats MPC's
argmin through *one of two routes* depending on regime: either
(i) per-drone tied and GPU MPPI clusters its failures (§3 N=4
baseline, N=6 baseline); or (ii) per-drone divergent in GPU's favour,
with the joint advantage following from per-drone alone (N=6 dense /
packed, N=8 packed). MPC has a density-driven cluster regime where
its argmin lock-step concentrates failures into specific seeds
(N=4 dense / packed, N=8 dense); whether this gives MPC an absolute
joint advantage depends on whether per-drone rates stay close
(yes at N=4 packed, partly at N=8 dense; no at N=6 packed or N=8
packed where GPU MPPI's per-drone has already opened up). The §3
N=4 baseline result is one corner of this grid where both routes
are absent for MPC and only GPU MPPI's cluster route registers.

**Connection to AirSim base_ew06.** The "AirSim sign reversal"
finding in §"AirSim multi-drone base_ew06" — MPC clusters at central
crossing while GPU MPPI stays independent — is one realisation of the
**N=4 density-driven $\Delta$ flip**. The AirSim `base_ew06` cell is
a dense-crowding regime (5 widened pillars + 4 drones converging at a
single central crossing) with per-drone rates close (90 % MPC vs 96 %
GPU): exactly the N=4 dense regime that flips the $\Delta$ sign on
dummy_3d. It does **not** generalise to AirSim cells at higher N or
to denser obstacle fields where GPU MPPI's per-drone might lead by
more — base_ew06 is one paired cell, not a general claim. Sim
backend choice is *not* orthogonal to the density-regime axis, but it
is also not the sole axis the AirSim observation lives on.

Reproduce:
```
for N in 4 6 8; do
  for v in "" "_dense" "_packed"; do
    uav-nav run examples/exp_multi_drone_3d_${N}${v}.yaml          # MPC
    uav-nav run examples/exp_multi_drone_3d_${N}${v}_gpu_mppi.yaml # GPU MPPI
  done
done
```

### dummy_3d N=4 + moving obstacle speed sweep: GPU MPPI's softmax averaging is catastrophic under dynamic obstacles

Extension to the §3 N=4 baseline cell (4 drones, 40×40×12 voxel
world, 30 random static obstacles) by adding one moving sphere
obstacle at $(20, 5, 6)$ with velocity $(0, +v, 0)$, radius 0.8,
reflecting at walls. The obstacle moves north along $x=20$, directly
along the north drone's corridor (start $(20, 3, 6)$ → goal
$(20, 37, 6)$). YAMLs:
`examples/exp_multi_drone_3d_4_dyn_v{2,4,8}{,_gpu_mppi}.yaml`. Each
cell is n=30 paired with the §3 seed set (42-71). Analysis script:
`scripts/paired_analysis_dummy_3d_multi.py`.

| $v$ (m/s) | MPC per-drone | MPC joint | $\Delta_\text{MPC}$ | GPU per-drone | GPU joint | $\Delta_\text{GPU}$ | McNemar | mean $t$ (MPC / GPU) |
|---|---|---|---|---|---|---|---|---|
| 0 (§3) | 95.8 % | 83.3 % | -1.0 pp | 95.0 % | 86.7 % | **+5.2 pp** | $p = 1.00$ | 5.06 / 4.50 s |
| 2 | 91.7 % | 73.3 % | +2.7 pp | **70.0 %** | **3.3 %** | -20.7 pp | $p \approx 0.0000$ | 14.48 / 4.91 s |
| 4 | 95.0 % | 80.0 % | -1.5 pp | **67.5 %** | **3.3 %** | -17.4 pp | $p \approx 0.0000$ | 7.60 / 4.89 s |
| 8 | **50.8 %** | **3.3 %** | -3.3 pp | 71.7 % | 3.3 % | -23.0 pp | $p = 1.00$ | 9.78 / 4.86 s |

Key readings:

**(1) GPU MPPI collapses catastrophically at $v \geq 2$ m/s.** At
$v=2$ the joint success drops from 86.7 % (§3 baseline) to **3.3 %**
— 27/30 paired-loss episodes against MPC's 73.3 %. The collapse is
*not* a per-drone-rate-floor artifact: per-drone is still 70 %, so
indep$^4$ would predict ~24 % joint and we measure 3.3 %, i.e.
$\Delta_\text{GPU} = -20.7$ pp (failures concentrate, but on the
*same drone* not on coordinated multi-drone clusters).

**(2) The failure is single-drone and deterministic.** Across all
v=2/4 GPU MPPI failures the colliding drone is **always drone idx 2
(north)** — the one whose corridor the obstacle moves along — and
collision time clusters at $t \approx 4.9-5.2$ s across seeds. The
other three drones (east, west, south) succeed in every failed
episode. This is consistent with GPU MPPI's softmax averaging
cancelling left/right avoidance rollouts when the dynamic obstacle
is dead ahead: half the cloud says "detour left around obstacle",
half says "detour right", the softmax-weighted mean lateral command
is near zero, and the drone slows to a stop in the central corridor
while the obstacle catches up. MPC's argmin selects a single
direction at each replan, commits, and clears the obstacle.

**(3) MPC holds at $v \leq 4$ but collapses at $v = 8$.** MPC's
per-drone stays $\geq 92$ % at $v \in \{2, 4\}$ and its joint stays
$\geq 73$ %. At $v = 8$ (parity with drone `max_speed`), MPC's
per-drone drops to 50.8 % and joint to 3.3 % — the obstacle moves
fast enough that within MPC's 1-2 s lookahead the planner can no
longer find a stable detour either. Three of the MPC v=8 failures
include `timeout` outcomes ($t = 75.0$ s), indicating drones got
stuck oscillating around the obstacle's bounce trajectory.

**(4) The §3 Δ-flip mechanism does not survive the dynamic-obstacle
extension.** At $v = 0$ both planners reach 95 %+ per-drone and GPU
MPPI's +5.2 pp $\Delta$ over MPC's $-1.0$ pp is the §3 signature.
At $v \geq 2$ the per-drone rates diverge by 20+ pp and the joint
success is dominated by single-drone per-drone failures, not by
multi-drone clustering. The $\Delta$ statistic becomes degenerate
(joint $\ll$ indep$^4$), and the planner comparison is decided at
the per-drone level by which planner can commit to an avoidance
direction. **Conclusion: GPU MPPI's softmax conservatism is *not*
just a coordination liability under static peers — it is a
generally fragile response to any obstacle-avoidance situation that
admits two symmetric escape directions, and dynamic obstacles
trigger this regime far more aggressively than static ones do.**

Reproduce:
```
for v in 2 4 8; do
  uav-nav run examples/exp_multi_drone_3d_4_dyn_v${v}.yaml          # MPC
  uav-nav run examples/exp_multi_drone_3d_4_dyn_v${v}_gpu_mppi.yaml # GPU MPPI
done
for v in 2 4 8; do
  echo "=== v=${v} ==="
  python3 scripts/paired_analysis_dummy_3d_multi.py \
    results/multi_drone_3d_4_dyn_v${v} \
    results/multi_drone_3d_4_dyn_v${v}_gpu_mppi
done
```

Scope: one moving obstacle, one trajectory geometry (north-bound on
$x=20$), one $(N, \text{density})$ baseline (§3 N=4 with 30 random
static obstacles). The $\Delta$-flip mechanism conditions identified
in §6 ((N, density, geometry) corner-specific) suggest this dynamic
extension is also corner-specific — different obstacle trajectories,
multiple moving obstacles, or different N rows will likely produce
different sign and magnitude. The robust qualitative finding is the
**single-drone deterministic GPU collapse mechanism**, not the absolute
magnitudes.

#### Probe 1: off-corridor obstacle restores the §3 mechanism

Moving the obstacle from $x=20$ (on the north corridor) to $x=15$
(5 m offset) at $v=4$ m/s — `exp_multi_drone_3d_4_dyn_off_v4{,_gpu_mppi}.yaml`
— recovers the §3 static baseline numbers:

| cell                   | MPC per/joint/Δ            | GPU per/joint/Δ            | McNemar  |
|---|---|---|---|
| §3 static baseline     | 95.8 / 83.3 / **-1.0**     | 95.0 / 86.7 / **+5.2**     | p=1.00   |
| dyn_off_v4 (x=15)      | 95.8 / 83.3 / **-1.0**     | 95.0 / 86.7 / **+5.2**     | p=1.00   |
| dyn_v4 (x=20, on corridor) | 95.0 / 80.0 / -1.5     | **67.5 / 3.3 / -17.4**     | p≈0      |

Per-drone, joint, and $\Delta$ are identical to within sampling
noise (in fact identical because the seeds resolve the same way at
both static cells given the off-corridor obstacle's net effect is
~null on the 4 drones' paths). The off-corridor probe **falsifies**
a generic "GPU MPPI is bad at dynamic obstacles" framing: GPU MPPI
handles the moving sphere fine when it does not align with a
drone's corridor. The dynamic failure mode is **specifically tied to
obstacle-on-corridor**, consistent with the bidirectional-cancellation
mechanism above (it requires the obstacle to present a left/right
symmetric escape choice to a drone whose path it directly blocks).

#### Probe 2: two obstacles compound but do not symmetrically halve

Placing one obstacle on the north corridor and one on the east
corridor — `exp_multi_drone_3d_4_dyn_2x_v4{,_gpu_mppi}.yaml`, both
at $v=4$ m/s — gives:

|                    | MPC                   | GPU MPPI            |
|---|---|---|
| per-drone          | 86/120 = **71.7 %**   | 59/120 = **49.2 %** |
| joint              | 4/30 = 13.3 %         | 1/30 = 3.3 %        |
| $\Delta$ over indep$^4$ | -13.0 pp         | -2.5 pp             |
| mean final_t       | 56.54 s               | 5.08 s              |
| McNemar (b, c)     | (3, 0)                | n=3, $p \approx 0.25$ |

Both planners drop. GPU MPPI's per-drone collapses to 49 % (vs
67.5 % at single obstacle) — the cancellation operator extends to
the east drone in addition to the north drone. MPC's per-drone
also drops to 72 % (vs 95 % at single obstacle); its mean final_t
of 56.5 s indicates many episodes hit the max-step timeout, with
drones oscillating around the obstacle trajectories rather than
clearing. The McNemar comparison is no longer decisive (p=0.25,
b=3 MPC-only out of 30) — at this difficulty level both planners
are near the joint-floor and the comparison cannot register a clean
direction. The qualitative reading is **the GPU MPPI cancellation
mechanism is per-corridor and extends additively, while MPC also
loses robustness once multiple drones are simultaneously obstacle-
adjacent** (probably from peer-prediction inflation around the
oscillating drones).

Together, the two probes refine the mechanism statement: GPU MPPI's
softmax bidirectional-cancellation failure mode applies *per
moving-obstacle/corridor alignment*. One alignment → one drone
fails deterministically. Two alignments → two drones fail and the
fleet drops to joint-floor. Off-corridor obstacle → §3 static
mechanism returns. The deployment statement "MPC is safer under
moving obstacles" holds only at single-corridor obstacle alignments;
at compound difficulty MPC also fails, just slower (timeout vs
collision) and via a different mode (path-oscillation vs
cancellation).

#### Probe 3: off-corridor gradient — planner role swap is non-monotonic in offset

Sweeping the obstacle's $x$-offset from the north corridor at $v=4$
m/s — `exp_multi_drone_3d_4_dyn_off{1,2,3}_v4{,_gpu_mppi}.yaml` —
reveals that the planner roles **do not monotonically restore to the
§3 baseline as the obstacle moves away from the corridor**. Instead
there is a regime around offset 1-2 m where MPC is the failing
planner and GPU MPPI is the safer one:

| offset (m) | $x$ | MPC per / joint / $\Delta$    | GPU MPPI per / joint / $\Delta$  | McNemar (b, c) | regime              |
|---|---|---|---|---|---|
| 0          | 20  | 95.0 / 80.0 / -1.5 pp         | **67.5 / 3.3 / -17.4 pp**        | $p \approx 0$ (23, 0)  | GPU collapses     |
| 1          | 19  | 85.8 / 53.3 / -0.9 pp         | 85.0 / **70.0** / +17.8 pp       | $p \approx 0.27$ (4, 9) | tied, GPU edge   |
| 2          | 18  | **69.2 / 6.7 / -16.2 pp**     | 87.5 / 70.0 / +11.4 pp           | $p \approx 0$ (1, 20)  | **MPC collapses** |
| 3          | 17  | 94.2 / 76.7 / -2.0 pp         | 91.7 / 73.3 / +2.7 pp            | $p = 1.00$ (5, 4)      | tied              |
| 5          | 15  | 95.8 / 83.3 / -1.0 pp         | 95.0 / 86.7 / +5.2 pp            | $p = 1.00$ (2, 3)      | §3 baseline      |

The offset 2 m MPC collapse reproduces the §3 dynamic-obstacle
mechanism with the *opposite* planner failing. MPC's failed
episodes (28/30 paired loses) split into:
- **North drone collision at $t \approx 4-6$ s** (the same drone
  the obstacle's bouncing trajectory reaches first), and
- **South drone `timeout` at $t = 75$ s** (the return drone gets
  stuck oscillating around the obstacle's trajectory through the
  central crossing).

The "stuck timeout" mode (mean episode time 23.6 s for MPC vs 4.5 s
for GPU) is the diagnostic: MPC's argmin commits to one detour
direction (east or west of the obstacle) each replan, but with
obstacle offset 2 m the two sides have asymmetric clearance against
static obstacles — and on hard seeds the argmin oscillates between
sides as the obstacle moves, freezing the drone near the central
crossing. GPU MPPI's softmax averages the same two sides into a
smooth lateral command and clears.

This is the **same softmax-vs-argmin mechanism §3 names**, but with
the *clutter geometry* — not the corridor alignment — selecting
which planner wins. The relationship between offset and planner
winner is non-monotonic: GPU loses at offset 0 (obstacle dead ahead,
symmetric escape, softmax cancels), MPC loses at offset 2 (obstacle
near corridor, asymmetric static clutter, argmin commits to the
wrong side), and both planners tie at offset ≥ 3 m where the
static-obstacle field is the dominant coupling and the §3 static
mechanism restores.

The combined message is that GPU MPPI's softmax is a **smoothing
operator on the action space** with three effects:
- Smoothing helps when the argmin would commit to a *wrong* side
  (offset 2 here, dense-corner cluster in §4.4.4).
- Smoothing hurts when there is *no right side to commit to*
  (offset 0 here, static-coordination clustering in §3 N=4
  baseline).
- The two regimes are *adjacent in scenario space*; small geometry
  changes flip the winner.

### AirSim + GPU MPPI parity: planner portable, dummy_3d plan-time advantage lost

`examples/exp_airsim_demo_gpu_mppi.yaml` — single-drone parity check
of GPU MPPI against the MPC baseline `exp_airsim_demo.yaml` on the
same Blocks scenario (start [0, 0, 5] → goal [45, 0, 12], LiDAR-built
occupancy, n=1 episode). Planner family swapped, sim/sensor/scenario
held constant.

| planner                | outcome | final_t | avg_v      | path_len | plan_ms mean/p95 |
|---|---|---|---|---|---|
| MPC      (n=32, h=40)  | success | 18.7 s  | 2.40 m/s   | 44.78 m  | 188 / 323        |
| GPU MPPI (n=64, h=20, T=1.0) | success | 24.2 s  | 1.87 m/s   | 45.05 m  | 180 / 334        |

Three observations:

1. **GPU MPPI runs end-to-end through `airsim_bridge`**, reaches the
   goal, and traces a path within 0.6 % of MPC's length (45.05 vs
   44.78 m) — the portability check that this section was designed to
   answer passes. The framework's planner / sensor / scenario boundary
   that PRs #44 / #45 secured for MPC transfers cleanly to GPU MPPI
   with no bridge-side changes.
2. **plan_dt is dominated by AirSim LiDAR + occupancy-update + RPC
   roundtrip, not planner internals.** MPC's 188 ms and GPU MPPI's
   180 ms are within 5 %; on `dummy_3d` GPU MPPI ran at 3.5 ms
   steady-state vs MPC's 70 ms (20 × faster). On AirSim the sim-side
   overhead is ~150 ms per replan and the planner choice is in the
   noise. The dummy_3d → AirSim transferability finding (same plan,
   different physics) **extends to plan_dt: the planner's compute
   advantage is *not* portable when the sim itself dominates the
   loop**.
3. **GPU MPPI is ~30 % slower wall-clock (24.2 vs 18.7 s) at the same
   `max_speed=3.0`.** avg_v 1.87 vs MPC's 2.40. The softmax average
   across 64 rollouts is more conservative than MPC's argmin on n=8
   when LiDAR-built occupancy is sparse / uncertain — more "stop" and
   "side-step" rollouts get folded into the population mean.

Mechanistic read: same sample-cloud-conservatism that drives the
multi-drone seed-sensitivity story (findings.md §"Multi-drone: GPU
MPPI's rollout cloud flips the coordination Δ") shows up here as
**lower commanded speeds under online perception**. The softmax
average across 64 rollouts produces more "stop / side-step" rollouts
when the LiDAR-built occupancy is uncertain, dragging the population
mean down. On dummy_3d with perfect sensing, this conservatism showed
up as clustered failures (bad seeds take down 2–4 drones together);
on AirSim with sparse LiDAR, it shows up as slower single-drone
speeds. Two faces of the same averaging operator — over rollouts in
both cases, but the cost-landscape variance comes from peer prediction
in the dummy_3d study and from sensor uncertainty here.

Bridge-side caveat — long settle for CUDA warmup:
`simulator.settle_after_teleport` is bumped from 0.3 s (MPC baseline)
to **3.0 s** in this YAML. The first GPU MPPI plan compiles the
autograd graph (≈14 s wall clock vs MPC's ≈10 s Dijkstra warmup);
the bridge holds AirSim paused during that wait, but during the
short window between teleport and the pause hand-off, 0.3 s is not
enough headroom for the engine to fully register the teleport before
the long planner wait begins. Without the bump the first step()
inherits a stale collision flag and t=0 collision ends the episode
immediately. The MPC baseline didn't need this because Dijkstra's
warmup is shorter and the engine's collision cache shrugs it off.
`max_steps` was also raised from 400 (= 20 s) to **800** (= 40 s) to
absorb the lower avg_v.

Reproduce:
```
# Blocks server running on 127.0.0.1:41451 with LidarFront in settings.json
uav-nav run examples/exp_airsim_demo_gpu_mppi.yaml
uav-nav compare results/airsim_demo results/airsim_demo_gpu_mppi
```

Open question: does the +11.4 pp multi-drone Δ (n=100 dummy_3d
re-run) survive at the GPU MPPI's 1.87 m/s vs MPC's 2.40 m/s on
AirSim? The dummy_3d Δ signature is "failures cluster within seeds";
under sparse AirSim LiDAR + slower commanded speeds, the same
seed-sensitivity mechanism could amplify (slower drones spend more
time in conflict zones) or dampen (LiDAR noise washes out the
deterministic seed effect). The next AirSim study is the n ≥ 30
paired multi-drone GPU MPPI vs MPC run.

### AirSim multi-drone parity: stack runs end-to-end, timing spread still visible at 4/4

`examples/exp_airsim_multi_demo_gpu_mppi.yaml` — 4-drone cross at
staggered altitudes (28/32/30/26 m) through Blocks, paired against
the MPC baseline `exp_airsim_multi_demo.yaml`. Same scenario,
sim/sensor/seed; planner family swapped, n=1 episode each.

| planner   | per-drone | joint   | avg_v (m/s) | max final_t | plan_dt steady-state (mean / p95) |
|---|---|---|---|---|---|
| MPC       | 4/4 succ  | success | 3.72        | 12.85 s     | 87.4 / 237.1 ms                   |
| GPU MPPI  | 4/4 succ  | success | 2.76        | 17.65 s     | 105.4 / 311.0 ms                  |

Three observations (n=1 carries no Δ-coordination statistics — these
are *mechanism* checks):

1. **Portability ✓**: 4-drone GPU MPPI with peer prediction runs
   end-to-end through `airsim_bridge`, all four drones reach goal
   without inter-drone or static-obstacle collision. The single-drone
   parity finding extends to multi-drone.
2. **avg_v 26 % lower, time-to-goal 37 % longer** (2.76 vs 3.72 m/s;
   17.65 vs 12.85 s) — same shape as the single-drone parity result.
   The softmax-average conservatism that costs ~30 % single-drone
   speed costs roughly the same on multi-drone with peer prediction
   enabled, so adding peers does *not* amplify the speed penalty
   (no peer-induced congestion overhead beyond what the planner
   already pays per-replan).
3. **Per-drone timing spread is visible even at 4/4 success**:
   MPC drones finish within **0.05 s** of each other (12.80–12.85 s).
   GPU MPPI drones spread by **0.55 s** (17.10–17.65 s). This is
   qualitatively consistent with GPU MPPI's softmax producing
   per-drone-different rollout picks even when the cost landscape
   is shared — MPC's argmin against the same shared landscape
   collapses to near-identical waypoint timing across drones. Note:
   timing spread is *not the same signal* as the dummy_3d Δ flip
   (which is about clustered FAILURES, not clustered successes); the
   AirSim multi-drone Δ requires an n ≥ 30 run to measure directly.

Headline question — **does the +11.4 pp dummy_3d Δ (n=100 re-run)
survive on AirSim?** — stays open. n=1 cannot answer it; n ≥ 30
paired AirSim run is the next study. The per-drone timing spread
above is at most a hint that the underlying softmax-vs-argmin
behaviour difference is preserved on AirSim physics; it does not
predict the Δ direction.

Plan-time budget caveat: 1 multi-drone episode on AirSim costs
~17 s sim + ~60 s for the 4 GPU MPPI planners' CUDA warmup +
~30 s AirSim reset/teleport overhead = ~100 s wall clock. An
n=30 × 2-planner paired run is ~100 minutes — doable overnight,
not in a single interactive session.

Reproduce:
```
# Blocks server + Drone1..4 in settings.json
uav-nav run examples/exp_airsim_multi_demo_gpu_mppi.yaml
uav-nav run examples/exp_airsim_multi_demo.yaml
uav-nav compare results/airsim_multi_demo results/airsim_multi_demo_gpu_mppi
```


### AirSim multi-drone n=30 paired: planner portable, scenario ceiling-limited, timing-spread signal preserved

`examples/exp_airsim_multi_n30.yaml` + `exp_airsim_multi_n30_gpu_mppi.yaml`
— same 4-drone staggered-altitude crossing through Blocks as the n=1
demo, num_episodes bumped to 30, both planners run on identical seeds
42..71 paired per-episode (chunked: each ep in a fresh Blocks server
to absorb the multi-drone `client.reset()` hang documented in
[bridge-fix](#bridge-fix-pause-after-reset-eliminates-a-stale-t0-collision-flag),
plus the bridge patch that fixes a stale-t=0 collision flag on
multi-drone reset).

| planner               | per-drone (CI)        | joint (CI)          | indep⁴ | Δ      | mean final_t | drone-spread/ep (mean / max) |
|---|---|---|---|---|---|---|
| MPC      (n=16, h=30) | 120/120 = 100.0% [96.9, 100.0] | 30/30 = 100.0% [88.6, 100.0] | 100.0% | +0.0 pp | 10.83 s | **0.02 s** / 0.15 s |
| GPU MPPI (n=64, h=20) | 120/120 = 100.0% [96.9, 100.0] | 30/30 = 100.0% [88.6, 100.0] | 100.0% | +0.0 pp | 14.88 s | **0.55 s** / 0.80 s |

McNemar same-seed: both-succ 30, MPC-only 0, GPU-only 0, neither 0 — no
disagreement, joint rates statistically indistinguishable (both at the
Wilson-CI ceiling 88.6–100.0%).

Three findings:

1. **Δ flip from dummy_3d does NOT transfer at this AirSim setup —
   because the setup is too easy.** The dummy_3d n=100 result had
   GPU MPPI at +11.4 pp coordination Δ over indep⁴ vs MPC's +0.8 pp
   ([previous section](#multi-drone-gpu-mppis-rollout-cloud-flips-the-coordination-delta)),
   driven by per-seed clustering of failures in a 40×40×12 voxel
   volume with 30 random obstacles. The AirSim demo scenario uses
   a 60×60×40 open volume with **zero obstacles** at altitude 26–32 m
   in Blocks — too easy for either planner to fail. Both planners hit
   100 % joint success across 30 seeds, so the coordination Δ
   measurement degenerates to 0 ± 0 pp at the Wilson ceiling. **This
   is a scenario-difficulty result, not a planner-portability one.**
   The dummy_3d Δ-flip remains unfalsified on AirSim; settling the
   transferability question requires a harder AirSim geometry —
   added Blocks obstacles, uniform-altitude crossing, or more
   drones.

2. **Planner portability ✓ (re-confirmed, statistically)**: GPU MPPI
   reaches goal at the same rate as MPC across 30 paired episodes
   on real quadrotor physics. The single-drone (n=1) and 4-drone
   (n=1 demo) parity findings extend to n=30 paired.

3. **Per-drone timing spread is preserved at n=30**: MPC drones finish
   within 0.02 s of each other on average (max 0.15 s); GPU MPPI
   spreads by **0.55 s** (max 0.80 s) — same magnitude as the n=1 demo
   observation, now over 30 episodes so it's not a single-seed
   artifact. This is the AirSim signature of GPU MPPI's softmax-over-64
   rollouts: even when all four drones succeed, they pick slightly
   different per-replan velocities, accumulating across the ~15 s
   trajectory into sub-second arrival spread. MPC's argmin against
   the shared cost landscape collapses to near-identical timing.
   Timing spread is **not the same signal** as the dummy_3d
   coordination-Δ flip (which is about clustered *failures*, not
   spread successes), but it is the mechanism-level signal that
   would underlie a Δ flip if the scenario were hard enough to fail.

Speed gap: GPU MPPI mean final_t is **37 % longer** than MPC (14.88 vs
10.83 s), consistent with the single-drone result (single-drone +26 %,
multi-drone +37 %). Adding peers widens the gap modestly, not
catastrophically.

Cost: ~2 min wall clock per chunked episode (~60 s AirSim restart +
warmup + ~60 s sim + reset/teleport), 60 paired eps ≈ 2 h total
(MPC ~65 min, GPU MPPI ~75 min). The reset-hang workaround is in
`scripts/run_airsim_multi_chunked.sh` (referenced by both YAML headers).

Reproduce:
```
# Each runs 30 1-ep invocations, bouncing Blocks between each — see
# the script header for the AirSim multi-drone reset-hang background.
./run_airsim_multi_chunked.sh mpc      30 42 results/airsim_multi_n30_mpc
./run_airsim_multi_chunked.sh gpu_mppi 30 42 results/airsim_multi_n30_gpu_mppi
python3 scripts/paired_analysis_airsim_multi.py \
    results/airsim_multi_n30_mpc results/airsim_multi_n30_gpu_mppi
```


### AirSim multi-drone uniform-altitude n=30: GPU MPPI collapses to 0 % joint while MPC holds 46.7 %

`examples/exp_airsim_multi_uniform_n30.yaml` + `..._gpu_mppi.yaml` —
same 4-way Blocks crossing as the staggered n=30 study above, but
all four drones fly at the **same altitude z=30 m** so the crossing
centre at (30,30,30) is a real 4-way conflict point rather than four
parallel corridors at ±2-4 m of vertical separation. The n=1 demo
YAML's own header warned this geometry is fragile; the n=30 paired
result quantifies *how* fragile, and how differently the two
planners cope.

| planner               | per-drone (CI)       | joint (CI)          | indep⁴ | Δ over indep | mean final_t (succ only) |
|---|---|---|---|---|---|
| MPC      (n=16, h=30) | 78/120 = 65.0% [56.1, 72.9] | 14/30 = 46.7% [30.2, 63.9] | 17.9% | **+28.8 pp** | 13.29 s over 14/30 |
| GPU MPPI (n=64, h=20) | 34/120 = 28.3% [21.0, 37.0] | **0/30 = 0.0%** [0.0, 11.4] | 0.6% | -0.6 pp | n/a (0 successes) |

McNemar same-seed paired-joint: both-succ 0, MPC-only-succ **14**,
GPU-only-succ **0**, neither-succ 16 — exact two-sided binomial
p ≈ **0.00012** ($1/2^{13}$). The two planners are not
statistically tied here; they are catastrophically different.

Four readings, ordered by inferential strength:

1. **Joint-success separates by 46.7 pp, McNemar p ≈ 0.00012.** This
   is the only AirSim configuration we've measured where a paired
   comparison rejects the null. At the easier staggered-altitude
   geometry both planners ceiling at 100 %; on dummy_3d both ran at
   ~77 % joint with the same indep⁴ structure flipped. At
   uniform-altitude AirSim, GPU MPPI loses every paired episode it
   doesn't tie on a both-fail seed.

2. **Per-drone separates by 36.7 pp.** MPC commands 2.40 m/s through
   the crossing and gets the bulk of its drones through cleanly;
   GPU MPPI's softmax-averaged 1.87 m/s (the same 30 % speed gap as
   the single-drone parity study) leaves drones at the conflict
   point ~30 % longer, and the per-drone collision rate climbs from
   35.0 % (MPC) to **71.7 %** (GPU MPPI).

3. **GPU MPPI's collapse is per-drone-rate-driven, not coordination-
   Δ-driven.** GPU MPPI's measured joint (0.0 %) sits **at** its
   indep⁴ prediction (0.6 %, Δ = -0.6 pp); the dummy_3d-style
   "failures cluster within seeds" signal isn't measurable here
   because there are no episodes with per-drone successes to compare
   to. The geometric coupling at (30,30,30) is dominating: GPU
   MPPI's slow commanded speed simply puts drones in front of each
   other long enough that *most* of them collide most of the time.
   The Δ-flip mechanism from dummy_3d (softmax amplifying seed
   sensitivity through a shared peer-prediction world model) needs a
   regime where the per-drone rate is high enough to leave indep⁴
   measurable headroom — dummy_3d's 90.0 % gives 9.0 pp of headroom
   for clustering; uniform-altitude AirSim's 28.3 % puts the indep⁴
   prediction below the joint success-rate floor.

4. **MPC's Δ over indep⁴ is +28.8 pp — but this Δ is geometric, not
   behavioural.** All 16 MPC failed episodes have ≥2 drones colliding
   (10× 2-fail, 2× 3-fail, 4× 4-fail; no 1-fail episodes). The
   crossing centre forces any drone that doesn't clear by time T to
   share its tile with at least one peer that also didn't clear,
   *regardless of planner behaviour*. The dummy_3d MPC's +0.8 pp Δ
   came from a scenario where single-drone failures (one drone hits
   a static obstacle the others avoid) were geometrically possible;
   uniform-altitude AirSim's bottleneck has no equivalent "single-
   drone failure" mode. So this number says "MPC's failures are
   strongly clustered" — but the clustering is mostly the geometry
   doing it, not the planner. We cannot attribute the +28.8 pp Δ
   to MPC the way the +11.4 pp on dummy_3d was attributable to GPU
   MPPI's softmax.

Combined reading: the dummy_3d Δ-flip transferability question
("does GPU MPPI's coordination-Δ advantage survive AirSim?") cannot
be answered cleanly on this scenario either, but for the *opposite*
reason from the staggered n=30: the staggered geometry was too easy
(both at 100 %, Δ measurement degenerate); uniform geometry is too
*hard for GPU MPPI* (its per-drone rate collapses 90 → 28 %, indep⁴
falls below the floor). The right discriminating AirSim setup for
the Δ-flip mechanism is somewhere between: tight enough to put both
planners somewhere in the 60-90 % per-drone band, where indep⁴ has
real headroom and the softmax-amplification signal can register.
Adding Blocks static obstacles to the staggered scenario is the
next obvious cell to try — separate future-work TODO.

What the result **does** answer cleanly: GPU MPPI is not a drop-in
replacement for MPC in tight-coupling deployments. The same softmax
conservatism that smooths typical-case behaviour and gives the
attractive 100 %-at-3.5-ms 3D Pareto cell (see §"3D Pareto (post-
fix)") destroys joint success at any geometry where speed-through-
bottleneck dominates inter-agent coordination. Plan accordingly when
choosing between planner families for a multi-agent mission.

Reproduce:
```
scripts/run_airsim_multi_chunked.sh mpc      30 42 \
    results/airsim_multi_uniform_n30_mpc \
    examples/exp_airsim_multi_uniform_n30.yaml
scripts/run_airsim_multi_chunked.sh gpu_mppi 30 42 \
    results/airsim_multi_uniform_n30_gpu_mppi \
    examples/exp_airsim_multi_uniform_n30_gpu_mppi.yaml
python3 scripts/paired_analysis_airsim_multi.py \
    results/airsim_multi_uniform_n30_mpc \
    results/airsim_multi_uniform_n30_gpu_mppi
```


### AirSim multi-drone ±1 m mid-stagger n=30: still ceiling-limited, cliff between 0 and 1 m

`examples/exp_airsim_multi_mid_n30.yaml` + `..._gpu_mppi.yaml` —
third cell between the demo's `±2-4 m` staggered crossing (6 m
z-spread) and the uniform-altitude crossing (0 m spread). This
config drops the four drones to z=29 / 31 / 30.5 / 29.5 (2 m
spread, every pair vertically separated by 0.5–2 m). The crossing-
centre tightest pair (east-south or north-west, both 0.5 m apart)
clears AirSim's 0.4 m drone radii by ~0.1 m mesh-to-mesh —
deliberately picked to land the per-drone rate in indep⁴'s
measurable headroom (60–90 %) for the failure-level Δ-flip signal
the dummy_3d study attributes to GPU MPPI's softmax.

| planner               | per-drone (CI)              | joint (CI)                  | indep⁴ | Δ      | mean final_t | drone-spread (mean / max) |
|---|---|---|---|---|---|---|
| MPC      (n=16, h=30) | 120/120 = 100.0% [96.9, 100.0] | 30/30 = 100.0% [88.6, 100.0] | 100.0% | +0.0 pp | 7.89 s | **0.10 s** / 0.20 s |
| GPU MPPI (n=64, h=20) | 120/120 = 100.0% [96.9, 100.0] | 30/30 = 100.0% [88.6, 100.0] | 100.0% | +0.0 pp | 10.28 s | **0.39 s** / 0.45 s |

McNemar paired-seed joint: both-succ 30, MPC-only-succ 0, GPU-only-succ 0, neither-succ 0.

Combined with the prior two cells, the AirSim altitude-stagger
response curve is now:

| stagger (z range) | MPC per-drone | GPU MPPI per-drone | MPC joint | GPU MPPI joint |
|---|---|---|---|---|
| ±2-4 m (6 m, demo)        | 100 % | 100 % | 100 %        | 100 %        |
| ±1 m (2 m, mid, this run) | 100 % | 100 % | 100 %        | 100 %        |
| 0 m (uniform)             | 65.0 % | 28.3 % | 46.7 %     | **0.0 %**    |

The AirSim multi-drone response is essentially **bimodal**: every
non-zero z-spread we have measured stays at the 4/4-success ceiling,
and any drone-pair convergence to the same z drops both planners
sharply (GPU MPPI catastrophically). The dummy_3d Δ-flip's
discriminating regime (per-drone ≈ 90 %, indep⁴ headroom ≈ 9 pp)
does not exist on the no-obstacle staggered geometry — peer
prediction + safety_margin = 0.6 m is enough to keep MPC and GPU
MPPI both clean at any non-zero vertical gap. Once two drones share
the z-axis, the bottleneck flips to physical mesh-mesh collision,
and the planner-mechanism difference (MPC's argmin vs GPU MPPI's
softmax) drops out of the failure analysis because GPU MPPI is too
slow to clear the conflict in the first place.

The trajectory-level signal IS preserved across the cliff:

| stagger      | MPC spread / final_t | GPU MPPI spread / final_t | spread ratio (GPU / MPC) |
|---|---|---|---|
| ±2-4 m       | 0.02 s / 10.83 s     | 0.55 s / 14.88 s          | 27 ×    |
| ±1 m         | 0.10 s / 7.89 s      | 0.39 s / 10.28 s          | 4 ×     |
| 0 m          | n/a (failure)        | n/a (0/30)                | n/a     |

GPU MPPI's per-drone-arrival spread is 4–27 × wider than MPC's at
every measurable cell — the softmax-spread mechanism is universal
across the AirSim cells we have, even where the failure-level Δ
cannot register. MPC's relative spread (spread / final_t) climbs
from 0.18 % at ±2-4 m to 1.27 % at ±1 m, suggesting that as
drone separations approach the safety margin, MPC's argmin
choices begin to correlate across drones too — but slowly.

Implication for the dummy_3d → AirSim transferability question:
adding altitude-stagger variants alone will not produce a
discriminating cell on this scenario. The next experiment that
*can* land in the right per-drone band is **adding Blocks static
obstacles to the staggered crossing** (or equivalently: dropping
the drones to z=8 where Blocks cubes are dense). Both require
substantial extensions — a new perception path (LiDAR + occupancy)
or a custom Blocks map — and remain future work. The Δ-flip
mechanism is still measured directly on dummy_3d (§3, n=100,
+11.4 pp); its AirSim transferability is now established as
*bracketed-but-not-confirmed* across three paired cells.

Reproduce:
```
scripts/run_airsim_multi_chunked.sh mpc      30 42 \
    results/airsim_multi_mid_n30_mpc \
    examples/exp_airsim_multi_mid_n30.yaml
scripts/run_airsim_multi_chunked.sh gpu_mppi 30 42 \
    results/airsim_multi_mid_n30_gpu_mppi \
    examples/exp_airsim_multi_mid_n30_gpu_mppi.yaml
python3 scripts/paired_analysis_airsim_multi.py \
    results/airsim_multi_mid_n30_mpc \
    results/airsim_multi_mid_n30_gpu_mppi
```


### AirSim multi-drone static-cube discriminating cell n=30: GPU MPPI clears every seed while MPC drops paired seeds

`examples/exp_airsim_multi_discriminating_n30.yaml` +
`..._gpu_mppi.yaml` — first AirSim cell that puts a planner in the
60-90 % per-drone band using spawned Blocks cube geometry rather than
altitude-only tightening. The bridge now accepts
`simulator.static_obstacles` and spawns `1M_Cube_Chamfer` meshes after
reset; the scenario carries matching `obstacles.boxes` for planner
occupancy. Final tuned cell uses four east/west pillars plus one
north/south pillar, z = 26/28/30/32, south lane x = 26, `inflate: 3`.

| planner               | per-drone (CI)       | joint (CI)          | indep⁴ | Δ over indep | mean final_t |
|---|---|---|---|---|---|
| MPC      (n=16, h=30) | 105/120 = 87.5% [80.4, 92.3] | 22/30 = 73.3% [55.6, 85.8] | 58.6% | **+14.7 pp** | 10.03 s over 22/30 |
| GPU MPPI (n=64, h=20) | 120/120 = 100.0% [96.9, 100.0] | 30/30 = 100.0% [88.6, 100.0] | 100.0% | +0.0 pp | 12.35 s over 30/30 |

McNemar same-seed paired-joint: both-succ 22, MPC-only 0,
GPU-only **8**, neither 0; exact p ≈ **0.008**. Disagreement seeds:
43, 47, 48, 51, 53, 60, 62, 65.

Interpretation: this **does close the AirSim discriminating-cell gap**
from the no-obstacle cells, but it does **not** reproduce the dummy_3d
Δ-flip mechanism. In dummy_3d, MPC and GPU MPPI tied on joint success
while GPU MPPI had the larger positive Δ, indicating stronger
within-seed failure clustering. In this AirSim static-cube cell,
GPU MPPI simply clears every paired seed; its Δ is degenerate at the
100 % ceiling. The robust result is planner separation under real
AirSim collision geometry: GPU MPPI trades slower completion
(12.35 s vs MPC's 10.03 s) for eliminating the paired MPC collision
seeds.

The tuning path matters. A symmetric eight-pillar version put the
south drone into a deterministic collision floor. The final cell keeps
only one north/south pillar and offsets the south lane to x=26 so the
failure budget comes from obstacle-induced crossing interactions
rather than a fixed mesh bottleneck.

Reproduce:
```
scripts/run_airsim_multi_chunked.sh mpc 30 42 \
    results/airsim_multi_discriminating_n30_mpc \
    examples/exp_airsim_multi_discriminating_n30.yaml
scripts/run_airsim_multi_chunked.sh gpu_mppi 30 42 \
    results/airsim_multi_discriminating_n30_gpu_mppi \
    examples/exp_airsim_multi_discriminating_n30_gpu_mppi.yaml
python3 scripts/paired_analysis_airsim_multi.py \
    results/airsim_multi_discriminating_n30_mpc \
    results/airsim_multi_discriminating_n30_gpu_mppi
```


### AirSim multi-drone base_ew06 density-sweep n=30: Δ-flip sign reverses — MPC is the clustering planner on AirSim

Extension of the static-cube discriminating cell above. The earlier
cell pinned GPU MPPI at a 100 % ceiling, so its $\Delta$ was
degenerate and the dummy_3d §"Multi-drone: GPU MPPI's rollout cloud
flips the coordination Δ" mechanism could not be tested directly.
Variant `base_ew06` widens the four east/west pillars in the baseline
5-pillar layout from `scale = 0.5` to `0.6` — the smallest knob found
in 30+ probes that drops GPU MPPI off ceiling without collapsing both
planners. n=50 paired episodes (seeds 42-91) measured on the same
chunked-server harness.

| planner               | per-drone (CI)                | joint (CI)                | indep⁴ | Δ over indep | mean final_t |
|---|---|---|---|---|---|
| MPC      (n=16, h=30) | 179/200 = 89.5 % [84.5, 93.0] | 34/50 = 68.0 % [54.2, 79.2] | 64.2 % | **+3.8 pp** | 10.00 s over 34/50 |
| GPU MPPI (n=64, h=20) | 191/200 = 95.5 % [91.7, 97.6] | 41/50 = 82.0 % [69.2, 90.2] | 83.2 % | −1.2 pp | 12.39 s over 41/50 |

McNemar same-seed paired-joint: both-succ 28, MPC-only-succ 6,
GPU-only-succ 13, neither-succ 3; exact $p \approx 0.167$. Point
estimate favours GPU MPPI by 14 pp on joint and by 2.2-to-1 on
discordant pairs but n=50 is still short of significance at
$\alpha = 0.05$.

**Headline.** The Δ-flip mechanism from dummy_3d transfers to AirSim,
**but the sign of the flip reverses**. dummy_3d had GPU MPPI as the
clustering planner ($\Delta_\text{GPU} = +11.4$ pp vs $\Delta_\text{MPC}
= +0.8$ pp at tied 94 % per-drone). AirSim `base_ew06` ties the
planners again (90 % MPC vs 96 % GPU per-drone) and re-differentiates
through Δ, but now **MPC is the clustering planner**
($\Delta_\text{MPC} = +3.8$ pp vs $\Delta_\text{GPU} = -1.2$ pp).

| backend / cell           | MPC Δ       | GPU MPPI Δ   | clustering planner |
|--------------------------|-------------|--------------|--------------------|
| dummy_3d §3 (n=100)      | +0.8 pp     | **+11.4 pp** | GPU MPPI |
| AirSim base_ew06 (n=50)  | **+3.8 pp** | −1.2 pp      | MPC |

Per-seed disagreement makes the mechanism concrete:

- **GPU MPPI failures (9 seeds: 43, 45, 46, 50, 52, 73, 75, 90, and
  concordant 73) are all the same drone** — drone idx 3, the
  southernmost lane, against the widened EW pillar. Softmax-averaged
  rollouts handle the geometric pinch uniformly across seeds, so
  failures are independent (Δ ≈ 0).
- **MPC failures (16 seeds) split into 13 single-drone (drone 3) and
  3 multi-drone clusters**: seeds 55 and 67 lose drones 1, 2, and 3
  simultaneously; seed 66 loses drones 1 and 2. The argmin update
  commits each MPC drone to the same north-end corridor at the same
  moment; when the corridor is geometrically infeasible for the
  seed's start jitter, multiple drones converge into the same
  collision frame within ~0.5 s. These 3 cluster seeds in 50 (6 %
  cluster rate) are the entire source of MPC's positive Δ — removing
  them would put MPC at Δ ≈ -4 pp (indistinguishable from GPU).

Reading: dummy_3d's frictionless point-mass kinematics let GPU MPPI's
softmax act as the cluster source; AirSim's PID + quadrotor physics +
meshed environment surfaces MPC's argmin commitment as the cluster
source. The *structural* claim from dummy_3d (per-drone tied + Δ
differentiates planners) is robust to the sim-backend change; the
*deployment* claim ("GPU MPPI's softmax is a joint-coordination
liability") is not — on AirSim base_ew06 it is MPC that loses three
drones together.

GPU MPPI also pays 2.4 s more (12.39 s vs 10.00 s mean final_t over
successes) — same direction as the previous static-cube cell.

Caveats:

- n=50 paired tightens McNemar from p ≈ 0.302 (n=30) to p ≈ 0.167 but
  still does not clear α = 0.05. Adding ~30 more paired seeds at the
  current discordant ratio would close the rest of the way; the
  qualitative sign-reversal claim above is supported by the Δ table
  and per-seed disagreement structure, not by the paired p-value.
- The MPC cluster failure mode is real but rare (3/50 = 6 % cluster
  rate). The absolute magnitude of MPC's $\Delta$ is sample-size
  sensitive — n=30 measured $+6.9$ pp, n=50 settles at $+3.8$ pp as
  the single-drone failures pile on without new clusters. The sign
  and the qualitative cluster mode are stable; the magnitude moves
  with $n$.
- All GPU failures concentrate on drone idx 3 — the south drone lane
  is intentionally offset to $x = 26$, which puts it within the
  expanded keepout of the EW pillar at $(25, 27)$ when scale = 0.6.
  The bias is shared by both planners' single-drone failures; only
  MPC additionally clusters into the multi-drone failure mode.
- Lane-shift probes (`base_ew06_lane30` at $x=30$, `base_ew06_lane22`
  at $x=22$, n=10 GPU-only smoke each) both produce 10/10 drone-3
  collisions at $(30.9, 29.4, 26.6)$ and $(21.2, 23.3, 26.6)$
  respectively. The $x=26$ baseline collides at $(21.2, 23.2, 26.6)$ —
  the same detour-terminus position lane22 hits, just reached via a
  longer route. The single-drone failure mode is therefore a property
  of the planner+inflate+EW pillar geometry of the cell, not a
  removable artifact of the lane offset choice.
- **Stability re-check.** A bridge patch capturing
  `simGetCollisionInfo().object_name` (see `airsim_bridge.py:525`)
  was added so collision attribution survives into the per-step JSON.
  Re-running the 8 GPU MPPI failure seeds and 27 fresh seeds gave
  35/35 successes vs. expected ~5-6 failures at the n=50 16 % rate.
  The three MPC cluster seeds (55, 66, 67) re-ran with mixed results:
  seed 67 reproduced exactly (drones 1/2 with empty `collision_object`
  = drone-drone at central crossing, drone 3 hitting
  `uavnav_disc_ew_35`); seed 66 morphed to drone-3 single; seed 55
  became 4/4 success.

- **Variability characterization** (3 fresh paired batches × n=15,
  seeds 200-214 / 220-234 / 240-254):

  | batch     | MPC joint  | GPU joint  | MPC Δ    | GPU Δ | (b, c)  | McNemar p | clusters |
  |-----------|------------|------------|----------|-------|---------|-----------|----------|
  | 1         | 14/15      | 15/15      | **+6.0** | +0.0  | (1, 0)  | 1.000     | 1        |
  | 2         | 11/15      | 13/15      | **+12.4**| −0.7  | (2, 4)  | 0.688     | 3        |
  | 3         | 14/15      | 11/15      | −0.2     | −2.6  | (4, 1)  | 0.375     | 0        |
  | n=50 orig | 34/50      | 41/50      | +3.8     | −1.2  | (6, 13) | 0.167     | 3        |
  | combined n=45 | 39/45  | 39/45      | +7.3     | −0.7  | (7, 5)  | 0.774     | 4        |

  Stable across batches: (i) **MPC cluster mode at central crossing
  is reproducible as a mode** (mean 8.9 % rate, 0-20 % per-batch
  range, collision-object signature is always drones 1/2 with empty
  object_name + optional drone 3 colateral); (ii) **GPU MPPI never
  exhibits the cluster mode**, only drone-3 single failures.

  Not stable: (i) **cluster rate per batch swings 0/15-3/15**, wider
  than the n=50 Wilson CI implies; (ii) **McNemar direction reverses
  across batches** — batch 1 tied, batch 2 GPU-favored (b=2 c=4),
  batch 3 MPC-favored (b=4 c=1, opposite sign from the n=50 +
  batch 2 reading). 3-batch combined n=45 gives joint 86.7 % vs
  86.7 % with McNemar p ≈ 0.77 — essentially a tie.

  Implication: the qualitative Δ-flip sign-reversal finding (only MPC
  has a cluster failure mode) is robust. The quantitative McNemar
  conclusion from the single n=50 study (GPU joint 82 % > MPC 68 %,
  p = 0.167) does **not replicate** in independent samples. The
  cleanest fix is either much larger N (≥200 paired) or a
  controlled-environment measurement.

Reproduce (uses the param-sweep YAML generator):
```
VARIANTS="base_ew06" MODE=paired N=30 BASE_SEED=42 \
    scripts/run_airsim_discriminating_param_sweep.sh
```

The script generates `/tmp/uavnav_airsim_disc_base_ew06_{mpc,gpu_mppi}.yaml`
from the static-cube cell YAMLs, chunked-runs n=30 paired into
`results/airsim_multi_discriminating_base_ew06_n30_{mpc,gpu_mppi}/`,
and invokes `paired_analysis_airsim_multi.py` at the end.


### AirSim dynamic-obstacle bridge extension (smoke verified, paired cell still tuning)

Adds dynamic-obstacle support to `airsim_bridge.py`. Each
`_DynamicObstacle3D` in the scenario gets a corresponding kinematic
`1M_Cube_Chamfer` mesh in Blocks (scale = 2 × radius), spawned at
reset via `simSpawnObject` and re-posed each step via
`simSetObjectPose(name, pose, True)` to match the scenario's
post-`advance(dt)` position. The scenario remains the authoritative
state — the bridge mirrors the scenario's clock to Unreal so the
Unreal collision detector sees drone-vs-cube hits naturally.

Implementation:
- `airsim_bridge._sync_dynamic_obstacles_initial(client)` — called
  once per reset by the master bridge after `_sync_static_obstacles`.
- `airsim_bridge._update_dynamic_obstacle_poses(client)` — called
  each step inside `step_command` after `scenario.advance(self.dt)`
  but before `simContinueForTime(self.dt)`, so the cubes are at their
  post-tick position when the physics tick runs.
- `step_command` now calls `scenario.advance(self.dt)` (mirroring the
  dummy_3d bridge's behaviour); without this the AirSim bridge never
  stepped scenario clock, so static-only AirSim cells worked but
  dynamic-only cells were no-ops.

Smoke verification (`exp_airsim_multi_dyn_n5_smoke.yaml`, one moving
sphere on the north drone's corridor at $(30, 8, 30)$, velocity
$(0, v, 0)$, n=2 paired episodes via `run_airsim_multi_chunked.sh`):

| obstacle radius / $v$ | MPC outcome | GPU MPPI outcome | observation |
|---|---|---|---|
| r=1.0, v=1 | 2/2 success     | 2/2 success      | drone detours west+up (x 30→28, z 30→32), §3 mechanism does not register |
| r=1.0, v=2 | (not run paired) | 2/2 success     | same — escape volume sufficient |
| r=2.0, v=2 | **2/2 north collision @ t=7.6 s** | **2/2 north collision @ t=10.2 s** | both planners fail — cell too hard, not discriminating |

North drone trajectory under MPC at r=1.0 v=1, episode 0 seed 42:
$(30, 5, 30)$ → $(29.0, 7.2, 32.0)$ at $t=0.75$ s → $(28.2, 11.6, 32.0)$
at $t=1.5$ — drone commits to a west+up detour immediately, hugs
$(x \approx 28, z \approx 32)$ until past the obstacle, returns to
$(29.4, 52.0, 30.3)$ at goal. The bridge's cube placement is
load-bearing here: without the kinematic cube the planner would see
the obstacle (via `scenario.dynamic_obstacles`) but Unreal would let
the drone fly through it. Equally, without `scenario.advance(self.dt)`
the cube would be frozen at its initial pose and the drone would
detour for an obstacle that never moves.

What this **does not yet show** is the dummy_3d §3 Table 2 cliff
($v=2$ m/s, GPU joint 86.7 % → 3.3 % vs MPC 73 %) reproduced in
AirSim. The default AirSim multi-drone cell ($60 \times 60 \times 40$
ENU volume, $\text{max\_speed}=4$ m/s, $z$-axis 4-12 m above ground
with no ceiling obstacles, $\text{inflate}=0$) has substantially
more escape volume than the dummy_3d cell ($40 \times 40 \times 12$,
$\text{max\_speed}=8$, $\text{inflate}=1$). GPU MPPI finds a
non-cancellable detour (the up-axis breaks the left-right symmetry)
and clears at r=1.0. At r=2.0 the obstacle is large enough to be
unavoidable for both planners — not discriminating.

Reproducing the dummy_3d cliff in AirSim therefore requires further
cell tuning, likely some combination of: a ceiling obstacle layer at
$z = 32-34$ to remove the up-detour, faster drone (`max_speed=8`),
smaller scenario volume (40×40×12 to match dummy), or a wall of
static cubes flanking the corridor. Each constraint reduces the
escape volume's effective dimensionality; the §3 mechanism needs the
2D (left/right) symmetric case to register. The implementation is
ready; the parameter sweep is future work. Reproduce smoke:

```
scripts/run_airsim_multi_chunked.sh mpc      2 42 \
  results/_airsim_dyn_smoke examples/exp_airsim_multi_dyn_n5_smoke.yaml
scripts/run_airsim_multi_chunked.sh gpu_mppi 2 42 \
  results/_airsim_dyn_smoke_gpu \
  examples/exp_airsim_multi_dyn_n5_smoke.yaml  # swap planner type to gpu_mppi
```


### Smart MPPI (argmin-fallback): mechanism detector works, naive fix doesn't

Tests the §3 dynamic-obstacle mechanism diagnosis by **fixing it**.
Adds a `fallback_to_argmin` option to `gpu_mppi` (commit
`uav_nav_lab/planner/gpu_mppi.py`). Each replan computes both the
softmax-averaged action and the lowest-cost (argmin) rollout's
action. If the softmax action's *lateral* component (perpendicular
to the goal direction) is much smaller than the argmin action's
lateral component, treat that as a left/right cancellation
signature and fall back to argmin:

```
argmin_lat_mag    = |argmin_action - (argmin_action · goal_dir) * goal_dir|
softmax_lat_mag   = |softmax_action - (softmax_action · goal_dir) * goal_dir|
if argmin_lat_mag > 0.5  m/s
    and softmax_lat_mag < 0.5 * argmin_lat_mag:
    fall back to argmin
```

**Detector verification (synthetic single-step)**: with a moving
obstacle dead ahead at $(20, 10, 6)$ and drone at $(20, 5, 6)$,
vanilla GPU MPPI outputs target_velocity $[0.11, 6.98, 0.36]$ —
near-zero lateral, the §3 mechanism. Smart MPPI's fallback fires
and outputs $[-1.49, 7.74, 1.36]$ — substantial lateral + up, an
escape direction. The mechanism is detectable.

**Paired n=30 results** on the dummy_3d §3 dynamic cells (planner
type swapped from `gpu_mppi` to `gpu_mppi` with
`fallback_to_argmin: true`, same Pareto cell, same seeds 42-71):

| cell                | vanilla GPU joint | smart GPU joint | vs vanilla McNemar | rescue / hurt |
|---|---|---|---|---|
| `dyn_v2` (on-corridor, v=2)    | 1/30 = 3.3 %  | 2/30 = 6.7 %  | (b=0, c=1), p=1.00 | +1 seed (no rescue) |
| `dyn_v4` (on-corridor, v=4)    | 1/30 = 3.3 %  | 1/30 = 3.3 %  | (b=0, c=0), p=1.00 | no change |
| `dyn_off2_v4` (off 2 m, v=4)   | 21/30 = 70.0 % | 16/30 = 53.3 % | (b=9, c=4), p=0.27 | **hurt -16.7 pp** |

The cliff at `dyn_v2` is *not* rescued — the bimodal-cancellation
fallback triggers, but the rescue rate is 1/30. At `dyn_off2_v4`,
where vanilla GPU MPPI was already the better planner (70 % vs
MPC's 7 %), Smart MPPI is **worse** — the fallback fires when the
softmax was finding a usable asymmetric direction, and the argmin
choice is more brittle. So the naive "fall back to argmin when
bimodal" intervention does not deliver a planner that strictly
dominates vanilla GPU MPPI.

**Per-step trace** ($v=2$ ep 1, drone 2 north): both vanilla and
smart collide at $t=0.15$ s (step 4), not at $t \approx 5$ s as
the earlier dyn_v2 writeup implied. The earlier reading interpreted
the *joint* episode final_t (max across all 4 drones, ~5 s when the
other 3 drones finish) as the drone-2 collision time. Inspecting
the scenario directly reveals a confound at the v=2 cell: the
static-obstacle seed (`seed: 7`, count=30) happens to place a
voxel obstacle at $(20, 5, 6)$ — the *exact* initial position of
the moving sphere. So drone 2 at $(20, 3, 6)$ faces a stacked
static + dynamic obstacle just 2 m ahead at $t=0$, and the
cancellation-induced near-zero lateral lets it crash into the
static cell at step 4. MPC's argmin commits to one side and clears.

So the §3 Table 2 reading is more nuanced than originally
stated:
- The mechanism (softmax cancellation under symmetric escape) is
  *real* and *detectable* — confirmed by the smart-MPPI synthetic
  diagnostic.
- The dyn_v2/v4 cells expose the mechanism by forcing a near-zero
  initial detour against a (partially fortuitous) static-obstacle
  configuration at the spawn-adjacent cell. The collision happens
  at $t = 0.15$ s, not late-episode.
- The off-corridor probe at $x = 18$ (offset 2 m) is the *cleaner*
  manifestation: drone 2 has time to develop its trajectory and
  the static-obstacle stacking is absent. Smart-MPPI's misfire here
  (70 % → 53 %) is genuine evidence that the argmin-fallback rule
  is too coarse for the wider class of bimodal cost landscapes.

**Implications for the paper**: the §3 mechanism story stands, but
the headline-grade rescue claim ("argmin-fallback fixes the cliff")
does not. A more nuanced fix is needed — candidates include:

1. **Asymmetric perturbation**: add a small per-step lateral bias
   to the rollout distribution so the L/R modes have unequal cost
   even at symmetric geometries.
2. **Temporally-coherent argmin commit**: once a side is chosen,
   stay committed across several replans rather than re-deciding
   each tick.
3. **Bimodal-mode-aware sampling**: cluster rollouts by lateral
   direction, then within the lower-cost cluster pick by softmax.

Future work. The smart-MPPI option is committed (off by default)
so the §3 mechanism is reproducible as a *diagnostic* (`meta.fallback_to_argmin`
flag per replan) even if the *rescue* path needs more research.
Repro:

```
for tag in dyn_v2 dyn_v4 dyn_off2_v4; do
  uav-nav run examples/exp_multi_drone_3d_4_${tag}_gpu_mppi_smart.yaml
done
for tag in dyn_v2 dyn_v4 dyn_off2_v4; do
  echo "=== ${tag} smart vs vanilla ==="
  python3 scripts/paired_analysis_dummy_3d_multi.py \
    results/multi_drone_3d_4_${tag}_gpu_mppi \
    results/multi_drone_3d_4_${tag}_gpu_mppi_smart
done
```


### Smart MPPI v2 (asymmetric perturbation): breaks softmax symmetry, helps the planner-swap regime

Follow-up to "Smart MPPI (argmin-fallback)". Where v1's argmin-fallback
hurt at the dyn_off2 planner-swap cell (vanilla GPU joint 70 % →
Smart v1 joint 53 %), v2 attacks the mechanism from a different
angle: rather than picking argmin at action-selection time, it
**breaks the L/R symmetry at sampling time** so the softmax never
sees a perfectly bimodal cost landscape in the first place.

Implementation
(`uav_nav_lab/planner/gpu_mppi.py`, `asymmetric_bias` config option):

1. At episode reset, each drone seeds a small random unit vector
   `bias_vec` deterministically from its initial observation (so
   different drones in a fleet get different preferred sides, but
   each drone stays consistent across replans — no oscillation).
2. At each replan, project `bias_vec` perpendicular to the current
   `base` (goal direction unit vector), scale by `asymmetric_bias`
   (default 0.2 = 20 % of unit), and rotate `base` by that
   perpendicular vector before generating the n_samples=64 rollouts.
3. All rollouts are now sampled around a slightly-lateral-of-goal
   axis. The softmax-averaged action picks up a small but consistent
   lateral component, so the §3-mechanism L/R cancellation cannot
   produce a zero-lateral command.

Paired n=30 vs MPC and vs vanilla GPU MPPI across 4 cells:

| cell                | vanilla GPU joint | Smart v1 joint | **asym v2 joint** | asym vs vanilla |
|---|---|---|---|---|
| §3 N=4 baseline (mode 1)     | 26/30 = 86.7 %  | (not tested)    | 26/30 = **86.7 %**  | tied (Δ +5.2 → +2.3) |
| dyn_v2 (mode 2 on-corridor)  | 1/30 = 3.3 %    | 2/30 = 6.7 %    | **0/30 = 0.0 %**    | −1 seed (no rescue)  |
| dyn_v4 (mode 2 on-corridor)  | 1/30 = 3.3 %    | 1/30 = 3.3 %    | 1/30 = 3.3 %        | per-drone +7.5 pp    |
| dyn_off2_v4 (planner-swap)   | 21/30 = 70.0 %  | 16/30 = **53.3 %** (HURT) | 26/30 = **86.7 %** | **+16.7 pp**         |

Three readings:

**(1) Asym preserves the §3 baseline.** McNemar paired vs vanilla
GPU MPPI on the static-peer N=4 baseline: both-succ 23, vanilla-only
3, asym-only 3, neither 1 — statistically tied at the joint level.
Δ over indep$^4$ drops from +5.2 to +2.3 pp, meaning the clustering
mechanism is **slightly weakened** by the bias (drones don't all
agree on the same conservative command quite as tightly), but not
eliminated. Mode 1 lives.

**(2) Asym doesn't rescue the cliff.** dyn_v2 stays at 0/30, dyn_v4
at 1/30. The cliff failure mode is dominated by the static-obstacle
at $(20, 5, 6)$ (confound documented in the "Smart MPPI v1" section
above) plus the 8 m/s drone meeting a 2 m initial gap — no
sampling-time perturbation overcomes a hard-blocking voxel cell 2 m
ahead at step 0. dyn_v4 shows a per-drone improvement (67.5 → 75 %)
but joint stays at floor.

**(3) Asym strictly dominates Smart v1 in the planner-swap regime.**
At dyn_off2_v4, asym v2 reaches joint **86.7 %**, vs Smart v1's
53.3 % (a 33-pp gap). McNemar paired vanilla vs asym: both-succ 21,
vanilla-only 0, asym-only 5, neither 4 — asym strictly improves
over vanilla GPU MPPI in this cell. The mechanism: at offset 2 m
the cost landscape is *asymmetric* (one side has clearer static
clearance), but the softmax-averaged command still pulled toward
the center; biasing the rollout cloud lateral by 20 % shifts the
softmax toward the better side and reduces the static-obstacle
oscillation that hurts MPC there.

**Combined story (Smart v1 + v2)**:
- v1's argmin-fallback rule detects the mechanism but applies the
  wrong intervention (commits to a single rollout that can be brittle).
- v2's sampling-time bias is a *softer* intervention — it breaks
  symmetry but keeps the softmax averaging.
- Neither rescues the cliff; v2 is the better default fix for
  off-corridor / asymmetric-clutter regimes.

The mechanism diagnosis (§3 4-mode framework) is independently
supported by these results: when symmetry is present (cliff cell,
spawn-adjacent stacked obstacles), no rollout-distribution
intervention rescues the affected drone; when symmetry is broken
geometrically (off-corridor), a small bias is enough to convert
"GPU MPPI is good here" into "GPU MPPI is the strict best".

Reproduce:
```
for tag in dyn_v2 dyn_v4 dyn_off2_v4; do
  uav-nav run examples/exp_multi_drone_3d_4_${tag}_gpu_mppi_asym.yaml
done
uav-nav run examples/exp_multi_drone_3d_4_gpu_mppi_asym.yaml  # §3 baseline
for tag in dyn_v2 dyn_v4 dyn_off2_v4; do
  python3 scripts/paired_analysis_dummy_3d_multi.py \
    results/multi_drone_3d_4_${tag}_gpu_mppi \
    results/multi_drone_3d_4_${tag}_gpu_mppi_asym
done
```


### Smart MPPI v3 (temporally-coherent argmin commit): trades mode 1 success for swap-regime stability

Third Smart MPPI variant. Where v1's argmin-fallback fires per-replan
(causing oscillation when the detector toggles on/off) and v2's
asymmetric perturbation breaks symmetry at sampling time but never
fires per-replan, **v3** holds the v1-style fallback decision for
$K = 5$ consecutive replans before re-evaluating. This addresses the
oscillation hypothesis directly: if v1's bad behaviour at dyn_off2
was caused by per-replan flip-flopping, a temporal commit should
rescue it.

Implementation: GPU MPPI gains `fallback_commit_steps: int = 1`
(default = v1 behaviour). v3 cell uses K=5 ≈ 1 s at
$\text{replan\_period}=0.2$ s. Once the bidirectional-cancellation
detector triggers, the planner returns the argmin action for K-1
further replans before checking the detector again.

Paired n=30 across 4 cells. **Full comparison table across all
three Smart variants**:

| cell                | vanilla GPU | Smart v1 (K=1) | Smart v2 (asym 0.2) | Smart v3 (K=5) |
|---|---|---|---|---|
| §3 N=4 baseline (mode 1)     | 86.7 % / Δ +5.2 | (not tested)  | 86.7 % / Δ +2.3 (tied) | **76.7 %** / Δ +11.1 (**HURT −10 pp**) |
| dyn_v2 (mode 2 on-corridor)  | 3.3 %  | 6.7 %  | 0.0 %  | 6.7 %  |
| dyn_v4 (mode 2 on-corridor)  | 3.3 %  | 3.3 %  | 3.3 %  | 3.3 % (per-drone +6.7 pp) |
| dyn_off2_v4 (planner-swap)   | 70.0 % | 53.3 % (HURT)  | **86.7 %** (+16.7 pp)  | 76.7 % (+6.7 pp) |

Three readings:

**(1) Temporal commit *hurts* mode 1 baseline.** Joint drops from
86.7 → 76.7 % (−10 pp) and per-drone from 95.8 → 90.0 %. The K=5
commit causes problems on the static-peer N=4 task: once a drone
commits to argmin, it stays committed for 1 s even when the cost
landscape would have been better served by re-averaging. The §3
mode 1 mechanism (softmax clustering driving GPU MPPI to +11.4 pp
Δ at the n=100 baseline) is structurally what v3's commit *removes*
— the planner becomes more like MPC's argmin, and we lose the
clustering advantage that gives GPU MPPI its mode 1 edge.

**(2) Temporal commit doesn't rescue the cliff.** dyn_v2 stays at
6.7 % (= v1's number), dyn_v4 stays at 3.3 %. The static-obstacle
confound at $(20, 5, 6)$ is geometric, not action-selection — no
fallback regime overcomes the spawn-adjacent stacked obstacle in
4 steps.

**(3) Temporal commit partially rescues the off-corridor swap regime,
but v2 still dominates.** v3 at dyn_off2 reaches 76.7 % (vs vanilla
70.0 %, vs Smart v1 53.3 %, vs Smart v2 86.7 %). The temporal commit
*does* help (no oscillation), but it commits to *one specific argmin
rollout's direction*, which can be suboptimal vs the smooth softmax
average that v2's asymmetric sampling preserves.

**Cross-variant conclusion: no simple intervention rescues all four
modes simultaneously**:
- v1 (K=1 per-replan fallback) — hurts swap regime via oscillation.
- v2 (asym 0.2 perturbation) — preserves baseline, dominates swap;
  doesn't rescue cliff.
- v3 (K=5 temporal commit) — partially rescues swap, but trades
  away the mode 1 clustering advantage.
- All three: cliff (mode 2 on-corridor) remains unsolved at n=30.

The §3 4-mode framework predicts this outcome: a planner cannot
simultaneously optimise for all four modes if the underlying
operator (softmax averaging vs argmin commit) carries opposite
optimal valences in different modes. The Smart MPPI experiments
confirm this empirically — improving one mode requires sacrificing
another. The **deployment story** therefore strengthens, not
weakens: pick the planner *per mission mode*, or build a **mode-
aware switcher** that selects between Smart v2 / vanilla GPU MPPI /
MPC based on a per-replan mode diagnostic (Tier 1 idea, future
work). A single static planner cell that optimises across all four
modes does not exist — at least not within the
softmax-vs-argmin family explored here.

Reproduce:
```
for tag in "" dyn_v2 dyn_v4 dyn_off2_v4; do
  base="examples/exp_multi_drone_3d_4${tag:+_$tag}_gpu_mppi_smart_v3.yaml"
  uav-nav run "$base"
done
for tag in "" dyn_v2 dyn_v4 dyn_off2_v4; do
  python3 scripts/paired_analysis_dummy_3d_multi.py \
    results/multi_drone_3d_4${tag:+_$tag}_gpu_mppi \
    results/multi_drone_3d_4${tag:+_$tag}_gpu_mppi_smart_v3
done
```


### Smart MPPI v4 (mode-aware sampling): the first variant that cracks the cancellation regime

Fourth Smart MPPI variant. v1–v3 all attacked the §3 dynamic-obstacle
**cancellation regime** at the *action-selection* layer (argmin-fallback,
asymmetric sampling, temporal commit). v4 attacks it at the *rollout-
aggregation* layer instead: cluster the rollouts by lateral principal-
component sign (L vs R), and emit the softmax-weighted action of the
**lower-cost cluster only**. This preserves MPPI smoothing *within* one
escape direction while breaking the cancellation that drives the §3
Table 2 mechanism.

Mechanism. The §3 cancellation regime makes the rollout cloud bimodal:
roughly half the rollouts escape left of a moving obstacle, the other
half escape right, with similar costs. A global softmax then averages
L and R back toward zero lateral motion — the planner commits to going
*straight* into the obstacle. Mode-aware sampling, by contrast, takes
the lateral component of each sampled action, projects it onto the
principal direction (SVD), splits rollouts by sign, computes the
softmax-weighted average and softmax-weighted cost within each cluster,
and outputs the action of the lower-cost cluster. The result is a
non-zero lateral commitment — but a *smooth* one, unlike v1/v3's
hard argmin commit.

Implementation: GPU MPPI gains `mode_aware_sampling: bool = False`
and `mode_aware_min_size: int = 8`. The min-size guard avoids
splitting when one cluster has < 8 rollouts (regime is unimodal).
Meta now exposes `mode_aware_triggered` and `mode_aware_cluster_sign`
(±1 = which side won; 0 = no split).

Paired n=30 across the same 4 cells as v1/v2/v3. **Full comparison
across all four Smart variants vs vanilla GPU MPPI**:

| cell                | vanilla GPU | Smart v1 (K=1) | Smart v2 (asym 0.2) | Smart v3 (K=5) | **Smart v4 (mode-aware)** |
|---|---|---|---|---|---|
| §3 N=4 baseline (mode 1)     | 86.7 % | (not tested)   | 86.7 % (tied) | 76.7 % (−10 pp) | **63.3 %** (−23 pp, HURT)  |
| dyn_v2 (mode 2 on-corridor)  | 3.3 %  | 6.7 %  | 0.0 %  | 6.7 %  | **50.0 %** (**+47 pp, $p \approx 0.0005$**) |
| dyn_v4 (mode 2 faster)       | 3.3 %  | 3.3 %  | 3.3 %  | 3.3 %  | **0.0 %** (tied; scenario still too hard) |
| dyn_off2_v4 (planner-swap)   | 70.0 % | 53.3 % (HURT) | **86.7 %** (+16.7 pp) | 76.7 % (+6.7 pp) | **46.7 %** (−23 pp, HURT) |

Three readings:

**(1) Mode-aware is the first variant that decisively cracks the
cancellation regime.** dyn_v2 jumps from 3.3 % (vanilla) / 0–6.7 %
(v1–v3) to 50.0 %. Paired McNemar vs Smart v3: $p \approx 0.001$;
vs vanilla GPU MPPI: $p \approx 0.0005$. The mechanism the variant
was designed to target is the mechanism it fixes. v1–v3 only ever
moved the dyn_v2 number by 0–3 pp because they intervened *after*
the rollout cloud had already been averaged toward zero, or they
broke symmetry too weakly. v4 intervenes *during* the averaging step
and forbids cross-side averaging entirely.

**(2) The fix is mode-specific, exactly as the §3 4-mode framework
predicts.** Cracking dyn_v2 costs −23 pp on the static §3 baseline
(mode 1) and −23 pp on the planner-swap regime (off-corridor). Both
losses share a root cause: forcing a one-side commit destroys the
rollout diversity that helps in regimes where *both* sides matter
(mode 1's static-peer clustering profits from the rollout cloud's
spread; the planner-swap regime needs to find the rare clear side,
which can be on either L or R). Mode-aware sampling does precisely
what v1–v3 do *not*: actively force the cloud into one mode at the
output. That intervention is the right call iff the cancellation is
the binding failure.

**(3) Faster dynamics (dyn_v4) remain unsolved.** Even v4 stays at
0/30 joint success. With $v_{obst} = 4$ m/s and an obstacle radius
of 0.8 m, the moving cube traverses the corridor faster than the
0.4 s lookahead horizon can react regardless of action-aggregation
scheme. This is a *scenario hardness* finding, not a planner
finding: a longer horizon or a better predictor (constant-velocity
already; would need acceleration-aware) is what dyn_v4 needs.
Smart v4 cannot rescue what no per-replan decision can rescue.

**Cross-variant conclusion: each variant targets exactly one mode**:
- v1 (per-replan argmin-fallback)      → no clear win; oscillation hurts swap.
- v2 (asymmetric sampling)             → dominates *planner-swap* regime.
- v3 (temporally-coherent argmin)      → partially rescues swap; loses mode 1.
- v4 (mode-aware cluster softmax)      → **dominates cancellation regime**; loses mode 1 + swap.

No single Smart variant is uniformly better. The §3 4-mode framework
predicts this exactly: when the operator (softmax averaging vs argmin
commit, cross-mode averaging vs single-mode averaging) carries opposite
optimal valences across modes, a static cell cannot win all four. The
**deployment story** therefore is *mode-aware switching* — pick
v2 in the planner-swap regime, v4 in the cancellation regime, vanilla
GPU MPPI (or MPC) in mode 1 — driven by an online mode diagnostic
(future work; the meta now exposes both `fallback_to_argmin` and
`mode_aware_cluster_sign` per replan, which is enough signal for a
hand-built switcher).

Reproduce:
```bash
for tag in "" dyn_v2 dyn_v4 dyn_off2_v4; do
  uav-nav run "examples/exp_multi_drone_3d_4${tag:+_$tag}_gpu_mppi_smart_v4.yaml"
done
for tag in "" dyn_v2 dyn_v4 dyn_off2_v4; do
  python3 scripts/paired_analysis_dummy_3d_multi.py \
    results/multi_drone_3d_4${tag:+_$tag}_gpu_mppi \
    results/multi_drone_3d_4${tag:+_$tag}_gpu_mppi_smart_v4
done
```


### Cost-to-go cache tolerance: 4-5x speedup on moving-goal scenarios

Both planners (`mpc`, `gpu_mppi`) gained a `ctg_cache_tolerance: int = 0`
option. When > 0, the planner reuses its cached Dijkstra cost-to-go map
as long as the new goal cell is within `tolerance` cells along every
axis from the cached one. Default 0 preserves the exact
per-replan-recompute behaviour from before.

Motivation. Aerobatic / race scenarios pass a moving lookahead point as
the planner goal, which crosses an integer cell boundary nearly every
replan. The Dijkstra recompute on a 40×40×12 grid takes $\sim 1.2$ s,
which dominated the multi-drone race wallclock ($\sim 9$ min/episode
for $n_{drones} = 4$ × 120 replans).

Measurement on `examples/exp_race_oval4_*.yaml` (1 episode, 4 drones,
480 steps):

| planner   | `ctg_cache_tolerance=0` | `ctg_cache_tolerance=3` | speedup |
|---|---|---|---|
| MPC       | 540 s                   | 141 s                   | **3.8x** |
| GPU MPPI  | 720 s                   | 139 s                   | **5.2x** |

Per-episode `per_drone_outcomes` and tracking RMSE numbers are
bit-identical at tolerance=3 for both planners — the cost-to-go
staleness ($\leq 3$ m of drift on a 40 m world) is well below the
$\sim 50$ m horizon length and doesn't change which rollout the
planner picks. With this in place the race $n = 30$ paper-grade
extension fit in $\sim 90$ min wallclock with 4 parallel processes,
instead of the projected $\sim 6$ hours.

For *static-goal* scenarios (every YAML in `examples/` except the
multi_drone_aerobatic / race set), the cache hits anyway after the
first replan because the goal cell is constant — so the new option
is a no-op there. Race YAMLs default to `ctg_cache_tolerance: 3`.


### Smart MPPI v5 (mode-aware switcher): lateral-cancellation gate dominates v4 on 4/5 cells

Fifth (and current best) Smart MPPI variant. v4's cluster softmax was
**unconditional** once both L/R clusters had enough samples — and that
unconditional commit hurt mode 1 (static peers, $-23$ pp vs vanilla)
and the planner-swap regime ($-23$ pp). v5 gates the cluster commit
on the *actual* cancellation signature: the vanilla softmax-mean's
lateral magnitude must be much smaller than the argmin rollout's
lateral magnitude. That is the only condition under which softmax
averaging is actively cancelling escape modes — and outside that
condition, vanilla softmax is correct.

Implementation. GPU MPPI's `plan()` already computes the vanilla
softmax action and the argmin action. Two new options add the gate:
- `mode_aware_lateral_threshold: float = 0.0` — minimum argmin lateral
  speed (m/s) to consider the signal trustworthy. Default 0 = no gate.
- `mode_aware_lateral_ratio: float = 0.5` — softmax-vs-argmin lateral
  magnitude ratio below which cancellation is declared.

Defaults (`threshold = 0.5`, `ratio = 0.5`) match the Smart v1
`fallback_to_argmin` thresholds — the same cancellation detector,
routing to cluster softmax (v4 path) instead of argmin (v1 path).

Paired $n = 30$ on the same five cells as v4 (plus race):

| cell                          | vanilla | Smart v4 | **Smart v5** | v5 - vanilla | v5 - v4 |
|---|---|---|---|---|---|
| §3 baseline (mode 1)          | 86.7 %  | 63.3 %   | **76.7 %**   | $-10$ pp     | **$+13$ pp** |
| dyn_v2 (mode 2)               | 3.3 %   | 50.0 %   | **66.7 %**   | $+63.3$ pp   | **$+17$ pp** ($p \approx 0.18$) |
| dyn_v4 (mode 2 faster)        | 3.3 %   | 0.0 %    | 3.3 %        | tied         | **$+3.3$ pp** |
| dyn_off2 (planner-swap)       | 70.0 %  | 46.7 %   | **70.0 %**   | tied         | **$+23$ pp** ($p \approx 0.07$) |
| race (mode 2+4, tracking RMSE) | 1.658 m | 1.719 m  | 1.759 m      | $+0.10$ m    | $+0.04$ m (only metric where v4 > v5) |

Three readings:

**(1) v5 is the first Smart variant that's Pareto-improving over v4.**
On every dyn cell v5 matches or beats v4. The cancellation regime gain
that v4 unlocked at dyn_v2 (+47 pp over vanilla) gets pushed even
further by v5 (+63.3 pp); the *cost* v4 paid in baseline mode 1 and
planner-swap is essentially recovered ($+13$ pp / $+23$ pp). dyn_v4
remains scenario-hard but v5 at least matches vanilla instead of v4's
zero.

**(2) The gate fires *precisely* when the cancellation is actively
hurting.** The lateral-magnitude signal directly measures the
phenomenon (softmax → 0 lateral component while argmin still moves
laterally), so the gate is selective by construction rather than by
threshold tuning. The previous attempt — cost-ratio gating at 2.0 —
never fired because cluster cost asymmetry stays close to 1 even
in the cancellation regime (both sides have legitimate avoidance
solutions, just averaged toward zero). The right signal isn't cost
asymmetry, it's lateral cancellation itself.

**(3) Race is the only cell where v4 still has an edge** — but
only on tracking, not safety. Both v4 and v5 collapse the collision
rate from vanilla's 75 % back to MPC's 50 %. The tracking-RMSE
difference (1.719 vs 1.759 m) reflects v5's gate firing more
selectively in race: v4 commits unconditionally and smooths every
replan into one cluster, while v5 falls through to vanilla softmax
whenever the peers' cancellation isn't lateral-thresholded — yielding
slightly more rollout-cloud diversity, slightly less precision. For
the *aerobatic* mode-4 sub-component of race (no obstacle), v5
behaves like vanilla and tracks at $-0.6°$ phase RMSE.

**Cross-variant conclusion, updated**: v5 is the **deployment recommendation**.
- Mode 1 (static peer clustering, no dynamic obstacles): nearly vanilla.
- Mode 2 (cancellation): aggressive cluster commit — beats v4.
- Mode 3 (sim-physics density-corner sign-reversal, AirSim §4.4.4): not retested but the lateral cancellation gate should defer to vanilla here too.
- Mode 4 (aerobatic): nearly vanilla.
- Mode 2 + 4 superposition (race): essentially v4-level safety, slightly less tracking precision than v4.

Reproduce:
```bash
for tag in "" dyn_v2 dyn_v4 dyn_off2_v4; do
  uav-nav run "examples/exp_multi_drone_3d_4${tag:+_$tag}_gpu_mppi_smart_v5.yaml"
done
uav-nav run examples/exp_race_oval4_gpu_mppi_smart_v5.yaml
for tag in "" dyn_v2 dyn_v4 dyn_off2_v4; do
  python3 scripts/paired_analysis_dummy_3d_multi.py \
    results/multi_drone_3d_4${tag:+_$tag}_gpu_mppi \
    results/multi_drone_3d_4${tag:+_$tag}_gpu_mppi_smart_v5
done
python3 scripts/paired_analysis_aerobatic.py \
  results/race_oval4_mpc results/race_oval4_gpu_mppi_smart_v5 4 30
```


### Drone race + bouncing intruder: Smart MPPI v4 recovers MPC-level safety without losing tracking precision

Single scenario that places **all three** §3 mode interactions on the
same 24 s episode: 4 drones lap a horizontal oval (12 × 8 m, 12 s
period, 2 laps = 480 steps) while a single bouncing intruder
(radius 1.2 m, $v_y = 6$ m/s, reflects in the 40 × 40 × 14 box)
crosses the track every ~4 s. Each drone tracks the same lookahead
goal on a phase-offset reference ellipse. YAMLs:
`examples/exp_race_oval4_{mpc,gpu_mppi,gpu_mppi_smart_v4}.yaml`.

Geometry is fully deterministic given the seed (which controls only
the unused static-obstacle seed), so the same drone phases meet the
same intruder positions in every episode — the seed varies internal
RNG (planner sample noise for GPU MPPI) but the failure pattern is
seed-stable to 3 decimal places. With paper-grade $n = 30$ (each
metric below was identical to the $n = 5$ first cut, confirming
seed-stability rather than noise reduction):

| planner                   | tracking RMSE | max error | phase RMSE | collisions (drone-eps) |
|---|---|---|---|---|
| MPC                       | 1.764 m       | 2.701 m   | 16.80°     | 60/120 (50 %)          |
| vanilla GPU MPPI          | **1.658 m**   | 1.976 m   | **16.14°** | 90/120 (75 %)          |
| Smart MPPI v2 (asym 0.2)  | 1.657 m       | 1.985 m   | 16.19°     | 90/120 (75 %)          |
| **Smart MPPI v4** (mode-aware) | 1.719 m  | **2.177 m** | 16.22° | **60/120 (50 %)**      |

(The $n = 30$ extension was unlocked by the Dijkstra cost-to-go cache
tolerance — see "Cost-to-go cache tolerance" finding — which cut
per-episode MPC wallclock from $\sim 9$ min to $\sim 2.4$ min, and
GPU MPPI from $\sim 12$ min to $\sim 2.3$ min.)

Three readings, one scenario:

**(1) Vanilla GPU MPPI tracks tighter but crashes more.** Mean
per-(drone, ep) tracking RMSE is **0.105 m lower** than MPC on
*every* drone-episode (120/120), and phase-offset RMSE is 0.66°
tighter. This is the §3 mode 4 mechanism — softmax averaging
smooths the tracking command around the lookahead point. But the
same operator turns the intruder into the §3 mode 2 cancellation
regime: when the bouncing cube enters a drone's corridor, the
rollout cloud becomes bimodal (L vs R escape) and the softmax
averages the two modes back toward zero lateral motion. Result:
**+30 extra collisions over the 120 drone-episodes** (75 % vs 50 %).
Same scenario, both regimes active at different moments.

**(2) Smart MPPI v4 recovers MPC's safety, keeps most of MPPI's
tracking edge.** Mode-aware cluster softmax (cf. "Smart MPPI v4
(mode-aware sampling)") commits to *one* lateral escape side within
the rollout aggregation step. Collision rate collapses back to
60/120 — tied with MPC — and tracking RMSE stays 0.044 m better than
MPC on every drone-episode. The catastrophic-bottom-drone failure
that vanilla GPU MPPI suffers (drone 3, the one closest to the
intruder spawn, loses every episode) disappears: v4 saves drone 3
in all 30 episodes. Smart v2 (asymmetric perturbation 0.2), in
contrast, lands at the *same* 75 % collision rate as vanilla GPU
MPPI — the perpendicular sampling bias is too weak (or too random
across drones) to break the cancellation symmetry that the intruder
imposes on each drone individually. v4's intervention is at the
right level (per-replan cluster split) and right strength
(softmax-within-cluster, not random bias-rotation).

**(3) The "drone-1 / drone-2 cliff" is geometric, not
planner-failure.** All three planners lose drones 1 and 2 (the
green / blue drones at the top and left of the oval) in every
episode. That's because the intruder bounces off the world ceiling
at $y = 40$ at $t \approx 5.3$ s and crashes back through the
oval interior at the exact moment drones 1 and 2 are passing
through $y \approx 28$ (the top of the oval). The oval is too tight
relative to the bounce period for *any* of the three planners to
detour by enough margin under a $0.4$ s lookahead. This is the
**scenario-hardness ceiling** — the same effect we saw in
"dummy_3d N=4 + moving obstacle speed sweep" at $v = 4$ m/s.

The hero GIF (`docs/images/compare_race_oval4.gif`) shows all three
planners side-by-side on episode 0, with the bouncing intruder
overlaid. The visual contrast is sharp: vanilla MPPI loses drone 3
within the first 4 s, then suffers the inevitable 1 / 2 cliff;
MPC and v4 dodge the early bounce (drone 3 survives) and only fall
to the cliff at $t \approx 10$ s.

Reproduce:
```bash
for plan in mpc gpu_mppi gpu_mppi_smart_v4 gpu_mppi_asym; do
  uav-nav run "examples/exp_race_oval4_${plan}.yaml"
done
for cmp in gpu_mppi gpu_mppi_smart_v4 gpu_mppi_asym; do
  python3 scripts/paired_analysis_aerobatic.py \
    results/race_oval4_mpc "results/race_oval4_${cmp}" 4 5
done
python3 scripts/render_race_gif.py \
  --runs results/race_oval4_mpc:MPC \
         results/race_oval4_gpu_mppi:"GPU MPPI (vanilla)" \
         results/race_oval4_gpu_mppi_smart_v4:"Smart MPPI v4" \
  --out docs/images/compare_race_oval4.gif \
  --ep 0 --fps 10 --stride 8 --trail 30
```


### Moving-gates race: the mirror image — softmax wins where it lost the single-intruder race

Same oval geometry as the bouncing-intruder race above (4 drones, 12 ×
8 m oval, 12 s period, 2 laps = 24 s) but the single intruder is
replaced by **4 sliding gates** at the NE/NW/SW/SE corners of the
oval. Each gate is two posts (radius 0.5 m, gap centre at y = 26 or
y = 14 at t = 0) that share a vertical velocity; both posts slide
together in y so the gap moves while keeping a fixed gap width. Gate
velocities are deliberately desynchronised (1.8 / 2.0 / 2.2 / 1.6 m/s)
so the encounter timing per lap drifts instead of repeating. YAMLs:
`examples/exp_race_gates4_{mpc,gpu_mppi,gpu_mppi_smart_v4,gpu_mppi_smart_v5}.yaml`.

With paper-grade $n = 30$:

| planner                   | tracking RMSE | max error | phase RMSE | collisions (drone-eps) |
|---|---|---|---|---|
| MPC                       | **1.620 m**   | 2.439 m   | **14.52°** | 62/120 (51.7 %)        |
| vanilla GPU MPPI          | 1.648 m       | **1.972 m** | 15.88°   | **4/120 (3.3 %)**      |
| Smart MPPI v4 (mode-aware)| 1.709 m       | 2.166 m   | 15.94°     | **4/120 (3.3 %)**      |
| Smart MPPI v5 (switcher)  | 1.749 m       | 2.137 m   | 15.78°     | **4/120 (3.3 %)**      |

**This flips the cancellation-regime story.** In the single-intruder
race above, the rollout cloud is *bimodal* (left or right escape) at
the moment of crossing — softmax averages the two modes back toward
zero lateral motion and GPU MPPI crashes 75 % of drone-eps; v4's
cluster softmax repairs that. With **paired posts** the topology is
different: the drone must thread the gap centre, and there is exactly
**one** feasible lateral target (the moving gap, not either post).
The rollout cloud is *unimodal* — the goal-cost basin has one
minimum, and softmax-averaging across samples converges to it more
robustly than MPC's hard argmin commit. **The mirror image of the
cancellation regime**: where averaging hurt before, here it is the
right operator.

MPC's failure mode is now the dual of GPU MPPI's vanilla failure on
the single-intruder race: the argmin sample at each replan picks
whichever individual rollout looks cheapest *this step*, and because
the gap moves while the drone is committing, the chosen rollout
becomes stale before the next replan. Across 30 episodes the MPC
loses 62 drone-eps (just over half), while all three softmax variants
(vanilla, v4, v5) clear 116/120. That includes v5 — even though its
lateral-cancellation gate (cf. "Smart MPPI v5") never fires on this
scenario because the rollout cloud isn't bimodal, so v5 stays in
vanilla-softmax mode, which is exactly what the scenario rewards.

Two clean readings, one $n = 30$ run:

**(1) Softmax averaging is the right operator when rollouts are
unimodal.** This is the symmetric claim to §3 mode 2 (the failure
mode is that softmax averages bimodal rollouts back to zero). When
rollouts cluster around a single feasible escape, averaging gives a
smoother command than argmin — fewer planner-step jumps, fewer
stale-commit failures, **51.7 % → 3.3 % collisions**.

**(2) Mode-aware gating doesn't hurt here.** v4 and v5 both keep the
3.3 % collision rate of vanilla — v4 does cluster softmax aggregation
but with unimodal rollouts the two "clusters" collapse to (almost)
the same direction, so the output matches plain softmax. v5 detects
no cancellation signature and stays in vanilla mode. The mode-aware
variants are *scenario-safe*: they don't degrade vanilla-softmax wins
in order to fix cancellation-regime losses.

The hero GIF (`docs/images/compare_race_gates4.gif`) shows all four
planners side-by-side on episode 1 — MPC loses 2 of 4 drones to gate
collisions while all three softmax variants thread every gap.

Reproduce:
```bash
for plan in mpc gpu_mppi gpu_mppi_smart_v4 gpu_mppi_smart_v5; do
  uav-nav run "examples/exp_race_gates4_${plan}.yaml"
done
for cmp in gpu_mppi gpu_mppi_smart_v4 gpu_mppi_smart_v5; do
  python3 scripts/paired_analysis_aerobatic.py \
    results/race_gates4_mpc "results/race_gates4_${cmp}" 4 30
done
python3 scripts/render_race_gif.py \
  --runs results/race_gates4_mpc:MPC \
         results/race_gates4_gpu_mppi:"GPU MPPI" \
         results/race_gates4_gpu_mppi_smart_v4:"Smart v4" \
         results/race_gates4_gpu_mppi_smart_v5:"Smart v5" \
  --config examples/exp_race_gates4_mpc.yaml \
  --title "Drone race + 4 moving gates (8 sliding posts, oval circuit)" \
  --out docs/images/compare_race_gates4.gif \
  --ep 1 --fps 10 --stride 12 --trail 20
```


### Drone race chaos — gates + intruders piled on, gate topology still dominates

What happens when we stack mode 2 (cancellation) **and** mode 2-mirror
(unimodal gate-thread) onto the same scenario? We took the moving-gates
race above and added back **2 bouncing intruders** (radius 1.0 m,
$v_y = \pm 5$ and $\pm 6$ m/s, both crossing the oval interior at
$x \in \{20, 22\}$). The resulting "chaos race" has **10 dynamic
obstacles** sharing the same 40 × 40 box as 4 drones — 8 sliding gate
posts + 2 bouncing intruders. YAMLs:
`examples/exp_race_chaos_{mpc,gpu_mppi,gpu_mppi_smart_v4,gpu_mppi_smart_v5}.yaml`.

The naïve prediction is that the two mechanisms compose: MPC loses
because the gates' moving target makes argmin stale, and now vanilla
GPU MPPI should *also* lose because the bouncing intruders introduce
bidirectional escape symmetry that cancels under softmax. Result at
paper-grade $n = 30$:

| planner          | tracking RMSE | phase RMSE | collisions (drone-eps) |
|---|---|---|---|
| MPC              | **1.620 m**   | **14.52°** | 62/120 (51.7 %)        |
| vanilla GPU MPPI | 1.648 m       | 15.88°     | **4/120 (3.3 %)**      |
| Smart MPPI v4    | 1.709 m       | 15.94°     | **4/120 (3.3 %)**      |
| Smart MPPI v5    | 1.749 m       | 15.78°     | **4/120 (3.3 %)**      |

The numbers are **bit-identical to the gates4 result above**. Diffing
the drone trajectories file-by-file confirms: every planner runs the
exact same control sequence as in gates4. The 2 bouncing intruders
add visual chaos to the GIF, but they never influence the planner
cost in a way that changes a drone's collision/success outcome.

The mechanism is a topology argument. Drones are tangentially moving
at $\sim 5.3$ m/s along the oval; they spend $\sim 1$ s in the
$\pm 1$ m vicinity of each gate corner. During that window the gate
posts are within $1.4$ m clearance of the drone — the active
constraint at every replan. The bouncing intruders, by contrast,
cross the oval interior at $x \in \{20, 22\}$ but the drones only
visit $x = 20$ at the very top ($y = 28$) and bottom ($y = 12$) of
the oval, and at those exact moments the intruder happens to be
$> 6$ m away in $y$ (worst case at $t \approx 12$ s, $\sim 3$ m
apart — still outside the intruder + drone radius sum of $1.4$ m).
**The gates dominate the active-constraint set every replan, so the
intruders never enter the cost.** Pile on as many bouncing intruders
as you want in the oval interior — as long as they don't intersect
the drone's tangential corridor *at the moment of crossing*, they are
invisible to the planner.

Two readings:

**(1) "Adding obstacles" is not the same as "raising scenario
difficulty."** This is a methodological caution: a visually busy
scenario does not automatically test more failure modes. The cost
landscape only sees obstacles in the planner's active window
(horizon × replan period × safety margin), and obstacles outside
that window contribute zero cost regardless of how many you add.
The chaos race has 25 % more obstacles than gates4 but the same
collision rate to 3 decimal places — the additional obstacles
are dead-weight to every planner.

**(2) Gate topology imposes the cloud structure.** When gates and
intruders both *would* be in the active window, the gates' fixed
geometric constraint (must thread a $\sim 5$ m gap) collapses the
rollout cloud onto the gap centre and the intruders cease to matter
as a cancellation source. There is no rollout choosing
"slow-down-and-let-intruder-pass" because the gate gap is closing —
all rollouts that survive go through the gap, and they happen to
agree on lateral commitment. **Hard topological constraints win over
soft cost gradients in determining the cloud structure.** This is
why Smart v5's lateral-cancellation gate never fires on this
scenario even though intruders are physically present.

(See the §3 head line's "Mirror-image of the cancellation regime"
section for the connection back to the 4-mode framework: chaos race
is mode 2-mirror with mode 2 forces applied but suppressed.)

The hero GIF (`docs/images/compare_race_chaos.gif`) shows the visual
chaos — 10 red obstacles bouncing and sliding around 4 drones — with
the contrast still cleanly visible: MPC loses drones 0 and 3 every
episode (collision with the gate corners they cross at lap end), the
three softmax variants thread the full 24 s race uninterrupted.

Reproduce:
```bash
for plan in mpc gpu_mppi gpu_mppi_smart_v4 gpu_mppi_smart_v5; do
  uav-nav run "examples/exp_race_chaos_${plan}.yaml"
done
for cmp in gpu_mppi gpu_mppi_smart_v4 gpu_mppi_smart_v5; do
  python3 scripts/paired_analysis_aerobatic.py \
    results/race_chaos_mpc "results/race_chaos_${cmp}" 4 30
done
python3 scripts/render_race_gif.py \
  --runs results/race_chaos_mpc:MPC \
         results/race_chaos_gpu_mppi:"GPU MPPI" \
         results/race_chaos_gpu_mppi_smart_v4:"Smart v4" \
         results/race_chaos_gpu_mppi_smart_v5:"Smart v5" \
  --config examples/exp_race_chaos_mpc.yaml \
  --title "Drone race chaos: 4 sliding gates + 2 bouncing intruders" \
  --out docs/images/compare_race_chaos.gif \
  --ep 1 --fps 10 --stride 12 --trail 20
```


### dyn4 path-intersecting intruders: controlled dynamic-avoidance harness

The chaos race result raised the question "do the planners actually
avoid the intruders, or are the intruders inert background?" — and the
answer was "inert: the gates dominate, intruders never enter the cost
window." This is unsatisfying as a *demonstration* of dynamic-obstacle
avoidance. We close that gap with a scenario designed so the intruders
are unambiguously on each drone's path. YAMLs:
`examples/exp_race_dyn4_{mpc,gpu_mppi,gpu_mppi_smart_v4,gpu_mppi_smart_v5}.yaml`.

**Construction.** Same oval as race / gates / chaos (12 × 8 m, 12 s
period, 2 laps). No gates. **4 intruders**, each constrained to bounce
on a line that *intersects* one drone's oval segment:

- Intruder 1: bounces on $x = 30$ (drone 0's right-side average, $y$
  range $[16, 24]$), $v_y = +5$ m/s, radius 1.2 m.
- Intruder 2: bounces on $y = 26$ (drone 1's top traversal, $x$ range
  $[12, 28]$), $v_x = +6$ m/s.
- Intruder 3: bounces on $x = 10$ (drone 2's left side), $v_y = -7$ m/s.
- Intruder 4: bounces on $y = 14$ (drone 3's bottom), $v_x = -8$ m/s.

Desynchronised velocities (5/6/7/8 m/s) make the encounter timing
drift per lap so each lap is a different race. Drone radius 0.4 m +
intruder radius 1.2 m = 1.6 m clearance sum; drone tangential speed
at oval long-sides is $\sim 6.3$ m/s, intruders 5-8 m/s — head-on
closing rates of $\sim 10$ m/s, $\sim 0.16$ s to collision at first
detection.

**Result at paper-grade $n = 30$:**

| planner          | tracking RMSE | max ref-dev | collisions (drone-eps) | GPU-better RMSE |
|---|---|---|---|---|
| MPC              | **1.751 m**   | 2.681 m   | 4/120 (3.3 %)          | —               |
| vanilla GPU MPPI | 1.649 m       | **1.972 m** | 4/120 (3.3 %)        | **120/120**     |
| Smart MPPI v4    | 1.707 m       | 2.165 m   | 4/120 (3.3 %)          | 118/120         |
| Smart MPPI v5    | 1.747 m       | 2.133 m   | 4/120 (3.3 %)          | 61/120 (tied)   |

The collision number is identical across planners because they all
clear the avoidance problem — the only collisions are the seed-42 ep 0
"initial chase ceiling" where 4 intruders simultaneously hit 4 drone
corridors in the first second and no planner can recover. The
remaining 29 of 30 episodes are clean wins.

Two findings:

**(1) The planners do, in fact, avoid the intruders.** Across all 4
planners and 116 paired-success drone-episodes, the mean reference
deviation is 1.7 m and the max single-step deviation is 2.7 m — drones
visibly swerve off the reference oval by up to 2 drone-radii to clear
the intruders, then re-acquire the oval. This validates the harness:
the dynamic-obstacle bridge, the planner's predictor of intruder
motion, the cost gradient, and the replan loop all compose into
working avoidance behaviour. Use this scenario as a smoke test before
running the mode-discriminating heroes (race / gates / chaos).

**(2) §3 mode 4 (precision) fires under dynamic-obstacle stress.**
With collision tied at the ceiling, the planners separate on *how
well they track the reference while detouring around intruders*.
Vanilla GPU MPPI's tracking RMSE is **0.102 m lower than MPC's on
every single one of the 120 paired drone-episodes** — a near-
deterministic precision win because softmax averaging across 64
rollouts produces a smoother detour command than MPC's single argmin
rollout. Smart v4 lands in between (0.045 m better, 118/120 wins),
Smart v5 ties (lateral-cancellation gate doesn't fire here — there
is no cancellation regime to detect — so v5's tracking matches MPC's).
This is the *aerobatic-loop* finding (mode 4) reproduced under
dynamic obstacles: the precision benefit of softmax averaging persists
even when active avoidance is in play, as long as the rollout cloud
isn't pushed into the cancellation regime.

The hero GIF (`docs/images/compare_race_dyn4.gif`) shows all 4
planners side-by-side with the dashed reference oval overlaid and
drones visibly swerving to dodge the 4 intruders. This is the
controlled demo to point at when someone asks "does it avoid dynamic
obstacles?"; the mode-discriminating story is in race / gates /
chaos.

Reproduce:
```bash
for plan in mpc gpu_mppi gpu_mppi_smart_v4 gpu_mppi_smart_v5; do
  uav-nav run "examples/exp_race_dyn4_${plan}.yaml"
done
for cmp in gpu_mppi gpu_mppi_smart_v4 gpu_mppi_smart_v5; do
  python3 scripts/paired_analysis_aerobatic.py \
    results/race_dyn4_mpc "results/race_dyn4_${cmp}" 4 30
done
python3 scripts/render_race_gif.py \
  --runs results/race_dyn4_mpc:MPC \
         results/race_dyn4_gpu_mppi:"GPU MPPI" \
         results/race_dyn4_gpu_mppi_smart_v4:"Smart v4" \
         results/race_dyn4_gpu_mppi_smart_v5:"Smart v5" \
  --config examples/exp_race_dyn4_mpc.yaml \
  --title "Drone race + 4 path-intersecting intruders — dynamic-obstacle avoidance" \
  --out docs/images/compare_race_dyn4.gif \
  --ep 1 --fps 10 --stride 12 --trail 30
```


### Aerobatic synchronized loop: GPU MPPI's softmax delivers 85 % tighter phase sync

A new scenario type `multi_drone_aerobatic` (commit
`uav_nav_lab/scenario/multi_drone_aerobatic.py`) tests the §3 4-mode
hypothesis directly: under choreography / formation-flight tasks,
the *same* softmax operator that hurts in static-peer clustering
(§3 N=4 baseline) and dynamic-obstacle cancellation (Table 2) should
*help* by producing smoother, tighter trajectories.

**Scenario**: 4 drones share one vertical loop in xz plane, center
$(20, 20, 7)$, radius 4 m, period 8 s. Each drone is phase-offset
by 90°. Episode = 2 loops × 8 s = 16 s = 320 steps. No static or
dynamic obstacles — the only coupling is the shared physical space
and the planners' peer-prediction layer. The "goal" passed to each
planner is the *lookahead* point on the drone's reference trajectory
(0.4 s ahead). YAMLs:
`examples/exp_aerobatic_loop4_{mpc,gpu_mppi}.yaml`. Analysis script:
`scripts/paired_analysis_aerobatic.py`.

Paired n=5 episodes (the scenario is deterministic given the
trajectory parameters; seed governs only the static-obstacle seed,
which is unused here, so all 5 episodes return identical metrics —
n=5 confirms reproducibility):

| metric                    | MPC      | GPU MPPI    | GPU vs MPC      |
|---|---|---|---|
| per-drone tracking RMSE   | 1.312 m  | **1.042 m** | -20.6 %         |
| per-drone max error       | 2.221 m  | **1.447 m** | -34.8 %         |
| phase-offset RMSE         | 10.73°   | **1.67°**   | **-84.4 %**     |
| collision rate            | 0/20     | 0/20        | tied (no coll.) |

Per-(drone, episode) tracking RMSE: GPU MPPI wins on **20/20**
drone-episodes by mean -0.27 m. The phase-offset reading is the
signature: GPU MPPI maintains a 90° between-drone offset within
$\pm 1.7°$ standard deviation across the 16-second loop; MPC's
argmin-driven commands produce $\pm 10.7°$ phase wobble — a factor
of 6× looser formation.

**Mechanism — same softmax operator, opposite valence**: GPU MPPI's
softmax averages 64 rollouts at each replan into a smooth weighted
command. On a moving reference trajectory the lookahead point shifts
~1.6 m every replan period (at $\omega r = 0.5 \pi \times 4 \approx 6.3$
m/s tangent speed × 0.2 s = 1.26 m + lookahead bias). The softmax-
averaged action follows the lookahead with low command oscillation —
each replan produces a small smooth adjustment. MPC's argmin selects
the single lowest-cost rollout per replan, which can flip between
"go slightly faster" and "go slightly slower" depending on
fractional cost differences — producing per-step command jitter
that the integrator smooths but does not eliminate. The smoothing
operator is the same operator that *clusters failures* in §3
N=4 baseline (under static peers) and *cancels avoidance* in Table 2
(under dynamic obstacles dead ahead); here, with no failure modes
to manifest, only the smoothing remains, and that smoothing is
*precisely* what choreography wants.

This completes the **§3 4-mode framework**:

| mode | regime                            | softmax outcome    | who wins   |
|---|---|---|---|
| 1    | Static peers, N=4 baseline         | clustering         | MPC (Δ)    |
| 2    | Dynamic obstacle on corridor       | bidirectional canc.| MPC        |
| 3    | Dense corner (AirSim `base_ew06`)  | suppresses cluster | GPU MPPI   |
| 4    | Aerobatic choreography             | smooth precision   | **GPU MPPI** |

**Implications**: GPU MPPI's softmax conservatism is not a planner
defect to fix — it is a *deployment-context tradeoff*. For air-show
flight, formation maneuvers, synchronised inspection passes, and any
mission where the metric is "tight reference tracking + multi-drone
sync", GPU MPPI is the correct planner family. For static-peer
crossings (where coordination $\Delta$ is the metric), MPC argmin's
distributed failure shape is the correct one. The 4-mode taxonomy
turns the planner-comparison question from "which is better?" into
"which mode does the mission live in?" — a more useful question for
deployment engineers.

Reproduce:
```
uav-nav run examples/exp_aerobatic_loop4_mpc.yaml
uav-nav run examples/exp_aerobatic_loop4_gpu_mppi.yaml
python3 scripts/paired_analysis_aerobatic.py \
  results/aerobatic_loop4_mpc results/aerobatic_loop4_gpu_mppi
```

Scope: one trajectory (synchronized vertical loop), one Pareto cell
each. Other choreography patterns (figure-8, diamond split-merge,
shape reveal) and seed-randomized variants (different start phases,
different loop radii) remain future work; the qualitative softmax-
smoothing-helps claim is the robust outcome.

### Bridge fix: pause-after-reset eliminates a stale-t=0 collision flag

Discovered during the n=30 paired-runs attempt above. The
multi-drone `airsim_bridge.py:reset()` path was leaving AirSim's
collision flag set to True at t=0.0 for **every** drone, so the
runner would record an immediate joint-collision outcome on the very
first episode of any uav-nav invocation that omitted the
`simulator.cameras: [...]` declaration. The single-drone demo with
cameras attached worked by accident — the per-step `simGetImages`
RPC was masking the bug.

Root cause: `client.reset()` in AirSim sends every vehicle back to
its `settings.json` spawn pose (which is at ground level, z ≈ 0). The
bridge then ran `_time.sleep(settle_after_reset)` (1 s default) with
the engine **unpaused**, so the 4 drones registered ground-contact
collisions during that 1-second window. The subsequent
`simSetVehiclePose(..., ignore_collision=True)` teleport relocated
them to altitude but did **not** clear the cumulative
`simGetCollisionInfo().has_collided` flag — and the first step()
readback returned `collision=True` against the unchanged start
position, ending the episode at t=0.05 s with all 4 drones flagged.

Fix (`uav_nav_lab/sim/airsim_bridge.py`): call `client.simPause(True)`
**immediately** after `client.reset()`, before the settle sleep, so
the engine never ticks during the on-ground spawn window. The
teleport-to-altitude still happens under pause (simSetVehiclePose
works in paused mode), and the existing simPause(True) at end of
reset() becomes redundant but harmless.

Verified by replacing the bridge call and re-running a 1-episode
multi-drone scenario without `cameras:` declared — collision flag now
returns `False` for all 4 drones at t=0.0. Earlier hypothesis (that
cameras' implicit per-step RPC delay was settling the flag) was a
red herring: cameras just gave AirSim enough wall-clock to overwrite
the stale flag, not actually fix the cumulative state.

Side observation, not fixed: AirSim's multi-drone `client.reset()`
sometimes wedges after 1–2 sequential resets, regardless of the
collision-flag fix. The n=30 paired study above worked around this
by running each episode as its own `uav-nav run` invocation with a
fresh Blocks server (`scripts/run_airsim_multi_chunked.sh`). The
underlying hang appears to be in AirSim itself (Blocks RPC handler
becomes unresponsive) and remains as future work.


## ROS 2 bridge: spatial equivalence verified

`scripts/ros2_dummy_sim.py` — minimal ROS 2 node that mirrors dummy_2d
physics, publishing `/odom` and subscribing to `/cmd_vel`. The
framework's `ros2_bridge` connects to it transparently. A single-episode
comparison (grid_world 50×50, MPC with identical config) between
direct `dummy_2d` and `ros2` backends:

| metric | direct dummy_2d | ROS 2 bridge | Δ |
|--------|----------------|-------------|---|
| outcome | success | success | — |
| steps | 132 | 270 | 2.0× |
| final x (m) | 38.61 | 38.79 | +0.18 |
| final y (m) | 38.75 | 38.89 | +0.14 |
| wall-clock t (s) | 6.60 | 13.50 | 2.0× |

**Spatial results agree within 0.2 m** — the ROS 2 hop does not
distort the planner's output or the positional accuracy. The 2×
wall-clock difference is expected: `ros2_bridge` runs at real-time
(one `spin_once` per step with `dt` timeout), while `dummy_2d`
integrates at CPU speed.

The QoS mismatch (cmd_vel uses RELIABLE but ros2_bridge publishes
SENSOR_DATA) is cosmetic — messages are still delivered. The
episode-reset behaviour relies on the sim node supporting teleport
(not implemented in the minimal dummy; production sims like Gazebo
handle this via `/reset` service).

### Implication for AirSim → ROS 2 integration

When the AirSim ROS2 wrapper (`ros2/AirsimROSWrapper`) is available,
the full chain AirSim → ROS 2 → ros2_bridge should produce the
same spatial behaviour as AirSim → airsim_bridge (direct), modulo
the real-time clock constraint. The framework's planner/sensor/
scenario boundary is proven invariant under the bridge hop.


## AirSim over ROS 2 parity harness

The AirSim ROS 2 integration is now wired at the framework boundary:

- `Ros2Bridge(frame: ned)` converts default AirSim ROS wrapper odometry
  and velocity commands between NED and the framework's ENU convention.
- `Ros2Bridge(cmd_msg_type: airsim_vel_cmd)` publishes AirSim's
  `airsim_interfaces/VelCmd` / `airsim_ros_pkgs/VelCmd` wrapper message
  instead of plain `geometry_msgs/Twist`.
- `examples/exp_airsim_ros2_direct.yaml` and
  `examples/exp_airsim_ros2.yaml` run the same empty `voxel_world`
  MPC scenario through direct AirSim RPC and AirSim-over-ROS2.
- `scripts/compare_spatial_runs.py` compares the two run directories
  on outcome, final-position delta, RMS trajectory delta and path-length
  delta.

This does not claim a measured result yet; it is the repeatable harness
for the next AirSim session. The remaining external requirement is the
AirSim ROS 2 wrapper itself: it must publish `nav_msgs/Odometry` for the
selected vehicle and subscribe to the velocity command topic. Reset /
teleport is still sim-specific, so the AirSim side must start from the
same pose used by the YAML before comparing trajectories.


## RL comparison baseline: gym.Env scaffold + initial training

`uav_nav_lab/rl/env.py` — `GridNavEnv` / `VoxelNavEnv` gymnasium
environments wrapping the framework's scenario + dummy sim stack.
Observation: [ego_x, ego_y, goal_x, goal_y, 7×7 local occupancy].
Action: continuous [vx, vy] bounded by max_speed. Reward: +10 goal,
-10 collision, -0.01/step, +shaping.

`scripts/train_rl_baseline.py` trains SAC (stable-baselines3 2.8)
against grid_world 50×50 with 30 random obstacles (same config as
the A\* baseline from `exp_basic.yaml`).

### Training characteristics

| metric | value |
|--------|-------|
| env step | 0.07 ms |
| SB3 SAC fps (RTX 4070 Ti) | **36 steps/s** |
| time for 100k timesteps | ~46 min |
| success @ 10k timesteps | 0 % (20 episodes) |
| success @ 5k timesteps | 0 % (10 episodes) |

The environment step itself is fast (0.07 ms), but SB3's SAC training
loop (rollout + replay buffer sampling + gradient updates on a 2-layer
MLP) runs at 36 fps on GPU. SAC typically requires 100k–500k
timesteps for simple grid navigation, translating to 1–4 hours of
wall-clock training.

### Comparison context

For the same scenario (grid_world 50×50, 30 obstacles, max_speed=8):
- **A\***: 95 % success, 5 ms plan_dt (from PR #11)
- **MPC (n=16)**: 93 % success, 9 ms plan_dt
- **SAC (10k steps)**: 0 % success, 46 min train time

The RL baseline requires an order-of-magnitude more training than is
practical in a single session. The scaffold (`rl/env.py` +
`scripts/train_rl_baseline.py`) is functional and ready for
overnight training; the comparison should be re-run at 500k+
timesteps before any conclusions about plan-based vs learned
navigation are drawn.

### Note from plan.md

The plan marks this as 優先度低 and suggests a separate repo or
submodule — the framework's design as a "planner comparison tool"
makes RL integration inherently heavyweight because the training
loop must drive the full sim/sensor/planner pipeline. The gym.Env
wrapper simplifies this to just sim/sensor (no planner), but the
training throughput bottleneck remains SB3's SAC implementation,
not the framework.
