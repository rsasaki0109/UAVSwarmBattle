# §1. Motivation

UAV planner comparisons in the recent literature exhibit a recurring
problem: two careful teams running ostensibly the same comparison
publish contradictory conclusions about which planner family wins.
Three structural causes account for most of these contradictions, and
none of them are visible from a single experiment run.

**Pareto cells, not planners.** A CPU MPC planner with $n_\text{samples}{=}8$
and horizon $h{=}40$ is a different object from the same planner at
$n_\text{samples}{=}16$ and $h{=}20$. Both are "CPU MPC" by name. On
the 50×50 2D dynamic-obstacle benchmark used throughout the present
work, the latter cell saturates the replan budget at 100 % success and
the former cell loses 12 pp at the same compute envelope. Reporting
"MPC achieves $x$ %" without naming the cell ties the headline to an
implicit pick that the reader cannot reproduce. We adopt the convention
of always reporting the cell explicitly and re-running every ablation
*at* the Pareto-optimal cell — not on a fixed, non-Pareto config (§4.1).

**Coordination Δ over independence, not joint success.** Joint
multi-drone success is the rate at which all $N$ drones complete an
episode without collision. It is the natural top-line number, but in
the absence of a peer prediction layer it is also the rate the
literature reports when planners are run independently — confounding
planner-level coordination with planner-level brittleness. The
indep$^N$ baseline, defined as $(\text{per-drone success rate})^N$,
removes that confound: it is the joint rate one would observe if the
$N$ drones' outcomes were uncorrelated. The Δ over indep$^N$ — joint
minus indep$^N$ — captures the coordination signal alone. Two planner
families at the same joint rate can have very different Δ over
indep$^N$: in our headline result (§3, n=100 paired) CPU MPC's joint
sits 0.8 pp above its indep$^4$ baseline while GPU MPPI's joint sits
**+11.4 pp** above its indep$^4$. The two planners do *not* coordinate
the same way under the same scenario, despite having statistically
indistinguishable joint rates.

**Implementation correctness gates every cell of a Pareto sweep.**
GPU MPPI rollouts in 3D voxel worlds initially appeared to suffer a
"speed cliff" at long horizons: success collapsed to 0 % when the
horizon extended past the static-obstacle scale of the scenario. The
phenomenon read as a fundamental property of MPPI's softmax averaging
against tall cost ridges. It was not — a goal-mask term that the CPU
MPC path naturally short-circuits (by detecting goal-reach and zeroing
remaining-horizon cost) was missing from the batched GPU path, so
GPU MPPI rollouts that reached goal early kept accruing cost beyond
arrival and the softmax weighted them down. Fixing the goal-mask
(commit 2a9d196) transforms the 3D Pareto frontier: GPU MPPI at
$(n{=}64{-}256, h{=}20)$ Pareto-dominates the CPU MPC baseline (3.5 ms
steady-state vs 70 ms, 100 % vs 88 %; §4.2). Every Pareto-cell number
in this paper was re-validated *after* that fix; the pre-fix tables in
the same harness's history would have led to an exactly inverted
recommendation between the two planner families.

**Sim transferability gates the deployment story.** A finding measured
in a point-mass `dummy_3d` sim is only as load-bearing as its
transferability to physics. We bridge to Microsoft AirSim (Blocks
Unreal env, SimpleFlight multirotor controller) and ROS 2 (`/cmd_vel`
+ `/odom`, optional sim-time anchoring) at the framework boundary —
same scenario / sensor / planner objects, the bridge is the only
swap. Three findings emerge that the dummy_3d-only literature could
not have produced: GPU MPPI's 20× plan-time edge over CPU MPC on
dummy_3d collapses to <5 % on AirSim because sim-side overhead
dominates (§4.4); altitude-only AirSim cells bracket the dummy_3d
multi-drone Δ-flip but degenerate at ceiling or floor; a static-cube
AirSim cell produces a real paired planner separation (GPU MPPI
30/30 joint vs MPC 22/30, McNemar p ≈ 0.008); and a density-swept
AirSim cell reproduces the Δ-flip mechanism **with the sign
reversed** — MPC becomes the multi-drone cluster source while GPU
MPPI does not (§4.4.4).

**The four causes interlock.** None is a research question on its own:
"Pareto cells matter" is methodology, not science. "Coordination Δ is
not joint success" is a definitional clarification. "Goal-mask bug fix"
is debugging. "AirSim ↔ dummy_3d transferability" is engineering. But
*together* they explain why the same comparison can be reported with
opposite winners by two careful teams, and why our headline result
(§3) — **GPU MPPI's softmax as a smoothing operator on the action
space with three regime-specific failure modes** (static-peer
clustering, dynamic-obstacle bidirectional cancellation, sim-physics
density-corner sign reversal) — is presentable at all as a finding
rather than a measurement artefact. The remainder of the paper walks
through the four prerequisite checks (§4) before staking the
headline (§3) on them, because without the prerequisites the headline
would have read as just one more entry in the contradictory-comparisons
literature.

**Why a single operator with three mode expressions is the right framing.**
A 1-direction headline (e.g. "GPU MPPI clusters multi-drone failures
more than CPU MPC") is what the §3 N=4 baseline cell measured at
n=100 paired episodes; that *direction* survives a re-run of the same
cell but does not survive moving to N=4 *dense* obstacles or to the
AirSim base_ew06 cell (MPC becomes the cluster source). What *does*
survive across regimes is the mechanism: GPU MPPI's softmax averages
over the rollout cost landscape, while CPU MPC's argmin commits to
one rollout per replan. Whether averaging helps or hurts is
regime-dependent: it amplifies peer-prediction clustering in one
static cell, suppresses MPC's dense-corner cluster mode in another,
and improves aerobatic tracking precision when failures are not the
binding metric. The **operator** is the load-bearing claim; the
*direction* in any single cell is downstream of it. The former
dynamic-obstacle mode is retracted pending a post-`1646e11` re-tune.
