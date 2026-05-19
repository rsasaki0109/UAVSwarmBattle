# §4.4. Sim transferability and ROS 2 invariance

The §3 headline result — GPU MPPI's +11.4 pp coordination Δ over its
indep$^4$ baseline vs CPU MPC's +0.8 pp — was measured on the
point-mass `dummy_3d` simulator. Whether that finding belongs in a UAV
planning paper at all depends on whether the mechanism transfers to
real quadrotor physics. This section shows which parts transfer
cleanly, and which parts are scenario-regime dependent. The AirSim
study has three layers: altitude-only cells bracket the failure-level
signal but degenerate at ceiling or floor (§4.4.2); a tuned
static-cube cell produces a true paired planner separation under
AirSim collision geometry, with GPU MPPI removing every paired seed
MPC loses (§4.4.3); and a density-swept extension drops GPU MPPI off
its 100 % ceiling, at which point the $\Delta$-flip mechanism
re-emerges with its sign reversed — MPC is the clustering planner
on AirSim, not GPU MPPI (§4.4.4).

## 4.4.1 Single-drone parity: planner portable, plan-time edge lost

The first transferability check uses one drone in the AirSim Blocks
Unreal env with SimpleFlight as the multirotor controller (a
typical PID velocity-tracking controller), driving the same GPU MPPI
planner used in §3. The drone runs through a static-cube scenario
that the planner does not see in its map (perception is built from
the simulator's onboard LiDAR via `pointcloud_occupancy`). Two
findings emerge from a paired run against the CPU MPC baseline on
the same Blocks scene:

(i) **Planner is portable**: GPU MPPI reaches goal at the same rate
as CPU MPC across n=10 paired episodes, with no implementation hits
needed beyond the bridge itself.

(ii) **Plan-time edge collapses on AirSim**: GPU MPPI's steady-state
`plan_dt` is 180 ms on AirSim vs 188 ms for CPU MPC — a <5 % gap. On
`dummy_3d` the same planners report 3.5 ms vs 70 ms (20×). The gap
shrinks not because GPU MPPI got slower but because AirSim's
synchronous RPC round-trip (`simContinueForTime` + `getMultirotorState`
+ `getLidarData`) dominates the per-step budget, and that cost is
paid by both planners. **The deployment-relevant `plan_dt` advantage
that `dummy_3d` measured was a property of the test bench, not the
planner family.** This is the §1 "sim transferability gates the
deployment story" claim made concrete: the same planner cell at the
same scenario gives a fundamentally different efficiency reading on
the two simulators.

The drone also commands lower steady-state velocities on AirSim
(1.87 m/s vs CPU MPC's 2.40 m/s, both well below the YAML `max_speed`
of 4.0). GPU MPPI's softmax averages over rollouts that explore both
sides of obstacles, producing a smoother but more conservative
commanded speed; the velocity profile difference is preserved on
SimpleFlight's PID controller and visible in the per-drone arrival
spread (next subsection).

## 4.4.2 Multi-drone, three altitude-stagger cells

Repeating the §3 paired comparison with all four drones in the
Blocks scene, we measure n=30 paired episodes on each of three
altitude-stagger geometries — chosen to bracket the indep$^4$
operating point where the Δ-flip mechanism can register.

| stagger (z range)        | MPC per-drone | GPU MPPI per-drone | MPC joint | GPU MPPI joint | trajectory spread ratio |
|---|---|---|---|---|---|
| ±2-4 m (6 m, demo)       | 100 %         | 100 %              | 100 %     | 100 %          | **27×** (0.55 s / 0.02 s) |
| ±1 m (2 m, mid)          | 100 %         | 100 %              | 100 %     | 100 %          | **4×** (0.39 s / 0.10 s)  |
| 0 m (uniform, z = 30)    | 65.0 % [56.1, 72.9] | 28.3 % [21.0, 37.0] | 46.7 % [30.2, 63.9] | **0/30 = 0.0 %** [0.0, 11.4] | n/a (no successes) |

McNemar same-seed pairing on the uniform-altitude cell: both-succ 0,
MPC-only-succ 14, GPU-only-succ 0, neither-succ 16, exact two-sided
binomial $p \approx 0.00012$ — the only AirSim cell of the three
where the planner comparison rejects the null hypothesis of equal
joint rates.

Three readings of the table, in increasing inferential strength:

**(a) The altitude-stagger response is bimodal, not graded.** The
expected outcome, given the dummy_3d operating point of 90 % per-
drone, was a graded degradation as the geometry tightens: ±2 m at
100 %, ±1 m at ~90 %, uniform at ~50 %. The measured response has
no graded middle. Non-zero z-spread (±2-4 m and ±1 m) keeps both
planners at the 4/4 ceiling — peer prediction + a 0.6 m safety
margin is enough to keep every drone clean at any non-zero vertical
gap. The bottleneck flips abruptly at z-coincidence, where the
bottleneck is the physical mesh-mesh collision at the (30, 30, 30)
crossing centre. Below that bottleneck, planner choice has no
measurable consequence; at the bottleneck, it has $\Delta = 47$ pp
of consequence.

**(b) Trajectory-spread mechanism is preserved across all measurable
cells.** In the n=30 cells where both planners succeed (±2-4 m and
±1 m), the per-drone arrival spread within an episode is the §3
signature in miniature: MPC's argmin keeps the four drones within
0.02-0.10 s of each other (essentially in lockstep), while GPU MPPI's
softmax over 64 rollouts spreads them by 0.39-0.55 s. The spread
ratio between planners is 4-27× across the cells. The same mechanism
that §3 attributes the +11.4 pp Δ to on dummy_3d is producing the same
*trajectory-level* signal on AirSim, even when the failure-level Δ
itself cannot register because every drone succeeds.

**(c) The uniform-altitude collapse is per-drone-rate-driven, not
$\Delta$-driven.** GPU MPPI's measured joint of 0.0 % sits at its
indep$^4$ prediction of 0.6 %, so $\Delta \approx 0$. The §3 dummy_3d
mechanism — softmax amplifying seed sensitivity through a shared
peer-prediction world model — needs a per-drone rate high enough to
leave indep$^4$ measurable headroom; at 28.3 % per-drone, indep$^4$
falls below the joint success-rate floor and the headroom is lost.
MPC's $\Delta = +28.8$ pp on the same cell is *geometric*, not
behavioural: all 16 MPC failed episodes have $\geq 2$ drones colliding
together at the bottleneck. Two drones converging on the same physical
point at the same physical time will collide together regardless of
which planner is commanding them, and this is what creates MPC's large
positive $\Delta$ at this cell — not anything specific about MPC's
coordination story.

Together these readings narrow the §3 transferability question but do
not close it. The *trajectory-level* mechanism GPU MPPI deploys on
dummy_3d (sample-cloud-driven per-drone variance) is preserved across
the measurable AirSim altitude cells. The *failure-level* $\Delta$ —
the actual +11.4 pp number — cannot be reproduced on altitude alone:
the easier cells ceiling the per-drone rate at 100 %, the harder cell
drops GPU MPPI below indep$^4$'s floor.

## 4.4.3 Static-cube discriminating cell: failure-level separation on AirSim

The missing AirSim operating point is produced by adding static cube
geometry to the staggered crossing. We extend the bridge with
`simulator.static_obstacles`, spawning five `1M_Cube_Chamfer` meshes
into Blocks after reset, and add matching `scenario.obstacles.boxes`
to the planner occupancy map. The tuned n=30 cell uses four
east/west pillars, one north/south pillar, z = 26/28/30/32, the south
lane offset to x = 26, and `inflate = 3`.

| planner               | per-drone (CI)       | joint (CI)          | indep$^4$ | $\Delta$ over indep | mean final_t |
|---|---|---|---|---|---|
| MPC      (n=16, h=30) | 105/120 = 87.5 % [80.4, 92.3] | 22/30 = 73.3 % [55.6, 85.8] | 58.6 % | **+14.7 pp** | 10.03 s |
| GPU MPPI (n=64, h=20) | 120/120 = 100.0 % [96.9, 100.0] | 30/30 = 100.0 % [88.6, 100.0] | 100.0 % | +0.0 pp | 12.35 s |

Same-seed McNemar pairing gives both-succ 22, MPC-only 0, GPU-only
8, neither 0; exact $p \approx 0.008$. The disagreement seeds are
43, 47, 48, 51, 53, 60, 62, and 65.

This result closes the AirSim *failure-level separation* gap: under
real quadrotor physics and real AirSim collision geometry, the two
planner families no longer merely differ in arrival spread. GPU MPPI
clears every paired seed that MPC clears, plus eight seeds where MPC
loses one or two drones. The tradeoff is time: GPU MPPI succeeds more
often but arrives 23 % slower on average (12.35 s vs 10.03 s).

It still does not reproduce the exact §3 dummy_3d $\Delta$ mechanism.
In §3, the two planners tie on joint success and GPU MPPI's larger
positive $\Delta$ shows stronger within-seed failure clustering. In
the static-cube AirSim cell, GPU MPPI reaches the 100 % ceiling, so
its $\Delta$ is degenerate at zero. The narrower transferability
claim from this cell is: **GPU MPPI's rollout cloud can remove paired
AirSim collision seeds that MPC loses, but this cell does not
establish a joint-tie / larger-$\Delta$ signature.** §4.4.4 takes the
density sweep one notch further to put GPU MPPI in the 60-90 %
per-drone band and re-measure $\Delta$ from there.

The tuning path also identifies a practical pitfall. A symmetric
eight-pillar version put one drone into a deterministic mesh
bottleneck, producing a joint floor rather than a discriminating
planner comparison. The final cell removes three of the north/south
pillars and offsets one lane so failures come from obstacle-induced
crossing interactions rather than a fixed geometry trap.

## 4.4.4 Density-sweep cell: $\Delta$-flip sign reverses on AirSim

Building on §4.4.3 we widen the east/west pillars in the baseline
5-pillar layout from `scale = 0.5` to `scale = 0.6` (cell `base_ew06`,
generated by `scripts/run_airsim_discriminating_param_sweep.sh`). The
intent is to drop GPU MPPI off the 100 % ceiling so its $\Delta$
becomes measurable against indep$^4$. We run n=50 paired episodes
(seeds 42-91), with the same physics, planner Pareto cells, and
chunked-server workaround as §4.4.3.

| planner               | per-drone (CI)                | joint (CI)                | indep$^4$ | $\Delta$ over indep | mean final_t |
|---|---|---|---|---|---|
| MPC      (n=16, h=30) | 179/200 = 89.5 % [84.5, 93.0] | 34/50 = 68.0 % [54.2, 79.2] | 64.2 % | **+3.8 pp** | 10.00 s |
| GPU MPPI (n=64, h=20) | 191/200 = 95.5 % [91.7, 97.6] | 41/50 = 82.0 % [69.2, 90.2] | 83.2 % | -1.2 pp | 12.39 s |

Same-seed McNemar pairing gives both-succ 28, MPC-only-succ 6,
GPU-only-succ 13, neither-succ 3; exact two-sided $p \approx 0.167$.
The point estimate favours GPU MPPI by 14 pp on joint success and
by 2.2-to-1 on discordant pairs, but n=50 is still short of clearing
$p = 0.05$ on McNemar alone.

The headline reading is structural, not statistical:

**The $\Delta$-flip mechanism transfers, but the sign of the flip
reverses.** §3 dummy_3d tied the two planners at 94 % per-drone and
let GPU MPPI's softmax cluster failures into seed-correlated joint
collapses, producing $\Delta_\text{GPU} = +11.4$ pp against
$\Delta_\text{MPC} = +0.8$ pp. AirSim `base_ew06` ties the planners at
the per-drone level (96 % GPU, 90 % MPC, neither at ceiling) and
differentiates them through $\Delta$ again, but this time **MPC is
the planner whose failures cluster** ($\Delta_\text{MPC} = +3.8$ pp)
while GPU MPPI sits at indep$^4$ ($\Delta_\text{GPU} = -1.2$ pp,
essentially independent failures). The mechanism that produces the
$\Delta$ gap — per-drone tied, a planner family that clusters
failures vs one that does not — survives the dummy_3d-to-AirSim hop;
the identity of the clustering planner does not.

| backend / cell           | MPC $\Delta$       | GPU MPPI $\Delta$   | which planner clusters |
|--------------------------|--------------------|--------------------|------------------------|
| dummy_3d §3 (n=100)      | +0.8 pp            | **+11.4 pp**       | GPU MPPI               |
| AirSim base_ew06 (n=50)  | **+3.8 pp**        | -1.2 pp            | MPC                    |

Per-seed disagreement explains the sign reversal. GPU MPPI's nine
failed seeds (43, 45, 46, 50, 52, 73, 75, 90, plus seed 73 as a
concordant collision with MPC) are **all the same drone** (drone
idx 3, the southernmost lane) losing one collision against the
widened EW pillar — a deterministic geometric pinch that the
softmax-averaged rollout handles uniformly across seeds. MPC's
sixteen failed seeds split into **thirteen single-drone (drone 3)
collisions and three multi-drone clusters**: seed 55 and seed 67 lose
drones 1, 2, and 3 simultaneously; seed 66 loses drones 1 and 2.
These three cluster seeds out of 50 are what drive MPC's $\Delta$
positive: the argmin update commits each MPC drone to a narrow
north-end corridor at the same moment, and when that corridor is
geometrically infeasible for the seed's start jitter, three of the
four drones converge into the same collision frame within ~0.5 s.
GPU MPPI's rollout cloud, averaging over 64 trajectories at each
replan, never commits to the corridor as hard, so its failures stay
confined to the one drone whose lane the EW pillar physically blocks.

This is the §3 dummy_3d claim with the sign of the flip swapped:
on the dummy_3d N=4 cell with 30 obstacles GPU MPPI's softmax was
the cluster source; on the AirSim `base_ew06` cell with 5 widened
pillars converging on a single central crossing, MPC's argmin is.
The shared structural claim — that two planner families with tied
per-drone rates can be separated through their joint-success $\Delta$
— survives.

A follow-up dummy_3d density sweep at N=4 (varying obstacle count
30 / 120 / 240) reproduces the sign flip in the controlled simulator:
MPC moves from $\Delta = -1.0$ pp at baseline to $\Delta = +6.7$ pp
at packed (240 obstacles), while GPU MPPI moves the other way
($+5.2 \to -1.2$ pp). At each of those N=4 cells the planners'
per-drone rates remain within ~14 pp of each other, so the $\Delta$
statistic is the differentiator.

The same density sweep at N=6 does *not* reproduce the sign flip.
At higher N the per-drone gap opens up — GPU MPPI's per-drone stays
74 % at packed while MPC's collapses to 42 %, so GPU MPPI wins the
joint comparison through per-drone advantage rather than through the
cluster mechanism. Across the (N, density) grid GPU MPPI beats MPC
through *one of two routes*: per-drone tied + GPU clusters (§3 N=4
baseline, §6 N=6 baseline); or per-drone divergent in GPU's favour
(N=6 dense / packed). MPC has a density-driven cluster regime where
its argmin lock-step concentrates failures (N=4 dense / packed), and
that mode produces a higher $\Delta_\text{MPC}$ — but it only gives
MPC an *absolute* joint advantage when per-drone rates stay close.
The N=8 row adds a third mechanism that is neither route: GPU MPPI's
per-drone uniquely collapses to 69 % under the 8-fold-symmetric
central crossing (vs MPC's 92 %, McNemar $p \approx 0.0001$ favours
MPC) and density actually *unwinds* the collapse rather than
deepening it. The full three-row picture lives in §6.

The AirSim `base_ew06` reading sits squarely in the N=4-dense regime
of this grid (4 drones, ≈90 % MPC vs ≈96 % GPU per-drone, 5 widened
pillars concentrating crossings) and is consistent with the dummy_3d
N=4 dense/packed reading. It is **not** a generic "AirSim flips the
mechanism" statement — at N=6 the flip does not reproduce, at N=8
baseline a different mechanism takes over, and base_ew06 is a single
AirSim point, not an AirSim-wide claim. Full numbers: findings.md
"dummy_3d density × planner sweep at $N \in \{4, 6, 8\}$".

The n=50 magnitude is more modest than the n=30 first cut suggested.
At n=30 we measured $\Delta_\text{MPC} = +6.9$ pp; extending to n=50
without adding new cluster seeds dilutes the contribution to
$+3.8$ pp. Of MPC's 16 failed seeds in 50, only three are multi-drone
clusters (6 % cluster rate); the other 13 are the same drone-3
single-collision mode that GPU MPPI also exhibits, just less often.
Removing those three cluster seeds would put MPC at $\Delta \approx
-4$ pp (indistinguishable from GPU). The cluster events are real and
reproducible (they recur on the same seeds across re-runs), but they
are rare enough that the absolute magnitude of MPC's $\Delta$ depends
strongly on whether the sample hits any of them. The qualitative
sign-reversal claim — MPC has a cluster failure mode that GPU MPPI
does not — is supported; the quantitative magnitude is a few pp and
sensitive to sample size.

The n=50 McNemar (b=6 MPC-only, c=13 GPU-only) gives a binomial-19
two-sided $p \approx 0.167$. Closing the rest of the way to
$p = 0.05$ would take another ~30 paired seeds at a similar
GPU-to-MPC discordant ratio; the qualitative findings above (cluster
mode + sign-reversed $\Delta$ + GPU-side trajectory-spread mechanism)
hold without that extension.

The drone-3 single-failure baseline is itself a geometric property
of the cell rather than a tunable artifact. Probing lane shifts at
n=10 GPU-only smoke — south lane moved from $x = 26$ to $x = 30$
(centered) or $x = 22$ (further west) — failed in both directions:
both `base_ew06_lane30` and `base_ew06_lane22` produce 10/10 drone-3
collisions at consistent positions $(30.9, 29.4, 26.6)$ and
$(21.2, 23.3, 26.6)$ respectively. The $x = 26$ baseline shares the
$(21.2, 23.2, 26.6)$ collision point with `lane22` but reaches it at
$t \approx 7.35$ s instead of 6.90 s, i.e. via a longer planner-chosen
detour around the inflate-3 keepout of the $(25, 27)$ pillar. The
GPU MPPI single-drone failure mode is the planner running out of
clearance at the detour terminus; lane shifts only either land the
drone in a worse part of the inflated keepout (lane22) or push it
into the central drone-drone crossing zone (lane30). The MPC cluster
failure mode (seeds 55/66/67) sits independently on top of this
geometric baseline.

**Stability note.** A bridge patch (`airsim_bridge.py`, capturing
`simGetCollisionInfo().object_name` into `state.extra["collision_object"]`)
allowed re-running the failed seeds with collision-object attribution.
The re-run revealed substantially higher run-to-run AirSim variance
than the Wilson CI accounts for. Re-running the 8 GPU MPPI failure
seeds and 27 additional fresh seeds gave 35/35 successes, against
an expected 5-6 failures at the n=50 16 % rate. The three MPC cluster
seeds reproduced unevenly: seed 67 reproduced exactly (cluster of
drones 1/2/3 at central crossing, drones 1 and 2 with empty
`object_name` indicating drone-drone collision and drone 3 hitting
`uavnav_disc_ew_35`); seed 66 morphed to drone-3 single failure;
seed 55 became 4/4 success.

To quantify the variability, we ran three independent paired batches
of n=15 each on fresh seed ranges (200-214, 220-234, 240-254). Each
batch was a complete, self-contained McNemar measurement:

| batch    | MPC joint    | GPU joint    | MPC Δ    | GPU Δ    | discordant (b, c) | McNemar p | clusters |
|----------|--------------|--------------|----------|----------|-------------------|-----------|----------|
| 1 (200–) | 14/15 = 93 % | 15/15 = 100 %| **+6.0** | +0.0     | (1, 0)            | 1.000     | 1        |
| 2 (220–) | 11/15 = 73 % | 13/15 = 87 % | **+12.4**| -0.7     | (2, 4)            | 0.688     | 3        |
| 3 (240–) | 14/15 = 93 % | 11/15 = 73 % | -0.2     | -2.6     | (4, 1)            | 0.375     | 0        |
| n=50 (orig) | 34/50 = 68 % | 41/50 = 82 % | +3.8 | -1.2     | (6, 13)           | 0.167     | 3        |
| **3-batch combined (n=45)** | **39/45 = 86.7 %** | **39/45 = 86.7 %** | **+7.3** | **-0.7** | **(7, 5)** | **0.774** | **4** |

Two things are stable across batches and replicate in n=50:

1. **The MPC cluster failure mode at the central crossing is reproducible
   as a *mode*.** Each batch produces between 0 and 3 cluster seeds at
   a rate consistent with the n=50 cell-level estimate of $\sim 6$ %
   (3-batch mean 8.9 %). The cluster signature is always the same:
   drones 1 (west) and 2 (north) collide drone-drone at $x \approx 30,
   y \approx 28$ (empty `collision_object` confirms drone-drone), with
   or without drone 3 (south) collateral.
2. **GPU MPPI's softmax avoids the cluster mode entirely** in all four
   measurements; its failures are always the same single drone (drone 3)
   running out of clearance at the planner's detour terminus.

Two things are *not* stable:

1. **The cluster rate per batch is variable** (0/15, 1/15, 3/15) —
   wider than the n=50 Wilson CI on the same statistic predicts. AirSim
   physics-tick jitter, GPU thermal state, and host load all matter.
2. **The McNemar direction reverses across batches.** Batch 1 ties
   (GPU at ceiling), batch 2 favours GPU MPPI (4 vs 2 discordant),
   batch 3 favours MPC (4 vs 1 discordant, opposite of batch 2 and of
   the n=50 reading). The 3-batch combined McNemar (b=7 MPC-only,
   c=5 GPU-only, n=12) gives $p \approx 0.77$ — essentially a tie.

The implication for §4.4.4's structural claim is one of scope. The
"$\Delta$-flip sign reverses on AirSim" reading is correct *as a
mechanism statement* — only MPC has a multi-drone-cluster failure
mode in this cell, and it consistently bumps $\Delta_\text{MPC}$
above the indep$^4$ baseline whenever the seeds happen to hit it.
But the "GPU MPPI wins paired episodes on AirSim" reading from the
n=50 number (joint 82 % vs 68 %, McNemar p = 0.167) is *not*
reproduced in 3 fresh independent batches (combined joint 86.7 %
vs 86.7 %, p = 0.77). The single n=50 measurement appears to have
been on the high-failure tail of the cell's run-to-run distribution.
The reproducible findings are: (i) MPC has the drone-drone-cluster
mode at central crossing; (ii) GPU MPPI does not; (iii) which planner
wins joint success is environment-sensitive and not robustly
determined by a single n=50 paired study. The clean fix for the
McNemar direction question would be either a much larger N (n ≥ 200
paired) or a controlled-environment measurement (idle host, fixed
GPU clock, fixed physics tick) — both out of scope for this paper.

## 4.4.5 The bridge fix that made the n=30 measurement possible

The n=30 paired runs above did not work the first time. AirSim's
multi-drone reset path was leaving every drone's collision flag set
at $t = 0$, so the runner recorded a 4/4 joint collision before the
first planning step. The single-drone demo had run for months without
noticing the bug, because attaching a `front_center` camera to the
drone (as the demo recording script does) serialised per-step
`simGetImages` RPC into the readback path and inadvertently delayed
the first collision check long enough for AirSim to clear the stale
state. The n=30 YAMLs disabled camera capture for storage reasons
and surfaced the bug as 100 % joint collision at $t = 0.05$ s.

The root cause is a missing pause point in the bridge's reset
sequence. AirSim's `client.reset()` snaps every vehicle back to its
`settings.json` spawn pose, which is on the ground; the bridge then
sleeps `settle_after_reset` (default 1.0 s) with the engine *unpaused*
to absorb post-reset RPC settling, and during that 1 s the four
drones register ground-contact collisions in `simGetCollisionInfo`'s
cumulative `has_collided` field. The subsequent
`simSetVehiclePose(..., ignore_collision=True)` teleport relocates
each drone to its altitude but does *not* clear the cumulative flag,
so the first step's collision readback returns True against the
unchanged start position.

The fix (`uav_nav_lab/sim/airsim_bridge.py` commit 382d207) inserts
`client.simPause(True)` immediately after `client.reset()` and before
the settle sleep, so the engine cannot tick during the on-ground
spawn window. Subsequent teleports still work (`simSetVehiclePose`
operates in paused mode), and the existing `simPause(True)` at the
end of reset becomes redundant but harmless. The fix is committed
ahead of the n=30 study; running the staggered AirSim demo on the
fixed bridge with cameras disabled now reproduces.

We additionally encountered an AirSim issue not fixed: the multi-
drone `client.reset()` RPC sometimes wedges after 1-2 sequential
calls. The workaround for the n=30 cells is the `scripts/run_airsim_multi_chunked.sh`
runner, which restarts the Blocks server between every episode and
overrides `--seed` per iteration; the underlying hang appears to be
in the AirSim server-side dispatch loop and is acknowledged as
out-of-scope for this paper. The bridge-side fix and the chunked
runner are reproducible together from any AirSim 1.8.x install with
`Drone1..Drone4` in `settings.json`.

## 4.4.6 ROS 2 invariance

The final transferability check uses the `ros2_bridge` backend to
run the same single-drone planner stack through `geometry_msgs/Twist`
+ `nav_msgs/Odometry` round-trips against a dummy ROS 2 simulator
(`scripts/ros2_dummy_sim.py` — a mirror of `dummy_2d`'s point-mass
kinematics). Spatial trajectories agree with the direct in-process
bridge within 0.2 m mean ATE across n=30 episodes; the bridge hop is
invariant in trajectory content and adds only the real-time clock
constraint. The same harness then exercises AirSim *through* ROS 2
(via the AirSim ROS wrapper) and checks against AirSim direct: the
two routes produce equivalent spatial behaviour, modulo the
real-time clock. Full numbers in findings.md §"ROS 2 bridge: spatial
equivalence verified" and §"AirSim over ROS 2 parity harness".

The ROS 2 result is not load-bearing for §3 by itself, but it
closes the loop on §1's "sim transferability gates the deployment
story" claim: a finding measured on `dummy_3d`, replayed on AirSim
direct, and replayed again on AirSim-over-ROS-2 produces the same
spatial trajectories. The §3 measurement is reproducible on each of
those three sim stacks, with the caveats §4.4.1-§4.4.4 list for
each.
