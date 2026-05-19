# §6. Limitations

The headline result in §3 is intentionally narrow. It is a paired
planner comparison at a validated Pareto cell on one four-drone
`dummy_3d` crossing task. The result is robust under the n=100 rerun:
joint success remains tied (78 % MPC, 77 % GPU MPPI) while the
coordination Δ separates (+0.8 pp vs +11.4 pp). What the result does
not claim is that GPU MPPI is universally better, or that the same
Δ shape appears unchanged under every simulator and geometry.

The AirSim transfer study sharpens that limitation rather than
removing it. Altitude-only AirSim cells bracket the dummy_3d regime
but do not reproduce it: non-zero z-spread ceilings both planners at
100 % joint success, while uniform z collapses GPU MPPI to 0/30 joint
because its slower softmax commands keep drones at the physical
bottleneck too long. The static-cube AirSim cell (§4.4.3) closes a
different gap: it demonstrates a significant failure-level planner
separation under real AirSim collision geometry (GPU MPPI 30/30 joint
vs MPC 22/30, McNemar p ≈ 0.008). The density-swept extension
(§4.4.4, `base_ew06`) drops GPU MPPI off ceiling at last and
re-measures $\Delta$ at tied per-drone rates. The qualitative finding
is that the $\Delta$-flip mechanism transfers but its sign reverses:
on AirSim it is MPC, not GPU MPPI, that exhibits a multi-drone
cluster failure mode. The shared structural claim (tied per-drone,
$\Delta$ differentiates) survives the sim-backend change; the
deployment-relevant claim ("GPU MPPI's softmax is a joint-coordination
liability") does not.

Tied to that result is a separate measurement-stability limitation we
surfaced and quantified rather than worked around. The AirSim
`base_ew06` cell has substantial **run-to-run** failure-rate
variability that is wider than a single n=50 paired study's Wilson CI
predicts. Three independent fresh paired batches of n=15 each
(§4.4.4 table) show the multi-drone cluster mode reproducing at
0/15-3/15 rate, and the McNemar direction itself reversing across
batches — batch 2 favors GPU MPPI, batch 3 favors MPC, batch 1 ties.
The 3-batch combined n=45 study gives joint success tied at 86.7 %
and McNemar $p \approx 0.77$, contradicting the single n=50 study's
GPU-favored reading. The robust paper-grade findings are therefore
the *qualitative* claims: only MPC has the central-crossing
drone-drone cluster mode (collision-object trace confirms drones 1
and 2 with empty `object_name` = vehicle-vehicle collision); GPU
MPPI's softmax avoids it. The *quantitative* "which planner wins
joint success" answer is environment-sensitive (GPU thermal state,
host load, AirSim physics tick jitter) and a single n=50 measurement
is insufficient to fix it. Closing the McNemar with statistical
certainty would require either $n \geq 200$ paired or a
controlled-environment harness — both out of scope here.

The AirSim infrastructure itself also imposes constraints. Multi-drone
`client.reset()` can wedge after one or two sequential resets in
Blocks, so all n=30 AirSim studies use `scripts/run_airsim_multi_chunked.sh`
to restart the server between episodes. The stale t=0 collision flag
is fixed in the bridge by pausing immediately after reset, but the
server-side reset hang is only worked around, not solved. These are
engineering limitations of the current AirSim stack, not planner
effects.

The §3 headline is four-drone but two follow-up sweeps narrow the
mechanism's scope further. The GPU MPPI N-scaling sweep across
$N \in \{2, 3, 4, 6, 8, 10, 12\}$ on dummy_3d (findings.md
"dummy_3d N-scaling paired") shows the mechanism is **non-monotonic
in N**: GPU MPPI's higher-$\Delta$ advantage holds at $N = 4$
(+5.2 vs −1.0 pp), $N = 6$ (+10.7 vs +7.5), and $N = 10$
(+24.3 vs +15.0 — the sweep maximum), reverses at $N = 2$
(McNemar p ≈ 0.008 favours MPC), at $N = 8$ where the
8-fold-symmetric central crossing uniquely collapses GPU MPPI's
per-drone to 69 % (McNemar p ≈ 0.0001 favours MPC), and at $N = 12$
where GPU's $\Delta$ drops back to +7.8 pp vs MPC's +15.2 pp.

The companion density sweep at $N \in \{4, 6, 8\}$ (findings.md
"dummy_3d density × planner sweep at $N \in \{4, 6, 8\}$") fills in
the 3×3 (N, density) grid and shows the sign-reversal effect is
**conditional and corner-specific**, not universal. At N=4 the
$\Delta$ sign flips with obstacle count — GPU MPPI's $\Delta$ goes
from $+5.2$ pp at baseline to $-1.2$ pp at 240 obstacles while MPC's
moves from $-1.0$ to $+6.7$ pp. At N=6 the same density sweep does
*not* flip — GPU MPPI keeps a 6-32 pp per-drone advantage across
densities and stays the cluster source. At N=8 baseline GPU MPPI
uniquely collapses (per-drone 69 % vs MPC's 92 %, McNemar
$p \approx 0.0001$), but at N=8 dense the per-drone rates re-tie
and at N=8 packed GPU MPPI's per-drone advantage returns (64 % vs
37 %) — with joint success floor-low in both denser cells, no
McNemar separation. The N=4 sign flip is therefore a coincidence of
"per-drone rates stay close enough that the $\Delta$ statistic takes
over"; that condition rarely holds elsewhere in the grid. The AirSim
`base_ew06` finding sits in the N=4 dense regime (5 widened pillars
concentrating 4 drones into one central crossing, MPC vs GPU
per-drone 90 vs 96 %) and reproduces the dummy_3d N=4 dense
behaviour; it is one paired cell, not an AirSim-wide statement.
Mapping the (N, density, drone-count symmetry) surface across
non-circular geometries and at finer resolution remains future work.

A separate axis we probed only at one cell is **obstacle dynamics**.
Extending the §3 N=4 baseline by adding one moving sphere obstacle
($(20, 5, 6)$, velocity $(0, +v, 0)$, $v \in \{2, 4, 8\}$ m/s; full
table in findings.md "dummy_3d N=4 + moving obstacle speed sweep")
collapses GPU MPPI's joint success from 86.7 % (§3) to **3.3 %** at
$v=2$ m/s, while MPC holds at 73 %; at $v=4$ the same pattern repeats
(GPU 3.3 %, MPC 80 %); at $v=8$ both planners collapse to the 3.3 %
floor. The GPU MPPI failure is deterministic and single-drone: across
all v=2/4 GPU MPPI failures, only drone idx 2 (north, whose corridor
the obstacle moves along) collides, at $t \approx 4.9-5.2$ s. The
mechanism is the same softmax-averaging operator §3 describes, but
now expressing as **bidirectional avoidance cancellation**: half the
rollouts say "detour left", half say "detour right", the softmax
mean lateral command is near zero, and the drone slows in the
central corridor while the obstacle catches up. MPC's argmin commits
to one side and clears. This dynamic-obstacle cell is therefore the
*opposite* deployment story from §3: under moving obstacles GPU
MPPI's softmax conservatism is no longer a coordination liability —
it is a generalised brittleness to any obstacle-avoidance situation
that admits two symmetric escape directions. The finding is from one
trajectory geometry, one $(N, \text{density})$ corner, and one moving
obstacle; like the static-grid mechanism, the *qualitative* story
(softmax cancels bidirectional commits) is the robust claim and the
absolute magnitudes are corner-specific.

Finally, this paper is simulation-only. The ROS 2 bridge and
AirSim-over-ROS-2 harness show spatial parity across software stacks,
but they do not validate sim-to-real transfer on PX4 hardware,
MAVROS, motion-capture feedback, or outdoor GNSS-denied flight. The
results should therefore be read as benchmark and simulator-transfer
evidence, not as a field deployment guarantee.
