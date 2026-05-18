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
bottleneck too long. The new static-cube AirSim cell closes a different
gap: it demonstrates a significant failure-level planner separation
under real AirSim collision geometry (GPU MPPI 30/30 joint vs MPC
22/30, McNemar p ≈ 0.008). However, it still does not reproduce the
exact §3 signature, because GPU MPPI reaches the 100 % ceiling and
therefore has Δ = 0 by construction. A static-cube density sweep that
also drops GPU MPPI into the 60-90 % per-drone band remains necessary
to test the joint-tie / larger-GPU-Δ mechanism on AirSim directly.

The AirSim infrastructure itself also imposes constraints. Multi-drone
`client.reset()` can wedge after one or two sequential resets in
Blocks, so all n=30 AirSim studies use `scripts/run_airsim_multi_chunked.sh`
to restart the server between episodes. The stale t=0 collision flag
is fixed in the bridge by pausing immediately after reset, but the
server-side reset hang is only worked around, not solved. These are
engineering limitations of the current AirSim stack, not planner
effects.

The study is still four-drone only for the GPU MPPI headline. MPC
N-scaling and density sweeps show that coordination Δ depends on
free volume per agent, but the full GPU MPPI N-scaling curve remains
future work. This matters because the §3 mechanism is a statement
about failure clustering, and clustering can change with N even when
per-drone success is held fixed.

Finally, this paper is simulation-only. The ROS 2 bridge and
AirSim-over-ROS-2 harness show spatial parity across software stacks,
but they do not validate sim-to-real transfer on PX4 hardware,
MAVROS, motion-capture feedback, or outdoor GNSS-denied flight. The
results should therefore be read as benchmark and simulator-transfer
evidence, not as a field deployment guarantee.
