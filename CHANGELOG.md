# Changelog

All notable changes to `uav-nav-lab` are in this file. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) loosely;
versions follow [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-05-17

Substantial expansion since v0.1.0: the framework gains a CUDA-backed
MPPI planner family, a measured-on-paired-seeds multi-drone
coordination story, three transferability sims (AirSim, ROS 2,
AirSim-over-ROS-2), and the prose draft of a paper organising the
above into a single argument. 88 commits across 6 months of work.

### Highlights

- **Multi-drone coordination Δ-flip** (`docs/findings.md` §"Multi-drone:
  GPU MPPI's rollout cloud flips the coordination Δ"): n=100 paired
  episodes on `dummy_3d` 40×40×12 with 4 drones in a cross pattern.
  Joint success ties between MPC and GPU MPPI (78.0 vs 77.0 %), but
  coordination Δ over indep⁴ separates by an order of magnitude
  (MPC **+0.8 pp** vs GPU MPPI **+11.4 pp**). Same joint rate, very
  different failure shape: GPU MPPI's softmax across 64 rollouts
  amplifies seed sensitivity so failures *cluster* within hard seeds
  while MPC's argmin spreads failures across episodes.

- **GPU MPPI planner** (`uav_nav_lab/planner/gpu_mppi.py`): PyTorch
  CUDA-batched rollouts at the 3D Pareto cell achieve 100 % success
  at 3.5 ms steady-state plan_dt — Pareto-dominating the CPU MPC 3D
  baseline (88 % / 70 ms). The earlier "speed collapse at h ≥ 40"
  was a goal-mask bug, not an MPPI property; fixed at commit 2a9d196.
  Per-drone subsampled rollout overlay (`viz_rollouts: 8`) renders
  via `uav-nav anim` in 2D and 3D, single-drone and multi-drone.

- **AirSim multi-drone transferability** (`docs/findings.md`
  §"AirSim multi-drone n=30 paired"): n=30 paired study across
  three altitude-stagger geometries (±2-4 m, ±1 m, 0 m). Response
  is *bimodal*: non-zero z-spread keeps both planners at 100 %
  joint; uniform z=30 drops MPC to 46.7 % joint and GPU MPPI to
  **0/30 = 0.0 %** joint (McNemar paired exact p ≈ 0.00012). The
  trajectory-spread mechanism (GPU MPPI per-drone arrival spread
  0.55 s vs MPC's 0.02 s) is preserved across all measurable
  cells (GPU/MPC ratio 4-27 ×).

- **Bridge fix: pause-after-reset** (`uav_nav_lab/sim/airsim_bridge.py`
  commit 382d207): the multi-drone reset path was leaving every
  drone's collision flag set to True at t=0, because drones registered
  ground-contact collisions during the `settle_after_reset` window
  with the engine unpaused, and the subsequent teleport's
  `ignore_collision=True` flag did NOT clear the cumulative state.
  Fix: pause AirSim immediately after `client.reset()`.

### Added

#### New planner backends
- **`gpu_mppi`** — PyTorch CUDA-batched MPPI with autograd-graph
  warmup, Fibonacci-sphere direction sampling in 3D, softmax-weighted
  action averaging at configurable temperature `T`. Supports
  `viz_rollouts: int` for per-drone subsampled rollout overlay.
- **`mppi`** — single-threaded CPU MPPI reference. Used to calibrate
  the GPU port.
- **`chomp`** — M⁻¹-preconditioned trajectory smoothing, cheapest
  non-trivial planner at 21 ms / replan, 53 % success on the 50×50
  bouncing-obstacle scenario.
- **`mpc_chomp`** — layered MPC + CHOMP. Honest null result on
  saturated MPC; lifts RRT init to 90 % at +50 % compute.
- **`rrt`, `rrt_star`** — continuous-space sampling; RRT beats grid
  A* by +53 pp at similar compute; RRT\* loses to plain RRT because
  it runs 2.3× the replan budget on dynamic-obstacle scenarios.

#### New simulator backends
- **`airsim`** — Microsoft AirSim Blocks Unreal env via msgpackrpc.
  Multi-drone via `simulator.vehicles: [Drone1, …]` paired with
  `multi_drone_voxel`. ENU↔NED conversion at the bridge boundary;
  optional LiDAR (`lidars: [name, …]`), cameras
  (`cameras: [{name, image_type}, …]`), depth (`depths: [{name,
  fov_deg, …}, …]`) pass-through. Multi-drone reset hang
  workaround in `scripts/run_airsim_multi_chunked.sh`.
- **`ros2`** — `geometry_msgs/Twist` + `nav_msgs/Odometry` over
  `rclpy`. AirSim-over-ROS-2 via `cmd_msg_type: airsim_vel_cmd`.
  `use_sim_time: true` anchors `state.t` on `/clock`. Mock adapter
  CI-testable without rclpy.

#### New sensor backends
- **`lidar`** — synthetic LiDAR for `dummy_3d` (mirror of the
  AirSim LiDAR shape for cross-sim experiments).
- **`pointcloud_occupancy`** — rasterizes `state.extra["lidar_points"]`
  into an occupancy grid for the planner. Consumes both AirSim and
  ROS 2 bridge LiDAR with no code change.
- **`depth_image_occupancy`** — same path for depth cameras
  (`state.extra["depth_images"]`).

#### Animation + viz
- `uav-nav anim` now renders multi-drone scenarios in 2D and 3D
  (rotating-camera GIF with per-drone palette).
- Per-drone GPU MPPI rollout overlay (cyan-tinted palette colour
  cloud + thicker best-rollout line) renders in both single-drone
  and multi-drone 3D anim. Configurable via the planner's
  `viz_rollouts` field.
- Dynamic-obstacle replay scatter (red spheres) added to
  multi-drone 3D anim.

#### Tooling
- `scripts/run_airsim_multi_chunked.sh` — per-episode AirSim
  bounce runner for n≥30 multi-drone paired studies.
- `scripts/paired_analysis_airsim_multi.py` — Wilson CI + McNemar
  paired stats from chunked-output directories.
- `scripts/record_airsim_demo.py`, `record_airsim_multi_demo.py`,
  `record_airsim_multi_compare.py`, `record_airsim_multi_obstacles_compare.py`
  — README hero GIF pipelines.
- `scripts/render_compare_gif.py` — side-by-side GIF compositor.
- `scripts/compare_spatial_runs.py` — AirSim-direct vs
  AirSim-over-ROS-2 spatial parity harness.

#### Paper drafts
- `docs/paper_a/outline.md`, `section_1_motivation.md`,
  `section_2_setup.md`, `section_3_headline.md`,
  `section_4_prerequisites.md`, `section_4_4_sim_transferability.md`.
- ~6 pages of prose drafting the multi-drone Δ-flip story with all
  prerequisite methodology, ready for §5 secondaries + §6 limitations.

### Changed

- **`planner.gpu_mppi`**: goal-mask fix (commit 2a9d196). Pre-fix
  Pareto reported 0 % at every h ≥ 40 cell — that was a bug, not
  an MPPI property. Post-fix 2D Pareto cell shifts to (n=128, h=40),
  3D Pareto cell to (n=64-256, h=20).
- **AirSim bridge reset path**: `simPause(True)` immediately after
  `client.reset()` to prevent the stale-t=0 collision flag (see
  Highlights above).
- **`runner.multi`**: passive-first dispatch eliminates the 1-tick
  multi-drone bridge lag, enabling the 4-way AirSim crossing.
- **README**: hero GIF row reorganised. AirSim multi-drone obstacle
  GIF + rollout viz GIF are the lead images; static / boring GIFs
  pruned. "More studies" bullets regrouped by theme. Roadmap
  refreshed with the current AirSim Δ-flip discriminating-cell TODO.

### Findings (long-form in `docs/findings.md`)

- MPC compute Pareto (2D + 3D) — sole optimal cells; the 3D plan_dt
  blow-up was a missing cost-to-go cache, not a CPU cliff.
- 3D perception-latency cliff — same corner as 2D, softened by
  escape volume; `velocity_window` optimum inverts (3D peaks at 1).
- Pareto config rewrites prior conclusions — methodological lesson
  on always re-validating ablations at the planner's Pareto cell.
- Wind miscalibration — diagonal-wins; +73 pp swing from awareness
  but no belief beats `sim_wind > max_speed` physics.
- Perception-latency saga — 4-step research arc including an honest
  negative result on Kalman ego (moving-average wins).
- Multi-drone N-scaling — peer prediction correlates failures the
  right way (+14.7 pp Δ at N=4); ablating prediction costs as much
  per-drone success as 8× obstacle density (49 pp).
- 3D escape volume — Δ vanishes in open volume, returns at 4-8×
  obstacle density. Boundary of when the multi-drone Δ claim applies.
- GPU MPPI temperature ablation — 3D Pareto cell tolerates T ∈
  [0.3, 10]; T=0.1 underperforms (softmax becomes argmin-like).
- AirSim + GPU MPPI parity (single-drone, multi-drone n=1, multi-
  drone n=30 × 3 cells) — see Highlights.

### Removed / Fixed

- Stale "n ≥ 30 paired AirSim re-run is open follow-up" framing in
  README hero captions. The n=30 work is now done across three
  geometries.
- Straight-line AirSim multi-drone demo GIFs pruned from README
  hero (still in `docs/images/` for reference).

## [0.1.0] - 2025-11-XX

Initial release. CPU MPC + A\* + straight-line planners, `dummy_2d`
and `dummy_3d` sims, `grid_world` and `voxel_world` scenarios, Wilson
95 % CI stats infrastructure, `uav-nav` CLI, GitHub Actions CI on
Python 3.10/3.11/3.12.

See the [v0.1.0 release notes](https://github.com/rsasaki0109/uav-nav-lab/releases/tag/v0.1.0)
for the original feature set.
