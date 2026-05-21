<div align="center">

# uav-nav-lab

**Python research framework for UAV motion planning.**
YAML-driven ablations with Wilson 95 % CIs by default.

> ⚠️ **Heads-up (2026-05-22)**: a critical multi-runner bug was found
> that froze dynamic obstacles in any episode following a total-wipeout
> episode (see commit `1646e11`). The old race / gates / dyn4 / chaos
> headline numbers ("MPC 51.7 % vs softmax 3.3 %") were artifacts of
> frozen obstacles. The hero GIF below shows the first re-tuned cell
> where MPC and CPU MPPI **visibly** avoid a dynamic intruder while also
> coordinating with each other — MPC stops & waits, MPPI swerves around
> (n=5 / 10 drone-episodes / 0 collisions for both planners). The old
> findings are being rewritten; see `docs/findings.md` for the current
> state of each result.

[![CI](https://github.com/rsasaki0109/uav-nav-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/rsasaki0109/uav-nav-lab/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/rsasaki0109/uav-nav-lab/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/rsasaki0109/uav-nav-lab)](https://github.com/rsasaki0109/uav-nav-lab/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Stars](https://img.shields.io/github/stars/rsasaki0109/uav-nav-lab?style=social)](https://github.com/rsasaki0109/uav-nav-lab/stargazers)

<img src="docs/images/compare_intersection_avoid.gif" alt="2 drones approach a 4-way intersection from N and E while a slow dynamic intruder sits at the centre; MPC stops and waits, MPPI swerves around — both succeed in all 10 drone-episodes" width="1080">

<i>Two drones enter a 4-way intersection from N (red) and E (green); a
slow dynamic intruder (large red square, 0.5 m/s) sits at the centre
and drifts E-W. Same stack, same seed, only the rollout aggregator
changes — MPC argmin on the left, MPPI softmax on the right. Two
different avoidance strategies emerge from the same cost: <b>MPC
brings the N drone to a stop and waits</b> for the E drone +
intruder to clear; <b>MPPI swerves both drones around</b> the
intruder simultaneously. <b>Both planners succeed in all 10 drone-
episodes (n=5 seeds × 2 drones, 0 collisions)</b>. The same contrast
scales to a 4-drone 4-way crossing (both pairs head-on, 20/20 drone-
episodes, see <code>compare_intersection_4way.gif</code> below).
Reproduce with <code>examples/exp_intersection_v1_{mpc,mppi}.yaml</code>.
&nbsp;<a href="docs/findings.md">Findings</a>
&middot; <a href="docs/paper_a/section_3_headline.md">§3 4-mode framework</a></i>

</div>

## 🚀 Quick start

```bash
git clone https://github.com/rsasaki0109/uav-nav-lab
cd uav-nav-lab
pip install -e '.[dev,viz]'        # numpy + pyyaml + matplotlib + pytest
# Optional: pip install -e '.[gpu]' (PyTorch for gpu_mppi), '.[rl]' (SB3)
pytest -q

uav-nav run     examples/exp_basic.yaml
uav-nav eval    results/basic_astar
uav-nav viz     results/basic_astar
```

A 2D heatmap sweep is one CLI invocation:

```bash
uav-nav sweep examples/exp_predictive.yaml \
  --param planner.horizon=20 --param planner.n_samples=16 \
  --param planner.max_speed=10,15,20,25,30 \
  --param planner.replan_period=0.1,0.2,0.5,1.0,2.0 \
  --param num_episodes=20 -j 4
uav-nav viz <out>     # → 6-panel sweep_summary.png
```

## 🛠️ CLI

| command | what |
|---|---|
| `uav-nav run <yaml>` | run all episodes, write per-episode JSONs + `summary.json` |
| `uav-nav eval <run_dir>` | recompute metrics, print Wilson 95 % CIs + planner-dt budget |
| `uav-nav compare <a> <b> ...` | side-by-side table with ± half-widths |
| `uav-nav sweep <yaml> --param k=spec` | Cartesian-product over `--param`s |
| `uav-nav viz <run_or_sweep>` | trajectory PNG per episode, or 6-panel sweep heatmap |
| `uav-nav anim <run_dir>` | animated GIF replay (2D) |
| `uav-nav video <run_dir>` | ffmpeg AirSim camera frames into per-episode MP4 |
| `uav-nav list` | enumerate registered planners / sensors / sims / scenarios |

`--param` syntax: `start:stop:step`, `a,b,c`, `[3,0]`, `true` / `false`, and
dotted keys like `planner.predictor.velocity_noise_std=0.0,0.5,1.0`.

## 🏗️ Architecture

```mermaid
flowchart LR
    YAML[experiment.yaml] -->|--param overrides| RUN[uav-nav run]
    RUN --> EPS[per-episode JSONs<br/>summary.json]
    EPS --> EVAL[uav-nav eval<br/>Wilson 95% CIs]
    EPS --> VIZ[uav-nav viz<br/>trajectory PNG]
    EPS --> ANIM[uav-nav anim<br/>animated GIF]
    YAML -->|Cartesian product| SWEEP[uav-nav sweep -j N]
    SWEEP --> CELLS[run_000…run_NNN]
    CELLS --> SVIZ[uav-nav viz<br/>6-panel heatmap]
    CELLS --> CMP[uav-nav compare]

    subgraph backends["pluggable backends (registry)"]
      SIM[sim] --- SCEN[scenario] --- PLAN[planner] --- SENS[sensor] --- PRED[predictor]
    end
    RUN -.uses.-> backends
    SWEEP -.uses.-> backends
```

| kind | shipped |
|---|---|
| sim | `dummy_2d`, `dummy_3d`, `airsim`, `ros2` |
| scenario | `grid_world`, `voxel_world`, `multi_drone_{grid,voxel,aerobatic}` |
| planner | `astar`, `straight`, `mpc`, `mppi`, `gpu_mppi`, `rrt`, `rrt_star`, `chomp`, `mpc_chomp` |
| sensor | `perfect`, `delayed`, `kalman_delayed`, `lidar`, `pointcloud_occupancy`, `depth_image_occupancy` |
| predictor | `constant_velocity`, `noisy_velocity`, `kalman_velocity` |

Add a backend by dropping a file with `@REGISTRY.register("name")` and a
`from_config(cfg)` classmethod — the CLI picks it up via `type: name`.

## 📊 Research findings

Full long-form write-ups in [`docs/findings.md`](docs/findings.md);
the working paper draft is under [`docs/paper_a/`](docs/paper_a/). The
active findings are grouped this way:

- **Static multi-drone coordination** — MPC argmin and GPU MPPI softmax
  can tie on joint success while producing different failure clustering
  (`Δ` over the independent-drone baseline). The sign depends on the
  `(N, density)` cell, so the result is a mechanism claim, not a
  universal planner ranking.
- **AirSim transferability** — the same coordination mechanism appears
  under AirSim physics, but dense static-cube cells can reverse which
  planner clusters failures. Absolute winner claims are treated as
  environment-sensitive.
- **Planner / sim framework** — YAML-driven paired runs cover CPU MPC,
  GPU MPPI, sampling planners, CHOMP variants, AirSim, ROS 2, and
  AirSim-over-ROS-2 parity checks.
- **Dynamic-obstacle race studies** — currently under repair after the
  `1646e11` multi-runner fix. Old race / gates / dyn4 / chaos numbers
  should not be cited until the scenarios are re-tuned and re-run.
- **Methodology** — Wilson 95 % CIs by default, McNemar paired tests
  for matched-seed comparisons, and Pareto-cell re-validation before
  making ablation claims.

<details>
<summary><b>Companion hero GIFs</b> — 4-way intersection ablation, multi-drone Δ-flip, single-drone 3D MPPI</summary>

<img src="docs/images/compare_intersection_4way.gif" width="720"><br>
<i><b>Intersection 4-way ablation</b> — extend the 2-drone hero to a
4-drone 4-way crossing (two head-on pairs N↔S + E↔W) with the same
slow centre intruder. <b>Both planners 5/5 episodes / 20/20 drone-
episodes / 0 collisions.</b> MPC has the S→N drone stop & wait while
the other three detour around the intruder; MPPI has all four drones
swerve simultaneously, each head-on pair offsetting in opposite
directions to braid around the intruder without anyone stopping.
Confirms the softmax-vs-argmin avoidance signature scales with peer
count. Reproduce with <code>examples/exp_intersection_4way_{mpc,mppi}.yaml</code>.</i>

<br><br>

<img src="docs/images/compare_multi_drone_3d_mpc_vs_gpu_mppi.gif" width="720"><br>
<i><b>§3 mode 1 multi-drone Δ-flip</b> (N=4 paired n=100, dummy_3d):
joint tied at 78 / 77 %, coordination Δ over indep⁴ separates by an
order of magnitude — MPC <b>+0.8 pp</b> vs GPU MPPI <b>+11.4 pp</b>.
GPU MPPI's softmax against a shared peer-prediction world model
clusters failures within seeds rather than spreading them.</i>

<br><br>

<img src="docs/images/compare_gpu_mppi_vs_mpc_3d.gif" width="720"><br>
<i><b>3D MPC vs GPU MPPI</b> single-drone navigation: rollout cloud
visible on the GPU MPPI side (light-blue spaghetti), single committed
trajectory on the MPC side. Both succeed; the visual shows the
algorithmic signature of each aggregator.</i>

</details>

<details>
<summary>⚠️ <b>Dynamic-obstacle hero GIFs (under repair)</b></summary>

The original race / gates / dyn4 / chaos GIFs were rendered against
the frozen-obstacle bug (fixed in <code>1646e11</code>). Re-runs with
the fix show those scenarios as designed are <b>uniformly 100 %
collision for every planner</b> — the moving-gate gap closes faster
than the planner's 0.4 s lookahead can detour around, and likewise
for the path-intersecting intruders. The "MPC vs softmax" contrast
on those scenarios was an artifact of the bug, not a real
planner-level finding.

The current hero GIF (<code>compare_intersection_avoid.gif</code> at
the top of the README) is the first re-tuned cell where both planners
visibly avoid a dynamic intruder while also coordinating with another
drone — MPC stops & waits, MPPI swerves around (n=5 / 10 drone-
episodes / 0 collisions for both). An earlier re-tune attempt on the
oval-race scenario (<code>compare_race_avoid.gif</code>) succeeded
statistically but did not show visible avoidance — drones at
period=19.8 mostly slipped past the bouncing intruders without an
obvious detour. The original gates4 / chaos / dyn4 scenarios still
need further re-tuning (wider gaps, slower gates) before their GIFs
go back up.

</details>

<details>
<summary><b>More demos</b> — aerobatic loop, multi-drone Δ-flip, AirSim</summary>

<table>
<tr><td><img src="docs/images/compare_aerobatic_loop4.gif" width="720"></td></tr>
<tr><td align="center"><i>§3 mode 4 — aerobatic loop, GPU MPPI delivers 84 % tighter phase sync.</i></td></tr>
<tr><td><img src="docs/images/compare_multi_drone_3d_mpc_vs_gpu_mppi.gif" width="720"></td></tr>
<tr><td align="center"><i>§3 mode 1 — joint tied at 78 / 77 %, Δ over indep⁴ +0.8 vs <b>+11.4 pp</b>.</i></td></tr>
<tr><td><img src="docs/images/compare_airsim_multi_obstacles.gif" width="720"></td></tr>
<tr><td align="center"><i>AirSim multi-drone FPV — MPC vs GPU MPPI through the same Blocks scenario.</i></td></tr>
</table>

</details>

## ✅ Status

v0.2.0 is tagged; CI runs on Python 3.10 / 3.11 / 3.12. The current
stack includes 4 sim backends, 6 sensors, 3 predictors, 9 planners, and
5 scenario families. Stable ablations are reproducible from the example
YAMLs and scripts; the re-tuned dynamic-obstacle hero is the
`compare_intersection_avoid.gif` 2-drone intersection (both planners
0/10 collisions, visibly different avoidance strategies). The older
race / gates4 / dyn4 / chaos scenarios remain marked under repair
after the `1646e11` bug fix.

**External backends:**

- **AirSim** (`uav_nav_lab/sim/airsim_bridge/`) — ENU ↔ NED bridge
  with deterministic stepping (`simPause` + `simContinueForTime`),
  multi-vehicle, LiDAR / cameras / depth, mock-injectable client for
  CI. See `examples/exp_airsim_*.yaml`.
- **ROS 2** (`uav_nav_lab/sim/ros2_bridge/`, requires `rclpy`) —
  Twist + Odometry round-trip, sim-time anchoring via `/clock`,
  AirSim-over-ROS-2 parity. See `examples/exp_ros2*.yaml`.

## 📄 License

Apache-2.0.
