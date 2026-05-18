# §7. Reproducibility map

Every quantitative claim in the paper is tied to a committed example
configuration or to a named runner script. The map below is the
appendix-level index: start from the paper section, run the listed
YAML(s), then compare against the named `docs/findings.md` section.

Unless noted otherwise, non-AirSim studies run through the normal CLI
entry points (`uav-nav run`, `uav-nav sweep`, or `uav-nav compare`) and
write result directories under `results/`. AirSim multi-drone studies
use `scripts/run_airsim_multi_chunked.sh` because Blocks can hang on
sequential multi-vehicle resets; the chunked runner restarts the server
between episodes.

| paper section | primary artifact(s) | findings.md anchor |
|---|---|---|
| §3 multi-drone Δ flip | `examples/exp_multi_drone_3d_4.yaml`, `examples/exp_multi_drone_3d_4_gpu_mppi.yaml` | "Multi-drone: GPU MPPI's rollout cloud flips the coordination Δ" |
| §4.1 2D MPC Pareto | `examples/exp_predictive.yaml` | "MPC compute Pareto" |
| §4.1 3D MPC Pareto | `examples/exp_3d_predictive.yaml` | "3D Pareto: the n_samples preference flips" |
| §4.1 2D GPU MPPI Pareto | `examples/exp_gpu_mppi_pareto.yaml` | "2D Pareto (post-fix)" |
| §4.1 3D GPU MPPI Pareto | `examples/exp_gpu_mppi_pareto_3d.yaml` | "3D Pareto (post-fix)" |
| §4.1 GPU MPPI temperature ablation | `examples/exp_gpu_mppi_temp_ablation_3d.yaml` | "Temperature ablation at the 3D Pareto cell" |
| §4.2 goal-mask correctness gate | commit `2a9d196`, `uav_nav_lab/planner/gpu_mppi.py` | "The goal-mask bug fix that changed every cell" |
| §4.3 CUDA plan-time reporting | same GPU MPPI Pareto configs; first replan dropped in analysis | "GPU MPPI: post-goal-mask fix unlocks long-horizon cells, 3D MPPI beats 3D MPC" |
| §4.4 dummy_3d vs AirSim transfer | `examples/exp_transfer_dummy.yaml`, `examples/exp_transfer_airsim.yaml` | "AirSim vs dummy_3d transferability: same plan, different physics" |
| §4.4 AirSim GPU MPPI single-drone parity | `examples/exp_airsim_demo.yaml`, `examples/exp_airsim_demo_gpu_mppi.yaml` | "AirSim + GPU MPPI parity" |
| §4.4 AirSim GPU MPPI multi-drone parity | `examples/exp_airsim_multi_demo.yaml`, `examples/exp_airsim_multi_demo_gpu_mppi.yaml` | "AirSim multi-drone parity" |
| §4.4 AirSim staggered-altitude n=30 | `examples/exp_airsim_multi_n30.yaml`, `examples/exp_airsim_multi_n30_gpu_mppi.yaml`, `scripts/run_airsim_multi_chunked.sh` | "AirSim multi-drone n=30 paired" |
| §4.4 AirSim uniform-altitude n=30 | `examples/exp_airsim_multi_uniform_n30.yaml`, `examples/exp_airsim_multi_uniform_n30_gpu_mppi.yaml`, `scripts/run_airsim_multi_chunked.sh` | "AirSim multi-drone uniform-altitude n=30: GPU MPPI collapses to 0 % joint while MPC holds 46.7 %" |
| §4.4 AirSim ±1 m mid-stagger n=30 | `examples/exp_airsim_multi_mid_n30.yaml`, `examples/exp_airsim_multi_mid_n30_gpu_mppi.yaml`, `scripts/run_airsim_multi_chunked.sh` | "AirSim multi-drone ±1 m mid-stagger n=30: still ceiling-limited, cliff between 0 and 1 m" |
| §4.4 AirSim static-cube discriminating cell | `examples/exp_airsim_multi_discriminating_n30.yaml`, `examples/exp_airsim_multi_discriminating_n30_gpu_mppi.yaml`, `scripts/run_airsim_multi_chunked.sh` | "AirSim multi-drone static-cube discriminating cell n=30" |
| §4.4.4 AirSim density-sweep cell `base_ew06` (Δ-flip sign reversal) | `examples/exp_airsim_multi_discriminating_n30{,_gpu_mppi}.yaml` (EW pillar scale 0.5→0.6 via param sweep), `scripts/run_airsim_discriminating_param_sweep.sh` | "AirSim multi-drone base_ew06 n=30: MPC Δ +6.9 pp clusters, GPU MPPI Δ -1.5 pp independent — Δ-flip sign reverses from dummy_3d" |
| §4.4 reset/collision bridge fix | `uav_nav_lab/sim/airsim_bridge.py`, `scripts/run_airsim_multi_chunked.sh` | "Bridge fix: pause-after-reset eliminates a stale-t=0 collision flag" |
| §4.4 ROS 2 spatial equivalence | `scripts/ros2_dummy_sim.py`, `examples/exp_basic.yaml`, `examples/exp_ros2.yaml` | "ROS 2 bridge: spatial equivalence verified" |
| §4.4 AirSim-over-ROS-2 harness | `examples/exp_airsim_ros2.yaml`, `examples/exp_airsim_ros2_direct.yaml` | "AirSim over ROS 2 parity harness" |
| §5.1 3D escape volume | `examples/exp_multi_drone_3d_4.yaml` | "3D escape volume erases the coordination Δ" |
| §5.1 3D density ablation | `examples/exp_multi_drone_3d_4.yaml`, `examples/exp_multi_drone_3d_4_dense.yaml`, `examples/exp_multi_drone_3d_4_packed.yaml` | "3D density ablation: bring escape volume back to non-trivial — Δ comes back too" |
| §5.2 peer-prediction ablation | `examples/exp_multi_drone_3d_4_dense_indep.yaml`, `examples/exp_multi_drone_3d_4_packed_indep.yaml` | "3D peer-prediction ablation: removing CV prediction is worse than 8× obstacle density" |
| §6 limitations / future work | no new run; summarizes the sections above plus SAC scaffold status | "RL comparison baseline: gym.Env scaffold + initial training" |

The AirSim static-cube discriminating pair is the newest reproducible
cell. Its paired command pattern is:

```bash
scripts/run_airsim_multi_chunked.sh mpc 30 0 \
  results/airsim_multi_discriminating_n30_mpc \
  examples/exp_airsim_multi_discriminating_n30.yaml

scripts/run_airsim_multi_chunked.sh gpu_mppi 30 0 \
  results/airsim_multi_discriminating_n30_gpu_mppi \
  examples/exp_airsim_multi_discriminating_n30_gpu_mppi.yaml

python scripts/paired_analysis_airsim_multi.py \
  results/airsim_multi_discriminating_n30_mpc \
  results/airsim_multi_discriminating_n30_gpu_mppi
```

The expected summary for that run is MPC per-drone 105/120 = 87.5 %
and joint 22/30 = 73.3 %, versus GPU MPPI per-drone 120/120 = 100 %
and joint 30/30 = 100 %, with GPU-only paired successes on eight
seeds and exact McNemar p ≈ 0.008.

The §4.4.4 density-sweep extension (`base_ew06`) is reproduced via
the variant generator:

```bash
VARIANTS="base_ew06" MODE=paired N=30 BASE_SEED=42 \
  scripts/run_airsim_discriminating_param_sweep.sh
```

The script generates `/tmp/uavnav_airsim_disc_base_ew06_{mpc,gpu_mppi}.yaml`
from the §4.4.3 YAMLs (scaling the four EW pillars from 0.5 to 0.6),
chunked-runs n=30 paired (seeds 42..71), and invokes
`paired_analysis_airsim_multi.py` automatically. Expected summary:
MPC per-drone 104/120 = 86.7 %, joint 19/30 = 63.3 %, Δ over indep⁴
= +6.9 pp; GPU MPPI per-drone 114/120 = 95.0 %, joint 24/30 = 80.0 %,
Δ over indep⁴ = -1.5 pp; McNemar paired both 14 / MPC-only 5 /
GPU-only 10 / neither 1, exact $p \approx 0.302$.
