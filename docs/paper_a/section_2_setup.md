# §2. Setup

All experiments in this paper are reproducible from a single Python
package, `uav-nav-lab`, with one `examples/exp_*.yaml` per cited
finding. A single CLI verb runs the experiment, computes Wilson
95 % CIs and McNemar paired statistics, and writes the per-episode
JSON logs that the figures replay. The harness is open source
(Apache-2.0) and the YAMLs cited below carry their own
result tables and reproduce commands in their headers, so each
table in this paper has a 1:1 mapping to a YAML in the repository.

## 2.1 Pluggable backends

The harness factors a planning experiment into five swappable
backends — simulator, scenario, planner, sensor, predictor — wired
into one configuration file. The CLI's `run` verb instantiates each
backend by `type` name from a registry, so swapping (for example)
the simulator from `dummy_3d` to `airsim` is a one-line YAML edit:

```yaml
simulator: { type: airsim, dt: 0.05, max_steps: 400, ... }
```

We report on four simulator backends in this work:
**`dummy_2d`** and **`dummy_3d`** (point-mass kinematics integrated at
the planner's `dt`, used for the Pareto sweeps and the headline
multi-drone study); **`airsim`** (Microsoft AirSim's Blocks Unreal env
with SimpleFlight multirotor controller, used for the transferability
checks); and **`ros2`** (publishes `geometry_msgs/Twist`, subscribes
to `nav_msgs/Odometry`, used to verify that the planner produces
identical trajectories when driven through a ROS 2 round-trip vs the
direct in-process call).

We report on five planner backends: **CPU MPC** (deterministic
sample-batch with `n_samples` × `horizon` rollouts, the Pareto-config
baseline for every comparison); **GPU MPPI** (PyTorch CUDA batched
rollouts with Fibonacci-sphere direction sampling in 3D, softmax-
weighted action averaging at temperature $T$, the focal alternative);
**CPU MPPI** (single-threaded reference implementation, used to
calibrate the GPU port); **A\*** (8-connected grid baseline);
and **SAC** (a reinforcement-learning baseline scaffold via
`stable-baselines3`). Three predictor backends — `constant_velocity`,
`noisy_velocity`, `kalman_velocity` — provide the multi-drone peer
prediction layer.

## 2.2 Scenarios

Three scenario backends produce the test cases:

`grid_world` is a 2D 50 × 50 cell grid with bouncing dynamic
obstacles, used for the planner head-to-head table (§4 appendix) and
the Pareto sweeps. `voxel_world` is its 3D analogue (40 × 40 × 12
cells with both static random obstacles and bouncing dynamic spheres),
used for the 3D Pareto sweeps and the GPU MPPI / CPU MPC plan-time
comparison. `multi_drone_voxel` extends `voxel_world` to N drones in
shared static + dynamic obstacle space; each drone gets its own
planner instance and consumes the other drones' positions through a
constant-velocity peer predictor.

The headline multi-drone study (§3) uses `multi_drone_voxel` at
N = 4, with the four drones flying a cross-pattern (east/west and
north/south pairs starting at opposite faces and converging on the
volume centre). The volume is 40 × 40 × 12 cells with 30 random
static obstacles; the AirSim transferability checks (§4.4) use
`multi_drone_voxel` on a 60 × 60 × 40 volume with the four drones
crossing at altitudes 26-32 m, well above the Blocks env's ground
geometry. Configurations are committed to `examples/`.

## 2.3 Metrics

We report binary success rates with Wilson 95 % intervals — they
behave better than normal-approximation intervals at small $n$ and
near 0/100 % rates, both of which occur frequently in our paired
comparisons. Continuous metrics (planner replan time `plan_dt`,
average velocity, time-to-goal) carry mean ± 1.96·SEM. Paired
comparisons across planners on the same seeds use the McNemar test
on joint outcomes; we report the 2 × 2 contingency table along with
the exact two-sided binomial p-value rather than the asymptotic
$\chi^2$ approximation, again because our $n$ is small enough that
the asymptote is meaningfully off.

Planner replan-time reporting deserves a clarification. The first
call of any GPU MPPI episode pays an autograd-graph compile (~14 s
on our hardware) plus a Dijkstra cost-to-go precompute (~50 ms);
neither is paid again within the episode. Including the first call
in plan_dt's mean makes the GPU MPPI numbers ~10× the steady-state.
Throughout, we report plan_dt's **steady-state mean and p95** with
the first call dropped; the warmup cost is acknowledged separately
in §4.3.

## 2.4 Coordination Δ

For an $N$-drone joint episode, define:

$$
\text{per-drone success} = \frac{\#(\text{drone-episodes with no collision})}{N \cdot \#\text{episodes}}
$$

$$
\text{joint success} = \frac{\#(\text{episodes with no drone collided})}{\#\text{episodes}}
$$

$$
\text{indep}^N = (\text{per-drone success})^N
$$

$$
\Delta = \text{joint success} - \text{indep}^N
$$

Interpretation: $\Delta = 0$ corresponds to per-drone failures
independently distributed across the $N$ drones in each episode.
$\Delta > 0$ means failures *cluster* within episodes (some seeds
take down multiple drones; other seeds let all $N$ through).
$\Delta < 0$ means failures *spread* across episodes (one drone
typically fails at a time, but most episodes have at least one
failure). The CLI verb `uav-nav eval` reports all four quantities
with Wilson CIs by default; `uav-nav compare A B` reports the same
quantities side-by-side along with the per-cell deltas. We will
return to this definition in §3 to ground the headline reading
that two planners with statistically identical joint rates can have
**dramatically different $\Delta$ values** — i.e. statistically
indistinguishable joint coordination outcomes with structurally
distinct failure shapes.

## 2.5 Reproducibility map

The full mapping from paper section → YAML file → `findings.md`
write-up is in §7 (Reproducibility map). The convention used
throughout: every numbered result table cites an `examples/exp_*.yaml`
in its caption, and that YAML's *file header* contains the same table
plus a one-line reproduce command. The harness's `compare` verb
reproduces every paired comparison cited in §3-§4.4 from two run
directories produced by `uav-nav run`.
