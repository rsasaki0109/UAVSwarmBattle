# Dynamic Obstacle Avoidance Survey

Date: 2026-05-25

This note exists because the README race GIF review exposed a real
methodology problem: a non-contact trajectory is not enough evidence
that the planner avoided a moving obstacle.  The replacement dynamic
obstacle work should be designed from papers and OSS systems first,
then rendered.

## Executive Takeaway

The literature does not treat dynamic-obstacle avoidance as "a path
missed the disk in the replay."  The stronger systems expose some
combination of:

- a nominal or no-obstacle path that would actually conflict,
- a time-indexed obstacle prediction or committed obstacle trajectory,
- a planner that reasons in space and time, not only in current-space
  distance,
- trajectory deviation and signed-clearance metrics around the
  encounter,
- success/failure rates across seeds, plus event logs for closest
  approach, gate passage, and collisions.

For `uav-nav-lab`, a README hero should not be accepted unless the
same seed has at least these controls:

1. `moving`: moving obstacle enabled, planner succeeds.
2. `no_obstacle`: obstacle removed, the same controller would enter the
   moving obstacle's original safety tube or pass substantially closer.
3. `frozen`: obstacle present but static at its initial or encounter
   pose, proving the behavior is not just static-obstacle clearance.
4. `wrong_prediction` or `no_prediction`: planner receives stale or
   zero velocity, showing that the dynamic predictor matters.

The current `race-simple` README hero only partially meets this bar. It
now has the full single-seed control set, and `no_prediction` fails on
the same scene, but the best no-obstacle virtual penetration is still
only `-0.0007 m`. It is acceptable as a debugging visual, not as the
final dynamic-obstacle hero.

## Source Patterns

| Source | What it does | What matters for us |
|---|---|---|
| [Polynomial-based Online Planning for Autonomous Drone Racing in Dynamic Environments](https://arxiv.org/abs/2306.14461) | Online polynomial replanning for racing under dynamic changes. It trades aggressive speed against flexible obstacle avoidance, keeps intermediate racing waypoints as hard constraints, and uses parallel multi-topology trajectory planning for dynamic obstacles. | A race hero needs more than one local line. It should show a race objective plus alternative topologies around a moving hazard. |
| [Time-Optimal Online Replanning for Agile Quadrotor Flight](https://aerial-core.eu/wp-content/uploads/2023/10/Time-Optimal_Online_Replanning_for_Agile_Quadrotor_Flight.pdf) | Real-time replanning for high-speed quadrotor racing, demonstrated with moving gates and wind disturbances. | Moving gates are a good race-native dynamic obstacle, but success is about online replanning under timing changes, not just disk avoidance. |
| [AirSim Drone Racing Lab](https://arxiv.org/abs/2003.05654) and [Microsoft Research page](https://www.microsoft.com/en-us/research/publication/airsim-drone-racing-lab/) | A racing benchmark with tracks, gates, race orchestration, sensor modalities, and benchmarking across planning, control, perception, and learning algorithms. | Hero visuals should be backed by race event logs: gate passage, lap time, collision, near miss, and per-agent state. |
| [MADER](https://github.com/mit-acl/mader) / [paper](https://arxiv.org/abs/2010.11061) | 3D decentralized trajectory planning for static obstacles, dynamic obstacles, and other agents. It separates trajectory intervals using outer polyhedral representations and constraints between polyhedra. | Dynamic obstacle avoidance is formulated as trajectory-vs-trajectory separation over time, not frame-by-frame nearest distance. |
| [PANTHER](https://github.com/mit-acl/panther) / [paper](https://arxiv.org/abs/2103.06372) | Perception-aware trajectory planner for multirotors in dynamic environments. It jointly optimizes translation and yaw so dynamic obstacles stay in the sensor FOV while being avoided. | If obstacles are unknown or sensed, the planner must also preserve observability. A convincing demo should show predicted/observed obstacle tracks. |
| [dyn_small_obs_avoidance](https://github.com/hku-mars/dyn_small_obs_avoidance) | Complete LiDAR UAV system with FAST-LIO, time-accumulated KD-tree mapping, and kinodynamic A*; reports avoidance of small dynamic bars at 50 Hz. | Dynamic avoidance systems report sensing stack, update rate, and obstacle size. We should log planner update rate and detection/prediction timing. |
| [Fast-Planner](https://github.com/HKUST-Aerial-Robotics/Fast-Planner) | Quadrotor fast-flight framework with kinodynamic path search, B-spline optimization, topological path search, and perception-aware planning. | A robust planner separates front-end topology search from back-end smoothing. Our current MPPI is missing explicit topology structure. |
| [Teach-Repeat-Replan](https://github.com/HKUST-Aerial-Robotics/Teach-Repeat-Replan) | Aggressive flight framework that turns rough routes into smooth repeat trajectories and locally replans around unmapped or moving obstacles. | Good race demos preserve the intended route while allowing local replan deviations. This maps directly to "race line vs detour" metrics. |
| [Nav2 MPPI controller](https://docs.ros.org/en/iron/p/nav2_mppi_controller/) | Production MPPI local controller with plugin critics and documented tuning tradeoffs between path alignment and obstacle avoidance. | Exact path tracking can remove the ability to deviate around dynamic obstacles. We should sweep and report path-align vs obstacle penalties instead of only temperature. |

## What Current `uav-nav-lab` Is Missing

### 1. Stronger controls

The `race_hero_causality_controls.py` no-sweeper control was the right
first correction, but it is not enough.  A final dynamic-obstacle
claim should include:

- `no_obstacle`: remove the obstacle entirely.
- `frozen_initial`: keep the obstacle at its start pose.
- `frozen_encounter`: keep the obstacle at the closest-approach pose.
- `wrong_velocity`: same obstacle position, velocity sign reversed or
  velocity set to zero only in the planner's perceived dynamics.
- `no_prediction`: obstacle exists in collision checking but the planner
  receives no future trajectory.

The expected outcome for a real dynamic avoidance hero is:

| arm | expected result |
|---|---|
| moving | success, positive signed clearance |
| no_obstacle | nominal race line enters or nearly enters original moving obstacle tube |
| frozen_initial | does not explain the same detour |
| frozen_encounter | shows static clearance is insufficient or different |
| wrong_velocity/no_prediction | lower clearance, collision, stop, or visibly different emergency behavior |

### 2. Event-level metrics

We need a reusable report, not one-off GIF tuning.  Add a script such
as `scripts/dynamic_encounter_report.py` that emits JSON with:

- obstacle travel distance during the encounter window,
- obstacle speed, radius, and predicted positions at each replan,
- moving-vs-control path delta over the encounter window,
- signed clearance to the moving obstacle tube for every arm,
- time-to-collision of the no-obstacle trajectory against the moving
  obstacle,
- first replan where obstacle enters the planner horizon,
- selected rollout clearance vs executed command clearance,
- reference error and race progress before/after the encounter,
- outcome classification: `dynamic_avoidance`, `static_clearance`,
  `lucky_non_contact`, `collision`, `peer_follow_on`, `timeout`.

### 3. Better scenario design

The hero should be generated from a cell where the no-obstacle
trajectory has a clear conflict, not a grazing `-0.0007 m` virtual
penetration.  Target acceptance thresholds:

- no-obstacle virtual min clearance <= `-0.5 m`,
- moving-run min clearance >= `+0.25 m`,
- moving-vs-no-obstacle max path delta >= `1.0 m`,
- obstacle travels at least `4.0 m` inside the GIF window,
- moving obstacle enters the planner horizon at least two replans
  before closest approach,
- the drone still maintains race progress, rather than stopping in
  place and letting the obstacle pass.

### 4. Planner capability gap

The current GPU MPPI planner samples actions around a local prior and
scores predicted dynamic obstacles, but it lacks explicit topological
branching.  The drone racing and Fast-Planner line of work suggests
the next serious implementation should be one of:

- add mode-aware / topology-aware rollout groups around the obstacle,
- add a short-horizon spatio-temporal corridor constraint,
- add a front-end kinodynamic path search that seeds MPPI with
  left/right/wait/accelerate branches,
- add a race-progress term so detours are evaluated against lap
  progress, not only waypoint tracking error.

For a fair MPPI-specific study, do not immediately replace the planner
with a B-spline optimizer.  First add instrumentation and scenario
controls; then decide whether topology-aware sampling is necessary.

## Recommended Next Work

### P0: stop using the current GIF as final evidence

Keep `docs/images/compare_race_temperature_avoid.gif` only as a
debugging placeholder.  The README caption already calls it a
single-seed visual control, but the next commit should probably demote
it further unless a stronger cell is found.

### P1: build the encounter report

Initial implementation added as `scripts/dynamic_encounter_report.py`.
It works on the current race-simple logs and compares arbitrary
`ROLE:LABEL:PATH` arms against one reference moving-obstacle config.
The first full-control report is
`docs/data/dynamic_encounter_report_p19p8_y5p0_35p0.json`.

The stronger control runs were generated with
`scripts/race_hero_control_variants.py` and summarized in
`docs/data/race_hero_control_variants.json`:

- `frozen_initial`: joint success `1/1`,
- `frozen_encounter`: joint collision `1/1`, min dynamic clearance
  `-0.45 m`,
- `wrong_velocity`: joint success `1/1`, but with a visibly different
  path delta,
- `no_prediction`: joint collision `1/1` (`1/4` drones succeed),
  min dynamic clearance `+0.01 m`.

Current verdict for the README race hero:

```text
weak_dynamic_avoidance_control
```

Reasons:

- moving-arm clearance is only `+0.10 m`, below the `+0.25 m` target,
- no-obstacle virtual penetration is only `-0.0007 m`, not the `<= -0.5 m`
  target,
- moving-vs-no-obstacle path delta is `0.81 m`, below the `1.0 m`
  target.

Interpretation: the `no_prediction` arm is useful evidence that dynamic
prediction matters in this scene, but the current no-obstacle conflict
is too shallow to carry a final README / paper-grade dynamic avoidance
claim. The next improvement is a control-first cell sweep, not another
GIF pass.

### P2: search cells by controls, not by GIF

Replace manual GIF inspection with a sweep:

1. Generate cells over obstacle start phase, speed, radius, and period.
2. Run `moving` and `no_obstacle` first.
3. Keep only cells where no-obstacle virtual clearance is meaningfully
   negative and moving succeeds.
4. Run `frozen` and `wrong_velocity` controls only for survivors.
5. Render GIF only after the JSON verdict is `dynamic_avoidance`.

Initial control-first sweep is now implemented as
`scripts/race_hero_control_sweep.py`. It reuses the no-sweeper ghost to
screen hypothetical sweeper tubes, then runs only selected moving arms.
Two first-pass outputs are tracked:

- `docs/data/race_hero_control_sweep_lowtemp.json`: eight
  low-temperature candidates with no-obstacle virtual clearance from
  `-1.02 m` to `-0.50 m`; all moving arms collided, so there were no
  survivors.
- `docs/data/race_hero_control_sweep_argmin_safety.json`: two
  stronger-controller candidates using argmin fallback, safety margin
  `0.8`, and obstacle weight `500`; the best local encounter reached
  `+0.30 m` focus-obstacle clearance and `1.58 m` path delta, but the
  episode still ended in collision, so it is not a valid hero.

Conclusion: simple phase/radius search can produce a strong no-obstacle
counterfactual, but the current local MPPI controller does not complete
those cells reliably. The next step should be planner capability
(topology/mode-aware rollout groups, spatio-temporal corridor, or race
progress-aware branch seeding), not another GIF pass.

### P3: only then update README

README hero acceptance criteria:

- at least one seed with the full control set,
- preferably n >= 10 showing the same event class is not a one-seed
  artifact,
- GIF overlays moving obstacle, predicted future obstacle, moving-run
  trajectory, no-obstacle ghost, and optionally selected rollout,
- caption includes exact clearance, path delta, and control verdict.

## Immediate Design Rule

Do not claim "dynamic obstacle avoidance" from a GIF unless the
no-obstacle ghost would have hit the moving obstacle tube by a visible
margin and the moving-obstacle run changes path before closest
approach.  Otherwise call it what it is: a contact/non-contact
counterfactual or a race tracking visual.
