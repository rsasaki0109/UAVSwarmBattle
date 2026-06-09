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

For `UAVSwarmBattle`, a README hero should not be accepted unless the
same seed has at least these controls:

1. `moving`: moving obstacle enabled, planner succeeds.
2. `no_obstacle`: obstacle removed, the same controller would enter the
   moving obstacle's original safety tube or pass substantially closer.
3. `frozen`: obstacle present but static at its initial or encounter
   pose, proving the behavior is not just static-obstacle clearance.
4. `wrong_prediction` or `no_prediction`: planner receives stale or
   zero velocity, showing that the dynamic predictor matters.

The earlier `race-simple` README hero only partially met this bar. It
had the full single-seed control set, and `no_prediction` failed on the
same scene, but the best no-obstacle virtual penetration was only
`-0.0007 m`. The current README hero moves to the stronger
`p19p8_y4p5_35p5_v1p5_r1p15` control-sweep survivor with
`score_collision_after_goal` enabled.

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

## What Current `UAVSwarmBattle` Is Missing

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

### P0: replace weak GIF-only evidence

The README GIF has been regenerated from the stronger post-goal
dynbranch survivor. It now has an n=10 moving-arm check, and the hero
seed clears the control-first thresholds: no-sweeper virtual penetration
`-0.61 m`, moving clearance `+0.47 m`, and path delta `5.55 m`.

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

Verdict for the previous `p19p8_y5p0_35p0` README race candidate:

```text
weak_dynamic_avoidance_control
```

Reasons:

- moving-arm clearance is only `+0.10 m`, below the `+0.25 m` target,
- no-obstacle virtual penetration is only `-0.0007 m`, not the `<= -0.5 m`
  target,
- moving-vs-no-obstacle path delta is `0.81 m`, below the `1.0 m`
  target.

Interpretation: the `no_prediction` arm was useful evidence that dynamic
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
Control-first outputs are tracked:

- `docs/data/race_hero_control_sweep_lowtemp.json`: eight
  low-temperature candidates with no-obstacle virtual clearance from
  `-1.02 m` to `-0.50 m`; all moving arms collided, so there were no
  survivors.
- `docs/data/race_hero_control_sweep_argmin_safety.json`: two
  stronger-controller candidates using argmin fallback, safety margin
  `0.8`, and obstacle weight `500`; the best local encounter reached
  `+0.30 m` focus-obstacle clearance and `1.58 m` path delta, but the
  episode still ended in collision, so it is not a valid hero.
- `docs/data/race_hero_control_sweep_dynbranch.json`: first
  topology-lite branch-sampling attempt. GPU MPPI now injects stop,
  slow, and lateral branch actions around near dynamic obstacles and
  logs `dynamic_branch_samples` per replan. The best tested candidate
  improved from `0/4` drone success to `2/4`, but joint outcome still
  collided, so it remains a planner-development result rather than a
  README hero.
- `docs/data/race_hero_control_sweep_postgoal_dynbranch.json`:
  same candidate with `score_collision_after_goal` enabled. This fixes
  the race-lookahead scoring hole where reaching the short dynamic goal
  masked a later collision inside the MPPI horizon. The run is joint
  success (`4/4` drones), with no-sweeper virtual clearance `-0.61 m`,
  moving clearance `+0.47 m`, and path delta `5.55 m`; it passes the
  single-seed control-first hero threshold and is now the README GIF.
- `docs/data/race_hero_control_sweep_postgoal_dynbranch_n10.json`:
  same moving-obstacle arm at n=10. It stays at `10/10` joint success
  (`40/40` drones), with no env / peer collisions.
- `docs/data/race_hero_control_variants_postgoal_dynbranch_n10.json`:
  n=10 controls from the same planner configuration. `frozen_initial`
  succeeds `10/10`; `frozen_encounter` fails `0/10` joint with `20/40`
  drone success and min dynamic clearance `-0.80 m`; `wrong_velocity`
  and `no_prediction` both succeed `10/10`. Interpretation: this cell
  is not strong evidence for velocity-prediction dependence. The robust
  positive is timing/post-goal collision scoring plus branch candidates
  avoiding a moving hazard that a frozen-at-encounter obstacle blocks.
- `docs/data/race_hero_control_sweep_postgoal_only_n10.json` and
  `docs/data/race_hero_control_sweep_dynbranch_n10.json`: direct
  scoring-vs-branch ablation. Post-goal scoring without dynamic branch
  sampling succeeds `10/10` (`40/40` drones, hero-seed clearance
  `+0.59 m`, path delta `6.35 m`). Dynamic branch sampling without
  post-goal scoring fails `0/10` (`20/40` drones, env collision `10`,
  peer collision `10`). This makes post-goal collision scoring the
  active fix for this cell; branch samples are not required here.
- `docs/data/race_hero_postgoal_generalization_n3.json`: fixed
  six-cell post-goal-only follow-up at n=3 per cell. Every moving arm
  succeeds (`18/18` joint, `72/72` drones, no env / peer collisions).
  Three cells pass the full control-first threshold (`-1.17`, `-1.03`,
  and `-0.61 m` no-obstacle virtual clearance with positive moving
  clearance and >1 m path delta); the other three still finish but are
  rejected as causal evidence because the ghost conflict is too shallow
  or absent.
- `docs/data/race_hero_postgoal_adversarial_screen.json` and
  `docs/data/race_hero_postgoal_adversarial_n1_top4.json`: broad
  radius/period screen looking for post-goal-only failures. The selected
  `r=1.75` cells have no-obstacle ghost clearance around `-1.77 m`, but
  the first four moving arms still succeed (`4/4` joint).
- `docs/data/race_hero_postgoal_extreme_radius_screen.json` and
  `docs/data/race_hero_postgoal_extreme_radius_n1_top2.json`: stricter
  `r=2.5` screen. The ghost clearance reaches `-2.52 m`, but the first
  two moving arms still succeed (`2/2` joint).
- `docs/data/race_hero_paired_sweeper_postgoal_allobs_n1_top4.json` and
  `docs/data/race_hero_paired_sweeper_postgoal_dynbranch_allobs_n1_top4.json`:
  all-obstacle paired-sweeper re-score using
  `scripts/race_hero_control_sweep.py --focus-obstacle -1`. The closest
  clearance is taken over both moving sweepers. Post-goal-only succeeds
  `4/4` with minimum all-obstacle clearance `+0.39 m`; post-goal plus
  branch also succeeds `4/4` with minimum all-obstacle clearance
  `+0.41 m`. The paired-sweeper GIF is
  `docs/images/race_hero_paired_sweeper_allobs_postgoal.gif`, rendered
  with both obstacle halos.
- `docs/data/race_hero_offset_gate_screen.json` and
  `docs/data/race_hero_offset_gate_postgoal_valid_allobs_n1.json`:
  offset-gate probe requiring the no-obstacle ghost to enter both
  sweeper safety halos (`--min-conflicting-obstacles 2`). Valid starts
  `y_high={11,13,14}` all succeed with post-goal-only control; the
  tightest moving all-obstacle clearance is `+0.37 m` and the largest
  path delta is `6.68 m`. The attempted `y_high=10` placement was
  rejected because it collides at `t=0`, not in the dynamic encounter.
  The offset-gate GIF is `docs/images/race_hero_offset_gate_allobs_postgoal.gif`.
- `docs/data/race_hero_third_blocker_postgoal_allobs_n1.json`,
  `docs/data/race_hero_third_blocker_r2_postgoal_allobs_n1.json`,
  `docs/data/race_hero_third_blocker_r3_postgoal_allobs_n1.json`, and
  `docs/data/race_hero_third_blocker_r3_postgoal_dynbranch_allobs_n1.json`:
  third-blocker probe using the new `--extra-obstacle` option. The
  blocker crosses the lower/east escape side used by the offset-gate
  solution. Even at radius `3.0 m`, post-goal-only succeeds with
  all-obstacle clearance `+0.49 m`; post-goal plus branch also succeeds
  with `+1.00 m`. The no-obstacle ghost enters all three safety halos.
  The GIF is `docs/images/race_hero_third_blocker_allobs_postgoal.gif`.
- `docs/data/race_hero_third_blocker_r3_postgoal_progress_wrt1000_wclean100_allobs_n10.json`:
  progress-weighted follow-up on the same r=3.0 third-blocker cell.
  GPU MPPI now has zero-default clean-reach tie-breaks:
  `w_reach_time` penalizes late local-goal arrival and `w_clean_ctg`
  penalizes clean rollouts that drift away after reaching the short
  race goal. The best tested arm (`w_reach_time=1000`,
  `w_clean_ctg=100`) keeps `10/10` joint success (`40/40` drones),
  preserves all-obstacle clearance (`+0.48 m` in the rendered seed),
  and reduces max path delta from `8.20 m` to `6.19 m`. The previous progress hero was
  `docs/images/race_hero_third_blocker_progress_allobs.gif`.
- `docs/data/race_hero_dynamic_gate_postgoal_progress_allobs_n10.json`:
  dynamic-gate follow-up for a clearer visual. Two extra moving blockers
  close around the ghost line near `x=24.5`, so the no-obstacle ghost
  enters four safety halos (`-1.77 / -1.32 / -0.63 / -1.00 m`). The
  progress-weighted run keeps `10/10` joint success (`40/40` drones),
  clears the closest halo by `+0.77 m` in the rendered seed, and has a
  `6.28 m` max path delta. The current README hero is
  `docs/images/race_hero_dynamic_gate_progress_allobs.gif`.
- `docs/data/race_hero_dynamic_gate_width_speed_n1_top4.json`,
  `docs/data/race_hero_dynamic_gate_width_speed_harder_n1_top4.json`,
  and `docs/data/race_hero_dynamic_gate_width_speed_gap0p8_vy0p64_n3.json`:
  width/speed limit sweep using
  `scripts/race_hero_dynamic_gate_sweep.py`. The first top-4
  (`gap=1.6/2.0`, `|v_y|=0.32/0.48`) all succeed at n=1. The harder
  top-4 (`gap=0.8/1.2`, `|v_y|=0.48/0.64`) also all succeed at n=1.
  The hardest tested cell, `gap0p8_vy0p64_t28p5`, remains `3/3` joint
  success (`12/12` drones), with ghost clearances
  `-1.77 / -1.32 / -1.36 / -1.54 m`, moving clearance `+0.42 m`, and
  max path delta `4.63 m`. Width/speed alone still does not expose a
  failure boundary.
- `docs/data/race_hero_dynamic_gate_two_stage_x27_center28_gap1p0_n1.json`,
  `docs/data/race_hero_dynamic_gate_two_stage_x27_center25p5_gap1p0_n1.json`,
  and `docs/data/race_hero_dynamic_gate_two_stage_x29_center29p7_t28_gap1p0_n1.json`:
  hand-placed second-row gate probes. All three remain `1/1` joint
  success, with moving clearances `+0.54 m`, `+0.47 m`, and `+0.51 m`.
  Manual second-row placement is still not enough; the next test should
  sweep second-row `(x, center_y, phase)` or add a slot/wall constraint.
- `docs/data/race_hero_dynamic_gate_second_row_grid_screen.json` and
  `docs/data/race_hero_dynamic_gate_second_row_grid_n1_top4.json`:
  second-row grid support in `scripts/race_hero_dynamic_gate_sweep.py`.
  On the hardest single gate (`gap0p8_vy0p64_t28p5`), a small grid over
  `x={27,29}`, `center_y={25.5,28.0,29.7}`, and `t={28.0,28.5}` still
  leaves all top-4 cells at `1/1` joint success. The closest moving
  clearance among the top-4 is `+0.45 m`.
- `docs/data/race_hero_dynamic_gate_slot_wall_x24_y27p5_n3.json`:
  structural slot/wall boundary. A short static box at center
  `(24.0, 27.5, 7)`, size `(5, 2, 14)` targets the lower escape route
  used by drone 3 after the hardest single dynamic gate. The dynamic
  blockers are still cleared (`+0.35 m` moving clearance), but the run
  fails `0/3` joint (`9/12` drones) by repeatable drone-3 environment
  collision at `t=29.80 s`. This is the first useful failure boundary:
  moving gates alone did not break the controller, but constraining the
  escape topology does.
- `docs/data/race_hero_base_pair_slot_wall_x24_y27p5_n3.json`: control
  split for the trimmed slot wall. The same wall without the extra
  dynamic gate succeeds `3/3` joint (`12/12` drones), with `+0.37 m`
  moving clearance and `6.17 m` path delta. Together with the hardest
  dynamic-gate-only n=3 success, this pins the failure on the
  gate+wall composition rather than on the static wall by itself.
- `scripts/race_hero_slot_wall_sweep.py` and
  `docs/data/race_hero_slot_wall_y_sweep_n1.json`: systematic split for
  slot-wall variants. At `x=24.0`, size `(5,2,14)`, wall centers
  `y=26.5` and `y=27.5` are `gate_wall_boundary` cases: base wall
  succeeds `1/1`, but gate+wall fails `0/1`. At `y=28.5`, the base wall
  already fails `0/1`, so that position is classified as
  `wall_too_blunt` and should not be used as evidence for dynamic-gate
  pressure.
- `docs/data/race_hero_slot_wall_x_sweep_n1.json`: x-axis follow-up at
  `y=27.5`, size `(5,2,14)`. `x=23.0` and `x=24.0` are
  `gate_wall_boundary` cases, while `x=25.0` is `wall_too_blunt`. The
  useful composition-boundary patch is therefore roughly `x=23-24`,
  `y=26.5-27.5` for this wall size.
- `docs/data/race_hero_slot_wall_sizex_sweep_n1.json`: size-x follow-up
  at `x=24.0`, `y=27.5`, `size_y=2`. `size_x=5.0` and `6.0` are
  `gate_wall_boundary` cases. `size_x=4.0` is now labeled
  `base_wall_failure`: the base wall fails `0/1` while gate+wall
  succeeds `1/1`, so it is not valid evidence that the dynamic gate made
  the scene harder.
- `docs/data/race_hero_slot_wall_x23_y27p5_sx5_n3.json` and
  `docs/data/race_hero_slot_wall_x24_y26p5_sx5_n3.json`: n=3 edge
  validation for the useful patch. Both keep `base_wall=3/3` joint
  success and `gate_wall=0/3` joint success (`9/12` drones), with
  positive moving clearances of `+1.46 m` and `+1.14 m`. This confirms
  the composition boundary is not isolated to the center
  `x=24.0,y=27.5` wall.
- `scripts/race_hero_slot_wall_failure_report.py` and
  `docs/data/race_hero_slot_wall_failure_mechanism.json`: mechanism
  report for the center cell plus the two n=3 edge cells. In all three,
  gate-wall collisions are `3/3`, the first 1 m path split occurs around
  `25.7-26.1 s`, and collision happens at `28.95-29.95 s`. The
  gate-wall arm has negative projected wall clearance at collision
  (`-0.15/-0.00/-0.02 m`), while the base-wall arm is still clear of the
  wall at the same time (`+0.91/+2.69/+1.04 m`). Extra dynamic-gate
  clearance stays positive (`+1.46/+1.14/+0.35 m`).
- `scripts/race_hero_slot_wall_rollout_horizon_report.py` and
  `docs/data/race_hero_slot_wall_rollout_horizon_report.json`: rollout
  horizon audit for the same three cells. The collision time enters the
  logged 2 s horizon at `27.10-28.10 s`, and visible wall-hit rollouts
  exist earlier (`25.10-26.10 s`). However, the logged best-visible
  rollout is clear at the final replan before impact in all three cases
  (`+0.18/+0.24/+0.90 m`), while other visible rollouts are deeply
  wall-hitting (`-1.66/-1.85/-1.88 m`). So the current bug is not merely
  horizon length; it is a closed-loop rollout/scoring mismatch near the
  constrained wall topology.
- The rollout-horizon report also reconstructs the command-limited
  post-step execution segment. At the actual wall clip time, the
  executed trajectory has wall clearance `-0.15/-0.00/-0.02 m`, while
  the logged best-visible rollout at the same time is still clear
  (`+0.44/+0.32/+1.30 m`). This points to a dynamics-model mismatch
  between constant-velocity MPPI rollouts and acceleration-limited
  simulator execution, amplified by the narrow wall topology.
- Implementation follow-up: GPU MPPI now has opt-in `rollout_max_accel`
  and the runner exposes current plant velocity through `set_current_state`.
  The stronger immediate fix for the slot-wall cell was static map
  inflation, not acceleration rollout alone. Passing `--inflate 1` through
  the sweep makes all three earlier boundary cells solve base-wall `3/3`
  and gate-wall `3/3`: x23/y27.5 gate clearance `+1.38 m`, x24/y26.5
  `+0.64 m`, and x24/y27.5 `+0.46 m`
  (`docs/data/race_hero_slot_wall_x23_y27p5_sx5_inflate1_n3.json`,
  `docs/data/race_hero_slot_wall_x24_y26p5_sx5_inflate1_n3.json`,
  `docs/data/race_hero_slot_wall_x24_y27p5_sx5_inflate1_n3.json`).
  The local root cause is therefore a swept-radius mismatch: planner
  occupancy used `inflate=0`, while the simulator checks collisions with a
  `0.4 m` drone radius.

Conclusion: simple phase/radius search can produce a strong no-obstacle
counterfactual, but the local MPPI controller needed two changes to
complete it reliably in the first implementation: branch rollout seeds,
and collision scoring beyond short race lookahead goals. The ablation
now narrows the claim further: this cell is a post-goal-scoring result,
not a branch-seeding or predictor-velocity result. The six-cell
follow-up shows that the scoring fix is not only a single hand-picked
cell. The next step is a broader adversarial sweep to find cells where
post-goal scoring alone fails and branch/corridor/topology sampling
becomes necessary. The first adversarial probes show that deeper ghost
penetration alone is not enough; the next design should narrow or
topologically split the available escape route. The all-obstacle
paired-sweeper check also shows that the existing mirrored pair is still
too easy for post-goal-only. The offset-gate and third-blocker probes
are stronger because multiple hazards conflict with the ghost, but they
are also solved by post-goal-only. The first static corridor wall probe
was too blunt for the periodic oval, so the current useful direction is
progress-weighted control plus slot/wall follow-ups. The single-gate
width/speed sweep did not break the controller down to `gap=0.8 m`,
`|v_y|=0.64 m/s`, and the first three hand-placed second rows plus a
small second-row grid also stayed collision-free. A trimmed static slot
wall finally gives a repeatable `0/3` joint boundary, and the wall-only
control split still succeeds `3/3`, so the boundary is the composition
of dynamic gate pressure plus constrained escape topology. The first
slot-wall y sweep confirms that `y=26.5-27.5` are useful composition
failures, while `y=28.5` is too blunt because it breaks the base scene.
The x sweep adds the same split along x: `x=23-24` are useful, `x=25`
is too blunt. The size-x sweep shows the useful band starts at
`size_x=5`; `size_x=4` perturbs the base scene instead of producing a
clean dynamic-gate boundary. Two edge cells from the useful patch now
also pass the n=3 split, so the slot-wall result is no longer just a
single center-cell observation. The mechanism report shows the same
pattern across those cells: dynamic gate pressure causes early route
divergence, then the static wall blocks the selected line. The rollout
horizon audit further narrows the next fix: extend-only horizon tuning
is unlikely to be sufficient because wall-hit candidates are already
inside the visible rollout set. The immediate fix target is static
constraint modeling first, rollout/execution model mismatch second, not
another GIF pass.

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
