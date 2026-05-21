# Handoff plan — uav-nav-lab

**Snapshot 2026-05-21.** This document hands off the state of the
repo to whoever picks up next (Codex / Claude-Code / a human). It
covers: where the project stands, what's working, what's broken,
and a prioritised task list with enough detail to start from cold.

---

## 0. The 60-second summary

- **`uav-nav-lab`** is a YAML-driven research framework for UAV
  motion planning. Same-seed paired ablations across
  sim / scenario / planner / sensor / predictor backends, Wilson
  95 % CIs by default, every example YAML carries its validated
  finding in the header.
- The framework is solid; the **§3 4-mode framework** (multi-drone
  GPU MPPI vs CPU MPC) was the main research thrust over the past
  few months. Modes 1 (clustering), 3 (sim-physics flip), 4
  (aerobatic precision) are validated. Mode 2 (dynamic-obstacle
  cancellation) and its "mirror image" (gate-thread) are **invalidated
  pending a scenario re-tune** — see Section 2.
- A critical multi-runner bug was found on 2026-05-21 that froze
  dynamic obstacles in any episode following a total-wipeout.
  The fix is in `1646e11`; all prior dynamic-obstacle multi-drone
  results need to be re-validated.
- Hero GIF currently shows the honest post-fix gates4 (everybody
  crashes) with a "scenario re-tune WIP" caption. Until the
  re-tune lands the GIF stays as the honest failure case rather
  than the prior bug-induced false positive.

---

## 1. What's working (do not touch unless it breaks)

These have been re-validated or were never affected by the
2026-05-21 bug.

| area | status | source of truth |
|---|---|---|
| §3 mode 1 — multi-drone clustering (N=4, n=100) | ✅ valid | `examples/exp_multi_drone_3d_4{,_gpu_mppi}.yaml`, findings.md "Multi-drone: GPU MPPI's rollout cloud flips the coordination Δ" |
| §3 mode 3 — AirSim density-corner sign-reversal (`base_ew06`) | ✅ valid | `examples/exp_airsim_multi_discriminating_central_n30*.yaml`, §4.4.4 |
| §3 mode 4 — aerobatic synchronized loop | ✅ valid | `examples/exp_aerobatic_loop4_*.yaml`, findings.md "Aerobatic synchronized loop" |
| Pareto cells (MPC 2D + 3D, GPU MPPI 2D + 3D) | ✅ valid | `examples/exp_predictive.yaml`, `exp_3d_predictive.yaml`, `exp_gpu_mppi_pareto*.yaml` |
| Planner head-to-head on dynamic obstacles (single-drone) | ✅ valid | single-drone, not multi.py path |
| AirSim transferability | ✅ valid | not affected (sim provides its own collision flag) |
| ROS 2 bridge | ✅ valid | unrelated to multi.py |
| Smart MPPI v1–v5 *on single-drone scenarios* | ✅ valid | `examples/exp_multi_drone_3d_4_dyn_v2*.yaml` etc. — N drones survive ep 0 with the master alive, so the bug never fires there |
| CLI / runner / sweep / viz / anim | ✅ valid | covered by CI |

---

## 2. What's broken or invalidated (must re-do)

### 2.1 The 2026-05-21 dynamic-obstacle bug

**Symptom:** every episode after a total-wipeout episode ran with
all dynamic obstacles frozen at their initial positions. Drones
appeared to "succeed" because the obstacles never moved into their
path.

**Root cause:** in `uav_nav_lab/runner/multi.py`, only sim 0
has `_advance_scenario=True`. When sim 0 (the master) died
mid-episode the runner handed mastership to the next unfinished
drone — but if all drones died in that episode, no hand-off
happened. The flag stayed `False` on every sim. Between episodes
the sim objects are reused, so `_advance_scenario` carried over.

**Fix:** at the top of `run_episode_multi`, restore
`sims[0]._advance_scenario = True` (and `False` for `sims[1:]`)
before calling `sim.reset()`. Committed as `1646e11`.

**Invalidated:**

- `examples/exp_race_oval4_*.yaml` (single bouncing intruder)
- `examples/exp_race_gates4_*.yaml` (4 sliding gates)
- `examples/exp_race_chaos_*.yaml` (gates + intruders)
- `examples/exp_race_dyn4_*.yaml` (4 path-intersecting intruders)
- `examples/exp_race_simple_mpc.yaml` (2 slow intruders, never went paper-grade)

All numbers reported in `docs/findings.md` and
`docs/paper_a/section_3_headline.md` for the "Drone race ...",
"Moving-gates race", "Drone race chaos", and "dyn4
path-intersecting intruders" sections were measured on frozen
obstacles and must be retired or re-validated.

### 2.2 Post-fix re-runs show 100 % collision

When the four affected scenarios are re-run with the fix:

| scenario | MPC | vanilla GPU MPPI | Smart v4 | Smart v5 |
|---|---|---|---|---|
| `race_gates4` | 120/120 | 120/120 | 120/120 | 120/120 |
| `race_dyn4` | 120/120 | 120/120 | (not re-run) | (not re-run) |
| `race_simple` (probe) | 12/12 (n=3) | not run | — | — |

The planner stack — MPC argmin / GPU MPPI softmax with 0.4 s
lookahead, drones tracking a fixed oval reference — cannot solve
the dynamic-obstacle scenarios as designed. Closing rates of
8–10 m/s combined with the oval-tracking constraint leave too
little spatial margin to detour. The "MPC 51.7 % vs softmax 3.3 %"
contrast that powered the §3 mode-2-mirror finding was an artifact
of frozen obstacles, not a real planner-level mechanism.

### 2.3 Knock-on doc invalidations

- `docs/findings.md` — sections "Drone race + bouncing intruder",
  "Moving-gates race", "Drone race chaos", "dyn4 path-intersecting
  intruders" carry invalidated numbers. Toc entries still link to
  these sections.
- `docs/paper_a/section_3_headline.md` — "Mode superposition in a
  single scenario: drone race + bouncing intruder", "Mirror-image
  of the cancellation regime: moving-gates race", "Mode
  superposition under topology dominance: chaos race", "Controlled
  avoidance harness: dyn4 path-intersecting intruders" sections
  carry invalidated numbers.
- `docs/paper_a/section_7_repro_map.md` — last 4-5 rows reference
  the affected YAMLs.

### 2.4 The hero GIF

`docs/images/compare_race_gates4.gif` was re-rendered against the
post-fix data on 2026-05-21. It shows every drone dying within
~1.5 s on both panes — honest but unflattering. Caption in the
README points at this PLAN.md as the explanation.

---

## 3. Prioritised task list

### P0 — restore a real winnable dynamic-obstacle scenario

**Goal:** the framework's headline claim ("4-mode operator analysis
of GPU MPPI vs CPU MPC") needs a working dynamic-obstacle cell, or
the §3 4-mode story collapses to 3 modes (1, 3, 4) and the framework
loses its main story arc.

Concrete steps:

1. **Diagnose why current scenarios are unwinnable.** The planner
   has 0.4 s lookahead and 0.2 s replan period. At 6.3 m/s drone
   tangential + ≥1.5 m/s obstacle, the relative closing rate per
   replan cycle is ≥1.5 m, vs a 1.4 m clearance sum. That's tight
   but should be solvable; the failure mode is probably that the
   oval-tracking goal pull (`dynamic_goal_at`) over-weights the
   obstacle avoidance term. Try
   `planner.w_goal: 0.3`, `planner.w_obs: 200` and see if the
   collision rate drops.
2. **Tune scenario geometry.** Three knobs to sweep:
   - Oval size: `scenario.radius` 12→16, `radius_y` 8→12. More
     swerve room.
   - Obstacle speed: `|velocity|` 1.5→0.8 m/s. Slower obstacles
     give the planner more lookahead margin.
   - Obstacle radius: 1.0→0.6 m. Lower clearance requirement.
3. **Run a 3×3 sweep** of `(oval_radius × obstacle_v × obstacle_r)`
   at n=5 first; pick the cell where MPC lands at 30–60 %
   collision (so there's headroom for softmax to differentiate).
   Then promote to n=30.
4. **Re-render the hero GIF** with the new scenario and ref-ghost
   overlay. The visual contract is: dashed reference oval +
   colored trail bending off it + open-ring marker showing where
   the drone *would* be + red blocks for obstacles + dead-drone
   ghost markers when applicable.

Files: `examples/exp_race_simple_mpc.yaml` is the starting point.
Sweep through `uav-nav sweep`.

**2026-05-21 pilot update:** first non-floor post-fix cell found and
committed as `examples/exp_race_simple_retuned_n5_{mpc,gpu_mppi}.yaml`.
Retune: oval `radius=16`, `radius_y=12`, `period=20`, `max_steps=800`,
planner weights `w_goal=0.3`, `w_obs=200`, two slow obstacles unchanged
at radius 1.0 and |v|=1.5 m/s. Full-duration n=5 (seeds 42-46):
MPC n=8,h=40 = 15/20 per-drone, 0/5 joint; GPU MPPI n=64,h=40 =
20/20 per-drone, 5/5 joint. This escapes the all-planner floor but is
GPU-ceiling / deterministic-MPC-drone-1-failure. Boundary probes show
`period=19.8` and `19.9` give both planners 3/3 joint success, while
`period=19.5` is floor-ish for both; the `period=20` MPC loss is a
narrow phase/geometry failure (drone 1 collision at t=29.6 s, no
named dynamic-obstacle collision object), not a smooth hardness
boundary. Next step: do not jump straight to a headline claim; either
run n=30 as a regression/pilot or tune obstacle phase/radius/speed to
create a less knife-edge partial-success band.

### P1 — rewrite the invalidated docs

Once P0 lands with a winnable scenario:

- Replace the four invalidated sections in `docs/findings.md` with
  the new numbers from the re-tuned scenario.
- Same for `docs/paper_a/section_3_headline.md` (in particular
  the "mirror-image" claim has to be either re-validated or
  retired).
- Update `docs/paper_a/section_7_repro_map.md` rows.
- Update README "Dynamic-obstacle hero GIFs (under repair)"
  details block.

### P2 — paper writeup (§1 / §2 / abstract)

The §3 4-mode framework with modes 1, 3, 4 is paper-ready *now* if
mode 2 is dropped or footnoted. Worth a draft to arXiv:

- `docs/paper_a/section_1_intro.md` (does not exist yet)
- `docs/paper_a/section_2_related.md` (does not exist yet)
- `docs/paper_a/abstract.md` (does not exist yet)

### P3 — Smart MPPI v6 (auto-topology aggregator)

Earlier roadmap item — defer until P0/P1 are done. The motivation
was to detect bimodality vs unimodality per-replan and switch
between softmax / argmin / cluster-softmax. Predicated on mode 2
being a real mechanism, which is now in question; revisit after
the re-tune.

### P4 — PX4 SITL bridge

Roadmap; nice-to-have for hardware transfer story but doesn't
block paper.

---

## 4. Where things live (cold-start map)

```
uav_nav_lab/
├── sim/        dummy_3d (point mass), airsim_bridge, ros2_bridge
├── scenario/   voxel_world, multi_drone_voxel, multi_drone_aerobatic
├── planner/    mpc.py, gpu_mppi.py, chomp.py, rrt.py, rrt_star.py
├── sensor/     perfect, delayed, kalman_delayed, lidar, pointcloud_occupancy
├── predictor/  constant_velocity, noisy_velocity, kalman_velocity
├── runner/     experiment.py (single drone), multi.py (multi-drone) ← bug was here
├── recorder.py
└── cli.py      single-file CLI: run / eval / compare / sweep / viz / anim

examples/      every YAML is a self-contained finding; copy + edit
scripts/       render_race_gif.py, paired_analysis_*.py, sweep helpers
docs/
├── findings.md             long-form per-finding writeups
├── images/                 GIFs and pareto plots
└── paper_a/
    ├── section_3_headline.md   the 4-mode framework
    ├── section_7_repro_map.md  YAML ↔ finding crossref
    └── (sections 1, 2, abstract — to write)
```

Key extension points:

- **New planner:** drop a file in `uav_nav_lab/planner/`, decorate
  with `@PLANNER_REGISTRY.register("name")`, add a
  `from_config(cfg)` classmethod. CLI picks it up via
  `planner.type: name` in YAML.
- **New scenario:** same pattern with `SCENARIO_REGISTRY`.
- **New ablation cell:** copy an existing YAML, edit, run
  `uav-nav run <yaml>`. Results go to `output.dir`.
- **Sweep:** `uav-nav sweep <yaml> --param k=spec` for Cartesian
  product. Specs: `a,b,c`, `start:stop:step`, `[3,0]`, `true`,
  `false`.

---

## 5. Reproduction recipes

### 5.1 The bug repro (smallest possible)

```bash
uav-nav run examples/exp_race_gates4_mpc.yaml --num-episodes 2
# Without 1646e11: ep 0 = all collisions, ep 1 = all successes (frozen gates)
# With    1646e11: ep 0 = all collisions, ep 1 = all collisions    (gates move)
```

The smoking gun is comparing the dynamic-obstacle y-position over
time in `episode_001_*_joint.json` against the analytical motion
from the YAML.

### 5.2 The §3 paper-grade cells that ARE still valid

```bash
# mode 1 — Δ-flip
uav-nav run examples/exp_multi_drone_3d_4.yaml
uav-nav run examples/exp_multi_drone_3d_4_gpu_mppi.yaml
python3 scripts/paired_analysis_dummy_3d_multi.py \
  results/multi_drone_3d_4 results/multi_drone_3d_4_gpu_mppi

# mode 4 — aerobatic loop
uav-nav run examples/exp_aerobatic_loop4_mpc.yaml
uav-nav run examples/exp_aerobatic_loop4_gpu_mppi.yaml
python3 scripts/paired_analysis_aerobatic.py \
  results/aerobatic_loop4_mpc results/aerobatic_loop4_gpu_mppi 4 20
```

### 5.3 The hero GIF render

```bash
python3 scripts/render_race_gif.py \
  --runs results/race_gates4_mpc:"MPC (argmin)" \
         results/race_gates4_gpu_mppi:"GPU MPPI (softmax)" \
  --config examples/exp_race_gates4_mpc.yaml \
  --title "<scenario title>" \
  --out docs/images/compare_race_gates4.gif \
  --ep 1 --fps 10 --stride 8 --trail 30
```

The render script (`scripts/render_race_gif.py`) supports
2- to 4-pane comparison, ref-ghost overlay, T_pad to right-pad
short trajectories, square obstacle markers sized to physical
radius, and `--config` to pull dynamic obstacles from YAML.

---

## 6. Open questions for whoever picks up next

1. **Is the §3 mode 2 mechanism real at all?** With the bug
   fixed, every scenario where it allegedly fired now loses every
   drone for every planner. Three possibilities: (a) the mechanism
   was always a frozen-obstacle artifact; (b) the mechanism is
   real but only at a narrower regime than the current YAMLs;
   (c) the mechanism is real but the planner stack lacks the
   compute budget to express it. P0 should disambiguate.
2. **Should the dyn4 / chaos scenarios be retired?** They were
   designed assuming the bug-induced "easy" scenario. If the
   re-tune still can't recover a winnable cell, retiring them
   honestly is better than keeping placeholder GIFs.
3. **arXiv timing.** Modes 1, 3, 4 are paper-ready. Mode 2 is
   not. Decide whether to ship the 3-mode paper now and follow
   up with mode 2 as a separate note, or wait for the full set.

---

## 7. Anti-recommendations (don't waste cycles on these)

- **Don't rerun all the AirSim n=30 paired cells.** The bug
  affects `runner/multi.py`; AirSim's collision flag comes from
  the bridge itself, not from `scenario.is_collision`, and the
  master-flag path is the same. Spot-check one cell instead.
- **Don't try to make gates4 winnable by raising `n_samples`.**
  We already swept up to 256 samples; the bottleneck is the
  lookahead / replan period vs closing rate, not sample diversity.
- **Don't delete the invalidated YAMLs.** They're useful as
  failure-mode regression tests and the commit history references
  them; just gate them behind a comment header noting they're
  awaiting re-tune.

---

*Last edit: 2026-05-21. Commit `1646e11` is the critical bug fix.*
