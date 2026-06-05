<div align="center">

# uav-nav-lab

**A Python lab for UAV motion planning that proves — or disproves — what actually works.**
Swap planners, sensors and swarm rules in YAML; settle every claim with seed-paired McNemar tests and Wilson 95 % CIs.

[![CI](https://github.com/rsasaki0109/uav-nav-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/rsasaki0109/uav-nav-lab/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/rsasaki0109/uav-nav-lab/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/rsasaki0109/uav-nav-lab)](https://github.com/rsasaki0109/uav-nav-lab/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Stars](https://img.shields.io/github/stars/rsasaki0109/uav-nav-lab?style=social)](https://github.com/rsasaki0109/uav-nav-lab/stargazers)

<img src="docs/images/swarm_convention_compare.gif" alt="Twelve drones swap across one hub: stock ORCA piles into the centre and collides; with a decentralised right-of-way the same fleet spirals into a clean roundabout" width="840">

<i><b>One rule turns a pile-up into a roundabout.</b> Twelve drones swap across a single hub. <b>Stock ORCA</b> (left) drives every path into the centre and collides; add a decentralised <b>right-of-way</b> (right) — each drone passes neighbours on a consistent side — and the same fleet spirals into a clean roundabout, collision-free. One of ~40 seed-paired findings — <a href="docs/findings.md">see them all</a>.</i>

</div>

## Why this exists

Most planning repos *ship* a method. This one *interrogates* it. Every headline is a paired, seed-controlled experiment with an exact p-value — and several overturn the textbook intuition:

- **The "optimal" planner is the dangerous one.** `rrt_star`'s shortest-path rewiring collides *more* than plain `rrt` in dynamic avoidance (21.7 % vs 76.7 %, p<1e-3, ~30× the compute) — the shortest path hugs minimum clearance.
- **The classical-planner ladder is a clearance ladder.** straight < astar < rrt\_star < rrt < mpc tracks path *directness*, not cleverness — the two "optimal" planners are the straightest and collide most.
- **Smarter prediction backfires under symmetry.** A goal-aware predictor wins head-on (+26 pp) but *inverts* on the antipodal swap (down to 1/40) — a correct shared symmetric forecast makes every drone mirror-swerve into the same hub.
- **The fix is a convention, not a better forecast.** A decentralised right-of-way (everyone veers the same way) turns the deadlock into a roundabout and reaches 100 %; once it is on, the predictor is *free* — smart and dumb forecasts tie.
- **ORCA's edge over RVO is structure, not continuity.** Refining RVO's sampling never smooths it; HRVO's side-commitment recovers 4.1× of the gain *and all the safety* while staying sampled — ORCA's LP only polishes the residual.
- **Risk-aversion's win is just ensembling.** CVaR-MPPI's collision drop is captured entirely by averaging sampled futures; the worst-case tail adds nothing significant.

Full write-ups — methods, tables, p-values — in **[`docs/findings.md`](docs/findings.md)** (≈40 studies). Working paper draft: [`docs/paper_a/`](docs/paper_a/).

## Gallery

<div align="center">
<table>
<tr>
<td align="center"><img src="docs/images/swarm_airsim_topdown.gif" width="330"><br><sub><b>Photorealistic AirSim</b> — 4 real quadrotors cross one hub in Unreal Engine, top-down chase-cam.</sub></td>
<td align="center"><img src="docs/images/swarm_flagship_all.gif" width="330"><br><sub><b>Everything at once</b> — 16 drones, four sweeping bodies, a gusting crosswind.</sub></td>
</tr>
<tr>
<td align="center"><img src="docs/images/swarm_3d_field.gif" width="330"><br><sub><b>3-D asteroid field</b> — 12 drones thread a drifting field of obstacles in full 3-D.</sub></td>
<td align="center"><img src="docs/images/swarm_3d_sphere.gif" width="330"><br><sub><b>3-D sphere swap</b> — 14 drones cross one centre, camera orbiting.</sub></td>
</tr>
</table>
</div>

| | | |
|---|---|---|
| <img src="docs/images/swarm_big_roundabout.gif" width="240"><br><sub>**18-drone roundabout** — an explicit shared ring clears the hub collision-free at any density.</sub> | <img src="docs/images/swarm_doorway.gif" width="240"><br><sub>**Doorway** — two opposing streams funnel through one gap.</sub> | <img src="docs/images/swarm_rollout_cloud.gif" width="240"><br><sub>**Sampling cloud** — each drone's fan of scored candidate velocities.</sub> |
| <img src="docs/images/swarm_obstacle_gauntlet.gif" width="240"><br><sub>**Obstacle gauntlet** — a dozen drones weave through six sweeping bodies.</sub> | <img src="docs/images/swarm_wind.gif" width="240"><br><sub>**Crosswind** — a gusting wind field bows every track.</sub> | <img src="docs/images/swarm_antipodal_orca_vs_hrvo.gif" width="240"><br><sub>**ORCA vs HRVO** at the hub — collide vs roundabout.</sub> |
| <img src="docs/images/swarm_crossing_rvo_vs_orca.gif" width="240"><br><sub>**RVO dance vs ORCA glide** — RVO's tracks kink, ORCA's stay smooth.</sub> | <img src="docs/images/compare_rrt_vs_rrt_star.gif" width="240"><br><sub>**RRT vs RRT\*** — the "optimal" path drives into the obstacle.</sub> | <img src="docs/images/compare_gpu_mppi_vs_mpc_3d.gif" width="240"><br><sub>**GPU-MPPI vs MPC in 3-D**, rollouts visualised.</sub> |

Every 2-D swarm clip above is one command — `scripts/render_swarm_gif.py` (no AirSim needed; `antipodal` / `crossing` / `doorway` scenarios, `--obstacles`, `--wind`, `--rollouts`, `+row` convention flags) — and the 3-D sphere/field is `scripts/render_swarm_3d_gif.py`. The photorealistic clip is real [AirSim](https://github.com/microsoft/AirSim) (Unreal Engine), recorded with `scripts/record_airsim_topdown_live.py` against a running Blocks server.

## Quick start

```bash
git clone https://github.com/rsasaki0109/uav-nav-lab
cd uav-nav-lab
pip install -e '.[dev,viz]'        # numpy + pyyaml + matplotlib + pytest
pytest -q

uav-nav run  examples/exp_basic.yaml
uav-nav eval results/basic_astar          # Wilson 95% CIs
uav-nav viz  results/basic_astar          # trajectory PNG / GIF
```

A 2-D heatmap sweep is one invocation:

```bash
uav-nav sweep examples/exp_predictive.yaml \
  --param planner.max_speed=10,15,20,25,30 \
  --param planner.replan_period=0.1,0.2,0.5,1.0,2.0 \
  --param num_episodes=20 -j 4
uav-nav viz <out>     # → 6-panel heatmap
```

## CLI

| command | what |
|---|---|
| `uav-nav run <yaml>` | run all episodes → per-episode JSONs + `summary.json` |
| `uav-nav eval <run>` | recompute metrics, print Wilson 95 % CIs + planner-dt budget |
| `uav-nav compare <a> <b> …` | side-by-side table with ± half-widths |
| `uav-nav sweep <yaml> --param k=spec` | Cartesian product over `--param`s |
| `uav-nav viz <run_or_sweep>` | trajectory PNG, or 6-panel sweep heatmap |
| `uav-nav anim / video <run>` | 2-D GIF replay / ffmpeg AirSim MP4 |
| `uav-nav list` | enumerate registered planners / sensors / sims / scenarios |

`--param` accepts `start:stop:step`, `a,b,c`, `[3,0]`, `true/false`, and dotted keys
like `planner.predictor.velocity_noise_std=0.0,0.5,1.0`.

## Architecture

Pluggable registry backends — add one by dropping a file with `@REGISTRY.register("name")`
and a `from_config(cfg)` classmethod; the CLI picks it up via `type: name`.

| kind | shipped |
|---|---|
| sim | `dummy_2d`, `dummy_3d`, `airsim`, `ros2` |
| scenario | `grid_world`, `voxel_world`, `multi_drone_{grid,voxel,aerobatic}` |
| planner | `astar`, `straight`, `mpc`, `mppi`, `cvar_mppi`, `gpu_mppi`, `rrt`, `rrt_star`, `chomp`, `mpc_chomp`, `warmup_select_mppi`, `orca`, `rvo`, `vo`, `hrvo`, `bvc`, `cbf`, `apf`, `roundabout` |
| sensor | `perfect`, `delayed`, `kalman_delayed`, `lidar`, `noisy_tracker`, `pointcloud_occupancy`, `depth_image_occupancy` |
| predictor | `constant_velocity`, `noisy_velocity`, `kalman_velocity`, `game_theoretic`, `constant_turn` |

Multi-drone runs step two-phase (plan all, then advance all); dynamic obstacles support
`linear` / `pursue` / `intercept` policies; the `noisy_tracker` sensor is the one that makes a
threat's *current* state uncertain — where a forecast can actually err.

## Status

Active research lab — APIs may shift between releases. The dynamic-obstacle race headlines were
re-grounded after a 2026-05 multi-runner fix; see [`docs/findings.md`](docs/findings.md) and
[`docs/dynamic_obstacle_oss_survey.md`](docs/dynamic_obstacle_oss_survey.md) for the audit trail.

## License

Apache-2.0 — see [LICENSE](LICENSE).
