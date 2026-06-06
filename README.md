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
- **"Team-size-agnostic" carrying is geometric, not learned.** Interrogating [TeamHOI](https://splionar.github.io/TeamHOI/) (CVPR 2026): N drones carry a rigid beam through a doorway. A *fixed* formation collapses to 0/60 for every N≥3 (the beam outgrows the gap); one that *reorients* the beam holds 57–60/60 flat across N=2–8 (p≤1.7e-18). What makes cooperative carrying scale to any team size is active formation reshaping — and it costs runway, not cleverness.
- **…and only where the workspace is convex.** Reorientation makes carrying size-agnostic at a *doorway*, but an *L-corner* obeys the classical ladder-around-a-corner bound `L_max = 2.83·width`: a beam longer than that cannot round the corner in any sequence of moves. So a non-convex passage imposes a hard ceiling `N_max ≈ 2.83·width/spacing` — at corridor width 4 m the corner ties the doorway to N=4 then collapses to 0/60 by N=6, and is restored only by *widening the corridor*, not by a cleverer team.
- **A learned teammate-token policy is only as symmetry-breaking as its teacher.** Distilling a [TeamHOI](https://splionar.github.io/TeamHOI/)-style permutation-invariant deep set (NumPy, behavioral cloning on *random scenes only*) from a **symmetric** avoider reimports — and *amplifies* — the antipodal deadlock (`8/1/0` of 60 at N=4/6/8, worse than its own teacher despite `bc_mse=1e-4`); the **same** architecture distilled from a right-of-way **convention** clears the unseen hub and generalises zero-shot to N=8 (`60/58/41`, p≤9e-13 between them). The teammate-token network neither creates nor cures the deadlock — it transports whatever convention the training signal had, bounding what "any team size" can come from.
- **…and the representation must be able to *represent* that handedness.** The teacher carrying the convention is necessary but *not sufficient*: quotient the left/right mirror out of the policy's frame (a reflection-canonical, chirality-free representation) and the **same convention teacher** distils to `2/4/0` of 60 — the deadlock floor — versus `60/58/41` for the chirality-preserving frame (p≤7e-18), at the same `bc_mse≈1e-4`. A learned convention needs the symmetry-breaker in *both* the training signal **and** the representation; remove either and the antipodal deadlock returns.
- **RL *discovers* the convention from a symmetric reward — but needs the same chirality-capable representation.** Train the same deep set by **REINFORCE** on a reflection-symmetric reward (no built-in handedness): in the chirality-capable frame it clears far above the symmetric-teacher floor (`24.8`/`14.3`/`9.3` of 30 at N=4/6/8 vs `4`/`0`/`0`) and settles on a self-generated side (handedness consistency `0.97`) — *spontaneous* symmetry breaking, no teacher required. So the convention is discoverable, not merely teachable. But the chirality-free frame collapses RL discovery to the floor too (`0.5`/`0` at N≥6): the representation requirement is the **common** necessary condition for both *taught* and *found* conventions. (Taught still beats found — BC-from-convention reaches `30/29/20`.)
- **A roundabout can be negotiated locally — but only where symmetry hands over agreement.** A faithful decentralised [Merry-Go-Round](https://arxiv.org/abs/2503.05848) (triggered on a local deadlock, ring centre = the *centroid of the ego's conflict cluster*, no global hub knowledge) breaks the antipodal CBF deadlock 40/40 at every N=4–20 and *ties the fixed-centre roundabout* (p=1.0) — agents agree on a common ring from sensing alone, because on the symmetric hub every local centroid coincides. The catch first looked like the same fact — on dense *unstructured* traffic the ungated trigger collapsed (N=16: 5/40, worse than stock) — but it was **over-triggering, not disagreement**: a **symmetry gate** that fires only on a genuine shared hub (ego *and* peers all crossing the cluster centroid) restores the off-switch the always-on convention can never have — it keeps the full 40/40 antipodal cure *and* is harmless on unstructured traffic (a tie with stock at every N).
- **ORCA's edge over RVO is structure, not continuity.** Refining RVO's sampling never smooths it; HRVO's side-commitment recovers 4.1× of the gain *and all the safety* while staying sampled — ORCA's LP only polishes the residual.
- **Risk-aversion's win is just ensembling.** CVaR-MPPI's collision drop is captured entirely by averaging sampled futures; the worst-case tail adds nothing significant.

Full write-ups — methods, tables, p-values — in **[`docs/findings.md`](docs/findings.md)** (≈40 studies). Working paper draft: [`docs/paper_a/`](docs/paper_a/).

## Gallery

<div align="center">
<img src="docs/images/swarm_airsim_dashboard.gif" width="840" alt="Foxglove-style live dashboard from one AirSim flight: FPV camera, LiDAR top-down, 4-drone scene, and min-separation telemetry">
<br><sub><b>Live AirSim dashboard</b> — one flight, four synced panels: Drone1's FPV camera, its LiDAR top-down, the 4-drone scene, and min-separation telemetry (the closest approach dips to ~2 m at the hub, above the 0.8 m collision line).</sub>
</div>

<div align="center">
<img src="docs/images/swarm_transport_doorway.gif" width="840" alt="Five drones carry a rigid beam through a doorway: held perpendicular the beam slams the wall; allowed to reorient, the same team rotates it to align with travel and threads the same gap">
<br><sub><b>Cooperative carrying through a doorway</b> — five drones carry a rigid beam, same seed both sides. <b>Fixed</b> orientation (left) slams the wall; <b>reorienting</b> (right) the beam to align with travel threads the same gap. The mechanism behind "team-size-agnostic" carrying (<a href="docs/findings.md#cooperative-carrying-scales-to-any-team-size-only-if-the-formation-can-reorient--testing-teamhois-size-agnostic-claim">TeamHOI probe</a>).</sub>
</div>

<div align="center">
<img src="docs/images/swarm_transport_corner.gif" width="840" alt="Carrying a beam around an L-corner: a 4-drone beam rounds the corridor corner cleanly while a 6-drone beam jams at the critical 45-degree configuration, its ends crossing the walls">
<br><sub><b>…but a corner has a hard ceiling.</b> The same reorientation that clears a doorway cannot beat the <b>ladder-around-a-corner</b> bound: a 4-drone beam (left) rounds the L-junction, a 6-drone beam (right) <b>jams</b> at the critical 45° pose — its length exceeds <code>L_max = 2.83·width</code>, so no reshaping fits it (<a href="docs/findings.md#reorientation-makes-a-straight-doorway-size-agnostic--an-l-corner-imposes-a-hard-ceiling-no-reshaping-beats">corner ceiling</a>).</sub>
</div>

<div align="center">
<img src="docs/images/swarm_bc_convention.gif" width="840" alt="Same teammate-token deep-set policy, two teachers: distilled from a symmetric avoider it piles into the hub and collides; distilled from a convention teacher the same network spirals into a clean roundabout">
<br><sub><b>A learned policy inherits the convention, not the architecture.</b> The <b>same</b> NumPy teammate-token deep set, behavior-cloned (<code>bc_mse=1e-4</code>) on random scenes only — then dropped on the unseen antipodal hub. From a <b>symmetric teacher</b> (left) it reimports the deadlock; from a <b>convention teacher</b> (right) it learns the right-of-way and spirals into a roundabout (a <a href="docs/findings.md#a-teammate-token-policy-is-only-as-symmetry-breaking-as-its-teacher--distilling-the-convention-transfers-the-antipodal-cure-distilling-a-symmetric-avoider-reimports-and-amplifies-the-deadlock">TeamHOI probe</a>).</sub>
</div>

<div align="center">
<img src="docs/images/swarm_airsim_policy.gif" width="840" alt="The learned teammate-token policy flown in photorealistic AirSim Blocks: four quadrotors swap across one hub and spiral into the convention's roundabout, rotors and shadows visible">
<br><sub><b>The lab's first learned policy, flown in photoreal 3-D.</b> The <b>same</b> convention-distilled teammate-token deep set above — now driving four quadrotors in <a href="https://github.com/microsoft/AirSim">AirSim</a> (Unreal Engine). The planar policy's rollout is replayed on the fleet while an external camera orbits the converging swarm: the <b>learned right-of-way roundabout</b>, in photorealistic 3-D (<code>scripts/record_airsim_swarm_policy.py</code>).</sub>
</div>

<div align="center">
<img src="docs/images/swarm_mgr_roundabout.gif" width="840" alt="The antipodal swap under two controllers: plain CBF freezes in a deadlocked clump at the hub; the decentralized triggered Merry-Go-Round forms a roundabout from local sensing and clears">
<br><sub><b>A roundabout negotiated from sensing alone.</b> Eight drones swap across one hub. <b>Plain CBF</b> (left) brakes everyone to a safe stop and <b>deadlocks</b> — a frozen clump. The <b>Merry-Go-Round</b> (right, <a href="https://arxiv.org/abs/2503.05848">Zhou et al. 2025</a>) is the <b>same</b> CBF, but each drone detects the local jam and agrees on a common ring <b>centre from sensing only</b> — no handed symmetry — and the fleet spirals through, matching the fixed-centre ring it was never told (<code>scripts/render_mgr_gif.py</code>; a <a href="docs/findings.md#a-decentralized-merry-go-round-negotiates-its-ring-from-sensing-alone--agents-agree-on-the-symmetric-hub-but-the-same-local-agreement-is-what-fails-on-unstructured-traffic">two-sided result</a>).</sub>
</div>

<div align="center">
<table>
<tr>
<td align="center"><img src="docs/images/swarm_airsim_cinematic.gif" width="225"><br><sub><b>AirSim chase-cam</b> — an external camera trails the lead quadrotor as the fleet crosses.</sub></td>
<td align="center"><img src="docs/images/swarm_airsim_orbit.gif" width="225"><br><sub><b>AirSim orbit</b> — the camera circles the fleet centroid as all four converge.</sub></td>
<td align="center"><img src="docs/images/swarm_airsim_topdown.gif" width="225"><br><sub><b>AirSim top-down</b> — the 4-drone hub crossing from a fixed overhead cam.</sub></td>
<td align="center"><img src="docs/images/swarm_airsim_lidar.gif" width="225"><br><sub><b>AirSim onboard LiDAR</b> — the 16-beam sensor reconstructs the world as a 3-D point cloud.</sub></td>
</tr>
<tr>
<td align="center"><img src="docs/images/swarm_flagship_all.gif" width="300"><br><sub><b>Everything at once</b> — 16 drones, four sweeping bodies, a gusting crosswind.</sub></td>
<td align="center"><img src="docs/images/swarm_3d_field.gif" width="300"><br><sub><b>3-D asteroid field</b> — 12 drones thread a drifting field of obstacles in full 3-D.</sub></td>
<td align="center"><img src="docs/images/swarm_3d_sphere.gif" width="300"><br><sub><b>3-D sphere swap</b> — 14 drones cross one centre, camera orbiting.</sub></td>
</tr>
</table>
</div>

| | | |
|---|---|---|
| <img src="docs/images/swarm_big_roundabout.gif" width="240"><br><sub>**18-drone roundabout** — an explicit shared ring clears the hub collision-free at any density.</sub> | <img src="docs/images/swarm_doorway.gif" width="240"><br><sub>**Doorway** — two opposing streams funnel through one gap.</sub> | <img src="docs/images/swarm_rollout_cloud.gif" width="240"><br><sub>**Sampling cloud** — each drone's fan of scored candidate velocities.</sub> |
| <img src="docs/images/swarm_obstacle_gauntlet.gif" width="240"><br><sub>**Obstacle gauntlet** — a dozen drones weave through six sweeping bodies.</sub> | <img src="docs/images/swarm_wind.gif" width="240"><br><sub>**Crosswind** — a gusting wind field bows every track.</sub> | <img src="docs/images/swarm_antipodal_orca_vs_hrvo.gif" width="240"><br><sub>**ORCA vs HRVO** at the hub — collide vs roundabout.</sub> |
| <img src="docs/images/swarm_crossing_rvo_vs_orca.gif" width="240"><br><sub>**RVO dance vs ORCA glide** — RVO's tracks kink, ORCA's stay smooth.</sub> | <img src="docs/images/compare_rrt_vs_rrt_star.gif" width="240"><br><sub>**RRT vs RRT\*** — the "optimal" path drives into the obstacle.</sub> | <img src="docs/images/compare_gpu_mppi_vs_mpc_3d.gif" width="240"><br><sub>**GPU-MPPI vs MPC in 3-D**, rollouts visualised.</sub> |

Every 2-D swarm clip above is one command — `scripts/render_swarm_gif.py` (no AirSim needed; `antipodal` / `crossing` / `doorway` scenarios, `--obstacles`, `--wind`, `--rollouts`, `+row` convention flags) — and the 3-D sphere/field is `scripts/render_swarm_3d_gif.py`. The photorealistic clips are real [AirSim](https://github.com/microsoft/AirSim) (Unreal Engine), recorded against a running Blocks server with `scripts/record_airsim_cinematic.py` (`--mode chase`/`orbit`), `scripts/record_airsim_topdown_live.py` (top-down), `scripts/record_airsim_lidar_live.py` (onboard LiDAR point cloud), `scripts/record_airsim_dashboard.py` (the Foxglove-style multi-panel dashboard), and `scripts/record_airsim_swarm_policy.py` (the learned teammate-token policy replayed on the fleet).

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
| planner | `astar`, `straight`, `mpc`, `mppi`, `cvar_mppi`, `gpu_mppi`, `rrt`, `rrt_star`, `chomp`, `mpc_chomp`, `warmup_select_mppi`, `orca`, `rvo`, `vo`, `hrvo`, `bvc`, `cbf`, `apf`, `roundabout`, `mgr` |
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
