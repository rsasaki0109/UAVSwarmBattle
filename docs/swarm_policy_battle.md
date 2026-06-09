# Swarm policy battle — OSS & paper roster

Head-to-head arena for decentralized swarm controllers on the lab's **50×50
antipodal** cells (peers-only and hub-crossing obstacle). Run:

```bash
python scripts/swarm_policy_battle_phase.py --episodes 20
python scripts/swarm_policy_battle_phase.py --scenario obstacle --episodes 20 --workers 4 --merge
python scripts/swarm_policy_battle_phase.py --scenario obstacle --arms orca_conv mpc_gt swarm_transformer navrl
```

NavRL arm setup (once):

```bash
bash scripts/setup_navrl_adapter.sh
```

Results → `results/swarm_policy_battle/phase.json`

## Latest full battle (n=20, seeds 6000–6019)

| Scenario | orca | orca_conv | hrvo | mgr | mpc_gt | navrl | **swarm_transformer** |
|----------|------|-----------|------|-----|--------|-------|----------------------|
| peers | 11/20 | 20/20 | 12/20 | 20/20 | 1/20 | — | **20/20** |
| obstacle | 1/20 | 0/20 | 2/20 | 0/20 | 12/20 | **0/20** | **20/20** |

NavRL uses the upstream single-agent checkpoint (~40 m maps, 4 m LiDAR); **0/20** on this geometry is expected domain gap. Re-run with parallel arms:

```bash
python scripts/swarm_policy_battle_phase.py --scenario obstacle --episodes 20 --workers 4 --merge
```

McNemar vs `swarm_transformer` on obstacle: mpc_gt p≈0.008 (8 seeds where xf wins alone).

Montage GIF: `python scripts/render_swarm_policy_battle_gif.py` → `docs/images/swarm_policy_battle_obstacle.gif`

## In-repo arms (ready to battle)

| Arm | Type | Lineage | Notes |
|-----|------|---------|-------|
| `orca` | reactive LP | van den Berg et al. 2011, [RVO2](https://github.com/snape/RVO2) | Canonical reciprocal baseline; hub deadlock N≥6 |
| `orca_conv` | ORCA + `lateral_bias` | this lab | Explicit right-of-way convention |
| `hrvo` | side-commitment VO | this lab | Local symmetry-break; decays with density |
| `mgr` | triggered Merry-Go-Round | Zhou et al. 2025 [arXiv:2503.05848](https://arxiv.org/abs/2503.05848) | Decentralized ring from local deadlock |
| `mpc_gt` | MPC + game-theoretic predictor | this lab | Sampled forecast + convention; obstacle teacher |
| `swarm_transformer` | teammate-token transformer | TeamHOI-style (this work) | BC + REINFORCE; pred threat tokens |
| `navrl` | LiDAR + PPO velocity | Xu et al. RA-L 2025 | Upstream checkpoint; 4 m rays |

## External OSS / papers (not yet wired)

| Project | Paper / venue | Task | Integration path |
|---------|---------------|------|------------------|
| [TeamHOI](https://github.com/sail-sg/TeamHOI) | CVPR 2026 | HOI carry, 2–8 humanoids | Token policy architecture reference; already mirrored in `swarm_transformer` |
| [CooHOI](https://github.com/Winston-Gu/CooHOI) | — | cooperative carry | Compare formation vs reorientation (see transport findings) |
| [RVO2](https://github.com/snape/RVO2) | ICRA 2011 | micro-agents | Already ported as `orca` / `rvo` / `hrvo` |
| [PathPlanning/ORCA-algorithm](https://github.com/PathPlanning/ORCA-algorithm) | — | ORCA + MAPF deadlock | MAPF unlock for hub deadlocks — future `orca_mapf` arm |
| [draca_planner](https://github.com/vinayakkapoor/draca_planner) | — | ORCA-distilled RL | Learned collision avoider; compare to `swarm_transformer` |
| [agentic-swarms](https://github.com/fabeha-raheel/agentic-swarms) | — | ROS leader–follower + ORCA | AirSim/ROS track, not antipodal geometry |

## Scenarios

| Key | Geometry | Stress |
|-----|----------|--------|
| `peers` | 6-drone antipodal, no static obs | Symmetry / convention |
| `obstacle` | same + hub-crossing dynamic body | Convention + non-peer threat |

Eval seeds: `6000 … 6000+episodes-1` (matches published transformer eval).

## Reading the table

- **joint** = all drones success
- Wilson 95 % CI on joint rate (paired seeds across arms)
- McNemar `p` vs `swarm_transformer` on shared seeds
