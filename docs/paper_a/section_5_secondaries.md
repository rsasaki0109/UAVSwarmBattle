# §5. Secondary findings: where coordination exists, and how MPPI should adapt

The §3 result is not a statement that GPU MPPI always changes
multi-drone coordination. It appears at a specific operating point:
four drones, a 3D volume tight enough that their routes interact, and
enough remaining free space that planners can still recover. Two
secondary ablations define that operating region. The second half of
this section asks a different follow-up: once MPPI is the family of
interest, can a benchmark cell tell us which softmax temperature to
use without sweeping every temperature?

## 5.1 Escape volume can erase coordination Δ

The same crossing pattern behaves differently when lifted from a
2D grid into an open 3D voxel volume. With CPU MPC and constant-
velocity peer prediction, N=4 in 2D has per-drone success 87.5 %,
joint success 73.3 %, and an independence baseline of 58.6 %, giving
Δ = +14.7 pp. The 3D version raises per-drone success to 95.8 % and
joint success to 83.3 %, but the independence baseline rises with it
to 84.3 %, leaving Δ = -1.0 pp. The absolute joint rate improves, yet
the coordination signal disappears.

That inversion is the point. In 2D, four crossing drones share the
same plane; at each intersection, peer prediction has to create a
yielding pattern. In open 3D, each drone can route over or under a
peer, so the route choice becomes mostly independent. The extra
dimension does not just make the problem easier; it removes the
interaction that coordination Δ was measuring. This is why the paper
reports both joint success and Δ. Joint success alone would say "3D
is better"; Δ says "3D made coordination unnecessary."

The density ablation confirms the mechanism. Keeping N=4 and the
same 3D world, increasing static obstacles from 30 to 120 drops
per-drone success from 95.8 % to 65.8 %, but Δ returns from -1.0 pp
to +8.0 pp. At 240 obstacles, per-drone success falls further to
46.7 % and Δ drops to +5.2 pp. Coordination therefore has a middle
regime: sparse worlds give independent escape routes, saturated
worlds lose too many drones to static obstacles, and intermediate
free volume is where peer-aware planning changes the joint outcome.

This boundary condition matters for interpreting §3. GPU MPPI's
larger Δ is meaningful only because the §3 cell lies in the middle:
the drones interact, but the per-drone planner still has viable
routes. A benchmark that is too sparse will conclude that
coordination is irrelevant; a benchmark that is too dense will
measure obstacle failure rather than coordination. Neither falsifies
the §3 mechanism. They locate its operating range.

## 5.2 Peer prediction is load-bearing in that middle regime

The density result says that coordination reappears when the 3D
escape volume is constrained. The direct ablation is to remove the
constant-velocity peer predictor at those constrained cells. At the
120-obstacle dense cell, prediction-on gives per-drone success
65.8 % and joint success 26.7 %. Turning prediction off drops
per-drone success to 16.7 % and joint success to 6.7 %. At the
240-obstacle packed cell, prediction-on gives 46.7 % per-drone and
10.0 % joint; prediction-off gives 18.3 % per-drone and 0.0 % joint.

The per-drone column is the important one. Removing peer prediction
at fixed density costs 49 pp of per-drone success in the dense cell,
which is the same scale as increasing obstacle count eightfold in the
prediction-on sweep. In crossing-pair scenarios, unmodelled peers are
not a minor perturbation; they are effectively a moving obstacle
field as severe as the static map itself. A planner that ignores
peer trajectories is not solving the same problem as one that
forecasts them.

This also prevents a common misread of Δ. The dense prediction-off
cell still has a positive Δ because per-drone success is so low that
the independence baseline is almost zero. That arithmetic does not
mean coordination succeeded. It means a few episodes happened to
survive after the planners ignored each other. The packed
prediction-off cell makes the mechanism clear: once chance runs out,
joint success is 0/30 and Δ collapses.

Together, §5.1 and §5.2 define the backdrop for §3. Free volume decides
whether coordination is needed; peer prediction decides whether a
classical MPC planner can exploit that need; GPU MPPI then changes
the *shape* of failures at a matched operating point. The paper's
headline Δ flip is therefore not an isolated planner trick. It is one
case inside a broader rule: coordination metrics only become
load-bearing when the scenario leaves enough, but not too much,
per-agent free volume.

## 5.3 One warmup episode predicts the MPPI temperature regime

The temperature sweeps that led to §3 also showed a less obvious
pattern: no fixed MPPI temperature is globally right. In one
intersection cell, high temperature (nearly uniform averaging) is best;
in a wave-like intruder cell, low temperature (argmin-like commitment)
is best; in a peer-dominated cell, all temperatures are nearly flat.
That looks at first like another sweep burden. The useful observation
is that the required temperature can be predicted from statistics the
vanilla MPPI planner already computes.

The rule uses a single episode at the vanilla temperature
(`temperature = 1.0`). For every replan we record two angles:

1. **top-2 disagreement**: the angular separation between the two
   highest-weighted rollout actions.
2. **chosen-vs-goal (cvg)**: the angle between vanilla MPPI's weighted
   action and the straight-to-goal direction.

The first angle is an applicability check. When top-2 disagreement is
very large, the rollout set is not choosing between coherent escape
modes; the landscape is chaotic, and temperature does not carry much
information. When top-2 disagreement is moderate, cvg predicts which
end of the softmax spectrum should win. Low cvg means the prior
straight-to-goal action is already right, so averaging many rollouts is
beneficial. High cvg means the prior misses and one specific evasion
direction matters, so argmin-like commitment is beneficial.

The calibrated rule used by `warmup_select_mppi` is:

| warmup signal | interpretation | selected temperature |
|---|---|---|
| mean top-2 > 50-60 deg | chaotic rollout landscape | keep vanilla, `t = 1.0` |
| otherwise, mean cvg < 12.5 deg | prior-aligned cell | uniform-like, `t = 10` |
| otherwise | prior misses, commit to a side | argmin-like, `t = 0.1` |

On the five calibration cells, the rule matches the measured best
temperature in every case:

| cell | top-2 | cvg | N+P prediction | measured best |
|---|---:|---:|---|---|
| intersection v1 | 29.1 deg | 9.2 deg | `t = 10` | `t = 10` (100 %) |
| wave intruders | 30.9 deg | 17.1 deg | `t = 0.1` | `t = 0.1` (70 %) |
| 4-way 3D | 33.7 deg | 4.8 deg | `t = 10` | `t = 10` (85 %) |
| peer-dominated | 83.9 deg | 24.9 deg | flat / vanilla | all MPPI temps tied (40 %) |
| chokepoint | 33.1 deg | 10.1 deg | `t = 10` | `t = 10` (95 %) |

This turns a post-hoc temperature sweep into a small adaptive
procedure: spend episode 0 measuring the cell, then run the remaining
episodes at the selected temperature. In multi-drone runs, the planner
pools the warmup samples across drones and picks one shared temperature
for the whole session. That matters because per-drone warmup signals
can straddle the threshold even when the cell-wide decision is clear.

## 5.4 The family-selector hypothesis fails, but in a useful direction

The next hypothesis was that the same warmup signal might decide not
only the MPPI temperature, but the planner family itself: MPC in some
cells, MPPI in others. That extension failed. Across the nine cells
where we have `{MPC, MPPI t=0.1, MPPI t=1.0, MPPI t=10}` plus a
warmup-select run, MPC is the empirical best planner in **0/9** cells.
The best MPPI temperature exceeds MPC in every row, by 10-60 pp and
35 pp on average:

| cell | MPC | best MPPI | gap |
|---|---:|---:|---:|
| intersection v1 | 55 % | 100 % (`t = 10`) | -45 pp |
| intersection wave | 45 % | 70 % (`t = 0.1`) | -25 pp |
| intersection chokepoint | 60 % | 95 % (`t = 10`) | -35 pp |
| 4-way 3D | 75 % | 85 % (`t = 10`) | -10 pp |
| peer-dominated | 30 % | 40 % (all MPPI temps) | -10 pp |
| city v1 | 30 % | 90 % (`t = 10`) | -60 pp |
| city wave | 50 % | 85 % (`t = 10`) | -35 pp |
| city chokepoint | 20 % | 45 % (`t = 0.1`) | -25 pp |
| city 3x3 | 45 % | 95 % (`t = 10`) | -50 pp |

So the negative result is the point: in these dynamic-prediction
regimes, there is no evidence that the adaptive system needs to switch
back to MPC. The useful prescription is narrower and stronger:
choose MPPI as the family, then use the N+P warmup rule to choose the
temperature.

There is one important caveat. The city chokepoint row is the only
confident temperature miss: the warmup signal is strongly prior-aligned
(cvg = 3.6 deg), so N+P selects `t = 10`, but the measured best is
`t = 0.1`. The failure is not a noisy-threshold problem. It is a
geometric scope condition: city walls and cube obstacles create a
forced 4 m gap, and averaging rollouts across both sides smears the
command into the obstacle. §6 treats that as a limitation of the rule,
not as a parameter to tune away.
