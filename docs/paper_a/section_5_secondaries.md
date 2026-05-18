# §5. Secondary findings: when coordination exists at all

The §3 result is not a statement that GPU MPPI always changes
multi-drone coordination. It appears at a specific operating point:
four drones, a 3D volume tight enough that their routes interact, and
enough remaining free space that planners can still recover. Two
secondary ablations define that operating region.

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
