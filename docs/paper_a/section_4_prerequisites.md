# §4. Prerequisites that made the comparison fair

The §3 headline — GPU MPPI's softmax-amplified +11.4 pp coordination
Δ over CPU MPC's +0.8 pp — only carries weight after four
prerequisite checks, listed below. Skipping any one of them would
have produced numbers within the same harness that would invert the
headline. The §4.4 sim-transferability prerequisite has its own
dedicated section; §4.1-4.3 follow here.

## §4.1 Pareto cells, not planners

Reporting "MPC achieves $x$ %" without naming the cell at which the
$x$ was measured is the most common reproducibility failure in our
own internal history. The same planner family — same code path, same
implementation — exhibits dramatically different success/cost tradeoffs
across `n_samples × horizon`. A cell-naïve comparison silently
embeds a non-Pareto operating point and skews the family's reported
quality by 10-50 pp.

To make our §3 comparison fair, we ran a full Pareto sweep for each
planner family in the relevant scenario before staking the headline
on a single cell. Four sweeps in particular are load-bearing:

| family | scenario | sweep grid | optimal cell | success | plan_dt steady |
|---|---|---|---|---|---|
| CPU MPC      | 2D `grid_world` 50×50          | $n \in \{4,8,16,32\}, h \in \{10,20,40\}$  | (16, 20) | 100 % | 35 ms |
| CPU MPC      | 3D `voxel_world` 40×40×12      | same                                       | (8, 40)  | 88 %  | 70 ms |
| GPU MPPI    | 2D `grid_world` 50×50          | $n \in \{32,64,128,256\}, h \in \{20,40,80\}$ | (128, 40) | 100 % | 3.0 ms |
| GPU MPPI    | 3D `voxel_world` 40×40×12      | same                                       | (64-256, 20) | 100 % | 3.5 ms |

Three patterns emerge across the four sweeps:

**(a) The optimal $h$ is not monotone in scenario dimension.** CPU
MPC's 2D Pareto cell sits at $h=20$; the 3D Pareto cell sits at
$h=40$. The 3D escape volume rewards longer planning horizons because
out-of-plane obstacle bypasses take more time-steps to commit to.
GPU MPPI exhibits the opposite — its 2D Pareto cell sits at $h=40$
and its 3D cell at $h=20$. GPU MPPI's softmax-weighted averaging at
long horizons in 3D produces "centerline" trajectories that get
trapped in the cost basin around the start-goal line; CPU MPC's
deterministic argmin does not have this failure mode, hence the
opposite $h$ preference.

**(b) $n_\text{samples}$ preferences flip with dimension.** CPU MPC
prefers smaller $n$ in 3D (8 vs 16 in 2D) — the 3D goal-bonus term
fires earlier and the small-batch argmin commits cleanly. GPU MPPI
prefers $n \in [64, 256]$ in both 2D and 3D, but the lower bound is
not interchangeable across dimensions: 2D at $n=32$ collapses to 0 %
(undersampling against single-frame cost ridges), 3D at the same
$n$ remains at 92 %.

**(c) The GPU MPPI 3D Pareto cell Pareto-dominates the CPU MPC 3D
Pareto cell.** GPU MPPI $(n=64, h=20)$ at 100 % / 3.5 ms beats CPU
MPC $(n=8, h=40)$ at 88 % / 70 ms on both axes — higher success,
20× lower steady-state plan time. The headline §3 comparison runs
these two cells against each other on a multi-drone scenario; the
fact that the GPU MPPI cell dominates *single-drone* is what makes
the §3 tie on *joint* success an interesting finding rather than a
matched-on-easier-baseline observation. The two planner families are
compared on each one's Pareto-optimal cell, not on a fixed config
or a config that handicaps one family.

Practical convention from §4.1 we adopt throughout: any cell cited
in a numerical claim names $n_\text{samples}$ and $h$ explicitly,
and the cell is sourced from the relevant Pareto sweep. We do not
cite numbers from non-Pareto cells unless making the methodological
point about cell sensitivity.

## §4.2 Implementation correctness gates every cell

The GPU MPPI cells reported in §4.1 are post-fix numbers. The pre-fix
GPU MPPI had a goal-mask bug that altered the conclusion of every
cell in the sweep — without altering any of the measurements at the
cell level.

The bug. The CPU MPC reference implementation, when a rollout enters
the goal radius mid-horizon, naturally short-circuits the remaining-
horizon cost: the goal-bonus term fires once on entry, and the
subsequent steps' obstacle/smoothing penalties either don't accrue
(the drone "stops" in the planner's internal state) or accrue but
are dominated by the goal bonus, so the rollout's total cost is
correctly small. The batched GPU port copies the cost-computation
loop literally, including the obstacle penalty summed over *all*
horizon steps. A GPU MPPI rollout that reaches goal at step 20 of a
horizon-40 plan keeps accruing collision-margin and smoothing penalty
over the remaining 20 steps — for any post-goal drift through a cost
ridge those terms can be arbitrarily large, so the rollout's total
cost looks bad and the softmax weighting drives it toward zero.

The phenotype. Pre-fix, the GPU MPPI 2D Pareto sweep returned 0 % at
every $h \geq 40$ cell. The reading we tentatively committed to was
"GPU MPPI suffers a speed collapse at long horizons" — a clean,
publishable, *wrong* claim. The 3D Pareto sweep returned the same
pattern. The MPC reference at the same cells returned 80-100 %.

The fix and its consequences. Adding the goal-mask term to the GPU
MPPI cost-accumulation loop (commit 2a9d196) transforms the Pareto
landscape:

- 2D optimum shifts to $(n{=}128, h{=}40)$, 100 % / 3.0 ms.
- 3D optimum shifts to $(n{=}64{-}256, h{=}20)$, 100 % / 3.5 ms.
- "Speed collapse at $h \geq 40$" disappears entirely.
- The 3D Pareto cell that pre-fix wasn't visible because of the bug
  now Pareto-dominates the CPU MPC baseline.

The methodological reading. Every measurement in our pre-fix 12-cell
sweep was numerically *correct* — the rates and plan-times were
honest readings of what the buggy planner did. The conclusion was
inverted because the bug affected only one planner family's expression
of the cost. **Re-running the comparison at the post-fix code while
trusting the pre-fix conclusion would have produced the opposite
recommendation.** This is the §1 "implementation correctness gates
every cell" claim's concrete instantiation: a single line of code
flipped the family-level winner across a 12-cell sweep, with no
change to any of the measurements *as such*. We re-validated every
cited cell after the fix.

## §4.3 Plan-time reporting on CUDA backends

CUDA-based planners (GPU MPPI in this work) carry a first-call
penalty that, if mishandled in the headline, makes them look 10-30×
slower than steady-state. The cause is autograd-graph compilation
plus a Dijkstra cost-to-go precompute on the first replan of an
episode. On our hardware (single RTX 4090), the compile costs
~14 s and the cost-to-go precompute ~50 ms. Steady-state plan_dt
for the same planner is 3.5 ms.

A naïve mean over per-replan times gives:

$$
\text{mean plan\_dt} = \frac{14000 + (n-1) \cdot 3.5}{n}
$$

where $n$ is the episode's replan count. For a 12 s episode at 5 Hz
($n=60$ replans), this gives a mean of 237 ms — two orders of
magnitude above steady state. Multi-drone amplifies the problem: with
4 GPU MPPI planners per episode, the first-call cost is paid 4 times
per episode, so a single-episode benchmark on 4 drones can report
plan_dt at $4 \times 14 \text{ s} / 240 = 233$ ms.

Convention adopted throughout: **steady-state mean is the headline
number, computed with the first replan of every episode dropped**.
We report it as `plan_dt (steady-state mean / p95)`. The first-call
cost is documented separately in the YAML headers for any
deployment-relevant deployment scenario. We make explicit two
configurations where the first-call cost is and is not amortizable:

(a) **Production deployment, warmed-up planner**: the planner stays
loaded across many episodes (or across the whole mission). The first
call cost is paid once at startup. Steady-state plan_dt is the right
metric.

(b) **Single-episode benchmarks with cold start**: the planner is
constructed fresh per episode and the autograd graph is rebuilt. Mean
plan_dt over the episode is then dominated by the first call. This
is the relevant metric for benchmarks that fork-new every episode
(some CI/test configurations do this).

The numbers we report in §3 and §4.4 use convention (a) — they are
the warmed-up planner's plan_dt, which is what the deployment cares
about. The first-call cost is mentioned in the §2 metrics section
and in the relevant YAML headers; the comparison between GPU MPPI's
3.5 ms and CPU MPC's 70 ms steady-state plan_dt is a fair comparison
*under the convention that both are amortized over the deployment*.
For a one-shot single-episode benchmark, the comparison would invert
in GPU MPPI's disfavor at small scenarios where the warmup dwarfs the
useful per-step compute.

This convention also explains why §4.4.1's AirSim plan_dt numbers
(180 ms vs 188 ms) look so similar across planners despite the
20× single-drone advantage on `dummy_3d`: the AirSim numbers are also
steady-state, and on AirSim the bridge's synchronous-RPC overhead
dominates the per-step budget for *both* planners. The convention
does not paper over the AirSim collapse; it just reports the
deployment-relevant steady-state without the warmup obscuring it.

---

§4.4 (sim transferability) follows as a separate document
(`docs/paper_a/section_4_4_sim_transferability.md`).
