#!/usr/bin/env bash
# Run short or full AirSim static-cube discriminating-cell sweeps.
#
# Default mode runs GPU MPPI only for a few seeds on the denser candidates.
# Use MODE=paired to run both planners for n=30 on every candidate.
#
# Examples:
#   scripts/run_airsim_discriminating_sweep.sh
#   N=10 BASE_SEED=42 CANDIDATES="dense packed" scripts/run_airsim_discriminating_sweep.sh
#   MODE=paired N=30 scripts/run_airsim_discriminating_sweep.sh

set -euo pipefail

MODE="${MODE:-gpu_smoke}"
N="${N:-5}"
BASE_SEED="${BASE_SEED:-42}"
CANDIDATES="${CANDIDATES:-central_29p375 central_29p25 central_half central_north central_west_thick central_west central_soft central mid dense packed}"

case "$MODE" in
  gpu_smoke)
    PLANNERS="gpu_mppi"
    ;;
  paired)
    PLANNERS="mpc gpu_mppi"
    ;;
  *)
    echo "MODE must be gpu_smoke or paired" >&2
    exit 2
    ;;
esac

for candidate in $CANDIDATES; do
  for planner in $PLANNERS; do
    suffix=""
    if [ "$planner" = "gpu_mppi" ]; then
      suffix="_gpu_mppi"
    fi
    yaml="examples/exp_airsim_multi_discriminating_${candidate}_n30${suffix}.yaml"
    out="results/airsim_multi_discriminating_${candidate}_${MODE}_${planner}"
    if [ "$MODE" = "paired" ]; then
      out="results/airsim_multi_discriminating_${candidate}_n30_${planner}"
    fi
    echo "=== candidate=${candidate} planner=${planner} n=${N} yaml=${yaml} ==="
    scripts/run_airsim_multi_chunked.sh "$planner" "$N" "$BASE_SEED" "$out" "$yaml"
  done

  if [ "$MODE" = "paired" ]; then
    python3 scripts/paired_analysis_airsim_multi.py \
      "results/airsim_multi_discriminating_${candidate}_n30_mpc" \
      "results/airsim_multi_discriminating_${candidate}_n30_gpu_mppi"
  fi
done
