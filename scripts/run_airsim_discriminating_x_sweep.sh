#!/usr/bin/env bash
# Sweep the physical x-position of the extra central AirSim pillar.
#
# This avoids committing one YAML pair per sub-cell. It starts from the
# central_29p375 YAMLs, rewrites only the generated experiment name, output
# directory, and the final static_obstacles entry's physical x-position, then
# delegates each generated YAML to run_airsim_multi_chunked.sh.
#
# Examples:
#   X_VALUES="29.42 29.45 29.47" scripts/run_airsim_discriminating_x_sweep.sh
#   MODE=paired N=30 X_VALUES="29.45" scripts/run_airsim_discriminating_x_sweep.sh

set -euo pipefail

MODE="${MODE:-gpu_smoke}"
N="${N:-3}"
BASE_SEED="${BASE_SEED:-42}"
X_VALUES="${X_VALUES:-29.42 29.45 29.47}"

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

slug_for_x() {
  printf '%s' "$1" | tr '.' 'p'
}

make_yaml() {
  local planner="$1"
  local x="$2"
  local slug="$3"
  local suffix=""
  if [ "$planner" = "gpu_mppi" ]; then
    suffix="_gpu_mppi"
  fi

  local src="examples/exp_airsim_multi_discriminating_central_29p375_n30${suffix}.yaml"
  local dst="/tmp/uavnav_airsim_disc_x_${slug}${suffix}.yaml"
  python3 - "$src" "$dst" "$x" "$slug" "$planner" <<'PY'
import sys
from pathlib import Path

import yaml

src, dst, raw_x, slug, planner = sys.argv[1:]
x = float(raw_x)
data = yaml.safe_load(Path(src).read_text(encoding="utf-8"))

data["name"] = f"airsim_multi_discriminating_x_{slug}_{planner}"
data["output"]["dir"] = f"results/airsim_multi_discriminating_x_{slug}_{planner}"

# The central boundary-probe mesh is intentionally the final static obstacle
# in the central_29p375 source pair.
central = data["simulator"]["static_obstacles"][-1]
central["name"] = f"uavnav_disc_ns_x_{slug}_33"
central["position"][0] = x

Path(dst).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
print(dst)
PY
}

for x in $X_VALUES; do
  slug="$(slug_for_x "$x")"
  for planner in $PLANNERS; do
    yaml_path="$(make_yaml "$planner" "$x" "$slug")"
    out="results/airsim_multi_discriminating_x_${slug}_${MODE}_${planner}"
    if [ "$MODE" = "paired" ]; then
      out="results/airsim_multi_discriminating_x_${slug}_n30_${planner}"
    fi
    echo "=== physical_x=${x} planner=${planner} n=${N} yaml=${yaml_path} ==="
    scripts/run_airsim_multi_chunked.sh "$planner" "$N" "$BASE_SEED" "$out" "$yaml_path"
  done

  if [ "$MODE" = "paired" ]; then
    python3 scripts/paired_analysis_airsim_multi.py \
      "results/airsim_multi_discriminating_x_${slug}_n30_mpc" \
      "results/airsim_multi_discriminating_x_${slug}_n30_gpu_mppi"
  fi
done
