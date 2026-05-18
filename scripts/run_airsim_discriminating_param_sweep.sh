#!/usr/bin/env bash
# Generate and run AirSim discriminating-cell parameter probes.
#
# The hand-written YAMLs established that the floor is driven by planner
# occupancy, not by the physical x-position alone. This runner keeps further
# probes out of examples/ by generating /tmp YAMLs from committed bases.
#
# Examples:
#   VARIANTS="occ29_inflate1 occ29_inflate2" scripts/run_airsim_discriminating_param_sweep.sh
#   VARIANTS="occ29_margin04 occ29_margin05" scripts/run_airsim_discriminating_param_sweep.sh
#   MODE=paired N=30 VARIANTS="occ29_inflate2" scripts/run_airsim_discriminating_param_sweep.sh

set -euo pipefail

MODE="${MODE:-gpu_smoke}"
N="${N:-3}"
BASE_SEED="${BASE_SEED:-42}"
VARIANTS="${VARIANTS:-base_ew05 base_ew06 base_ew07 base_ns01 base_ns02 base_ns03}"

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

make_yaml() {
  local planner="$1"
  local variant="$2"
  local suffix=""
  if [ "$planner" = "gpu_mppi" ]; then
    suffix="_gpu_mppi"
  fi

  local src="examples/exp_airsim_multi_discriminating_central_soft_n30${suffix}.yaml"
  if [[ "$variant" == base_* ]]; then
    src="examples/exp_airsim_multi_discriminating_n30${suffix}.yaml"
  fi
  local dst="/tmp/uavnav_airsim_disc_${variant}_${planner}.yaml"
  python3 - "$src" "$dst" "$variant" "$planner" <<'PY'
import sys
from pathlib import Path

import yaml

src, dst, variant, planner = sys.argv[1:]
data = yaml.safe_load(Path(src).read_text(encoding="utf-8"))

data["name"] = f"airsim_multi_discriminating_{variant}_{planner}"
data["output"]["dir"] = f"results/airsim_multi_discriminating_{variant}_{planner}"

boxes = data["scenario"]["obstacles"]["boxes"]
meshes = data["simulator"]["static_obstacles"]
central_box = boxes[-1]
central_mesh = meshes[-1]

def set_central_cell(x: int, y: int = 33) -> None:
    central_box["min"] = [x, y, 22]
    central_box["max"] = [x, y, 38]

def set_central_mesh_position(x: float, y: float = 33.5) -> None:
    central_mesh["position"][0] = x
    central_mesh["position"][1] = y
    slug_x = str(x).replace(".", "p")
    slug_y = str(y).replace(".", "p")
    central_mesh["name"] = f"uavnav_disc_ns_x_{slug_x}_y_{slug_y}"

def set_central_mesh_x(x: float) -> None:
    set_central_mesh_position(x, central_mesh["position"][1])

def set_ew_scale(scale: float) -> None:
    for mesh in meshes[:4]:
        mesh["scale"][0] = scale
        mesh["scale"][1] = scale

def set_south_lane_x(x: float) -> None:
    south = data["scenario"]["drones"][3]
    south["start"][0] = x
    south["goal"][0] = x

def set_baseline_ns_scale(scale: float) -> None:
    meshes[-1]["scale"][0] = scale
    meshes[-1]["scale"][1] = scale

def parse_scale_token(token: str) -> float:
    if not token.isdigit():
        raise SystemExit(f"bad EW scale token in variant: {variant}")
    return int(token) / (10 ** (len(token) - 1))

if variant.startswith("base_ew06_lane"):
    # base_ew06_lane30 / base_ew06_lane22: EW pillars at scale 0.6 +
    # south-drone lane override (default x=26 → centered or further
    # west). Used to disentangle drone-3 geometric pinch from MPC's
    # multi-drone cluster failure mode in §4.4.4.
    set_ew_scale(0.6)
    set_south_lane_x(float(variant.rsplit("lane", 1)[1]))
elif variant.startswith("base_ew"):
    set_ew_scale(parse_scale_token(variant.rsplit("ew", 1)[1]))
elif variant.startswith("base_ns"):
    set_baseline_ns_scale(parse_scale_token(variant.rsplit("ns", 1)[1]))
elif variant == "occ29_inflate1":
    data["planner"]["inflate"] = 1
elif variant == "occ29_inflate2":
    data["planner"]["inflate"] = 2
elif variant == "occ29y34_inflate2":
    data["planner"]["inflate"] = 2
    set_central_cell(29, 34)
    set_central_mesh_position(29.5, 34.5)
elif variant == "occ30y33_inflate2":
    data["planner"]["inflate"] = 2
    set_central_cell(30, 33)
    set_central_mesh_position(30.5, 33.5)
elif variant == "occ29_inflate2_second31":
    data["planner"]["inflate"] = 2
    boxes.append({"min": [31, 33, 22], "max": [31, 33, 38]})
    meshes.append(
        {
            "name": "uavnav_disc_ns_31_33",
            "asset": "1M_Cube_Chamfer",
            "position": [31.5, 33.5, 30.5],
            "scale": [0.05, 0.05, 17.0],
        }
    )
elif variant == "occ29_inflate2_second32":
    data["planner"]["inflate"] = 2
    boxes.append({"min": [32, 33, 22], "max": [32, 33, 38]})
    meshes.append(
        {
            "name": "uavnav_disc_ns_32_33",
            "asset": "1M_Cube_Chamfer",
            "position": [32.5, 33.5, 30.5],
            "scale": [0.05, 0.05, 17.0],
        }
    )
elif variant == "occ29_inflate2_second31y34":
    data["planner"]["inflate"] = 2
    boxes.append({"min": [31, 34, 22], "max": [31, 34, 38]})
    meshes.append(
        {
            "name": "uavnav_disc_ns_31_34",
            "asset": "1M_Cube_Chamfer",
            "position": [31.5, 34.5, 30.5],
            "scale": [0.05, 0.05, 17.0],
        }
    )
elif variant.startswith("occ29_inflate2_ew"):
    data["planner"]["inflate"] = 2
    set_ew_scale(parse_scale_token(variant.rsplit("ew", 1)[1]))
elif variant == "occ29_wobs100":
    data["planner"]["w_obs"] = 100.0
elif variant == "occ29_wobs50":
    data["planner"]["w_obs"] = 50.0
elif variant == "occ29_margin04":
    data["planner"]["safety_margin"] = 0.4
elif variant == "occ29_margin05":
    data["planner"]["safety_margin"] = 0.5
elif variant.startswith("occ28_x29p45_ew"):
    # ew04 -> 0.4, ew035 -> 0.35, ew06 -> 0.6.
    scale = parse_scale_token(variant.rsplit("ew", 1)[1])
    set_central_cell(28)
    set_central_mesh_x(29.45)
    set_ew_scale(scale)
else:
    raise SystemExit(f"unknown variant: {variant}")

Path(dst).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
print(dst)
PY
}

for variant in $VARIANTS; do
  for planner in $PLANNERS; do
    yaml_path="$(make_yaml "$planner" "$variant")"
    out="results/airsim_multi_discriminating_${variant}_${MODE}_${planner}"
    if [ "$MODE" = "paired" ]; then
      out="results/airsim_multi_discriminating_${variant}_n30_${planner}"
    fi
    echo "=== variant=${variant} planner=${planner} n=${N} yaml=${yaml_path} ==="
    scripts/run_airsim_multi_chunked.sh "$planner" "$N" "$BASE_SEED" "$out" "$yaml_path"
  done

  if [ "$MODE" = "paired" ]; then
    python3 scripts/paired_analysis_airsim_multi.py \
      "results/airsim_multi_discriminating_${variant}_n30_mpc" \
      "results/airsim_multi_discriminating_${variant}_n30_gpu_mppi"
  fi
done
