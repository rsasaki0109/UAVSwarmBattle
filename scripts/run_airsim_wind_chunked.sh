#!/usr/bin/env bash
# Chunked runner for the AirSim N=4 antipodal-under-crosswind study.
# One episode per process, Blocks bounced between episodes (RPC degrades after
# a few sequential resets). Per-episode joint JSON in seed_NNN/ (chunked layout).
#
# Usage:
#   scripts/run_airsim_wind_chunked.sh <lateral_bias> <wind_x> <wind_y> <n> <base_seed> <out_dir>
# Env: AIRSIM_BLOCKS_DIR, UAVNAV_VENV
set -u

BIAS="${1:-0.0}"
WX="${2:-4.0}"
WY="${3:-4.0}"
N="${4:-12}"
BASE_SEED="${5:-42}"
OUT_DIR="${6:-results/airsim_wind}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLOCKS_DIR="${AIRSIM_BLOCKS_DIR:-/media/sasaki/aiueo/airsim/Blocks/LinuxNoEditor}"
VENV="${UAVNAV_VENV:-/media/sasaki/aiueo/ai_coding_ws/uav-nav-lab/.venv}"
TPL="$REPO_ROOT/examples/exp_airsim_antipodal_wind_n4.yaml"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$REPO_ROOT"

WORK_YAML="/tmp/_airsim_wind_b${BIAS}_w${WX}x${WY}.yaml"
sed -e "s|^num_episodes: .*|num_episodes: 1|" \
    -e "s|LATERAL_BIAS|${BIAS}|" -e "s|WIND_X|${WX}|" -e "s|WIND_Y|${WY}|" \
    "$TPL" > "$WORK_YAML"

mkdir -p "$OUT_DIR"

kill_blocks() {
  for p in /proc/[0-9]*/comm; do
    [ -r "$p" ] || continue
    if [ "$(cat "$p" 2>/dev/null)" = "Blocks" ]; then
      bpid="${p#/proc/}"; bpid="${bpid%/comm}"; kill -9 "$bpid" 2>/dev/null
    fi
  done
  sleep 2
}
start_airsim() {
  kill_blocks
  (cd "$BLOCKS_DIR" && setsid ./Blocks.sh -RenderOffscreen -ResX=720 -ResY=405 \
      > /tmp/_blocks_wind.log 2>&1 < /dev/null &)
  for i in $(seq 1 60); do
    if ss -tln 2>/dev/null | grep -q :41451; then sleep 6; return 0; fi
    sleep 1
  done
  echo "[$(date +%T)] AirSim failed to listen within 60s" >&2; return 1
}

for i in $(seq 0 $((N - 1))); do
  SEED=$((BASE_SEED + i))
  EP_OUT="$OUT_DIR/seed_${SEED}"
  if [ -f "$EP_OUT/episode_000_joint.json" ]; then
    echo "[$(date +%T)] seed $SEED done, skip"; continue
  fi
  echo "[$(date +%T)] === bias=$BIAS wind=($WX,$WY) seed=$SEED (i=$i/$N) ==="
  rm -rf "$EP_OUT"
  start_airsim || { echo "AirSim start failed; aborting"; break; }
  timeout 200 "$VENV/bin/uav-nav" run "$WORK_YAML" --seed "$SEED" \
      --output-dir "$EP_OUT" > "$EP_OUT.log" 2>&1
  RC=$?
  if [ -f "$EP_OUT/episode_000_joint.json" ]; then
    "$VENV/bin/python" - "$EP_OUT/episode_000_joint.json" << 'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f'  -> seed={d["meta"]["seed"]} per={d["per_drone_outcomes"]} joint={d["outcome"]} t={d["final_t"]:.1f}')
PY
  else
    echo "  -> NO joint.json, rc=$RC; tail:"; tail -4 "$EP_OUT.log" 2>/dev/null
  fi
done
kill_blocks
echo "[$(date +%T)] [done] bias=$BIAS wind=($WX,$WY) all $N seeds"
