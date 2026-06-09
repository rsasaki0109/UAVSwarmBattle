#!/usr/bin/env bash
# Chunked runner for the AirSim N=4 antipodal convention study.
#
# AirSim's Blocks RPC degrades/wedges after a handful of sequential
# client.reset() calls in one process (a 20-episode single run crawled to
# 1 episode in 13 min). So, like scripts/run_airsim_multi_chunked.sh, we
# bounce the Blocks server between EVERY episode and run each as its own
# `uav-nav run --seed` invocation. Per-episode joint JSON lands in a
# seed_NNN/ subdir (chunked layout) for paired analysis.
#
# Usage:
#   scripts/run_airsim_antipodal_chunked.sh <lateral_bias> <n_episodes> <base_seed> <out_dir>
# Env:
#   AIRSIM_BLOCKS_DIR : path to Blocks/LinuxNoEditor (has Blocks.sh)
#   UAVNAV_VENV       : path to the venv (default: main checkout's .venv)
set -u

BIAS="${1:-0.0}"
N="${2:-16}"
BASE_SEED="${3:-42}"
OUT_DIR="${4:-results/airsim_antipodal_stock}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLOCKS_DIR="${AIRSIM_BLOCKS_DIR:-/media/sasaki/aiueo/airsim/Blocks/LinuxNoEditor}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${UAVNAV_VENV:-$ROOT/.venv}"
TPL="$REPO_ROOT/examples/exp_airsim_antipodal_n4.yaml"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$REPO_ROOT"

WORK_YAML="/tmp/_airsim_antipodal_chunk_b${BIAS}.yaml"
sed -e "s|^num_episodes: .*|num_episodes: 1|" \
    -e "s|^  lateral_bias: .*|  lateral_bias: ${BIAS}|" \
    "$TPL" > "$WORK_YAML"

mkdir -p "$OUT_DIR"

# Kill the Blocks binary by /proc/N/comm match (avoids pkill self-match).
# The user authorised bouncing this (idle, user-launched) Blocks instance.
kill_blocks() {
  for p in /proc/[0-9]*/comm; do
    [ -r "$p" ] || continue
    if [ "$(cat "$p" 2>/dev/null)" = "Blocks" ]; then
      bpid="${p#/proc/}"; bpid="${bpid%/comm}"
      kill -9 "$bpid" 2>/dev/null
    fi
  done
  sleep 2
}

start_airsim() {
  kill_blocks
  (cd "$BLOCKS_DIR" && setsid ./Blocks.sh -RenderOffscreen -ResX=720 -ResY=405 \
      > /tmp/_blocks_antipodal.log 2>&1 < /dev/null &)
  for i in $(seq 1 60); do
    if ss -tln 2>/dev/null | grep -q :41451; then
      sleep 6   # engine warmup grace
      return 0
    fi
    sleep 1
  done
  echo "[$(date +%T)] AirSim failed to listen within 60s" >&2
  return 1
}

for i in $(seq 0 $((N - 1))); do
  SEED=$((BASE_SEED + i))
  EP_OUT="$OUT_DIR/seed_${SEED}"
  if [ -f "$EP_OUT/episode_000_joint.json" ]; then
    echo "[$(date +%T)] seed $SEED already done, skipping"; continue
  fi
  echo "[$(date +%T)] === bias=$BIAS seed=$SEED (i=$i/$N) ==="
  rm -rf "$EP_OUT"
  start_airsim || { echo "AirSim start failed; aborting"; break; }
  timeout 180 "$VENV/bin/uav-nav" run "$WORK_YAML" --seed "$SEED" \
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
echo "[$(date +%T)] [done] bias=$BIAS all $N seeds processed"
