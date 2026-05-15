#!/usr/bin/env bash
# Chunked runner for AirSim multi-drone n=30 paired study.
#
# AirSim's Blocks RPC handler wedges after 1-2 sequential multi-drone
# `client.reset()` calls — the second reset never returns. Workaround
# is to bounce the Blocks server between every episode and run each as
# its own `uav-nav run` invocation with --seed overriding the YAML's
# base seed. The per-episode JSON outputs (episode_000_*.json) land in
# a `seed_NNN/` subdir, and `scripts/paired_analysis_airsim_multi.py`
# globs `seed_*/episode_000_joint.json` to produce the paired-seed
# Wilson CI + McNemar comparison.
#
# Usage:
#   scripts/run_airsim_multi_chunked.sh <mpc|gpu_mppi> <n_episodes> \
#                                       <base_seed> <out_dir>
# Env:
#   AIRSIM_BLOCKS_DIR: path to Blocks/LinuxNoEditor (default /tmp/airsim-blocks/...)

set -u

PLANNER="${1:-mpc}"
N="${2:-30}"
BASE_SEED="${3:-42}"
OUT_DIR="${4:-results/airsim_multi_n30_mpc}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLOCKS_DIR="${AIRSIM_BLOCKS_DIR:-/tmp/airsim-blocks/Blocks/LinuxNoEditor}"

case "$PLANNER" in
  mpc)
    TPL="$REPO_ROOT/examples/exp_airsim_multi_n30.yaml"
    ;;
  gpu_mppi)
    TPL="$REPO_ROOT/examples/exp_airsim_multi_n30_gpu_mppi.yaml"
    ;;
  *)
    echo "unknown planner: $PLANNER"; exit 1 ;;
esac

mkdir -p "$OUT_DIR"

WORK_YAML="/tmp/_airsim_chunk_${PLANNER}.yaml"
sed 's|^num_episodes: 30$|num_episodes: 1|' "$TPL" > "$WORK_YAML"

# Find/kill Blocks binary via /proc/N/comm to avoid pkill self-match.
kill_blocks() {
  for p in /proc/[0-9]*/comm; do
    [ -r "$p" ] || continue
    if [ "$(cat "$p" 2>/dev/null)" = "Blocks" ]; then
      bpid="${p#/proc/}"
      bpid="${bpid%/comm}"
      kill -9 "$bpid" 2>/dev/null
    fi
  done
  sleep 2
}

start_airsim() {
  kill_blocks
  (cd "$BLOCKS_DIR" && setsid ./Blocks.sh -RenderOffscreen -ResX=720 -ResY=405 \
      > /tmp/_blocks_chunk.log 2>&1 < /dev/null &)
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

export PATH="$HOME/.local/bin:$PATH"

for i in $(seq 0 $((N - 1))); do
  SEED=$((BASE_SEED + i))
  EP_OUT="$OUT_DIR/seed_${SEED}"
  if [ -f "$EP_OUT/episode_000_joint.json" ]; then
    echo "[$(date +%T)] seed $SEED already done, skipping"
    continue
  fi
  echo "[$(date +%T)] === seed=$SEED ($PLANNER, i=$i/$N) ==="
  rm -rf "$EP_OUT"
  start_airsim || { echo "AirSim start failed; aborting"; break; }
  timeout 240 uav-nav run "$WORK_YAML" --seed "$SEED" --output-dir "$EP_OUT" \
      > "$EP_OUT.log" 2>&1
  RC=$?
  kill_blocks
  if [ -f "$EP_OUT/episode_000_joint.json" ]; then
    python3 - "$EP_OUT/episode_000_joint.json" << 'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f'  → seed={d["meta"]["seed"]} per={d["per_drone_outcomes"]} joint={d["outcome"]} t={d["final_t"]}')
PY
  else
    echo "  → NO joint.json, rc=$RC, tail of log:"
    tail -5 "$EP_OUT.log" 2>/dev/null
  fi
done

echo "[$(date +%T)] [done] $PLANNER all $N seeds processed"
