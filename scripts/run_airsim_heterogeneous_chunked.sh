#!/usr/bin/env bash
# Chunked runner for the AirSim N=4 HETEROGENEOUS (MPC+CBF) convention study.
# One episode per process with the Blocks server bounced between episodes
# (AirSim's RPC degrades after a few sequential resets). Per-episode joint
# JSON lands in seed_NNN/ (chunked layout) for paired analysis.
#
# Usage:
#   scripts/run_airsim_heterogeneous_chunked.sh <mpc_bias> <cbf_bias> <n> <base_seed> <out_dir>
#     off arm : mpc_bias=0.0 cbf_bias=0.0   (no shared convention)
#     on  arm : mpc_bias=2.0 cbf_bias=0.5   (shared right-of-way)
# Env: AIRSIM_BLOCKS_DIR, UAVNAV_VENV
set -u

MPC_BIAS="${1:-0.0}"
CBF_BIAS="${2:-0.0}"
N="${3:-12}"
BASE_SEED="${4:-42}"
OUT_DIR="${5:-results/airsim_het_off}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLOCKS_DIR="${AIRSIM_BLOCKS_DIR:-/media/sasaki/aiueo/airsim/Blocks/LinuxNoEditor}"
VENV="${UAVNAV_VENV:-/media/sasaki/aiueo/ai_coding_ws/uav-nav-lab/.venv}"
TPL="$REPO_ROOT/examples/exp_airsim_heterogeneous_n4.yaml"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$REPO_ROOT"

WORK_YAML="/tmp/_airsim_het_m${MPC_BIAS}_c${CBF_BIAS}.yaml"
sed -e "s|^num_episodes: .*|num_episodes: 1|" \
    -e "s|MPC_BIAS|${MPC_BIAS}|g" -e "s|CBF_BIAS|${CBF_BIAS}|g" \
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
      > /tmp/_blocks_het.log 2>&1 < /dev/null &)
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
  echo "[$(date +%T)] === mpc=$MPC_BIAS cbf=$CBF_BIAS seed=$SEED (i=$i/$N) ==="
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
echo "[$(date +%T)] [done] mpc=$MPC_BIAS cbf=$CBF_BIAS all $N seeds"
