#!/usr/bin/env bash
# Clone NavRL and install the PyTorch stack for the navrl planner adapter.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/NavRL"
if [ ! -d "$DEST/.git" ]; then
  mkdir -p "$ROOT/third_party"
  git clone --depth 1 https://github.com/Zhefan-Xu/NavRL.git "$DEST"
fi
python3 -m pip install -e "$ROOT[navrl]"
python3 -m pip install einops
echo "NavRL adapter ready: $DEST/quick-demos/ckpts/navrl_checkpoint.pt"
