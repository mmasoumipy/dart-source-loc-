#!/usr/bin/env bash
# Run all 6 LOSO folds sequentially. Extra args are passed to src.train, e.g.
#   scripts/run_loso.sh                                   # YOLO26-m, seed 0
#   scripts/run_loso.sh --model weights/yolo26s-obb.pt --batch 8 --name loso_s
#   SEED=1 scripts/run_loso.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/logs
for k in 1 2 3 4 5 6; do
  echo "=== fold $k  $(date)"
  .venv/bin/python -m src.train --fold "$k" --seed "${SEED:-0}" "$@" > "runs/logs/fold${k}_s${SEED:-0}_$(date +%H%M).log" 2>&1
done
echo "=== all folds done $(date)"
