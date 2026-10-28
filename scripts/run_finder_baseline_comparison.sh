#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-${FINDER_DEVICE:-cuda:0}}"
RESUME="${RESUME:-never}"
if [[ "$RESUME" != "never" ]]; then
  echo "FINDER Table 2 has no checkpoint-resume support; use RESUME=never." >&2
  exit 2
fi
if [[ "$DEVICE" == cuda:* ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
else
  export CUDA_VISIBLE_DEVICES=""
fi
export FINDER_RESUME="$RESUME"
export PYTHONHASHSEED=93
export FINDER_DEVICE="$DEVICE"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
RUN_ROOT="${FINDER_OUTPUT_DIR:-${RUN_ROOT:-$ROOT/runs/baseline_comparison}}"
mkdir -p "$RUN_ROOT/folds1to4" "$RUN_ROOT/folds5to10"
# The published run restarted training seed 93 before fold 5.
FINDER_FOLD_START=1 FINDER_FOLD_END=4 FINDER_OUTPUT_DIR="$RUN_ROOT/folds1to4" \
  "$PYTHON_BIN" finder/baseline_comparison/c02_model_train.py
export FINDER_FOLD_START=5 FINDER_FOLD_END=10 FINDER_OUTPUT_DIR="$RUN_ROOT/folds5to10"
exec "$PYTHON_BIN" finder/baseline_comparison/c02_model_train.py
