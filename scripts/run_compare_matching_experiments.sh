#!/usr/bin/env bash
set -euo pipefail

# Defaults
CONFIG=${1:-./configs/compare_matching.yaml}
PYTHON_BIN=${PYTHON_BIN:-python}
dataset="./assets/matching/crop_1350x1012"
downscale=1
lg_thresh=0.1

# Model comparison (edit weights as needed). Use list of "model|weight" entries to allow duplicates.
MODEL_WEIGHTS=(
  "MagicLeap|./weights/MagicLeap/superpoint_v1.pth"
  "SuperpointNet|./weights/SuperpointNet/leon_esuperPointNet_400000_checkpoint.pth"
  "SuperpointNet|./weights/SuperpointNet/superPointNet_200000_checkpoint.pth"
  "SuperpointNet_gauss2|./weights/SuperpointNet_gauss2/superPointNet_gauss2_170000_checkpoint.pth"
  "SuperpointNet_gauss2|./weights/SuperpointNet_gauss2/22_F_gpu2_bs32_45000_checkpoint.pth"
  "SuperpointNet_gauss2|./weights/SuperpointNet_gauss2/18_F_retrain_true_185000_checkpoint.pth"
)

run_exp() {
  echo "[RUN] $PYTHON_BIN compare_matching.py --config $CONFIG $*"
  $PYTHON_BIN compare_matching.py --config "$CONFIG" "$@" --dataset "$dataset" --downscale "$downscale" --lightglue-filter-threshold "$lg_thresh"
}

for entry in "${MODEL_WEIGHTS[@]}"; do
  IFS='|' read -r model weight <<< "$entry"
  
  # SuperPoint detection threshold sweep
  for sp_thresh in 0.005 0.0005; do
    if [[ -f "$weight" ]]; then
      run_exp --superpoint-model-name "$model" --superpoint-weights-path "$weight" --superpoint-detection-threshold "$sp_thresh"
    else
      echo "[WARN] Skipping $model (missing weights at $weight)" >&2
    fi
  done
done