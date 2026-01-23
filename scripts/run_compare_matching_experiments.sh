#!/usr/bin/env bash
set -euo pipefail

# Defaults
CONFIG=${1:-./configs/compare_matching.yaml}
PYTHON_BIN=${PYTHON_BIN:-python}
dataset="./assets/matching/crop_1350x1012"
downscale=1
lg_thresh=0.05
sp_thresh=0.0
disk_thresh=0.0
aliked_thresh=0.0
aliked_model="aliked-t16"
nn_thresh=0.9
ratio_thresh=0.9

# Model comparison (edit weights as needed). Use list of "model|weight" entries to allow duplicates.
MODEL_WEIGHTS=(
  "MagicLeap|./weights/MagicLeap/superpoint_v1.pth"
)

  # "MagicLeap|./weights/MagicLeap/superpoint_v1.pth"
  # "SuperpointNet|./weights/SuperpointNet/leon_esuperPointNet_400000_checkpoint.pth"
  # "SuperpointNet|./weights/SuperpointNet/superPointNet_200000_checkpoint.pth"
  # "SuperpointNet_gauss2|./weights/SuperpointNet_gauss2/superPointNet_gauss2_170000_checkpoint.pth"
  # "SuperpointNet_gauss2|./weights/SuperpointNet_gauss2/22_F_gpu2_bs32_45000_checkpoint.pth"
  # "SuperpointNet_gauss2|./weights/SuperpointNet_gauss2/18_F_retrain_true_185000_checkpoint.pth"

run_exp() {
  echo "[RUN] $PYTHON_BIN compare_matching.py --config $CONFIG $*"
  $PYTHON_BIN compare_matching.py --config "$CONFIG" "$@" --dataset "$dataset" --downscale "$downscale" --lightglue-filter-threshold "$lg_thresh" --superpoint-detection-threshold "$sp_thresh" --disk-detection-threshold "$disk_thresh" --aliked-model-name "$aliked_model" --aliked-detection-threshold "$aliked_thresh"
}

for entry in "${MODEL_WEIGHTS[@]}"; do
  IFS='|' read -r model weight <<< "$entry"

  # for ratio_thresh in 0.7 0.8 0.9 1; do
  
    if [[ -f "$weight" ]]; then
      # # bi 
      run_exp --superpoint-model-name "$model" --superpoint-weights-path "$weight" --output-dir "bi"
      
      # # bi_ratio
      # run_exp --superpoint-model-name "$model" --superpoint-weights-path "$weight" --ratio-nn-thresh "$ratio_thresh" --output-dir "bi_ratio_$ratio_thresh"
      
      # # bi_nn_thresh
      # run_exp --superpoint-model-name "$model" --superpoint-weights-path "$weight" --nn-match-threshold "$nn_thresh" --output-dir "bi_nn_thresh_$nn_thresh"
      
      # # bi_ratio_nn_thresh
      # run_exp --superpoint-model-name "$model" --superpoint-weights-path "$weight" --ratio-nn-thresh "$ratio_thresh" --nn-match-threshold "$nn_thresh" --output-dir "bi_ratio${ratio_thresh}_nn-thresh${nn_thresh}"
      else
      echo "[WARN] Skipping $model (missing weights at $weight)" >&2
     fi
  # done
done
