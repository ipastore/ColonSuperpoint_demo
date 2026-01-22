# LightGlue weight alignment notes (2025-02-07)

## Context
- Comparing official `sift_lightglue` weights with custom checkpoints trained without scale/orientation inputs caused key/shape mismatches.
- CUDA/OpenCV SIFT scale/orientation conventions differed; conversions and matching scripts were updated accordingly.
- Tools were added to align/compare checkpoints and to optionally build a no-scale/ori matcher for the right-hand weights in comparisons.

## Problems observed
- Scale conventions: OpenCV SIFT outputs diameter and degrees; LightGlue SIFT expects sigma and radians (pycolmap style).
- Precomputed CUDA SIFT features were rescaled without adjusting `scales`.
- Checkpoint alignment: tar checkpoints included prefixes (`module.`, `matcher.`) and buffers (`confidence_thresholds`); official `.pth` did not.
- Shape mismatch: custom training without `add_scale_ori` produced `posenc.Wr.weight` of shape `(32,2)` vs expected `(32,4)` and transformer dims matching a no-scale/ori config.
- Large missing/extra key reports when loading a no-scale/ori checkpoint into the default SIFT matcher (different model topology).

## Solutions applied
- Scale/orientation handling:
  - Harmonize OpenCV SIFT scales to sigma (2*sigma → sigma) and rescale `scales` alongside `keypoints` for precomputed features.
  - Orientation handling remains in radians; CUDA SIFT bin loader converts degrees → radians.
- Conversion/comparison tooling:
  - `tools/convert_to_pth/align_and_compare.py` strips prefixes, optionally keeps extras, compares value differences, and saves aligned `.pth` by default; `--compare-only` to skip saving, `--keep-extras` to retain buffers like `confidence_thresholds`.
  - `compare_lightglue_weights.py` logs state_dict key counts and missing/extra/dropped shape mismatches.
- No-scale/ori matcher option:
  - Right-hand matcher can be built with `--no-scale-ori` (default dims: `input_dim=128`, `descriptor_dim=256`, `add_scale_ori=False`) to load checkpoints trained without scales/oris.

## How to use
- Align/inspect a checkpoint against the official `.pth`:
  - Compare only: `python tools/convert_to_pth/align_and_compare.py <tar> <official.pth> --compare-only`
  - Align and save (drops extras by default): `python tools/convert_to_pth/align_and_compare.py <tar> <official.pth>`
  - Keep tar-only buffers (e.g., `confidence_thresholds`): add `--keep-extras`.
- Run matcher comparison with a no-scale/ori right matcher:
  - `python compare_lightglue_weights.py --config configs/compare_lightglue_weights.yaml --run-name <name> --no-scale-ori --right-weights <aligned_no_scale_ckpt>`

## Remaining gaps / decisions
- A checkpoint trained with `add_scale_ori=True` is required to eliminate the `posenc.Wr.weight` shape mismatch entirely; padding/reshaping is possible but not recommended without retraining.
- `confidence_thresholds` buffer is optional for loading but can be kept during conversion with `--keep-extras`.
