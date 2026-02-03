"""CLI to compare official and finetuned LightGlue weights on CudaSIFT."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import compare_matching as cm
from compare_matching import (
    LG_MATCH_COLOR,
    detach_to_cpu,
    gather_image_sets,
    lightglue_matches_to_tensor,
    load_preprocessed_image,
    load_sift_bin,
    prepare_axes,
    plot_matches_on_axes,
    rescale_feature_dict,
)
from utils.LightGlue.lightglue import LightGlue
from utils.LightGlue.lightglue import viz2d
from utils.LightGlue.lightglue.utils import rbd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_OFFICIAL_WEIGHTS = Path("weights/lightglue/sift_lightglue_official.pth")
DEFAULT_FINETUNED_WEIGHTS = Path(
    "weights/lightglue/long26_gt_pos_null_10_lg1e-03_lr5e-05_5bin.pth"
)
RowData = Tuple[str, dict, dict, torch.Tensor, np.ndarray, np.ndarray]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for LightGlue weight comparison.

    Returns:
        Parsed CLI namespace with all options.
    """
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=str,
        help="YAML file with default options (keys must match CLI flags).",
    )

    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(parents=[config_parser], description=__doc__)
    parser.add_argument(
        "--cuda-sift-dataset",
        type=str,
        help="Root folder containing *_sift.bin files for CUDA SIFT.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Destination directory for comparison figures.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for the output subfolder.",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        choices=[1, 2, 4, 8],
        help="Image downscale factor (1 leaves original size).",
    )
    parser.add_argument(
        "--official-weights",
        type=str,
        help="LightGlue weights path for the official matcher.",
    )
    parser.add_argument(
        "--finetuned-weights",
        type=str,
        help="LightGlue weights path for the finetuned matcher.",
    )
    parser.add_argument(
        "--lightglue-filter-threshold",
        type=float,
        help="LightGlue filter threshold.",
    )
    parser.add_argument(
        "--official-lightglue-filter-threshold",
        type=float,
        help="Override filter threshold for the official weights.",
    )
    parser.add_argument(
        "--finetuned-lightglue-filter-threshold",
        type=float,
        help="Override filter threshold for the finetuned weights.",
    )
    parser.add_argument(
        "--lightglue-depth-confidence",
        type=float,
        default=-1.0,
        help="LightGlue depth confidence (default -1 to disable early stop).",
    )
    parser.add_argument(
        "--lightglue-width-confidence",
        type=float,
        default=-1.0,
        help="LightGlue width confidence (default -1 to disable pruning).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="Computation device.",
    )

    config_args, remaining = config_parser.parse_known_args()
    config_data: Dict[str, object] = {}
    if config_args.config:
        config_path = Path(config_args.config)
        if not config_path.exists():
            parser.error(f"Config file not found: {config_path}")
        config_data = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(config_data, dict):
            parser.error("Config file must map option names to values.")

    if config_data:
        parser.set_defaults(**config_data)

    opt = parser.parse_args(remaining)
    opt.config = config_args.config

    if opt.device is None:
        opt.device = default_device

    defaults = {
        "downscale": 1,
        "cuda_sift_dataset": "./assets/matching/no_crop_1440x1080",
        "output_dir": "./matching_outputs/lightglue_cudasift_weights",
        "lightglue_filter_threshold": LightGlue.default_conf["filter_threshold"],
        "lightglue_depth_confidence": LightGlue.default_conf["depth_confidence"],
        "lightglue_width_confidence": LightGlue.default_conf["width_confidence"],
        "official_weights": str(DEFAULT_OFFICIAL_WEIGHTS),
        "finetuned_weights": str(DEFAULT_FINETUNED_WEIGHTS),
    }
    for key, value in defaults.items():
        if getattr(opt, key, None) is None:
            setattr(opt, key, value)

    if opt.official_lightglue_filter_threshold is None:
        opt.official_lightglue_filter_threshold = opt.lightglue_filter_threshold
    if opt.finetuned_lightglue_filter_threshold is None:
        opt.finetuned_lightglue_filter_threshold = opt.lightglue_filter_threshold

    return opt


def build_lightglue(
    weights_path: Path,
    device: torch.device,
    *,
    filter_threshold: float,
    depth_confidence: float,
    width_confidence: float,
) -> LightGlue:
    """Instantiate LightGlue and load a raw .pth state dict."""
    resolved = weights_path.expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"LightGlue weights not found: {resolved}")

    sift_conf = LightGlue.features["sift"]
    matcher = (
        LightGlue(
            features=None,
            input_dim=sift_conf["input_dim"],
            add_scale_ori=sift_conf.get("add_scale_ori", True),
            filter_threshold=filter_threshold,
            depth_confidence=depth_confidence,
            width_confidence=width_confidence,
        )
        .eval()
        .to(device)
    )
    state_dict = torch.load(str(resolved), map_location="cpu")
    matcher.load_state_dict(state_dict, strict=False)
    return matcher


def run_lightglue_matches(
    matcher: LightGlue, feats0: dict, feats1: dict
) -> torch.Tensor:
    """Run LightGlue matching and return CPU match indices."""
    output = matcher({"image0": feats0, "image1": feats1})
    return lightglue_matches_to_tensor(rbd(output)["matches"])


def process_pair(
    image0_path: Path,
    image1_path: Path,
    downscale: int,
    matcher_official: LightGlue,
    matcher_finetuned: LightGlue,
    official_label: str,
    finetuned_label: str,
) -> List[RowData]:
    """Compute CudaSIFT matches for both LightGlue variants."""
    def image_to_bin_path(path: Path) -> Path:
        return path.with_name(path.stem + "_sift.bin")

    cuda_image0 = load_preprocessed_image(image0_path, downscale)
    cuda_image1 = load_preprocessed_image(image1_path, downscale)
    cuda_image0_np = cuda_image0.permute(1, 2, 0).cpu().numpy()
    cuda_image1_np = cuda_image1.permute(1, 2, 0).cpu().numpy()

    image_size0 = (cuda_image0.shape[-1], cuda_image0.shape[-2])
    image_size1 = (cuda_image1.shape[-1], cuda_image1.shape[-2])

    feats0_bin_raw = load_sift_bin(image_to_bin_path(image0_path), image_size0)
    feats1_bin_raw = load_sift_bin(image_to_bin_path(image1_path), image_size1)

    matches_official = run_lightglue_matches(
        matcher_official, feats0_bin_raw, feats1_bin_raw
    )
    matches_finetuned = run_lightglue_matches(
        matcher_finetuned, feats0_bin_raw, feats1_bin_raw
    )

    feats0_bin = rescale_feature_dict(detach_to_cpu(rbd(feats0_bin_raw)), downscale)
    feats1_bin = rescale_feature_dict(detach_to_cpu(rbd(feats1_bin_raw)), downscale)

    return [
        (
            official_label,
            feats0_bin,
            feats1_bin,
            matches_official,
            cuda_image0_np,
            cuda_image1_np,
        ),
        (
            finetuned_label,
            feats0_bin,
            feats1_bin,
            matches_finetuned,
            cuda_image0_np,
            cuda_image1_np,
        ),
    ]


def weight_label(weights_path: Optional[Path], default: str) -> str:
    """Build a short label to describe a weight path."""
    if weights_path is None:
        return default
    if weights_path.is_dir():
        return weights_path.name or default
    return weights_path.stem or default


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available.", file=sys.stderr)
        return 1

    global DEVICE
    DEVICE = torch.device(args.device)
    cm.DEVICE = DEVICE
    torch.set_grad_enabled(False)

    cuda_dataset_root = Path(args.cuda_sift_dataset).expanduser().resolve()
    if not cuda_dataset_root.exists():
        print(f"CudaSIFT dataset not found: {cuda_dataset_root}", file=sys.stderr)
        return 1

    image_sets = gather_image_sets(cuda_dataset_root)

    official_weights = Path(args.official_weights).expanduser()
    finetuned_weights = Path(args.finetuned_weights).expanduser()

    matcher_official = build_lightglue(
        official_weights,
        DEVICE,
        filter_threshold=args.official_lightglue_filter_threshold,
        depth_confidence=args.lightglue_depth_confidence,
        width_confidence=args.lightglue_width_confidence,
    )
    matcher_finetuned = build_lightglue(
        finetuned_weights,
        DEVICE,
        filter_threshold=args.finetuned_lightglue_filter_threshold,
        depth_confidence=args.lightglue_depth_confidence,
        width_confidence=args.lightglue_width_confidence,
    )

    official_label = weight_label(official_weights, "official")
    finetuned_label = weight_label(finetuned_weights, "finetuned")

    base_dir = Path(args.output_dir).expanduser().resolve()
    run_name = args.run_name or f"{official_label}_vs_{finetuned_label}"
    run_root = base_dir / run_name
    if run_root.exists():
        raise RuntimeError(
            f"Output directory '{run_root}' already exists. "
            "Choose a different --run-name."
        )
    run_root.mkdir(parents=True, exist_ok=False)

    config_snapshot = {
        "cuda_sift_dataset": str(cuda_dataset_root),
        "downscale": args.downscale,
        "official_weights": str(official_weights),
        "finetuned_weights": str(finetuned_weights),
        "device": args.device,
        "output_dir": str(run_root),
        "lightglue_filter_threshold": args.lightglue_filter_threshold,
        "official_lightglue_filter_threshold": args.official_lightglue_filter_threshold,
        "finetuned_lightglue_filter_threshold": args.finetuned_lightglue_filter_threshold,
        "lightglue_depth_confidence": args.lightglue_depth_confidence,
        "lightglue_width_confidence": args.lightglue_width_confidence,
    }
    (run_root / "config_used.yaml").write_text(
        yaml.safe_dump(config_snapshot, sort_keys=True)
    )

    for folder, images in image_sets:
        out_subfolder = run_root / folder.name
        out_subfolder.mkdir(parents=True, exist_ok=True)
        print(f"Processing {folder} -> {out_subfolder}")

        for img_path0, img_path1 in itertools.combinations(images, 2):
            rows = process_pair(
                img_path0,
                img_path1,
                args.downscale,
                matcher_official,
                matcher_finetuned,
                official_label,
                finetuned_label,
            )

            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            for row_idx, (
                label,
                feats0_row,
                feats1_row,
                matches_row,
                img0_np,
                img1_np,
            ) in enumerate(rows):
                left_img, right_img = axes[row_idx, 0], axes[row_idx, 1]
                prepare_axes(left_img, right_img, img0_np, img1_np)
                plot_matches_on_axes(
                    left_img,
                    right_img,
                    feats0_row,
                    feats1_row,
                    matches_row,
                    f"CudaSIFT+LG ({label})",
                    LG_MATCH_COLOR,
                )

            fig.tight_layout(pad=0.5)
            base0 = img_path0.stem
            base1 = img_path1.stem
            out_path = out_subfolder / f"{base0}__{base1}.png"
            viz2d.save_plot(str(out_path))
            plt.close(fig)

    print(f"All comparisons saved under {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
