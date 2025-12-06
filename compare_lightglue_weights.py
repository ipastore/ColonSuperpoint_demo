"""CLI to compare two LightGlue weights on SIFT and CudaSIFT features."""

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
    harmonize_sift_scales,
)
from utils.LightGlue.lightglue import LightGlue, SIFT
from utils.LightGlue.lightglue import viz2d
from utils.LightGlue.lightglue.utils import rbd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_RIGHT_WEIGHTS = Path("weights/lightglue")
RowData = Tuple[str, dict, dict, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]


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
        "--dataset",
        type=str,
        help="Path to image dataset.",
    )
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
        "--left-weights",
        type=str,
        help="Optional custom LightGlue weights path for the left matcher.",
    )
    parser.add_argument(
        "--right-weights",
        type=str,
        help="LightGlue weights path for the right matcher.",
    )
    parser.add_argument(
        "--lightglue-filter-threshold",
        type=float,
        help="LightGlue filter threshold.",
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
        "--no-scale-ori",
        action="store_true",
        help="Instantiate LightGlue without scale/orientation inputs (add_scale_ori=False).",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=None,
        help="Override LightGlue input_dim (defaults to 128 when --no-scale-ori is set).",
    )
    parser.add_argument(
        "--descriptor-dim",
        type=int,
        default=None,
        help="Override LightGlue descriptor_dim (defaults to 256 when --no-scale-ori is set).",
    )
    parser.add_argument(
        "--sift-max-keypoints",
        type=int,
        help="Maximum SIFT keypoints (OpenCV backend).",
    )
    parser.add_argument(
        "--sift-contrast-threshold",
        type=float,
        help="OpenCV SIFT contrast threshold.",
    )
    parser.add_argument(
        "--sift-edge-threshold",
        type=float,
        help="OpenCV SIFT edge threshold.",
    )
    parser.add_argument(
        "--sift-n-octave-layers",
        type=int,
        help="OpenCV SIFT octave layers.",
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

    opt = parser.parse_args(remaining)
    opt.config = config_args.config

    for key, value in config_data.items():
        if getattr(opt, key, None) is None:
            setattr(opt, key, value)

    if opt.device is None:
        opt.device = default_device

    defaults = {
        "downscale": 1,
        "cuda_sift_dataset": "./assets/matching/no_crop_1440x1080",
        "output_dir": "./matching_outputs/lightglue_weights",
        "lightglue_filter_threshold": LightGlue.default_conf["filter_threshold"],
        "lightglue_depth_confidence": LightGlue.default_conf["depth_confidence"],
        "lightglue_width_confidence": LightGlue.default_conf["width_confidence"],
        "right_weights": str(DEFAULT_RIGHT_WEIGHTS),
    }
    for key, value in defaults.items():
        if getattr(opt, key, None) is None:
            setattr(opt, key, value)

    missing = [name for name in ["dataset"] if getattr(opt, name, None) is None]
    if missing:
        parser.error(
            "Missing required options (provide via CLI or config): "
            + ", ".join(missing)
        )

    return opt


def load_lightglue_with_weights(
    weights_path: Optional[Path],
    device: torch.device,
    *,
    filter_threshold: float,
    depth_confidence: float,
    width_confidence: float,
    no_scale_ori: bool,
    input_dim_override: Optional[int],
    descriptor_dim_override: Optional[int],
) -> LightGlue:
    """Instantiate LightGlue with optional custom weights.

    Args:
        weights_path: Optional checkpoint path or directory with a single .pth file.
        device: Target device for the matcher.
        filter_threshold: LightGlue filter threshold.
        depth_confidence: LightGlue depth confidence.
        width_confidence: LightGlue width confidence.

    Returns:
        A LightGlue matcher on the requested device.
    """
    if no_scale_ori:
        input_dim = input_dim_override or 128
        descriptor_dim = descriptor_dim_override or 256
        matcher = LightGlue(
            features=None,
            input_dim=input_dim,
            descriptor_dim=descriptor_dim,
            add_scale_ori=False,
            n_layers=LightGlue.default_conf["n_layers"],
            num_heads=LightGlue.default_conf["num_heads"],
            flash=LightGlue.default_conf["flash"],
            mp=LightGlue.default_conf["mp"],
            depth_confidence=depth_confidence,
            width_confidence=width_confidence,
            filter_threshold=filter_threshold,
        ).eval().to(device)
    else:
        matcher = LightGlue(
            features="sift",
            filter_threshold=filter_threshold,
            depth_confidence=depth_confidence,
            width_confidence=width_confidence,
        ).eval().to(device)
    base_checksum = matcher_checksum(matcher)
    expected_keys = len(matcher.state_dict())

    if weights_path is None:
        print(
            f"LightGlue loaded with default (official) weights; "
            f"state_dict_keys={expected_keys}"
        )
        return matcher

    resolved = weights_path.expanduser()
    candidate = resolved
    if resolved.is_dir():
        patterns = ["*.pth", "*.pth.tar"]
        ckpts = sorted(
            {p for pattern in patterns for p in resolved.glob(pattern)}
        )
        if not ckpts:
            raise FileNotFoundError(f"No .pth or .pth.tar files found under {resolved}")
        candidate = ckpts[0]

    if not candidate.exists():
        raise FileNotFoundError(f"LightGlue weights not found: {candidate}")

    raw_obj = torch.load(str(candidate), map_location="cpu")
    state_dict = _extract_state_dict(raw_obj)
    state_dict = _strip_prefix(state_dict, "module.")
    state_dict = _strip_prefix(state_dict, "matcher.")
    state_dict, mismatched = _filter_compatible_state_dict(
        state_dict, matcher.state_dict()
    )
    load_result = matcher.load_state_dict(state_dict, strict=False)
    changed = matcher_checksum(matcher) != base_checksum
    loaded_keys = len(state_dict)
    # Align terminology with align_and_compare: keys present in the model
    # state_dict but absent in the loaded weights are "missing in weights".
    # Keys present in the weights but not expected by the model are "extra in weights".
    missing_keys = list(load_result.missing_keys)
    extra_in_weights = list(load_result.unexpected_keys)
    dropped_details = [
        (k, f"ckpt_shape={shape_ckpt}", f"model_shape={shape_model}")
        for (k, shape_ckpt, shape_model) in mismatched
    ]
    print(
        f"Loaded LightGlue from {candidate}; checksum_changed={changed}; "
        f"state_dict_keys_loaded={loaded_keys}; "
        f"state_dict_keys_expected={expected_keys}; "
        f"missing_in_weights={len(missing_keys)}; "
        f"extra_in_weights={len(extra_in_weights)}; "
        f"dropped_mismatched={len(mismatched)}"
    )
    if missing_keys or extra_in_weights or mismatched:
        if missing_keys:
            print(format_key_list("  missing_in_weights", missing_keys, limit=10))
        if extra_in_weights:
            print(format_key_list("  extra_in_weights", extra_in_weights, limit=10))
        if dropped_details:
            print(
                format_key_list(
                    "  dropped_mismatched",
                    dropped_details,
                    formatter=lambda x: f"{x[0]} ({x[1]} -> {x[2]})",
                    limit=10,
                )
            )
    return matcher


def matcher_checksum(matcher: LightGlue) -> float:
    """Compute a lightweight checksum over model parameters."""
    return float(
        sum(p.detach().float().sum().item() for p in matcher.parameters())
    )


def parameter_count(matcher: LightGlue) -> int:
    """Count total parameters (all elements) in the matcher."""
    return int(sum(p.numel() for p in matcher.parameters()))


def format_key_list(
    title: str, items: list, *, formatter=lambda x: str(x), limit: int = 10
) -> str:
    """Format key lists with a capped preview to avoid huge prints."""
    if not items:
        return f"{title}: 0"
    shown = items[:limit]
    suffix = "" if len(items) <= limit else f" ... (+{len(items) - limit} more)"
    preview = ", ".join(formatter(x) for x in shown)
    return f"{title}: {len(items)} [{preview}{suffix}]"


def _extract_state_dict(obj: object) -> dict:
    """Infer a state dict from common checkpoint layouts."""
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "matcher"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise KeyError(
        "Could not find state_dict in checkpoint "
        "(checked state_dict, model_state_dict, model, matcher)."
    )


def _strip_prefix(sd: dict, prefix: str) -> dict:
    if all(k.startswith(prefix) for k in sd.keys()):
        return {k.replace(prefix, "", 1): v for k, v in sd.items()}
    return sd


def _filter_compatible_state_dict(
    state_dict: dict, reference: dict
) -> tuple[dict, list[tuple[str, tuple, tuple]]]:
    """Drop entries whose shapes do not match the reference model."""
    ref_shapes = {k: tuple(v.shape) for k, v in reference.items() if hasattr(v, "shape")}
    filtered = {}
    mismatched = []
    for key, value in state_dict.items():
        if key in ref_shapes and tuple(value.shape) != ref_shapes[key]:
            mismatched.append((key, tuple(value.shape), ref_shapes[key]))
            continue
        filtered[key] = value
    return filtered, mismatched


def run_lightglue_matches(
    matcher: LightGlue, feats0: dict, feats1: dict
) -> torch.Tensor:
    """Run LightGlue matching and return CPU match indices.

    Args:
        matcher: LightGlue matcher to run.
        feats0: Feature dictionary for the first image.
        feats1: Feature dictionary for the second image.

    Returns:
        A (K, 2) tensor of matched keypoint indices on CPU.
    """
    output = matcher({"image0": feats0, "image1": feats1})
    return lightglue_matches_to_tensor(rbd(output)["matches"])


def process_pair(
    image0_path: Path,
    image1_path: Path,
    downscale: int,
    extractor_sift: SIFT,
    matcher_left: LightGlue,
    matcher_right: LightGlue,
    cuda_dataset_root: Path,
) -> List[RowData]:
    """Compute SIFT/CudaSIFT matches for both LightGlue variants.

    Args:
        image0_path: Path to the first image.
        image1_path: Path to the second image.
        downscale: Image downscale factor applied to visualization and SIFT.
        extractor_sift: OpenCV SIFT extractor.
        matcher_left: LightGlue matcher using left weights.
        matcher_right: LightGlue matcher using right weights.
        cuda_dataset_root: Root path containing CUDA SIFT binaries and images.

    Returns:
        Row descriptors for plotting the comparison grid.
    """
    image0 = load_preprocessed_image(image0_path, downscale)
    image1 = load_preprocessed_image(image1_path, downscale)
    image0_np = image0.permute(1, 2, 0).cpu().numpy()
    image1_np = image1.permute(1, 2, 0).cpu().numpy()

    feats0_sift = extractor_sift.extract(image0)
    feats1_sift = extractor_sift.extract(image1)
    feats0_sift = harmonize_sift_scales(feats0_sift, extractor_sift.conf.backend)
    feats1_sift = harmonize_sift_scales(feats1_sift, extractor_sift.conf.backend)
    matches_sift_left = run_lightglue_matches(matcher_left, feats0_sift, feats1_sift)
    matches_sift_right = run_lightglue_matches(matcher_right, feats0_sift, feats1_sift)
    feats0_sift = detach_to_cpu(rbd(feats0_sift))
    feats1_sift = detach_to_cpu(rbd(feats1_sift))

    image_size = feats0_sift["image_size"]

    def image_to_bin_path(path: Path) -> Path:
        return path.with_name(path.stem + "_sift.bin")

    cuda_image0_path = cuda_dataset_root / image0_path.parent.name / image0_path.name
    cuda_image1_path = cuda_dataset_root / image1_path.parent.name / image1_path.name
    feats0_bin_raw = load_sift_bin(image_to_bin_path(cuda_image0_path), image_size)
    feats1_bin_raw = load_sift_bin(image_to_bin_path(cuda_image1_path), image_size)
    matches_bin_left = run_lightglue_matches(
        matcher_left, feats0_bin_raw, feats1_bin_raw
    )
    matches_bin_right = run_lightglue_matches(
        matcher_right, feats0_bin_raw, feats1_bin_raw
    )
    feats0_bin = rescale_feature_dict(detach_to_cpu(rbd(feats0_bin_raw)), downscale)
    feats1_bin = rescale_feature_dict(detach_to_cpu(rbd(feats1_bin_raw)), downscale)

    cuda_image0_np = load_preprocessed_image(cuda_image0_path, downscale)
    cuda_image0_np = cuda_image0_np.permute(1, 2, 0).cpu().numpy()
    cuda_image1_np = load_preprocessed_image(cuda_image1_path, downscale)
    cuda_image1_np = cuda_image1_np.permute(1, 2, 0).cpu().numpy()

    return [
        (
            "OpenCV SIFT",
            feats0_sift,
            feats1_sift,
            matches_sift_left,
            matches_sift_right,
            image0_np,
            image1_np,
        ),
        (
            "CudaSIFT",
            feats0_bin,
            feats1_bin,
            matches_bin_left,
            matches_bin_right,
            cuda_image0_np,
            cuda_image1_np,
        ),
    ]


def weight_label(weights_path: Optional[Path], default: str) -> str:
    """Build a short label to describe a weight path.

    Args:
        weights_path: Path or directory for the weights.
        default: Fallback label when no path is provided.

    Returns:
        A compact label suitable for plot titles.
    """
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

    dataset_root = Path(args.dataset).expanduser().resolve()
    cuda_dataset_root = Path(args.cuda_sift_dataset).expanduser().resolve()
    if not cuda_dataset_root.exists():
        print(f"CudaSIFT dataset not found: {cuda_dataset_root}", file=sys.stderr)
        return 1

    image_sets = gather_image_sets(dataset_root)

    left_weights = (
        Path(args.left_weights).expanduser() if args.left_weights else None
    )
    right_weights = (
        Path(args.right_weights).expanduser() if args.right_weights else None
    )

    matcher_kwargs_left = {
        "filter_threshold": args.lightglue_filter_threshold,
        "depth_confidence": args.lightglue_depth_confidence,
        "width_confidence": args.lightglue_width_confidence,
        "no_scale_ori": False,
        "input_dim_override": None,
        "descriptor_dim_override": None,
    }
    matcher_kwargs_right = {
        "filter_threshold": args.lightglue_filter_threshold,
        "depth_confidence": args.lightglue_depth_confidence,
        "width_confidence": args.lightglue_width_confidence,
        "no_scale_ori": args.no_scale_ori,
        "input_dim_override": args.input_dim,
        "descriptor_dim_override": args.descriptor_dim,
    }
    matcher_left = load_lightglue_with_weights(left_weights, DEVICE, **matcher_kwargs_left)
    matcher_right = load_lightglue_with_weights(right_weights, DEVICE, **matcher_kwargs_right)
    left_checksum = matcher_checksum(matcher_left)
    right_checksum = matcher_checksum(matcher_right)
    print(
        f"LightGlue checksum diff (left vs right): {left_checksum != right_checksum} "
        f"(left={left_checksum:.4f}, right={right_checksum:.4f})"
    )

    sift_kwargs = {"backend": "opencv"}
    if args.sift_max_keypoints is not None:
        sift_kwargs["max_num_keypoints"] = args.sift_max_keypoints
    if args.sift_contrast_threshold is not None:
        sift_kwargs["detection_threshold"] = args.sift_contrast_threshold
    if args.sift_edge_threshold is not None:
        sift_kwargs["edge_threshold"] = args.sift_edge_threshold
    if args.sift_n_octave_layers is not None:
        sift_kwargs["num_octaves"] = args.sift_n_octave_layers
    extractor_sift = SIFT(**sift_kwargs).eval().to(DEVICE)

    left_label = weight_label(left_weights, "official")
    right_label = weight_label(right_weights, "weights")

    base_dir = Path(args.output_dir).expanduser().resolve()
    run_name = args.run_name or f"{left_label}_vs_{right_label}"
    run_root = base_dir / run_name
    if run_root.exists():
        raise RuntimeError(
            f"Output directory '{run_root}' already exists. "
            "Choose a different --run-name."
        )
    run_root.mkdir(parents=True, exist_ok=False)

    config_snapshot = {
        "dataset": str(dataset_root),
        "cuda_sift_dataset": str(cuda_dataset_root),
        "downscale": args.downscale,
        "left_weights": str(left_weights) if left_weights else None,
        "right_weights": str(right_weights) if right_weights else None,
        "device": args.device,
        "output_dir": str(run_root),
        "lightglue_filter_threshold": args.lightglue_filter_threshold,
        "lightglue_depth_confidence": args.lightglue_depth_confidence,
        "lightglue_width_confidence": args.lightglue_width_confidence,
        "sift_max_keypoints": args.sift_max_keypoints,
        "sift_contrast_threshold": args.sift_contrast_threshold,
        "sift_edge_threshold": args.sift_edge_threshold,
        "sift_n_octave_layers": args.sift_n_octave_layers,
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
                extractor_sift,
                matcher_left,
                matcher_right,
                cuda_dataset_root,
            )

            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            for row_idx, (
                label,
                feats0_row,
                feats1_row,
                matches_left_row,
                matches_right_row,
                img0_np,
                img1_np,
            ) in enumerate(rows):
                left_img, left_pair = axes[row_idx, 0], axes[row_idx, 1]
                right_img, right_pair = axes[row_idx, 2], axes[row_idx, 3]
                prepare_axes(left_img, left_pair, img0_np, img1_np)
                prepare_axes(right_img, right_pair, img0_np, img1_np)
                plot_matches_on_axes(
                    left_img,
                    left_pair,
                    feats0_row,
                    feats1_row,
                    matches_left_row,
                    f"{label}+LG ({left_label})",
                    LG_MATCH_COLOR,
                )
                plot_matches_on_axes(
                    right_img,
                    right_pair,
                    feats0_row,
                    feats1_row,
                    matches_right_row,
                    f"{label}+LG ({right_label})",
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
