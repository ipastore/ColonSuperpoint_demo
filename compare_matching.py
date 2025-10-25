"""CLI tool to compare SIFT/CudaSIFT and SuperPoint matches using LightGlue and NN."""

from __future__ import annotations

import argparse
import itertools
import struct
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from kornia.color import rgb_to_grayscale

from models import SUPERPOINT_MODEL_CHOICES, build_superpoint_model
from utils.LightGlue.lightglue import LightGlue, SIFT
from utils.LightGlue.lightglue import viz2d
from utils.LightGlue.lightglue.superpoint import (
    sample_descriptors,
    simple_nms,
    top_k_keypoints,
)
from utils.LightGlue.lightglue.utils import (
    Extractor,
    load_image,
    match_pair,
    rbd,
    read_image,
    resize_image,
    numpy_image_to_torch,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NN_MATCH_COLOR = "dodgerblue"
LG_MATCH_COLOR = "lime"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def nn_match_two_way(
    desc1: np.ndarray,
    desc2: np.ndarray,
    nn_thresh: float,
    ratio_thresh: Optional[float] = None,
) -> np.ndarray:
    """Two-way nearest-neighbour matching with optional Lowe ratio filtering."""

    assert desc1.shape[0] == desc2.shape[0], "Descriptor dimensions must match."
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
        return np.zeros((3, 0))
    if nn_thresh < 0.0:
        raise ValueError("nn_thresh should be non-negative")

    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    keep = scores < nn_thresh
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)

    if ratio_thresh is not None and ratio_thresh < 1.0 and desc2.shape[1] > 1:
        second_best = np.partition(dmat, 1, axis=1)[:, 1]
        second_best = np.clip(second_best, 1e-12, None)
        ratios = scores / second_best
        keep = np.logical_and(keep, ratios <= ratio_thresh)

    idx = idx[keep]
    scores = scores[keep]
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    matches = np.zeros((3, int(keep.sum())), dtype=np.float32)
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches


def detach_to_cpu(data: dict) -> dict:
    """Detach tensor values to CPU for downstream numpy/plotting usage."""
    return {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
    }


def descriptors_to_matrix(feats: dict) -> np.ndarray:
    """Return descriptors as a (D, N) float32 array for NN matching."""
    desc = feats.get('descriptors')
    if desc is None:
        return np.zeros((0, 0), dtype=np.float32)
    if isinstance(desc, torch.Tensor):
        desc = desc.detach().cpu().numpy()
    else:
        desc = np.asarray(desc)
    if desc.ndim != 2:
        return np.zeros((0, 0), dtype=np.float32)

    keypoints = feats.get("keypoints")
    num_keypoints = None
    if isinstance(keypoints, torch.Tensor):
        num_keypoints = keypoints.shape[0]
    elif keypoints is not None:
        num_keypoints = np.asarray(keypoints).shape[0]
    if num_keypoints is not None and desc.shape[0] == num_keypoints:
        desc = desc.T
    return desc.astype(np.float32, copy=False)


def run_nn_matching(
    feats0: dict, feats1: dict, threshold: float, ratio_thresh: Optional[float]
) -> torch.Tensor:
    """Compute two-way NN matches with optional ratio_thresh and return indices as a tensor of shape (K, 2)."""
    desc0 = descriptors_to_matrix(feats0)
    desc1 = descriptors_to_matrix(feats1)
    if desc0.size == 0 or desc1.size == 0:
        return torch.empty((0, 2), dtype=torch.long)
    matches = nn_match_two_way(desc0, desc1, threshold, ratio_thresh)
    if matches.shape[1] == 0:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.from_numpy(matches[:2].T.astype(np.int64))


def prepare_axes(ax_left, ax_right, image0_np: np.ndarray, image1_np: np.ndarray) -> None:
    """Display images on the provided axes and hide ticks/borders."""
    ax_left.imshow(image0_np)
    ax_right.imshow(image1_np)
    for ax in (ax_left, ax_right):
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])
        ax.set_axis_off()
        for spine in ax.spines.values():
            spine.set_visible(False)


def lightglue_matches_to_tensor(matches: torch.Tensor) -> torch.Tensor:
    """Convert LightGlue match indices to a CPU tensor of shape (K, 2)."""
    if matches is None:
        return torch.empty((0, 2), dtype=torch.long)
    matches = matches.detach().cpu()
    if matches.ndim == 0:
        return torch.empty((0, 2), dtype=torch.long)
    if matches.ndim == 1:
        matches = matches.unsqueeze(0)
    return matches.long()


def plot_matches_on_axes(
    ax_left,
    ax_right,
    feats0: dict,
    feats1: dict,
    matches: torch.Tensor,
    title_prefix: str,
    match_color: str,
) -> None:
    keypoints0 = feats0.get("keypoints")
    keypoints1 = feats1.get("keypoints")
    if keypoints0 is None or keypoints1 is None:
        ax_left.set_title(f"{title_prefix}: 0 matches", fontsize=10)
        ax_right.set_title("")
        return
    if isinstance(keypoints0, torch.Tensor):
        keypoints0 = keypoints0.cpu()
    if isinstance(keypoints1, torch.Tensor):
        keypoints1 = keypoints1.cpu()

    matches = matches.cpu()
    if matches.ndim == 1:
        matches = matches.unsqueeze(0)

    n0 = keypoints0.shape[0]
    n1 = keypoints1.shape[0]
    matched0 = torch.zeros(n0, dtype=torch.bool)
    matched1 = torch.zeros(n1, dtype=torch.bool)

    if matches.numel() > 0:
        idx0 = matches[:, 0].long().clamp(max=max(n0 - 1, 0))
        idx1 = matches[:, 1].long().clamp(max=max(n1 - 1, 0))
        matched0[idx0] = True
        matched1[idx1] = True
        matched_kpts0 = keypoints0[idx0]
        matched_kpts1 = keypoints1[idx1]
    else:
        matched_kpts0 = torch.empty((0, 2))
        matched_kpts1 = torch.empty((0, 2))

    nm_kpts0 = keypoints0[~matched0]
    nm_kpts1 = keypoints1[~matched1]

    viz2d.plot_keypoints([nm_kpts0], colors="yellow", ps=2, axes=[ax_left])
    viz2d.plot_keypoints([nm_kpts1], colors="yellow", ps=2, axes=[ax_right])
    if matched_kpts0.numel() > 0:
        viz2d.plot_matches(
            matched_kpts0, matched_kpts1, color=match_color, lw=0.2, axes=(ax_left, ax_right)
        )

    kp_left = keypoints0.shape[0]
    kp_right = keypoints1.shape[0]

    ax_left.set_title(f"{title_prefix}: {matched_kpts0.shape[0]} matches", fontsize=10)
    ax_left.text(
        0.01,
        0.99,
        f"kpts: {kp_left}",
        transform=ax_left.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="white",
        bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=2),
    )
    ax_right.set_title("")
    ax_right.text(
        0.99,
        0.99,
        f"kpts: {kp_right}",
        transform=ax_right.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="white",
        bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=2),
    )


def load_sift_bin(path, image_size):
    with open(path, "rb") as f:
        (N,) = struct.unpack("<I", f.read(4))
        rec = 4 + 128
        data = np.fromfile(f, dtype=np.float32, count=N * rec).reshape(N, rec)
    keypoints = (
        torch.from_numpy(data[:, :2].copy()).to(dtype=torch.float32).unsqueeze(0).to(DEVICE)
    )  # (1, N, 2)
    scales = (
        torch.from_numpy(data[:, 2].copy()).to(dtype=torch.float32).unsqueeze(0).to(DEVICE)
    )  # (1, N)
    # convert orientations from degrees to radians using numpy, then to torch
    oris_np = np.deg2rad(data[:, 3].copy())
    oris = (
        torch.from_numpy(oris_np).to(dtype=torch.float32).unsqueeze(0).to(DEVICE)
    )  # (1, N)
    descriptors = (
        torch.from_numpy(data[:, 4:].copy()).to(dtype=torch.float32).unsqueeze(0).to(DEVICE)
    )  # (1, N, 128)
    # Normalize image_size to (1, 2)
    if isinstance(image_size, torch.Tensor):
        image_size_t = image_size.to(dtype=torch.float32)
        if image_size_t.dim() == 1:
            image_size_t = image_size_t.unsqueeze(0)
    else:
        image_size_t = torch.tensor(image_size, dtype=torch.float32).unsqueeze(0)
    image_size_t = image_size_t.to(DEVICE)
    features = {
        'keypoints': keypoints,
        'scales': scales,
        'oris': oris,
        'descriptors': descriptors,
        'image_size': image_size_t,
    }
    return features


class SuperPointFromWeights(Extractor):
    """Minimal LightGlue-compatible extractor wrapping repository SuperPoint models."""

    default_conf = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "max_num_keypoints": 2048,
        "detection_threshold": 0.0005,
        "remove_borders": 4,
    }
    preprocess_conf = {"resize": 1024}
    required_data_keys = ["image"]

    def __init__(self, model_name: str, weights_path: str, device: torch.device, **conf) -> None:
        super().__init__(**conf)
        self.model_name = model_name
        self.device = torch.device(device)
        self.net = build_superpoint_model(model_name, weights_path, self.device)
        self.net.eval()

    @torch.no_grad()
    def forward(self, data: dict) -> dict:
        image = data["image"].to(self.device)
        if image.shape[1] == 3:
            image = rgb_to_grayscale(image)

        logits, dense_descriptors = self.net(image)
        scores = torch.softmax(logits, dim=1)[:, :-1]
        batch, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(batch, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(batch, h * 8, w * 8)
        scores = simple_nms(scores, self.conf.nms_radius)

        if self.conf.remove_borders:
            pad = self.conf.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1

        keypoints = []
        keypoint_scores = []
        for i in range(batch):
            ys, xs = torch.where(scores[i] > self.conf.detection_threshold)
            if ys.numel() == 0:
                keypoints.append(scores.new_zeros((0, 2)))
                keypoint_scores.append(scores.new_zeros((0,)))
                continue
            kp = torch.stack([ys, xs], dim=-1)
            sc = scores[i, ys, xs]
            if self.conf.max_num_keypoints is not None:
                kp, sc = top_k_keypoints(kp, sc, self.conf.max_num_keypoints)
            keypoints.append(kp.float())
            keypoint_scores.append(sc)

        keypoints = [torch.flip(k, [1]) for k in keypoints]
        dense_descriptors = F.normalize(dense_descriptors, p=2, dim=1, eps=1e-8)
        descriptors = [
            sample_descriptors(k[None], dense_descriptors[i][None], 8)[0]
            for i, k in enumerate(keypoints)
        ]

        return {
            "keypoints": torch.stack(keypoints, 0),
            "keypoint_scores": torch.stack(keypoint_scores, 0),
            "descriptors": torch.stack(descriptors, 0).transpose(-1, -2).contiguous(),
        }


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=str,
        help="YAML file with default options (keys must match CLI flags).",
    )

    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(parents=[config_parser], description=__doc__)
    parser.add_argument("--dataset", type=str, default=None, help="Path to the dataset root")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory for comparison figures",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name of the run/output subfolder",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        choices=[1, 2, 4, 8],
        default=None,
        help="Image downscale factor (1 leaves original size)",
    )
    parser.add_argument("--lightglue-filter-threshold", type=float, default=None)
    parser.add_argument("--lightglue-depth-confidence", type=float, default=None)
    parser.add_argument("--lightglue-width-confidence", type=float, default=None)
    parser.add_argument("--sift-lowe-thresh", type=float, default=None)
    parser.add_argument("--sift-max-keypoints", type=int, default=None)
    parser.add_argument("--sift-contrast-threshold", type=float, default=None)
    parser.add_argument("--sift-edge-threshold", type=float, default=None)
    parser.add_argument("--sift-n-octave-layers", type=int, default=None)
    parser.add_argument("--nn-match-threshold", type=float, default=None)
    parser.add_argument(
        "--superpoint-model-name",
        type=str,
        default=None,
        choices=SUPERPOINT_MODEL_CHOICES,
    )
    parser.add_argument("--superpoint-weights-path", type=str, default=None)
    parser.add_argument("--superpoint-max-keypoints", type=int, default=None)
    parser.add_argument("--superpoint-detection-threshold", type=float, default=None)
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Computation device",
    )

    config_args, remaining = config_parser.parse_known_args()
    config_data = {}
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

    missing = [
        name
        for name in [
            "dataset",
            "downscale",
            "lightglue_filter_threshold",
            "lightglue_depth_confidence",
            "lightglue_width_confidence",
            "sift_lowe_thresh",
            "sift_max_keypoints",
            "sift_contrast_threshold",
            "sift_edge_threshold",
            "sift_n_octave_layers",
            "nn_match_threshold",
            "superpoint_model_name",
            "superpoint_weights_path",
            "superpoint_max_keypoints",
            "superpoint_detection_threshold",
        ]
        if getattr(opt, name, None) is None
    ]
    if missing:
        parser.error(
            "Missing required options (provide via CLI or config): "
            + ", ".join(missing)
        )

    return opt


def gather_image_sets(root: Path) -> List[Tuple[Path, List[Path]]]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset not found: {root}")

    def image_list(folder: Path) -> List[Path]:
        return sorted(
            [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        )

    primary_images = image_list(root)
    if len(primary_images) >= 2:
        return [(root, primary_images)]

    image_sets: List[Tuple[Path, List[Path]]] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        images = image_list(sub)
        if len(images) >= 2:
            image_sets.append((sub, images))
    if not image_sets:
        raise RuntimeError(f"No valid image folders found under {root}")
    return image_sets


def load_preprocessed_image(path: Path, downscale: int) -> torch.Tensor:
    if downscale <= 1:
        return load_image(str(path)).to(DEVICE)

    image = read_image(str(path))
    h, w = image.shape[:2]
    target_h = max(1, int(round(h / downscale)))
    target_w = max(1, int(round(w / downscale)))
    resized, _ = resize_image(image, (target_h, target_w))
    return numpy_image_to_torch(resized).to(DEVICE)


def rescale_feature_dict(feats: dict, factor: int) -> dict:
    if factor == 1:
        return feats
    scale = 1.0 / factor
    if "keypoints" in feats and isinstance(feats["keypoints"], torch.Tensor):
        feats["keypoints"] = feats["keypoints"] * scale
    if "image_size" in feats and isinstance(feats["image_size"], torch.Tensor):
        feats["image_size"] = feats["image_size"] * scale
    return feats


def process_pair(
    image0_path: Path,
    image1_path: Path,
    downscale: int,
    extractor_sift: SIFT,
    matcher_sift_lg: LightGlue,
    superpoint_extractor: Optional[SuperPointFromWeights],
    matcher_superpoint_lg: Optional[LightGlue],
    nn_threshold: float,
    ratio_thresh: Optional[float],
    superpoint_label: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, dict, dict, torch.Tensor, torch.Tensor, str]]]:
    image0 = load_preprocessed_image(image0_path, downscale)
    image1 = load_preprocessed_image(image1_path, downscale)
    image0_np = image0.permute(1, 2, 0).cpu().numpy()
    image1_np = image1.permute(1, 2, 0).cpu().numpy()

    feats0_sift, feats1_sift, matches_sift_lg_batch = match_pair(extractor_sift, matcher_sift_lg, image0, image1)
    feats0_sift = detach_to_cpu(feats0_sift)
    feats1_sift = detach_to_cpu(feats1_sift)
    matches_sift_lg = lightglue_matches_to_tensor(matches_sift_lg_batch["matches"])
    matches_sift_nn = run_nn_matching(feats0_sift, feats1_sift, nn_threshold, ratio_thresh)

    def image_to_bin_path(path: Path) -> Path:
        return path.with_name(path.stem + "_sift.bin")

    image_size = feats0_sift["image_size"]
    feats0_bin_raw = load_sift_bin(image_to_bin_path(image0_path), image_size)
    feats1_bin_raw = load_sift_bin(image_to_bin_path(image1_path), image_size)
    matches_bin_lg_batch = matcher_sift_lg({"image0": feats0_bin_raw, "image1": feats1_bin_raw})
    feats0_bin = detach_to_cpu(rbd(feats0_bin_raw))
    feats1_bin = detach_to_cpu(rbd(feats1_bin_raw))
    feats0_bin = rescale_feature_dict(feats0_bin, downscale)
    feats1_bin = rescale_feature_dict(feats1_bin, downscale)
    matches_bin_lg = lightglue_matches_to_tensor(rbd(matches_bin_lg_batch)["matches"])
    matches_bin_nn = run_nn_matching(feats0_bin, feats1_bin, nn_threshold, ratio_thresh)

    superpoint_row = None
    if superpoint_extractor is not None and matcher_superpoint_lg is not None:
        feats0_sp, feats1_sp, matches_sp_lg_batch = match_pair(
            superpoint_extractor,
            matcher_superpoint_lg,
            image0,
            image1,
        )
        feats0_sp = detach_to_cpu(feats0_sp)
        feats1_sp = detach_to_cpu(feats1_sp)
        matches_sp_lg = lightglue_matches_to_tensor(matches_sp_lg_batch["matches"])
        matches_sp_nn = run_nn_matching(feats0_sp, feats1_sp, nn_threshold, ratio_thresh)
        superpoint_row = (feats0_sp, feats1_sp, matches_sp_nn, matches_sp_lg)

    rows = [
        ("SIFT", feats0_sift, feats1_sift, matches_sift_nn, matches_sift_lg),
        ("CudaSIFT", feats0_bin, feats1_bin, matches_bin_nn, matches_bin_lg),
    ]
    if superpoint_row is not None:
        feats0_sp, feats1_sp, matches_sp_nn, matches_sp_lg = superpoint_row
        rows.append(
            (
                superpoint_label or "SuperPoint",
                feats0_sp,
                feats1_sp,
                matches_sp_nn,
                matches_sp_lg,
            )
        )

    return image0_np, image1_np, rows


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available.", file=sys.stderr)
        return 1

    global DEVICE
    DEVICE = torch.device(args.device)
    torch.set_grad_enabled(False)

    dataset_root = Path(args.dataset).resolve()
    image_sets = gather_image_sets(dataset_root)

    weights_path = Path(args.superpoint_weights_path).expanduser()
    dataset_token = dataset_root.name
    weights_token = weights_path.stem if weights_path.name else "weights"
    base_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path("./matching_outputs").resolve()
    )
    if args.run_name:
        run_token = args.run_name
        run_root = base_dir / (
            f"{dataset_token}_{args.superpoint_model_name}_{weights_token}_spth{args.superpoint_detection_threshold}_{run_token}"
        )
    else:
        run_root = base_dir / (
            f"{dataset_token}_{args.superpoint_model_name}_{weights_token}_spth{args.superpoint_detection_threshold}"
        )
    if run_root.exists():
        raise RuntimeError(
            f"Output directory '{run_root}' already exists. Please choose a different --run-name."
        )
    run_root.mkdir(parents=True, exist_ok=False)

    config_snapshot = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    config_snapshot.update(
        {
            "dataset_resolved": str(dataset_root),
            "superpoint_weights_resolved": str(weights_path),
            "run_directory": str(run_root),
        }
    )
    snapshot_path = run_root / "config_used.yaml"
    snapshot_path.write_text(yaml.safe_dump(config_snapshot, sort_keys=True))

    extractor_sift = SIFT(
        max_num_keypoints=args.sift_max_keypoints,
        detection_threshold=args.sift_contrast_threshold,
        edge_threshold=args.sift_edge_threshold,
        num_octaves=args.sift_n_octave_layers,
        backend="opencv",
    ).eval().to(DEVICE)
    matcher_sift_lg = LightGlue(
        features="sift",
        filter_threshold=args.lightglue_filter_threshold,
        depth_confidence=args.lightglue_depth_confidence,
        width_confidence=args.lightglue_width_confidence,
    ).eval().to(DEVICE)

    superpoint_extractor = None
    matcher_superpoint_lg = None
    if args.superpoint_weights_path:
        if not weights_path.exists():
            print(f"SuperPoint weights not found: {weights_path}", file=sys.stderr)
            return 1
        superpoint_extractor = SuperPointFromWeights(
            args.superpoint_model_name,
            str(weights_path),
            device=DEVICE,
            max_num_keypoints=args.superpoint_max_keypoints,
            detection_threshold=args.superpoint_detection_threshold,
        ).eval()
        matcher_superpoint_lg = LightGlue(
            features="superpoint",
            filter_threshold=args.lightglue_filter_threshold,
            depth_confidence=args.lightglue_depth_confidence,
            width_confidence=args.lightglue_width_confidence,
        ).eval().to(DEVICE)

    for folder, images in image_sets:
        out_subfolder = run_root / folder.name
        out_subfolder.mkdir(parents=True, exist_ok=True)
        print(f"Processing {folder} -> {out_subfolder}")

        for img_path0, img_path1 in itertools.combinations(images, 2):
            image0_np, image1_np, rows = process_pair(
                img_path0,
                img_path1,
                args.downscale,
                extractor_sift,
                matcher_sift_lg,
                superpoint_extractor,
                matcher_superpoint_lg,
                args.nn_match_threshold,
                args.sift_lowe_thresh,
                args.superpoint_model_name if superpoint_extractor is not None else None,
            )

            rows_nn_counts = [matches_nn.shape[0] for _, _, _, matches_nn, _ in rows]
            rows_lg_counts = [matches_lg.shape[0] for _, _, _, _, matches_lg in rows]
            print(
                f"  Pair {img_path0.name} vs {img_path1.name}: "
                f"NN matches {rows_nn_counts}, LG matches {rows_lg_counts}"
            )

            fig, axes = plt.subplots(len(rows), 4, figsize=(16, 4 * len(rows)))
            axes = np.atleast_2d(axes)

            for row_idx, (label, feats0_row, feats1_row, matches_nn_row, matches_lg_row) in enumerate(rows):
                nn_left, nn_right = axes[row_idx, 0], axes[row_idx, 1]
                lg_left, lg_right = axes[row_idx, 2], axes[row_idx, 3]
                prepare_axes(nn_left, nn_right, image0_np, image1_np)
                prepare_axes(lg_left, lg_right, image0_np, image1_np)
                plot_matches_on_axes(
                    nn_left,
                    nn_right,
                    feats0_row,
                    feats1_row,
                    matches_nn_row,
                    f"{label}+NN",
                    NN_MATCH_COLOR,
                )
                plot_matches_on_axes(
                    lg_left,
                    lg_right,
                    feats0_row,
                    feats1_row,
                    matches_lg_row,
                    f"{label}+LG",
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
