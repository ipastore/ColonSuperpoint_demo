import os
import struct
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from models import build_superpoint_model
from utils.LightGlue.lightglue import LightGlue, SIFT
from utils.LightGlue.lightglue import viz2d
from utils.LightGlue.lightglue.superpoint import (
    sample_descriptors,
    simple_nms,
    top_k_keypoints,
)
from utils.LightGlue.lightglue.utils import Extractor, load_image, match_pair, rbd
from kornia.color import rgb_to_grayscale

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SUPERPOINT_MODEL_NAME = "MagicLeap"  # e.g. "MagicLeap", "SuperpointNet", "SuperpointNet_gauss2"
SUPERPOINT_WEIGHTS_PATH = "./weights/MagicLeap/superpoint_v1.pth"  

# e.g. "./weights/magicleap_superpoint.pth"
SUPERPOINT_MAX_KEYPOINTS = 2048
SUPERPOINT_DETECTION_THRESHOLD = 0.015
SUPERPOINT_MATCH_COLOR = 'lime'
NN_MATCH_THRESHOLD = 0.7
NN_MATCH_COLOR = 'dodgerblue'
LG_MATCH_COLOR = 'lime'


def nn_match_two_way(desc1: np.ndarray, desc2: np.ndarray, nn_thresh: float) -> np.ndarray:
    """Two-way nearest-neighbor matching for L2-normalized descriptors."""
    assert desc1.shape[0] == desc2.shape[0], 'Descriptor dimensions must match.'
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
        return np.zeros((3, 0))
    if nn_thresh < 0.0:
        raise ValueError('nn_thresh should be non-negative.')
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    keep = scores < nn_thresh
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    matches = np.zeros((3, int(keep.sum())))
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
    keypoints = feats.get('keypoints')
    num_keypoints = None
    if isinstance(keypoints, torch.Tensor):
        num_keypoints = keypoints.shape[0]
    elif keypoints is not None:
        num_keypoints = np.asarray(keypoints).shape[0]
    if num_keypoints is not None and desc.shape[0] == num_keypoints:
        desc = desc.T
    return desc.astype(np.float32, copy=False)


def run_nn_matching(feats0: dict, feats1: dict, threshold: float) -> torch.Tensor:
    """Compute two-way NN matches and return indices as a tensor of shape (K, 2)."""
    desc0 = descriptors_to_matrix(feats0)
    desc1 = descriptors_to_matrix(feats1)
    if desc0.size == 0 or desc1.size == 0:
        return torch.empty((0, 2), dtype=torch.long)
    matches = nn_match_two_way(desc0, desc1, threshold)
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
    """Overlay keypoints and matches on a pair of axes."""
    keypoints0 = feats0.get('keypoints')
    keypoints1 = feats1.get('keypoints')
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
        viz2d.plot_matches(matched_kpts0, matched_kpts1, color=match_color, lw=0.2, axes=(ax_left, ax_right))

    ax_left.set_title(f"{title_prefix}: {matched_kpts0.shape[0]} matches", fontsize=10)
    ax_right.set_title("")


def load_sift_bin(path, image_size):
    with open(path, 'rb') as f:
        (N,) = struct.unpack('<I', f.read(4))
        # Each record = 2 + 128 float32s
        rec = 4 + 128  # x,y,scale,orientation + 128-d desc
        data = np.fromfile(f, dtype=np.float32, count=N*rec).reshape(N, rec)
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
        'descriptor_dim': 256,
        'nms_radius': 4,
        'max_num_keypoints': 2048,
        'detection_threshold': 0.0005,
        'remove_borders': 4,
    }
    preprocess_conf = {'resize': None}
    required_data_keys = ['image']

    def __init__(
        self,
        model_name: str,
        weights_path: str,
        device: torch.device = None,
        **conf,
    ) -> None:
        super().__init__(**conf)
        self.model_name = model_name
        self.device = torch.device(device or DEVICE)
        self.net = build_superpoint_model(model_name, weights_path, self.device)
        self.net.eval()

    @torch.no_grad()
    def forward(self, data: dict) -> dict:
        image = data['image']
        if image.device != self.device:
            image = image.to(self.device)
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
            'keypoints': torch.stack(keypoints, 0),
            'keypoint_scores': torch.stack(keypoint_scores, 0),
            'descriptors': torch.stack(descriptors, 0).transpose(-1, -2).contiguous(),
        }

extractor_sift = SIFT(
    max_num_keypoints=2048,
    backend='opencv',
    detection_threshold=0.0000667,
).eval().to(DEVICE)

matcher_sift_lg = LightGlue(
    features='sift',
    depth_confidence=-1,
    width_confidence=-1,
).eval().to(DEVICE)

superpoint_extractor = None
matcher_superpoint_lg = None
if SUPERPOINT_MODEL_NAME and SUPERPOINT_WEIGHTS_PATH:
    superpoint_extractor = SuperPointFromWeights(
        SUPERPOINT_MODEL_NAME,
        SUPERPOINT_WEIGHTS_PATH,
        device=DEVICE,
        max_num_keypoints=SUPERPOINT_MAX_KEYPOINTS,
        detection_threshold=SUPERPOINT_DETECTION_THRESHOLD,
    ).eval()
    matcher_superpoint_lg = LightGlue(
        features='superpoint',
        depth_confidence=-1,
        width_confidence=-1,
    ).eval().to(DEVICE)

matching_folder = './assets/matching/examples'
output_folder = './matching_outputs'
superpoint_folder_name = (
    f"{SUPERPOINT_MODEL_NAME}_{Path(SUPERPOINT_WEIGHTS_PATH).stem}_{SUPERPOINT_DETECTION_THRESHOLD}"
    if SUPERPOINT_MODEL_NAME and SUPERPOINT_WEIGHTS_PATH
    else 'no_superpoint'
)
output_root = os.path.join(output_folder, superpoint_folder_name)
os.makedirs(output_root, exist_ok=True)


def is_image_file(name):
    ext = os.path.splitext(name)[1].lower()
    return ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# iterate over subfolders and match all pairs per subfolder
for entry in os.scandir(matching_folder):
    if not entry.is_dir():
        continue

    subfolder = entry.path
    images = [
        os.path.join(subfolder, f.name)
        for f in os.scandir(subfolder)
        if f.is_file() and is_image_file(f.name)
    ]
    images.sort()
    if len(images) < 2:
        continue

    out_subfolder = os.path.join(output_root, os.path.basename(subfolder))
    os.makedirs(out_subfolder, exist_ok=True)

    for img_path0, img_path1 in combinations(images, 2):
        image0 = load_image(img_path0).to(DEVICE)
        image1 = load_image(img_path1).to(DEVICE)

        print(f"Matching {img_path0} and {img_path1}")

        image0_np = image0.permute(1, 2, 0).cpu().numpy()
        image1_np = image1.permute(1, 2, 0).cpu().numpy()

        feats0_sift, feats1_sift, matches_sift_lg_batch = match_pair(extractor_sift, matcher_sift_lg, image0, image1)
        feats0_sift = detach_to_cpu(feats0_sift)
        feats1_sift = detach_to_cpu(feats1_sift)
        matches_sift_lg = lightglue_matches_to_tensor(matches_sift_lg_batch['matches'])
        matches_sift_nn = run_nn_matching(feats0_sift, feats1_sift, NN_MATCH_THRESHOLD)

        def image_to_bin_path(img_path: str) -> str:
            base, _ = os.path.splitext(img_path)
            return f"{base}_sift.bin"

        image_size = feats0_sift['image_size']
        feats0_bin_raw = load_sift_bin(image_to_bin_path(img_path0), image_size)
        feats1_bin_raw = load_sift_bin(image_to_bin_path(img_path1), image_size)
        matches_bin_lg_batch = matcher_sift_lg({"image0": feats0_bin_raw, "image1": feats1_bin_raw})
        feats0_bin = detach_to_cpu(rbd(feats0_bin_raw))
        feats1_bin = detach_to_cpu(rbd(feats1_bin_raw))
        matches_bin_lg = lightglue_matches_to_tensor(rbd(matches_bin_lg_batch)['matches'])
        matches_bin_nn = run_nn_matching(feats0_bin, feats1_bin, NN_MATCH_THRESHOLD)

        superpoint_row = None
        if superpoint_extractor is not None:
            feats0_sp_lg, feats1_sp_lg, matches_sp_lg_batch = match_pair(
                superpoint_extractor,
                matcher_superpoint_lg,
                image0,
                image1,
            )
            feats0_sp = detach_to_cpu(feats0_sp_lg)
            feats1_sp = detach_to_cpu(feats1_sp_lg)
            matches_sp_lg = lightglue_matches_to_tensor(matches_sp_lg_batch['matches'])
            matches_sp_nn = run_nn_matching(feats0_sp, feats1_sp, NN_MATCH_THRESHOLD)
            superpoint_row = (feats0_sp, feats1_sp, matches_sp_nn, matches_sp_lg)

        rows = [
            ("SIFT", feats0_sift, feats1_sift, matches_sift_nn, matches_sift_lg, LG_MATCH_COLOR),
            ("CudaSIFT", feats0_bin, feats1_bin, matches_bin_nn, matches_bin_lg, LG_MATCH_COLOR),
        ]
        if superpoint_row is not None:
            feats0_sp, feats1_sp, matches_sp_nn, matches_sp_lg = superpoint_row
            rows.append((SUPERPOINT_MODEL_NAME or "SuperPoint", feats0_sp, feats1_sp, matches_sp_nn, matches_sp_lg, SUPERPOINT_MATCH_COLOR))

        fig, axes = plt.subplots(len(rows), 4, figsize=(16, 4 * len(rows)))
        axes = np.atleast_2d(axes)

        for row_idx, (label, feats0_row, feats1_row, matches_nn_row, matches_lg_row, lg_color) in enumerate(rows):
            nn_left, nn_right = axes[row_idx, 0], axes[row_idx, 1]
            lg_left, lg_right = axes[row_idx, 2], axes[row_idx, 3]
            prepare_axes(nn_left, nn_right, image0_np, image1_np)
            prepare_axes(lg_left, lg_right, image0_np, image1_np)
            plot_matches_on_axes(nn_left, nn_right, feats0_row, feats1_row, matches_nn_row, f"{label}+NN", NN_MATCH_COLOR)
            plot_matches_on_axes(lg_left, lg_right, feats0_row, feats1_row, matches_lg_row, f"{label}+LG", lg_color)

        fig.tight_layout(pad=0.5)

        base0 = os.path.splitext(os.path.basename(img_path0))[0]
        base1 = os.path.splitext(os.path.basename(img_path1))[0]
        out_path = os.path.join(out_subfolder, f"{base0}__{base1}.png")
        viz2d.save_plot(out_path)
        print(f"Saved matches to {out_path}")
        plt.close(fig)
