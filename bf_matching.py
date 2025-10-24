import os
import struct
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

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
MATCHER = 'lg'  # Options: 'lg' (LightGlue) or 'nn' (two-way nearest neighbor)
SUPERPOINT_MODEL_NAME = "MagicLeap"  # e.g. "MagicLeap", "SuperpointNet", "SuperpointNet_gauss2"
SUPERPOINT_WEIGHTS_PATH = "./weights/MagicLeap/superpoint_v1.pth"  

# e.g. "./weights/magicleap_superpoint.pth"
SUPERPOINT_MAX_KEYPOINTS = 2048
SUPERPOINT_DETECTION_THRESHOLD = 0.015
SUPERPOINT_MATCH_COLOR = 'lime'
NN_MATCH_THRESHOLD = 0.7


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


def maybe_rescale_keypoints(feats: dict, target_size) -> dict:
    """Rescale keypoints to match the displayed image dimensions if needed."""
    keypoints = feats.get('keypoints')
    image_size = feats.get('image_size')
    if keypoints is None or image_size is None or not isinstance(keypoints, torch.Tensor):
        return feats

    if isinstance(image_size, torch.Tensor):
        size = image_size.view(-1).float().cpu()
    else:
        size = torch.tensor(image_size, dtype=torch.float32)
    if size.numel() < 2:
        return feats

    src_w, src_h = float(size[0]), float(size[1])
    dst_w, dst_h = map(float, target_size)
    if abs(src_w - dst_w) < 1e-3 and abs(src_h - dst_h) < 1e-3:
        return feats

    scale = torch.tensor([dst_w / src_w, dst_h / src_h], dtype=keypoints.dtype, device=keypoints.device)
    feats['keypoints'] = keypoints * scale
    feats['image_size'] = torch.tensor([dst_w, dst_h], dtype=scale.dtype)
    return feats


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
            kp = torch.stack([xs, ys], dim=-1)
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

use_lightglue = MATCHER.lower() == 'lg'
matcher_sift = None
if use_lightglue:
    matcher_sift = LightGlue(
        features='sift',
        depth_confidence=-1,
        width_confidence=-1,
    ).eval().to(DEVICE)

superpoint_extractor = None
matcher_superpoint = None
if SUPERPOINT_MODEL_NAME and SUPERPOINT_WEIGHTS_PATH:
    superpoint_extractor = SuperPointFromWeights(
        SUPERPOINT_MODEL_NAME,
        SUPERPOINT_WEIGHTS_PATH,
        device=DEVICE,
        max_num_keypoints=SUPERPOINT_MAX_KEYPOINTS,
        detection_threshold=SUPERPOINT_DETECTION_THRESHOLD,
    ).eval()
    if use_lightglue:
        matcher_superpoint = LightGlue(
            features='superpoint',
            depth_confidence=-1,
            width_confidence=-1,
        ).eval().to(DEVICE)

matching_folder = './assets/matching/examples'
output_folder = './matching_outputs'

def is_image_file(name):
    ext = os.path.splitext(name)[1].lower()
    return ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# iterate over subfolders and match all pairs per subfolder
for entry in os.scandir(matching_folder):
    if not entry.is_dir():
        continue

    subfolder = entry.path
    # collect images directly under this subfolder (non-recursive)
    images = [os.path.join(subfolder, f.name) for f in os.scandir(subfolder) if f.is_file() and is_image_file(f.name)]
    images.sort()
    if len(images) < 2:
        continue

    # ensure output subfolder exists
    out_subfolder = os.path.join(output_folder, os.path.basename(subfolder))
    os.makedirs(out_subfolder, exist_ok=True)

    for img_path0, img_path1 in combinations(images, 2):
        # load images
        image0 = load_image(img_path0).to(DEVICE)
        image1 = load_image(img_path1).to(DEVICE)

        print(f"Matching {img_path0} and {img_path1}")

        matcher_label = 'LG' if use_lightglue else 'NN'

        # compute matches A->B direction (SIFT)
        if use_lightglue and matcher_sift is not None:
            feats0_sift, feats1_sift, matches01 = match_pair(extractor_sift, matcher_sift, image0, image1)
            sift_feats0 = detach_to_cpu(feats0_sift)
            sift_feats1 = detach_to_cpu(feats1_sift)
            sift_matches = matches01['matches'].cpu().long()
        else:
            feats0_raw = extractor_sift.extract(image0)
            feats1_raw = extractor_sift.extract(image1)
            sift_feats0 = detach_to_cpu(rbd(feats0_raw))
            sift_feats1 = detach_to_cpu(rbd(feats1_raw))
            desc0 = sift_feats0['descriptors'].T.numpy()
            desc1 = sift_feats1['descriptors'].T.numpy()
            matches_np = nn_match_two_way(desc0, desc1, NN_MATCH_THRESHOLD)
            if matches_np.shape[1]:
                sift_matches = torch.from_numpy(matches_np[:2].T.astype(np.int64))
            else:
                sift_matches = torch.empty((0, 2), dtype=torch.long)

        if sift_matches.ndim == 1:
            sift_matches = sift_matches.unsqueeze(0)

        target_size_left = (image0.shape[-1], image0.shape[-2])
        target_size_right = (image1.shape[-1], image1.shape[-2])

        # compute matches from binary data (CudaSIFT)
        def image_to_bin_path(img_path):
            base, ext = os.path.splitext(img_path)
            return f"{base}_sift.bin"

        image_size = sift_feats0['image_size']
        img0_bin = image_to_bin_path(img_path0)
        img1_bin = image_to_bin_path(img_path1)
        feats0_bin = load_sift_bin(img0_bin, image_size)
        feats1_bin = load_sift_bin(img1_bin, image_size)

        if use_lightglue and matcher_sift is not None:
            matches01_bin = matcher_sift({"image0": feats0_bin, "image1": feats1_bin})
            feats0_bin, feats1_bin, matches01_bin = [rbd(x) for x in [feats0_bin, feats1_bin, matches01_bin]]
            feats0_bin = detach_to_cpu(feats0_bin)
            feats1_bin = detach_to_cpu(feats1_bin)
            bin_matches = matches01_bin['matches'].cpu().long()
        else:
            feats0_bin = detach_to_cpu(rbd(feats0_bin))
            feats1_bin = detach_to_cpu(rbd(feats1_bin))
            desc0_bin = feats0_bin['descriptors'].T.numpy()
            desc1_bin = feats1_bin['descriptors'].T.numpy()
            matches_np_bin = nn_match_two_way(desc0_bin, desc1_bin, NN_MATCH_THRESHOLD)
            if matches_np_bin.shape[1]:
                bin_matches = torch.from_numpy(matches_np_bin[:2].T.astype(np.int64))
            else:
                bin_matches = torch.empty((0, 2), dtype=torch.long)

        if bin_matches.ndim == 1:
            bin_matches = bin_matches.unsqueeze(0)


        sp_results = None
        if superpoint_extractor is not None:
            if use_lightglue and matcher_superpoint is not None:
                feats0_sp, feats1_sp, matches01_sp = match_pair(
                    superpoint_extractor,
                    matcher_superpoint,
                    image0,
                    image1,
                )
                feats0_sp = detach_to_cpu(feats0_sp)
                feats1_sp = detach_to_cpu(feats1_sp)
                sp_matches = matches01_sp['matches'].cpu().long()
            else:
                feats0_raw_sp = superpoint_extractor.extract(image0)
                feats1_raw_sp = superpoint_extractor.extract(image1)
                feats0_sp = detach_to_cpu(rbd(feats0_raw_sp))
                feats1_sp = detach_to_cpu(rbd(feats1_raw_sp))
                desc0_sp = feats0_sp['descriptors'].T.numpy()
                desc1_sp = feats1_sp['descriptors'].T.numpy()
                matches_np_sp = nn_match_two_way(desc0_sp, desc1_sp, NN_MATCH_THRESHOLD)
                if matches_np_sp.shape[1]:
                    sp_matches = torch.from_numpy(matches_np_sp[:2].T.astype(np.int64))
                else:
                    sp_matches = torch.empty((0, 2), dtype=torch.long)

            if sp_matches.ndim == 1:
                sp_matches = sp_matches.unsqueeze(0)
            maybe_rescale_keypoints(feats0_sp, target_size_left)
            maybe_rescale_keypoints(feats1_sp, target_size_right)
            sp_results = (feats0_sp, feats1_sp, sp_matches)

        num_rows = 3 if sp_results is not None else 2
        fig, axes = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows))
        axes = np.atleast_2d(axes)

        def prep_axes(row_idx):
            left, right = axes[row_idx, 0], axes[row_idx, 1]
            left.imshow(image0.permute(1, 2, 0).cpu().numpy())
            right.imshow(image1.permute(1, 2, 0).cpu().numpy())
            for ax_ in (left, right):
                ax_.get_yaxis().set_ticks([])
                ax_.get_xaxis().set_ticks([])
                ax_.set_axis_off()
                for spine in ax_.spines.values():
                    spine.set_visible(False)
            return left, right

        # Row 0: SIFT+LG
        ax_left, ax_right = prep_axes(0)
        n0 = sift_feats0['keypoints'].shape[0]
        n1 = sift_feats1['keypoints'].shape[0]
        matched0 = torch.zeros(n0, dtype=torch.bool)
        matched1 = torch.zeros(n1, dtype=torch.bool)
        if sift_matches.numel() > 0:
            matched0[sift_matches[..., 0]] = True
            matched1[sift_matches[..., 1]] = True
            p0_ab = sift_feats0['keypoints'][sift_matches[..., 0]]
            p1_ab = sift_feats1['keypoints'][sift_matches[..., 1]]
        else:
            p0_ab = torch.empty((0, 2))
            p1_ab = torch.empty((0, 2))
        nm_kpts0 = sift_feats0['keypoints'][~matched0]
        nm_kpts1 = sift_feats1['keypoints'][~matched1]
        viz2d.plot_keypoints([nm_kpts0, nm_kpts1], colors="yellow", ps=2, axes=(ax_left, ax_right))
        if p0_ab.numel() > 0:
            viz2d.plot_matches(p0_ab, p1_ab, color="lime", lw=0.2, axes=(ax_left, ax_right))
        viz2d.add_text(0, f"SIFT+{matcher_label}: {p0_ab.shape[0]} matches", fs=16)

        # Row 1: CudaSIFT+LG
        ax_left_cuda, ax_right_cuda = prep_axes(1)
        n0b = feats0_bin['keypoints'].shape[0]
        n1b = feats1_bin['keypoints'].shape[0]
        matched0b = torch.zeros(n0b, dtype=torch.bool)
        matched1b = torch.zeros(n1b, dtype=torch.bool)
        if bin_matches.numel() > 0:
            matched0b[bin_matches[..., 0]] = True
            matched1b[bin_matches[..., 1]] = True
            points0_bin = feats0_bin['keypoints'][bin_matches[..., 0]]
            points1_bin = feats1_bin['keypoints'][bin_matches[..., 1]]
        else:
            points0_bin = torch.empty((0, 2))
            points1_bin = torch.empty((0, 2))
        nm_kpts0b = feats0_bin['keypoints'][~matched0b]
        nm_kpts1b = feats1_bin['keypoints'][~matched1b]
        viz2d.plot_keypoints([nm_kpts0b, nm_kpts1b], colors="yellow", ps=2, axes=(ax_left_cuda, ax_right_cuda))
        if points0_bin.numel() > 0:
            viz2d.plot_matches(points0_bin, points1_bin, color="lime", lw=0.2, axes=(ax_left_cuda, ax_right_cuda))
        viz2d.add_text(2, f"CudaSIFT+{matcher_label}: {points0_bin.shape[0]} matches", fs=16)

        if sp_results is not None:
            feats0_sp, feats1_sp, matches_sp = sp_results
            ax_left_sp, ax_right_sp = prep_axes(2)
            n0_sp = feats0_sp['keypoints'].shape[0]
            n1_sp = feats1_sp['keypoints'].shape[0]
            matched0_sp = torch.zeros(n0_sp, dtype=torch.bool)
            matched1_sp = torch.zeros(n1_sp, dtype=torch.bool)
            if matches_sp.numel() > 0:
                matched0_sp[matches_sp[..., 0]] = True
                matched1_sp[matches_sp[..., 1]] = True
                p0_sp = feats0_sp['keypoints'][matches_sp[..., 0]]
                p1_sp = feats1_sp['keypoints'][matches_sp[..., 1]]
                sp_match_count = p0_sp.shape[0]
                if p0_sp.numel() > 0:
                    viz2d.plot_matches(
                        p0_sp,
                        p1_sp,
                        color=SUPERPOINT_MATCH_COLOR,
                        lw=0.2,
                        axes=(ax_left_sp, ax_right_sp),
                    )
            else:
                sp_match_count = 0
            nm_kpts0_sp = feats0_sp['keypoints'][~matched0_sp]
            nm_kpts1_sp = feats1_sp['keypoints'][~matched1_sp]
            viz2d.plot_keypoints(
                [nm_kpts0_sp, nm_kpts1_sp],
                colors="yellow",
                ps=2,
                axes=(ax_left_sp, ax_right_sp),
            )
            viz2d.add_text(
                4,
                f"{SUPERPOINT_MODEL_NAME}+{matcher_label}: {sp_match_count} matches",
                fs=16
            )

        fig.tight_layout(pad=0.5)

        base0 = os.path.splitext(os.path.basename(img_path0))[0]
        base1 = os.path.splitext(os.path.basename(img_path1))[0]
        out_path = os.path.join(out_subfolder, f"{base0}__{base1}.png")
        viz2d.save_plot(out_path)
        print(f"Saved matches to {out_path}")
        plt.close(fig)
