from utils.LightGlue.lightglue import LightGlue, SIFT
from utils.LightGlue.lightglue.utils import load_image, rbd
from utils.LightGlue.lightglue.utils import match_pair
from utils.LightGlue.lightglue import viz2d
import matplotlib.pyplot as plt
import os
from itertools import combinations
import numpy as np
import struct
import torch

def load_sift_bin(path, image_size):
    with open(path, 'rb') as f:
        (N,) = struct.unpack('<I', f.read(4))
        # Each record = 2 + 128 float32s
        rec = 4 + 128  # x,y,scale,orientation + 128-d desc
        data = np.fromfile(f, dtype=np.float32, count=N*rec).reshape(N, rec)
    keypoints = (
        torch.from_numpy(data[:, :2].copy()).to(dtype=torch.float32).unsqueeze(0).cuda()
    )  # (1, N, 2)
    scales = (
        torch.from_numpy(data[:, 2].copy()).to(dtype=torch.float32).unsqueeze(0).cuda()
    )  # (1, N)
    # convert orientations from degrees to radians using numpy, then to torch
    oris_np = np.deg2rad(data[:, 3].copy())
    oris = (
        torch.from_numpy(oris_np).to(dtype=torch.float32).unsqueeze(0).cuda()
    )  # (1, N)
    descriptors = (
        torch.from_numpy(data[:, 4:].copy()).to(dtype=torch.float32).unsqueeze(0).cuda()
    )  # (1, N, 128)
    # Normalize image_size to (1, 2)
    if isinstance(image_size, torch.Tensor):
        image_size_t = image_size.to(dtype=torch.float32)
        if image_size_t.dim() == 1:
            image_size_t = image_size_t.unsqueeze(0)
    else:
        image_size_t = torch.tensor(image_size, dtype=torch.float32).unsqueeze(0)
    image_size_t = image_size_t.cuda()
    features = {
        'keypoints': keypoints,
        'scales': scales,
        'oris': oris,
        'descriptors': descriptors,
        'image_size': image_size_t,
    }
    return features


extractor_sift = SIFT(max_num_keypoints=2048, backend='opencv', detection_threshold=0.0000667).eval().cuda()  # load the extractor
matcher_sift = LightGlue(features='sift', depth_confidence=-1, width_confidence=-1).eval().cuda()  # load the matcher

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
        image0 = load_image(img_path0).cuda()
        image1 = load_image(img_path1).cuda()

        print(f"Matching {img_path0} and {img_path1}")

        # compute matches A->B direction
        feats0, feats1, matches01 = match_pair(extractor_sift, matcher_sift, image0, image1)
        m_ab = matches01['matches']
        p0_ab = feats0['keypoints'][m_ab[..., 0]]
        p1_ab = feats1['keypoints'][m_ab[..., 1]]

        # compute matches from binary data
        def image_to_bin_path(img_path):
            base, ext = os.path.splitext(img_path)
            return f"{base}_sift.bin"

        image_size = feats0['image_size']
        img0_bin = image_to_bin_path(img_path0)
        img1_bin = image_to_bin_path(img_path1)
        feats0_bin = load_sift_bin(img0_bin, image_size)
        feats1_bin = load_sift_bin(img1_bin, image_size)

        matches01_bin = matcher_sift({"image0": feats0_bin, "image1": feats1_bin})
        feats0_bin, feats1_bin, matches01_bin = [rbd(x) for x in [feats0_bin, feats1_bin, matches01_bin]]  # remove batch dimension
        matches01_bin = matches01_bin['matches']  # indices with shape (K,2)
        points0_bin = feats0_bin['keypoints'][matches01_bin[..., 0]]  # coordinates in image #0, shape (K,2)
        points1_bin = feats1_bin['keypoints'][matches01_bin[..., 1]]  # coordinates in image #1, shape (K,2)

        # Create figure with 2 rows (A->B and B->A) and 2 columns (left and right images)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Row 0: A->B direction
        ax_left_ab = axes[0, 0]
        ax_right_ab = axes[0, 1]
        
        ax_left_ab.imshow(image0.permute(1, 2, 0).cpu().numpy())
        ax_right_ab.imshow(image1.permute(1, 2, 0).cpu().numpy())
        
        for ax_ in (ax_left_ab, ax_right_ab):
            ax_.get_yaxis().set_ticks([])
            ax_.get_xaxis().set_ticks([])
            ax_.set_axis_off()
            for spine in ax_.spines.values():
                spine.set_visible(False)
        
        # plot non-matched keypoints (extractor) in yellow
        n0 = feats0['keypoints'].shape[0]
        n1 = feats1['keypoints'].shape[0]
        matched0 = torch.zeros(n0, dtype=torch.bool, device=feats0['keypoints'].device)
        matched1 = torch.zeros(n1, dtype=torch.bool, device=feats1['keypoints'].device)
        matched0[m_ab[..., 0]] = True
        matched1[m_ab[..., 1]] = True
        nm_kpts0 = feats0['keypoints'][~matched0]
        nm_kpts1 = feats1['keypoints'][~matched1]
        viz2d.plot_keypoints([nm_kpts0, nm_kpts1], colors="yellow", ps=2, axes=(ax_left_ab, ax_right_ab))

        # plot matched keypoints and lines (extractor) in lime
        viz2d.plot_matches(p0_ab, p1_ab, color="lime", lw=0.2, axes=(ax_left_ab, ax_right_ab))
        viz2d.add_text(0, f"SIFT+LG: {len(p0_ab)} matches", fs=16)
        
        # Row 1: B->A direction
        ax_left_ab = axes[1, 0]
        ax_right_ab = axes[1, 1]
        
        ax_left_ab.imshow(image0.permute(1, 2, 0).cpu().numpy())
        ax_right_ab.imshow(image1.permute(1, 2, 0).cpu().numpy())
        
        for ax_ in (ax_left_ab, ax_right_ab):
            ax_.get_yaxis().set_ticks([])
            ax_.get_xaxis().set_ticks([])
            ax_.set_axis_off()
            for spine in ax_.spines.values():
                spine.set_visible(False)
        
        # plot non-matched keypoints (binary) in yellow
        n0b = feats0_bin['keypoints'].shape[0]
        n1b = feats1_bin['keypoints'].shape[0]
        matched0b = torch.zeros(n0b, dtype=torch.bool, device=feats0_bin['keypoints'].device)
        matched1b = torch.zeros(n1b, dtype=torch.bool, device=feats1_bin['keypoints'].device)
        matched0b[matches01_bin[..., 0]] = True
        matched1b[matches01_bin[..., 1]] = True
        nm_kpts0b = feats0_bin['keypoints'][~matched0b]
        nm_kpts1b = feats1_bin['keypoints'][~matched1b]
        viz2d.plot_keypoints([nm_kpts0b, nm_kpts1b], colors="yellow", ps=2, axes=(ax_left_ab, ax_right_ab))

        # plot matched keypoints and lines (binary) in lime
        viz2d.plot_matches(points0_bin, points1_bin, color="lime", lw=0.2, axes=(ax_left_ab, ax_right_ab))
        viz2d.add_text(2, f"CudaSIFT+LG: {len(points0_bin)} matches", fs=16)

        fig.tight_layout(pad=0.5)

        base0 = os.path.splitext(os.path.basename(img_path0))[0]
        base1 = os.path.splitext(os.path.basename(img_path1))[0]
        out_path = os.path.join(out_subfolder, f"{base0}__{base1}.png")
        viz2d.save_plot(out_path)
        print(f"Saved matches to {out_path}")
        plt.close(fig)