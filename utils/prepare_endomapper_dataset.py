"""Utility to generate cropped Endomapper datasets for matching experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import cv2

FIRST_CROP = {
    "width": 1350,
    "height": 1012,
    "x": 70,
    "y": 34,
}

SECOND_CROP = {
    "width": 1344,
    "height": 992,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop Endomapper frames twice to remove vignette and enforce multiples of 8."
        )
    )
    parser.add_argument(
        "input_root",
        nargs="?",
        default="./assets/matching",
        help="Root directory containing matching examples (default: ./assets/matching)",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Where to write the cropped datasets (default: <input_root>/cropped)",
    )
    return parser.parse_args()


def crop_first_stage(image):
    x, y = FIRST_CROP["x"], FIRST_CROP["y"]
    w, h = FIRST_CROP["width"], FIRST_CROP["height"]
    return image[y : y + h, x : x + w].copy()


def crop_second_stage(image):
    target_w, target_h = SECOND_CROP["width"], SECOND_CROP["height"]
    h, w = image.shape[:2]
    if w < target_w or h < target_h:
        raise ValueError(
            f"Image smaller than desired crop: got {(w, h)}, needed {(target_w, target_h)}"
        )
    x = (w - target_w) // 2
    y = (h - target_h) // 2
    return image[y : y + target_h, x : x + target_w].copy()


def iter_images(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def process_image(src: Path, dst_first: Path, dst_second: Path) -> None:
    dst_first.parent.mkdir(parents=True, exist_ok=True)
    dst_second.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(src))
    if image is None:
        raise RuntimeError(f"Unable to read image: {src}")
    first = crop_first_stage(image)
    second = crop_second_stage(first)
    cv2.imwrite(str(dst_first), first)
    cv2.imwrite(str(dst_second), second)


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    if not input_root.exists():
        print(f"Input path not found: {input_root}", file=sys.stderr)
        return 1

    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else input_root / "cropped"
    )
    first_dir = output_root / f"crop_{FIRST_CROP['width']}x{FIRST_CROP['height']}"
    second_dir = output_root / f"crop_{SECOND_CROP['width']}x{SECOND_CROP['height']}"

    total = 0
    for image_path in iter_images(input_root):
        rel_path = image_path.relative_to(input_root)
        dst_first = first_dir / rel_path
        dst_second = second_dir / rel_path
        process_image(image_path, dst_first, dst_second)
        total += 1

    print(
        f"Processed {total} images. Outputs: \n  First crop -> {first_dir}\n  Second crop -> {second_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
