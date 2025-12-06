"""Align a .pth.tar checkpoint to match an official .pth and compare them.

This combines conversion and comparison:
* loads a training checkpoint (.pth/.pth.tar) and an official .pth
* extracts state_dicts (default tar key: 'model')
* strips module/matcher prefixes by default to match LightGlue SIFT exports
* optionally drops tar-only keys so it matches the official structure
* reports extra keys and value mismatches
* optionally writes the aligned tar weights to disk
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch


def _infer_state_dict(
    obj: object, prefer_key: str | None = None
) -> Dict[str, torch.Tensor]:
    """Extract a state_dict from a checkpoint-like object."""
    if isinstance(obj, dict):
        if prefer_key and prefer_key in obj:
            return obj[prefer_key]
        for k in ("model_state_dict", "state_dict"):
            if k in obj:
                return obj[k]
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj  # type: ignore[return-value]
    raise KeyError(
        "Could not infer state_dict. Available: "
        f"{sorted(obj.keys()) if isinstance(obj, dict) else type(obj)}"
    )


def _strip_module(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Remove a leading prefix (optionally followed by a dot) from keys if present."""
    stripped = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            new_key = k[len(prefix) :]
            if new_key.startswith("."):
                new_key = new_key[1:]
            stripped[new_key] = v
        else:
            stripped[k] = v
    return stripped


def _apply_prefixes(sd: Dict[str, torch.Tensor], prefixes: Iterable[str]) -> Dict[str, torch.Tensor]:
    out = dict(sd)
    for pref in prefixes:
        out = _strip_prefix(out, pref)
    return out


def _load_state(path: Path, key: str | None, strip_module: bool) -> Dict[str, torch.Tensor]:
    obj = torch.load(str(path), map_location="cpu")
    sd = _infer_state_dict(obj, prefer_key=key) if not isinstance(obj, dict) else _infer_state_dict(obj, prefer_key=key)
    return _strip_module(sd) if strip_module else sd


def _compare(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
    *,
    rtol: float,
    atol: float,
) -> Tuple[list, list, list, Tuple[str | None, float]]:
    keys_a = set(a.keys())
    keys_b = set(b.keys())
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    inter = sorted(keys_a & keys_b)

    values_diffs = []
    biggest_diff = (None, 0.0)
    for k in inter:
        ta, tb = a[k], b[k]
        if ta.shape != tb.shape:
            values_diffs.append((k, f"shape {tuple(ta.shape)} != {tuple(tb.shape)}"))
            continue
        diff = torch.max(torch.abs(ta - tb)).item()
        if not torch.allclose(ta, tb, rtol=rtol, atol=atol):
            values_diffs.append((k, f"max_abs_diff={diff:.3e}"))
        if diff > biggest_diff[1]:
            biggest_diff = (k, diff)
    return only_a, only_b, values_diffs, biggest_diff


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Align a .tar checkpoint to an official .pth (default LightGlue SIFT) "
            "by stripping prefixes and dropping extras, then compare."
        )
    )
    parser.add_argument("tar", type=Path, help="Path to .pth.tar (or .pth) checkpoint")
    parser.add_argument("pth", type=Path, help="Path to official/raw .pth state_dict")
    parser.add_argument("--tar-key", type=str, default="model", help="Key inside tar")
    parser.add_argument("--pth-key", type=str, default=None, help="Key inside pth if not raw")
    parser.add_argument("--no-strip-module", action="store_true", help="Keep leading 'module.' prefixes")
    parser.add_argument(
        "--strip-prefix-tar",
        action="append",
        default=["matcher."],
        help="Prefix to strip from tar keys (repeatable). Default: matcher.",
    )
    parser.add_argument(
        "--strip-prefix-pth",
        action="append",
        default=[],
        help="Prefix to strip from pth keys (repeatable). Default: none.",
    )
    parser.add_argument(
        "--drop-extras",
        action="store_true",
        default=True,
        help="Drop tar-only keys so it matches pth structure (default: on).",
    )
    parser.add_argument(
        "--keep-extras",
        action="store_false",
        dest="drop_extras",
        help="Keep tar-only keys (override --drop-extras).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the aligned tar state_dict (.pth). "
        "Defaults to tar path with .pth suffix when not in --compare-only.",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only compare; do not write an aligned state dict even if --output is set.",
    )
    parser.add_argument("--rtol", type=float, default=0.0, help="Relative tol")
    parser.add_argument("--atol", type=float, default=1e-7, help="Absolute tol")
    args = parser.parse_args()

    strip_module = not args.no_strip_module
    sd_tar = _load_state(args.tar, key=args.tar_key, strip_module=strip_module)
    sd_pth = _load_state(args.pth, key=args.pth_key, strip_module=strip_module)

    sd_tar = _apply_prefixes(sd_tar, args.strip_prefix_tar or [])
    sd_pth = _apply_prefixes(sd_pth, args.strip_prefix_pth or [])

    if args.drop_extras:
        keep_keys = set(sd_pth.keys())
        sd_tar = {k: v for k, v in sd_tar.items() if k in keep_keys}

    only_tar, only_pth, values_diffs, biggest_diff = _compare(
        sd_tar, sd_pth, rtol=args.rtol, atol=args.atol
    )

    def _print_keys(label: str, items):
        print(f"{label}: {len(items)}")
        if not items:
            return
        if len(items) <= 10:
            print("  keys:", items)
        else:
            print("  e.g.:", items[:10], f"... (+{len(items) - 10} more)")

    _print_keys("extra in tar", only_tar)
    _print_keys("extra in pth", only_pth)
    print(f"compared params: {len(sd_tar.keys() & sd_pth.keys())}")
    if values_diffs:
        print(
            f"parameters differ (same shape, different values) for {len(values_diffs)} shared keys; "
            f"e.g., {values_diffs[0][0]} (max_abs_diff={biggest_diff[1]:.3e})"
        )
    else:
        print("parameters match for all shared keys")

    if args.compare_only:
        args.output = None

    # Default output path when not comparing only
    if args.output is None and not args.compare_only:
        out_path = args.tar
        if out_path.suffix == "":
            out_path = out_path.with_suffix(".pth")
        else:
            out_path = out_path.with_suffix(".pth")
        args.output = out_path

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sd_tar, str(args.output))
        print(f"Aligned tar saved to: {args.output}")


if __name__ == "__main__":
    main()
