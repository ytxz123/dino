"""从 RC 原始数据目录构建 family manifest。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import json

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unimapgen.dataset_build_refactor.common import format_progress, log_error, log_event, log_warning, require_existing_path, validate_ratio, write_json, write_jsonl
from unimapgen.dataset_build_refactor.rc_dataset import read_binary_mask, read_raster_metadata
from unimapgen.dataset_build_refactor.tiling import (
    annotate_tile_windows_with_mask,
    compute_mask_bbox,
    expand_bbox,
    generate_tile_windows,
    select_tile_windows,
)


DEFAULT_IMAGE_RELPATH = "patch_tif/0.tif"
DEFAULT_MASK_RELPATH = "patch_tif/0_edit_poly.tif"
DEFAULT_LANE_RELPATH = "label_check_crop/Lane.geojson"
DEFAULT_INTERSECTION_RELPATH = "label_check_crop/Intersection.geojson"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RC family manifest from GeoTIFF samples.")
    parser.add_argument("--dataset-root", type=str, default="")
    parser.add_argument("--train-root", type=str, default="")
    parser.add_argument("--val-root", type=str, default="")
    parser.add_argument("--output-manifest", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    parser.add_argument("--image-relpath", type=str, default=DEFAULT_IMAGE_RELPATH)
    parser.add_argument("--mask-relpath", type=str, default=DEFAULT_MASK_RELPATH)
    parser.add_argument("--lane-relpath", type=str, default=DEFAULT_LANE_RELPATH)
    parser.add_argument("--intersection-relpath", type=str, default=DEFAULT_INTERSECTION_RELPATH)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--tile-size-px", type=int, default=896)
    parser.add_argument("--overlap-px", type=int, default=232)
    parser.add_argument("--keep-margin-px", type=int, default=116)
    parser.add_argument("--review-crop-pad-px", type=int, default=64)
    parser.add_argument("--tile-min-mask-ratio", type=float, default=0.02)
    parser.add_argument("--tile-min-mask-pixels", type=int, default=256)
    parser.add_argument("--tile-max-per-sample", type=int, default=0)
    parser.add_argument("--search-within-review-bbox", action="store_true")
    parser.add_argument("--fallback-to-all-if-empty", action="store_true")
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    return parser.parse_args()


def resolve_split_root(*, split: str, dataset_root: Path | None, train_root: str, val_root: str) -> Path | None:
    if split == "train" and str(train_root).strip():
        return Path(str(train_root).strip()).resolve()
    if split == "val" and str(val_root).strip():
        return Path(str(val_root).strip()).resolve()
    if dataset_root is None:
        return None
    return (dataset_root / split).resolve()


def build_family_for_sample(
    *,
    split: str,
    sample_dir: Path,
    image_relpath: str,
    mask_relpath: str,
    lane_relpath: str,
    intersection_relpath: str,
    mask_threshold: int,
    tile_size_px: int,
    overlap_px: int,
    keep_margin_px: int,
    review_crop_pad_px: int,
    tile_min_mask_ratio: float,
    tile_min_mask_pixels: int,
    tile_max_per_sample: int,
    search_within_review_bbox: bool,
    fallback_to_all_if_empty: bool,
) -> Dict | None:
    image_path = sample_dir / image_relpath
    if not image_path.is_file():
        log_warning("Manifest", f"sample={sample_dir.name} skip missing image path={image_path}")
        return None
    mask_path = sample_dir / mask_relpath
    lane_path = sample_dir / lane_relpath
    intersection_path = sample_dir / intersection_relpath
    try:
        raster_meta = read_raster_metadata(image_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read raster metadata for sample={sample_dir.name} image={image_path}") from exc
    review_mask = read_binary_mask(mask_path, threshold=int(mask_threshold)) if mask_path.is_file() else None
    review_bbox = compute_mask_bbox(review_mask)
    region_bbox = None
    if bool(search_within_review_bbox) and review_bbox is not None:
        region_bbox = expand_bbox(review_bbox, pad_px=int(review_crop_pad_px), width=int(raster_meta.width), height=int(raster_meta.height))
    windows = generate_tile_windows(
        width=int(raster_meta.width),
        height=int(raster_meta.height),
        tile_size_px=int(tile_size_px),
        overlap_px=int(overlap_px),
        region_bbox=region_bbox,
        keep_margin_px=int(keep_margin_px),
    )
    windows = annotate_tile_windows_with_mask(windows, review_mask)
    windows = select_tile_windows(
        windows,
        min_mask_ratio=float(tile_min_mask_ratio),
        min_mask_pixels=int(tile_min_mask_pixels),
        max_tiles=int(tile_max_per_sample),
        fallback_to_all_if_empty=bool(fallback_to_all_if_empty),
    )
    if not windows:
        log_warning("Manifest", f"sample={sample_dir.name} no valid windows selected after mask filtering")
        return None
    unique_xs = sorted({int(window.x0) for window in windows})
    unique_ys = sorted({int(window.y0) for window in windows})
    col_map = {int(value): index for index, value in enumerate(unique_xs)}
    row_map = {int(value): index for index, value in enumerate(unique_ys)}
    patches: List[Dict] = []
    for patch_id, window in enumerate(windows):
        x0, y0, x1, y1 = window.bbox
        keep_x0, keep_y0, keep_x1, keep_y1 = window.keep_bbox
        center_x = int(round(0.5 * float(x0 + x1)))
        center_y = int(round(0.5 * float(y0 + y1)))
        patches.append(
            {
                "patch_id": int(patch_id),
                "row": int(row_map[int(y0)]),
                "col": int(col_map[int(x0)]),
                "center_x": center_x,
                "center_y": center_y,
                "crop_box": {"x_min": int(x0), "y_min": int(y0), "x_max": int(x1), "y_max": int(y1), "center_x": center_x, "center_y": center_y},
                "keep_box": {"x_min": int(keep_x0), "y_min": int(keep_y0), "x_max": int(keep_x1), "y_max": int(keep_y1)},
                "mask_ratio": float(window.mask_ratio),
                "mask_pixels": int(window.mask_pixels),
            }
        )
    sample_id = str(sample_dir.name)
    return {
        "family_id": f"{sample_id}__rc_tiles",
        "split": str(split),
        "source_sample_id": sample_id,
        "source_image": image_path.name,
        "source_image_path": str(image_path.resolve()),
        "source_mask_path": str(mask_path.resolve()) if mask_path.is_file() else "",
        "source_lane_path": str(lane_path.resolve()) if lane_path.is_file() else "",
        "source_intersection_path": str(intersection_path.resolve()) if intersection_path.is_file() else "",
        "image_size": [int(raster_meta.width), int(raster_meta.height)],
        "crop_size": int(tile_size_px),
        "paper_grid": {},
        "tiling": {
            "tile_size_px": int(tile_size_px),
            "overlap_px": int(overlap_px),
            "keep_margin_px": int(keep_margin_px),
            "review_crop_pad_px": int(review_crop_pad_px),
            "search_within_review_bbox": bool(search_within_review_bbox),
            "tile_min_mask_ratio": float(tile_min_mask_ratio),
            "tile_min_mask_pixels": int(tile_min_mask_pixels),
            "tile_max_per_sample": int(tile_max_per_sample),
            "row_count": int(len(unique_ys)),
            "col_count": int(len(unique_xs)),
        },
        "patches": patches,
    }


def main() -> None:
    args = parse_args()
    validate_ratio("--tile-min-mask-ratio", float(args.tile_min_mask_ratio))
    dataset_root = Path(str(args.dataset_root).strip()).resolve() if str(args.dataset_root).strip() else None
    output_manifest = Path(args.output_manifest).resolve()
    if dataset_root is None and not str(args.train_root).strip() and not str(args.val_root).strip():
        raise ValueError("You must provide --dataset-root or at least one of --train-root / --val-root.")
    log_event("Manifest", f"start output_manifest={output_manifest}")
    families: List[Dict] = []
    split_counts: Dict[str, Dict[str, int]] = {}
    for split in [str(item) for item in args.splits]:
        split_root = resolve_split_root(split=split, dataset_root=dataset_root, train_root=str(args.train_root), val_root=str(args.val_root))
        if split_root is None or not split_root.is_dir():
            log_warning("Manifest", f"split={split} skip missing split dir path={split_root}")
            continue
        require_existing_path(split_root, kind="dir")
        sample_dirs = [path for path in sorted(split_root.iterdir()) if path.is_dir()]
        if int(args.max_samples_per_split) > 0:
            sample_dirs = sample_dirs[: int(args.max_samples_per_split)]
        log_event("Manifest", f"split={split} root={split_root} sample_count={len(sample_dirs)}")
        family_count = 0
        patch_count = 0
        for index, sample_dir in enumerate(sample_dirs, start=1):
            if index == 1 or index == len(sample_dirs) or index % 20 == 0:
                log_event("Manifest", f"split={split} sample_progress={format_progress(index, len(sample_dirs))} sample_id={sample_dir.name}")
            try:
                family = build_family_for_sample(
                    split=split,
                    sample_dir=sample_dir,
                    image_relpath=str(args.image_relpath),
                    mask_relpath=str(args.mask_relpath),
                    lane_relpath=str(args.lane_relpath),
                    intersection_relpath=str(args.intersection_relpath),
                    mask_threshold=int(args.mask_threshold),
                    tile_size_px=int(args.tile_size_px),
                    overlap_px=int(args.overlap_px),
                    keep_margin_px=int(args.keep_margin_px),
                    review_crop_pad_px=int(args.review_crop_pad_px),
                    tile_min_mask_ratio=float(args.tile_min_mask_ratio),
                    tile_min_mask_pixels=int(args.tile_min_mask_pixels),
                    tile_max_per_sample=int(args.tile_max_per_sample),
                    search_within_review_bbox=bool(args.search_within_review_bbox),
                    fallback_to_all_if_empty=bool(args.fallback_to_all_if_empty),
                )
            except Exception as exc:
                log_error("Manifest", f"split={split} sample_id={sample_dir.name} failed: {exc}")
                raise
            if family is None:
                continue
            families.append(family)
            family_count += 1
            patch_count += len(family["patches"])
        split_counts[split] = {"families": family_count, "patches": patch_count, "samples_scanned": len(sample_dirs)}
        log_event("Manifest", f"split={split} done families={family_count} patches={patch_count}")
    if not families:
        raise RuntimeError("No family records were generated. Check input roots, image-relpath, and mask filter thresholds.")
    family_count = write_jsonl(output_manifest, families)
    summary = {
        "dataset_root": str(dataset_root) if dataset_root is not None else "",
        "output_manifest": str(output_manifest),
        "splits": [str(item) for item in args.splits],
        "family_count": int(family_count),
        "tile_size_px": int(args.tile_size_px),
        "overlap_px": int(args.overlap_px),
        "keep_margin_px": int(args.keep_margin_px),
        "review_crop_pad_px": int(args.review_crop_pad_px),
        "counts_by_split": split_counts,
    }
    write_json(output_manifest.with_suffix(".summary.json"), summary)
    log_event("Manifest", f"done family_count={family_count} summary={output_manifest.with_suffix('.summary.json')}")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
