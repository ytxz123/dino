from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from dataset_generation.common.io_utils import write_json, write_jsonl
from dataset_generation.common.raster_utils import read_binary_mask, read_raster_metadata
from dataset_generation.manifest.tiling import (
    annotate_tile_windows_with_mask,
    assign_ownership_keep_boxes,
    audit_tile_window_selection,
    compute_mask_bbox,
    expand_bbox,
    generate_tile_windows,
)


def build_family_manifest(
    dataset_root: Path,
    splits: Sequence[str],
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
    max_samples_per_split: int,
    shard_index: int = 0,
    num_shards: int = 1,
    split_roots: Optional[Dict[str, Path]] = None,
) -> List[Dict]:
    """构建 family manifest，保持旧主链的数据结构和字段语义。"""
    families: List[Dict] = []
    shard_index = max(0, int(shard_index))
    num_shards = max(1, int(num_shards))
    if shard_index >= num_shards:
        raise ValueError(f"Invalid shard config: shard_index={shard_index} num_shards={num_shards}")
    for split in splits:
        explicit_root = None if split_roots is None else split_roots.get(str(split))
        split_root = Path(explicit_root).resolve() if explicit_root is not None else Path(dataset_root).resolve() / str(split)
        if not split_root.is_dir():
            print(f"[Manifest] skip split={split} reason=missing_dir path={split_root}", flush=True)
            continue
        sample_dirs = [path for path in sorted(split_root.iterdir()) if path.is_dir()]
        if int(max_samples_per_split) > 0:
            sample_dirs = sample_dirs[: int(max_samples_per_split)]
        if num_shards > 1:
            sample_dirs = [path for index, path in enumerate(sample_dirs) if index % num_shards == shard_index]
        print(
            f"[Manifest] split={split} sample_count={len(sample_dirs)} root={split_root} shard={shard_index + 1}/{num_shards}",
            flush=True,
        )
        split_family_count = 0
        split_patch_count = 0
        for sample_index, sample_dir in enumerate(sample_dirs, start=1):
            sample_id = str(sample_dir.name)
            print(f"[Manifest] split={split} sample={sample_index}/{len(sample_dirs)} sample_id={sample_id} stage=scan", flush=True)
            image_path = sample_dir / image_relpath
            mask_path = sample_dir / mask_relpath
            lane_path = sample_dir / lane_relpath
            intersection_path = sample_dir / intersection_relpath
            if not image_path.is_file():
                print(f"[Manifest] split={split} sample_id={sample_id} stage=skip reason=missing_image path={image_path}", flush=True)
                continue
            raster_meta = read_raster_metadata(image_path)
            review_mask = read_binary_mask(mask_path, threshold=mask_threshold) if mask_path.is_file() else None
            review_bbox = compute_mask_bbox(review_mask) if review_mask is not None else None
            region_bbox = None
            if bool(search_within_review_bbox) and review_bbox is not None:
                region_bbox = expand_bbox(review_bbox, pad_px=int(review_crop_pad_px), width=int(raster_meta.width), height=int(raster_meta.height))
            region = (0, 0, int(raster_meta.width), int(raster_meta.height)) if region_bbox is None else tuple(int(value) for value in region_bbox)
            tile_windows = generate_tile_windows(
                width=int(raster_meta.width),
                height=int(raster_meta.height),
                tile_size_px=int(tile_size_px),
                overlap_px=int(overlap_px),
                region_bbox=region,
                keep_margin_px=int(keep_margin_px),
            )
            tile_windows = annotate_tile_windows_with_mask(tile_windows=tile_windows, mask=review_mask)
            selected_windows, tile_audits = audit_tile_window_selection(
                tile_windows=tile_windows,
                min_mask_ratio=float(tile_min_mask_ratio),
                min_mask_pixels=int(tile_min_mask_pixels),
                max_tiles=int(tile_max_per_sample) if int(tile_max_per_sample) > 0 else None,
                fallback_to_all_if_empty=bool(fallback_to_all_if_empty),
            )
            if len(selected_windows) == 0 and bool(fallback_to_all_if_empty):
                fallback_region = (0, 0, int(raster_meta.width), int(raster_meta.height))
                selected_windows = annotate_tile_windows_with_mask(
                    tile_windows=generate_tile_windows(
                        width=int(raster_meta.width),
                        height=int(raster_meta.height),
                        tile_size_px=int(tile_size_px),
                        overlap_px=int(overlap_px),
                        region_bbox=fallback_region,
                        keep_margin_px=int(keep_margin_px),
                    ),
                    mask=review_mask,
                )
                tile_audits = [
                    {
                        "candidate_index": int(index),
                        "selected": True,
                        "reason": "selected",
                        "bbox": [int(value) for value in window.bbox],
                        "keep_bbox": [int(value) for value in window.keep_bbox],
                        "mask_ratio": float(window.mask_ratio),
                        "mask_pixels": int(window.mask_pixels),
                    }
                    for index, window in enumerate(selected_windows)
                ]
                region = fallback_region
            unique_xs = sorted({int(window.x0) for window in selected_windows})
            unique_ys = sorted({int(window.y0) for window in selected_windows})
            col_map = {int(x0): index for index, x0 in enumerate(unique_xs)}
            row_map = {int(y0): index for index, y0 in enumerate(unique_ys)}
            patches: List[Dict] = []
            for patch_id, window in enumerate(selected_windows):
                x0, y0, x1, y1 = window.bbox
                keep_x0, keep_y0, keep_x1, keep_y1 = window.keep_bbox
                patches.append(
                    {
                        "patch_id": int(patch_id),
                        "row": int(row_map[int(y0)]),
                        "col": int(col_map[int(x0)]),
                        "center_x": int(round(0.5 * float(x0 + x1))),
                        "center_y": int(round(0.5 * float(y0 + y1))),
                        "crop_box": {
                            "x_min": int(x0),
                            "y_min": int(y0),
                            "x_max": int(x1),
                            "y_max": int(y1),
                            "center_x": int(round(0.5 * float(x0 + x1))),
                            "center_y": int(round(0.5 * float(y0 + y1))),
                        },
                        "keep_box": {
                            "x_min": int(keep_x0),
                            "y_min": int(keep_y0),
                            "x_max": int(keep_x1),
                            "y_max": int(keep_y1),
                        },
                        "mask_ratio": float(window.mask_ratio),
                        "mask_pixels": int(window.mask_pixels),
                    }
                )
            patches = assign_ownership_keep_boxes(patches=patches)
            split_family_count += 1
            split_patch_count += len(patches)
            families.append(
                {
                    "family_id": f"{sample_id}__geo_current_sw",
                    "split": str(split),
                    "source_sample_id": sample_id,
                    "source_image": image_path.name,
                    "source_image_path": str(image_path),
                    "source_mask_path": str(mask_path) if mask_path.is_file() else "",
                    "source_lane_path": str(lane_path) if lane_path.is_file() else "",
                    "source_intersection_path": str(intersection_path) if intersection_path.is_file() else "",
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
                    "crop_bbox": [int(value) for value in region],
                    "patches": patches,
                    "tile_audits": tile_audits,
                }
            )
            print(
                f"[Manifest] split={split} sample_id={sample_id} stage=done family_index={split_family_count} patch_count={len(patches)} total_split_patches={split_patch_count}",
                flush=True,
            )
        print(f"[Manifest] split={split} completed families={split_family_count} patches={split_patch_count}", flush=True)
    return families


def save_family_manifest(output_manifest: Path, families: Sequence[Dict], dataset_root: Path, splits: Sequence[str], tile_size_px: int, overlap_px: int, keep_margin_px: int, review_crop_pad_px: int, shard_index: int, num_shards: int) -> Dict:
    """写出 family_manifest.jsonl 和它的 summary。"""
    family_count = write_jsonl(output_manifest, families)
    summary = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "output_manifest": str(output_manifest.resolve()),
        "splits": [str(split) for split in splits],
        "tile_size_px": int(tile_size_px),
        "overlap_px": int(overlap_px),
        "keep_margin_px": int(keep_margin_px),
        "review_crop_pad_px": int(review_crop_pad_px),
        "family_count": int(family_count),
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
    }
    write_json(output_manifest.with_suffix(".summary.json"), summary)
    return summary