from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TileWindow:
    """一个候选 patch 窗口，包含 crop_box、keep_box 和 mask 统计。"""

    x0: int
    y0: int
    x1: int
    y1: int
    keep_x0: int
    keep_y0: int
    keep_x1: int
    keep_y1: int
    mask_ratio: float = 0.0
    mask_pixels: int = 0

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return int(self.x0), int(self.y0), int(self.x1), int(self.y1)

    @property
    def keep_bbox(self) -> Tuple[int, int, int, int]:
        return int(self.keep_x0), int(self.keep_y0), int(self.keep_x1), int(self.keep_y1)


def compute_mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """计算 review mask 的最小外接矩形。"""
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_bbox(bbox: Optional[Tuple[int, int, int, int]], pad_px: int, width: int, height: int) -> Tuple[int, int, int, int]:
    """把 review bbox 外扩，并裁回图像边界。"""
    if bbox is None:
        return 0, 0, int(width), int(height)
    x0, y0, x1, y1 = bbox
    pad = max(0, int(pad_px))
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(int(width), x1 + pad),
        min(int(height), y1 + pad),
    )


def build_axis_centers_for_region(
    region_start: int,
    region_end: int,
    crop_size_px: int,
    base_start_px: int,
    base_stride_px: int,
    axis_count: int,
) -> List[int]:
    """保留旧的一维中心点生成逻辑，便于后续扩展 family 网格模式。"""
    crop_size_px = max(1, int(crop_size_px))
    half = crop_size_px // 2
    min_center = int(region_start) + half
    max_center = int(region_end) - half
    if max_center < min_center:
        return []
    stride = max(1, int(base_stride_px))
    anchor = int(region_start) + int(base_start_px)
    min_count = max(1, int(axis_count))
    centers = [int(anchor + stride * index) for index in range(min_count)]
    centers = sorted({int(center) for center in centers if min_center <= int(center) <= max_center})
    if not centers:
        center = int(round(0.5 * float(min_center + max_center)))
        return [int(np.clip(center, min_center, max_center))]
    first = int(centers[0])
    while first - stride >= min_center:
        first -= stride
        centers.insert(0, int(first))
    last = int(centers[-1])
    while last + stride <= max_center:
        last += stride
        centers.append(int(last))
    if centers[0] > min_center:
        centers.insert(0, int(min_center))
    if centers[-1] < max_center:
        centers.append(int(max_center))
    output: List[int] = []
    for center in centers:
        clipped = int(np.clip(int(center), min_center, max_center))
        if not output or output[-1] != clipped:
            output.append(clipped)
    return output


def build_family_patches_from_centers(
    x_centers: Sequence[int],
    y_centers: Sequence[int],
    crop_size_px: int,
    family_grid_size: int,
) -> List[Dict]:
    """保留旧的基于中心点网格构建 family patch 的逻辑。"""
    crop_size_px = max(1, int(crop_size_px))
    half = crop_size_px // 2
    x_centers = [int(value) for value in x_centers]
    y_centers = [int(value) for value in y_centers]
    if len(x_centers) == 0 or len(y_centers) == 0:
        return []
    grid_size = max(1, min(int(family_grid_size), len(x_centers), len(y_centers)))
    families: List[Dict] = []
    max_row0 = max(0, len(y_centers) - grid_size)
    max_col0 = max(0, len(x_centers) - grid_size)
    for row0 in range(max_row0 + 1):
        for col0 in range(max_col0 + 1):
            patches: List[Dict] = []
            for row in range(grid_size):
                for col in range(grid_size):
                    center_x = int(x_centers[col0 + col])
                    center_y = int(y_centers[row0 + row])
                    patches.append(
                        {
                            "patch_id": int(row * grid_size + col),
                            "row": int(row),
                            "col": int(col),
                            "center_x": int(center_x),
                            "center_y": int(center_y),
                            "crop_box": {
                                "x_min": int(center_x - half),
                                "y_min": int(center_y - half),
                                "x_max": int(center_x + half),
                                "y_max": int(center_y + half),
                                "center_x": int(center_x),
                                "center_y": int(center_y),
                            },
                        }
                    )
            families.append({"row0": int(row0), "col0": int(col0), "grid_size": int(grid_size), "patches": patches})
    return families


def assign_ownership_keep_boxes(patches: Sequence[Dict]) -> List[Dict]:
    """根据上下左右邻居修正 keep_box，保持旧 ownership 语义。"""
    patch_map = {(int(patch["row"]), int(patch["col"])): patch for patch in patches}
    output: List[Dict] = []
    for patch in patches:
        row = int(patch["row"])
        col = int(patch["col"])
        crop_box = patch["crop_box"]
        left = float(crop_box["x_min"])
        right = float(crop_box["x_max"])
        top = float(crop_box["y_min"])
        bottom = float(crop_box["y_max"])
        if (row, col - 1) in patch_map:
            left = 0.5 * (float(patch["center_x"]) + float(patch_map[(row, col - 1)]["center_x"]))
        if (row, col + 1) in patch_map:
            right = 0.5 * (float(patch["center_x"]) + float(patch_map[(row, col + 1)]["center_x"]))
        if (row - 1, col) in patch_map:
            top = 0.5 * (float(patch["center_y"]) + float(patch_map[(row - 1, col)]["center_y"]))
        if (row + 1, col) in patch_map:
            bottom = 0.5 * (float(patch["center_y"]) + float(patch_map[(row + 1, col)]["center_y"]))
        copied = dict(patch)
        copied["keep_box"] = {
            "x_min": int(round(left)),
            "y_min": int(round(top)),
            "x_max": int(round(right)),
            "y_max": int(round(bottom)),
        }
        output.append(copied)
    return output


def sliding_positions(start: int, end: int, tile_size: int, limit: int, stride: int) -> List[int]:
    """生成一维滑窗左上角序列，末尾不足一个 tile 时贴边补齐。"""
    start = max(0, int(start))
    end = min(int(limit), int(end))
    tile_size = max(1, int(tile_size))
    stride = max(1, int(stride))
    if end - start <= tile_size:
        return [int(start)]
    positions = list(range(int(start), max(int(start), int(end - tile_size)) + 1, stride))
    last = max(int(start), int(end - tile_size))
    if not positions or positions[-1] != last:
        positions.append(last)
    return [int(position) for position in positions]


def compute_keep_bbox(bbox: Tuple[int, int, int, int], width: int, height: int, keep_margin_px: int) -> Tuple[int, int, int, int]:
    """先按 margin 生成一个初始 keep_box。"""
    x0, y0, x1, y1 = [int(value) for value in bbox]
    margin = max(0, int(keep_margin_px))
    return (
        max(0, x0 + margin),
        max(0, y0 + margin),
        min(int(width), x1 - margin),
        min(int(height), y1 - margin),
    )


def generate_tile_windows(
    width: int,
    height: int,
    tile_size_px: int,
    overlap_px: int,
    region_bbox: Optional[Tuple[int, int, int, int]],
    keep_margin_px: int,
) -> List[TileWindow]:
    """按旧规则在区域内生成所有候选 patch。"""
    stride = max(1, int(tile_size_px) - int(overlap_px))
    rx0, ry0, rx1, ry1 = (0, 0, int(width), int(height)) if region_bbox is None else tuple(int(value) for value in region_bbox)
    xs = sliding_positions(start=rx0, end=rx1, tile_size=int(tile_size_px), limit=int(width), stride=int(stride))
    ys = sliding_positions(start=ry0, end=ry1, tile_size=int(tile_size_px), limit=int(height), stride=int(stride))
    windows: List[TileWindow] = []
    for y0 in ys:
        for x0 in xs:
            x1 = min(int(width), int(x0 + tile_size_px))
            y1 = min(int(height), int(y0 + tile_size_px))
            keep_bbox = compute_keep_bbox(bbox=(x0, y0, x1, y1), width=int(width), height=int(height), keep_margin_px=int(keep_margin_px))
            windows.append(
                TileWindow(
                    x0=int(x0),
                    y0=int(y0),
                    x1=int(x1),
                    y1=int(y1),
                    keep_x0=int(keep_bbox[0]),
                    keep_y0=int(keep_bbox[1]),
                    keep_x1=int(keep_bbox[2]),
                    keep_y1=int(keep_bbox[3]),
                )
            )
    return windows


def annotate_tile_windows_with_mask(tile_windows: Sequence[TileWindow], mask: Optional[np.ndarray]) -> List[TileWindow]:
    """为每个窗口统计 mask 覆盖率和 mask 像素数。"""
    if mask is None:
        return list(tile_windows)
    output: List[TileWindow] = []
    for window in tile_windows:
        x0, y0, x1, y1 = window.bbox
        crop = mask[y0:y1, x0:x1]
        output.append(
            TileWindow(
                x0=window.x0,
                y0=window.y0,
                x1=window.x1,
                y1=window.y1,
                keep_x0=window.keep_x0,
                keep_y0=window.keep_y0,
                keep_x1=window.keep_x1,
                keep_y1=window.keep_y1,
                mask_ratio=float(crop.mean()) if crop.size > 0 else 0.0,
                mask_pixels=int(crop.sum()) if crop.size > 0 else 0,
            )
        )
    return output


def audit_tile_window_selection(
    tile_windows: Sequence[TileWindow],
    min_mask_ratio: float,
    min_mask_pixels: int,
    max_tiles: Optional[int],
    fallback_to_all_if_empty: bool,
) -> Tuple[List[TileWindow], List[Dict]]:
    """保留旧的选窗审计规则，方便后续 summary 和排查。"""
    all_windows = list(tile_windows)
    filtered = [
        window
        for window in all_windows
        if float(window.mask_ratio) >= float(min_mask_ratio) or int(window.mask_pixels) >= int(min_mask_pixels)
    ]
    used_fallback = len(filtered) == 0 and bool(fallback_to_all_if_empty)
    candidates = filtered if filtered else (list(all_windows) if bool(fallback_to_all_if_empty) else [])
    candidates = sorted(candidates, key=lambda item: (float(item.mask_ratio), int(item.mask_pixels)), reverse=True)
    selected = list(candidates)
    if max_tiles is not None and int(max_tiles) > 0:
        selected = selected[: int(max_tiles)]
    selected_keys = {window.bbox for window in selected}
    candidate_keys = {window.bbox for window in candidates}
    audits: List[Dict] = []
    for index, window in enumerate(all_windows):
        key = window.bbox
        if key in selected_keys:
            reason = "selected"
        elif (not used_fallback) and key not in candidate_keys:
            reason = "below_mask_threshold"
        elif max_tiles is not None and int(max_tiles) > 0 and key in candidate_keys:
            reason = "truncated_by_max_tiles"
        else:
            reason = "discarded"
        audits.append(
            {
                "candidate_index": int(index),
                "selected": bool(key in selected_keys),
                "reason": str(reason),
                "bbox": [int(value) for value in window.bbox],
                "keep_bbox": [int(value) for value in window.keep_bbox],
                "mask_ratio": float(window.mask_ratio),
                "mask_pixels": int(window.mask_pixels),
            }
        )
    return selected, audits