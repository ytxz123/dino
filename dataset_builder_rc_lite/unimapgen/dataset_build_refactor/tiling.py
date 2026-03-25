"""RC 滑窗与 keep_box helper。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TileWindow:
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


def compute_mask_bbox(mask: np.ndarray | None) -> Optional[Tuple[int, int, int, int]]:
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_bbox(bbox: Optional[Tuple[int, int, int, int]], pad_px: int, width: int, height: int) -> Tuple[int, int, int, int]:
    if bbox is None:
        return 0, 0, int(width), int(height)
    x0, y0, x1, y1 = bbox
    pad = max(0, int(pad_px))
    return max(0, x0 - pad), max(0, y0 - pad), min(int(width), x1 + pad), min(int(height), y1 + pad)


def sliding_positions(start: int, end: int, tile_size: int, limit: int, stride: int) -> List[int]:
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
    x0, y0, x1, y1 = [int(value) for value in bbox]
    margin = max(0, int(keep_margin_px))
    keep_x0 = max(0, x0 + margin)
    keep_y0 = max(0, y0 + margin)
    keep_x1 = min(int(width), x1 - margin)
    keep_y1 = min(int(height), y1 - margin)
    if keep_x1 <= keep_x0 or keep_y1 <= keep_y0:
        return int(x0), int(y0), int(x1), int(y1)
    return keep_x0, keep_y0, keep_x1, keep_y1


def generate_tile_windows(width: int, height: int, tile_size_px: int, overlap_px: int, region_bbox: Optional[Tuple[int, int, int, int]], keep_margin_px: int) -> List[TileWindow]:
    stride = max(1, int(tile_size_px) - int(overlap_px))
    rx0, ry0, rx1, ry1 = (0, 0, int(width), int(height)) if region_bbox is None else tuple(int(value) for value in region_bbox)
    xs = sliding_positions(start=rx0, end=rx1, tile_size=int(tile_size_px), limit=int(width), stride=int(stride))
    ys = sliding_positions(start=ry0, end=ry1, tile_size=int(tile_size_px), limit=int(height), stride=int(stride))
    windows: List[TileWindow] = []
    for y0 in ys:
        for x0 in xs:
            x1 = min(int(width), int(x0 + tile_size_px))
            y1 = min(int(height), int(y0 + tile_size_px))
            keep_bbox = compute_keep_bbox((x0, y0, x1, y1), width=int(width), height=int(height), keep_margin_px=int(keep_margin_px))
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


def annotate_tile_windows_with_mask(tile_windows: Sequence[TileWindow], mask: np.ndarray | None) -> List[TileWindow]:
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


def select_tile_windows(tile_windows: Sequence[TileWindow], min_mask_ratio: float, min_mask_pixels: int, max_tiles: int, fallback_to_all_if_empty: bool) -> List[TileWindow]:
    candidates = [
        window
        for window in tile_windows
        if float(window.mask_ratio) >= float(min_mask_ratio) or int(window.mask_pixels) >= int(min_mask_pixels)
    ]
    if not candidates and bool(fallback_to_all_if_empty):
        candidates = list(tile_windows)
    candidates = sorted(candidates, key=lambda item: (float(item.mask_ratio), int(item.mask_pixels)), reverse=True)
    if int(max_tiles) > 0:
        candidates = candidates[: int(max_tiles)]
    return list(candidates)
