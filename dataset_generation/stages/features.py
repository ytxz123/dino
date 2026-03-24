from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from dataset_generation.common.defaults import DEFAULT_INTERSECTION_RELPATH, DEFAULT_LANE_RELPATH
from dataset_generation.common.geojson_utils import load_sample_global_lines
from dataset_generation.common.geometry_utils import (
    canonicalize_line_direction,
    clamp_points_float_rect,
    clip_polygon_ring_to_rect,
    clip_polyline_to_rect,
    deduplicate_points,
    ensure_closed_ring,
    point_boundary_side,
    resample_polyline_keep_tail,
    simplify_points_for_json,
    sort_lines,
)
from dataset_generation.common.raster_utils import RasterMetadata, read_binary_mask, read_rgb_geotiff


def load_family_raster_and_mask(family: Dict, band_indices: Sequence[int], mask_threshold: int) -> Tuple[np.ndarray, RasterMetadata, Optional[np.ndarray]]:
    """读取 family 对应的大图和审核 mask，并把 mask 外像素置黑。"""
    image_hwc, raster_meta = read_rgb_geotiff(Path(family["source_image_path"]).resolve(), band_indices=tuple(int(index) for index in band_indices))
    mask_path = str(family.get("source_mask_path", "")).strip()
    review_mask = read_binary_mask(Path(mask_path), threshold=int(mask_threshold)) if mask_path else None
    if review_mask is not None:
        image_hwc = image_hwc.copy()
        image_hwc[review_mask <= 0] = 0
    return image_hwc, raster_meta, review_mask



def family_global_lines(family: Dict, raster_meta: RasterMetadata, include_lane: bool, include_intersection: bool) -> List[Dict]:
    """读取一个 family 的 Lane 和 Intersection，并统一转成整图像素几何。"""
    image_path = Path(family["source_image_path"]).resolve()
    sample_dir = image_path.parents[1]
    lane_path_text = str(family.get("source_lane_path", "")).strip()
    intersection_path_text = str(family.get("source_intersection_path", "")).strip()
    lane_path = Path(lane_path_text) if lane_path_text else sample_dir / DEFAULT_LANE_RELPATH
    intersection_path = Path(intersection_path_text) if intersection_path_text else sample_dir / DEFAULT_INTERSECTION_RELPATH
    lane_rel = str(lane_path.resolve().relative_to(sample_dir)) if lane_path.exists() else DEFAULT_LANE_RELPATH
    intersection_rel = str(intersection_path.resolve().relative_to(sample_dir)) if intersection_path.exists() else DEFAULT_INTERSECTION_RELPATH
    return load_sample_global_lines(
        sample_dir=sample_dir,
        raster_meta=raster_meta,
        lane_relpath=lane_rel,
        intersection_relpath=intersection_rel,
        include_lane=bool(include_lane),
        include_intersection=bool(include_intersection),
    )



def build_patch_image(raw_image_hwc: np.ndarray, patch: Dict) -> Image.Image:
    """按 crop_box 从整图裁出 patch 图像。"""
    crop_box = patch["crop_box"]
    crop = raw_image_hwc[
        int(crop_box["y_min"]):int(crop_box["y_max"]),
        int(crop_box["x_min"]):int(crop_box["x_max"]),
    ]
    return Image.fromarray(np.asarray(crop, dtype=np.uint8))



def _line_piece_cut_flags_after_clip(source_points: np.ndarray, clipped_points: np.ndarray) -> Tuple[bool, bool]:
    """根据裁切前后首尾点是否变化，判断 piece 两端是否为 cut。"""
    source = np.asarray(source_points, dtype=np.float32)
    clipped = np.asarray(clipped_points, dtype=np.float32)
    if source.ndim != 2 or clipped.ndim != 2 or source.shape[0] == 0 or clipped.shape[0] == 0:
        return False, False
    tol = 1e-3
    return (not np.allclose(clipped[0], source[0], atol=tol), not np.allclose(clipped[-1], source[-1], atol=tol))



def build_patch_segments_global(global_lines: Sequence[Dict], rect_global: Tuple[float, float, float, float], resample_step_px: float, boundary_tol_px: float) -> List[Dict]:
    """把整图几何裁到当前 patch 或 keep_box 对应的全局矩形内。"""
    output: List[Dict] = []
    for feature in global_lines:
        geometry_type = str(feature.get("geometry_type", "line"))
        points_global = np.asarray(feature.get("points_global", []), dtype=np.float32)
        if geometry_type == "polygon":
            ring = ensure_closed_ring(points_global)
            if ring.ndim != 2 or ring.shape[0] < 4:
                continue
            for clipped_ring in clip_polygon_ring_to_rect(ring, rect_global):
                normalized = ensure_closed_ring(deduplicate_points(clipped_ring))
                if normalized.ndim != 2 or normalized.shape[0] < 4:
                    continue
                output.append(
                    {
                        "category": str(feature.get("category", "intersection_polygon")),
                        "geometry_type": "polygon",
                        "points_global": normalized.astype(np.float32),
                        "start_type": "closed",
                        "end_type": "closed",
                    }
                )
            continue
        for clipped_piece in clip_polyline_to_rect(points_global, rect_global):
            piece = np.asarray(clipped_piece, dtype=np.float32)
            if piece.ndim != 2 or piece.shape[0] < 2:
                continue
            cut_start, cut_end = _line_piece_cut_flags_after_clip(points_global, piece)
            if float(resample_step_px) > 0.0:
                piece = resample_polyline_keep_tail(piece, step_px=float(resample_step_px))
            if piece.ndim != 2 or piece.shape[0] < 2:
                continue
            start_side = point_boundary_side(piece[0], rect_global, tol_px=float(boundary_tol_px))
            end_side = point_boundary_side(piece[-1], rect_global, tol_px=float(boundary_tol_px))
            start_type = "cut" if cut_start or start_side is not None else "start"
            end_type = "cut" if cut_end or end_side is not None else "end"
            piece, start_type, end_type = canonicalize_line_direction(piece, start_type=start_type, end_type=end_type)
            piece = deduplicate_points(piece)
            if piece.ndim != 2 or piece.shape[0] < 2:
                continue
            output.append(
                {
                    "category": str(feature.get("category", "lane_line")),
                    "geometry_type": "line",
                    "points_global": piece.astype(np.float32),
                    "start_type": str(start_type),
                    "end_type": str(end_type),
                }
            )
    return sort_lines(output)



def build_owned_segments_by_patch(family: Dict, global_lines: Sequence[Dict], resample_step_px: float, boundary_tol_px: float) -> Dict[int, List[Dict]]:
    """按 patch 的 keep_box 生成每个 patch 真正拥有的几何片段。"""
    output: Dict[int, List[Dict]] = {}
    for patch in sorted(list(family.get("patches", [])), key=lambda item: int(item["patch_id"])):
        keep_box = patch["keep_box"]
        rect_global = (
            float(keep_box["x_min"]),
            float(keep_box["y_min"]),
            float(keep_box["x_max"]),
            float(keep_box["y_max"]),
        )
        output[int(patch["patch_id"])] = build_patch_segments_global(
            global_lines=global_lines,
            rect_global=rect_global,
            resample_step_px=float(resample_step_px),
            boundary_tol_px=float(boundary_tol_px),
        )
    return output



def local_lines_to_uv(lines: Sequence[Dict], patch: Dict, quantize: bool) -> List[Dict]:
    """把全局或本地浮点线段整理成 patch-local UV 坐标。"""
    crop_box = patch["crop_box"]
    patch_width = float(crop_box["x_max"] - crop_box["x_min"])
    patch_height = float(crop_box["y_max"] - crop_box["y_min"])
    offset = np.asarray([float(crop_box["x_min"]), float(crop_box["y_min"])], dtype=np.float32)[None, :]
    output: List[Dict] = []
    for line in lines:
        points_key = "points_global" if "points_global" in line else "points"
        points_array = np.asarray(line.get(points_key, []), dtype=np.float32)
        if points_key == "points_global":
            local_source = points_array - offset
        else:
            local_source = points_array
        if str(line.get("geometry_type", "line")) == "polygon":
            local = clamp_points_float_rect(local_source, patch_width=patch_width, patch_height=patch_height)
            local = ensure_closed_ring(deduplicate_points(local))
            if local.ndim != 2 or local.shape[0] < 4:
                continue
            points = simplify_points_for_json(local, patch_size=int(max(patch_width, patch_height))) if quantize else [[float(x), float(y)] for x, y in local.tolist()]
            output.append({
                "category": str(line.get("category", "intersection_polygon")),
                "start_type": "closed",
                "end_type": "closed",
                "points": points,
            })
            continue
        local = clamp_points_float_rect(local_source, patch_width=patch_width, patch_height=patch_height)
        local = deduplicate_points(local)
        if local.ndim != 2 or local.shape[0] < 2:
            continue
        points = simplify_points_for_json(local, patch_size=int(max(patch_width, patch_height))) if quantize else [[float(x), float(y)] for x, y in local.tolist()]
        output.append(
            {
                "category": str(line.get("category", "lane_line")),
                "start_type": str(line.get("start_type", "start")),
                "end_type": str(line.get("end_type", "end")),
                "points": points,
            }
        )
    return sort_lines(output)



def build_patch_target_lines_float(segments: Sequence[Dict], patch: Dict) -> List[Dict]:
    """输出 patch-local 浮点 target lines。"""
    return local_lines_to_uv(segments, patch=patch, quantize=False)



def build_patch_target_lines_quantized(segments: Sequence[Dict], patch: Dict) -> List[Dict]:
    """输出 patch-local 整数 target lines。"""
    return local_lines_to_uv(segments, patch=patch, quantize=True)



def build_patch_target_lines(segments: Sequence[Dict], patch: Dict) -> List[Dict]:
    """当前训练集默认写入量化后的 target lines。"""
    return build_patch_target_lines_quantized(segments=segments, patch=patch)



def extract_state_lines(patch: Dict, family: Dict, owned_segments_by_patch: Dict[int, List[Dict]], trace_points: int, boundary_tol_px: float) -> List[Dict]:
    """从左邻和上邻 patch 的 owned segments 中提取 handoff state。"""
    patches = sorted(list(family.get("patches", [])), key=lambda item: int(item["patch_id"]))
    patch_map = {(int(item["row"]), int(item["col"])): item for item in patches}
    row = int(patch["row"])
    col = int(patch["col"])
    crop_box = patch["crop_box"]
    keep_box = patch["keep_box"]
    crop_rect_global = (
        float(crop_box["x_min"]),
        float(crop_box["y_min"]),
        float(crop_box["x_max"]),
        float(crop_box["y_max"]),
    )
    local_keep_rect = (
        float(keep_box["x_min"] - crop_box["x_min"]),
        float(keep_box["y_min"] - crop_box["y_min"]),
        float(keep_box["x_max"] - crop_box["x_min"]),
        float(keep_box["y_max"] - crop_box["y_min"]),
    )
    patch_width = float(crop_box["x_max"] - crop_box["x_min"])
    patch_height = float(crop_box["y_max"] - crop_box["y_min"])
    offset = np.asarray([float(crop_box["x_min"]), float(crop_box["y_min"])], dtype=np.float32)[None, :]
    neighbors: List[Tuple[Dict, str]] = []
    if (row, col - 1) in patch_map:
        neighbors.append((patch_map[(row, col - 1)], "left"))
    if (row - 1, col) in patch_map:
        neighbors.append((patch_map[(row - 1, col)], "top"))
    output: List[Dict] = []
    for neighbor_patch, handoff_side in neighbors:
        for segment in owned_segments_by_patch.get(int(neighbor_patch["patch_id"]), []):
            if str(segment.get("geometry_type", "line")) != "line":
                continue
            for piece in clip_polyline_to_rect(np.asarray(segment.get("points_global", []), dtype=np.float32), crop_rect_global):
                local = clamp_points_float_rect(np.asarray(piece, dtype=np.float32) - offset, patch_width=patch_width, patch_height=patch_height)
                if local.ndim != 2 or local.shape[0] < 2:
                    continue
                boundary_index = None
                if point_boundary_side(local[0], local_keep_rect, tol_px=float(boundary_tol_px)) == handoff_side:
                    boundary_index = 0
                elif point_boundary_side(local[-1], local_keep_rect, tol_px=float(boundary_tol_px)) == handoff_side:
                    boundary_index = -1
                if boundary_index is None:
                    continue
                if boundary_index == -1:
                    local = local[::-1].copy()
                trace = local[: max(2, int(trace_points))]
                if trace.ndim != 2 or trace.shape[0] < 2:
                    continue
                output.append(
                    {
                        "source_patch": int(neighbor_patch["patch_id"]),
                        "category": str(segment.get("category", "lane_line")),
                        "start_type": "cut",
                        "end_type": "cut",
                        "points": [[float(x), float(y)] for x, y in trace.tolist()],
                    }
                )
    deduped: List[Dict] = []
    seen = set()
    for line in sort_lines(output):
        key = (int(line.get("source_patch", -1)), tuple((round(float(point[0]), 3), round(float(point[1]), 3)) for point in line.get("points", [])))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line)
    return deduped
