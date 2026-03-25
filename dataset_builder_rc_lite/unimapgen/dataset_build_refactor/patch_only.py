"""RC patch-only 导出 helper。"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .common import make_sharegpt_record
from .geometry import (
    canonicalize_line_direction,
    clamp_points_float_rect,
    clip_polygon_ring_to_rect,
    clip_polyline_to_rect,
    dedup_points,
    ensure_closed_ring,
    point_boundary_side,
    resample_polyline_keep_tail,
    simplify_for_json,
    sort_lines,
)


PATCH_ONLY_PROMPT_TEMPLATE = """<image>
Please construct the complete road-structure line map in the current satellite patch."""

PATCH_ONLY_SYSTEM_PROMPT = (
    "You are a road-structure reconstruction assistant for satellite-image patches.\n"
    "Predict the complete patch-local line map from the current image.\n"
    "The output JSON schema is {\"lines\": [...]} .\n"
    "Each line must stay in patch-local UV coordinates.\n"
    "Use patch-local integer UV coordinates where one pixel equals one unit.\n"
    "Use category lane_line for roads and intersection_polygon for intersections.\n"
    "Return only valid JSON and no extra text."
)


def _line_piece_cut_flags_after_clip(source_points: np.ndarray, clipped_points: np.ndarray) -> Tuple[bool, bool]:
    source = np.asarray(source_points, dtype=np.float32)
    clipped = np.asarray(clipped_points, dtype=np.float32)
    if source.ndim != 2 or clipped.ndim != 2 or source.shape[0] == 0 or clipped.shape[0] == 0:
        return False, False
    tol = 1e-3
    return (not np.allclose(clipped[0], source[0], atol=tol), not np.allclose(clipped[-1], source[-1], atol=tol))


def build_patch_segments_global(global_features: Sequence[Dict], rect_global: Tuple[float, float, float, float], resample_step_px: float, boundary_tol_px: float) -> List[Dict]:
    output: List[Dict] = []
    for feature in global_features:
        geometry_type = str(feature.get("geometry_type", "line"))
        points_global = np.asarray(feature.get("points_global", []), dtype=np.float32)
        if geometry_type == "polygon":
            ring = ensure_closed_ring(points_global)
            if ring.ndim != 2 or ring.shape[0] < 4:
                continue
            for clipped_ring in clip_polygon_ring_to_rect(ring, rect_global):
                normalized = ensure_closed_ring(dedup_points(clipped_ring))
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
            piece = dedup_points(piece)
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


def build_patch_target_lines(segments_global: Sequence[Dict], patch: Dict, quantize: bool = True) -> List[Dict]:
    crop_box = patch["crop_box"]
    patch_width = float(crop_box["x_max"] - crop_box["x_min"])
    patch_height = float(crop_box["y_max"] - crop_box["y_min"])
    patch_size = int(max(patch_width, patch_height))
    offset = np.asarray([float(crop_box["x_min"]), float(crop_box["y_min"])], dtype=np.float32)[None, :]
    output: List[Dict] = []
    for segment in segments_global:
        local = np.asarray(segment.get("points_global", []), dtype=np.float32) - offset
        if str(segment.get("geometry_type", "line")) == "polygon":
            local = clamp_points_float_rect(local, patch_width=patch_width, patch_height=patch_height)
            local = ensure_closed_ring(dedup_points(local))
            if local.ndim != 2 or local.shape[0] < 4:
                continue
            points = simplify_for_json(local, patch_size=patch_size) if quantize else [[float(x), float(y)] for x, y in local.tolist()]
            output.append({"category": str(segment.get("category", "intersection_polygon")), "start_type": "closed", "end_type": "closed", "points": points})
            continue
        local = clamp_points_float_rect(local, patch_width=patch_width, patch_height=patch_height)
        local = dedup_points(local)
        if local.ndim != 2 or local.shape[0] < 2:
            continue
        points = simplify_for_json(local, patch_size=patch_size) if quantize else [[float(x), float(y)] for x, y in local.tolist()]
        output.append(
            {
                "category": str(segment.get("category", "lane_line")),
                "start_type": str(segment.get("start_type", "start")),
                "end_type": str(segment.get("end_type", "end")),
                "points": points,
            }
        )
    return sort_lines(output)


def make_patch_only_record(*, sample_id: str, image_rel_path: str, target_lines: Sequence[Dict], system_prompt: str) -> Dict:
    return make_sharegpt_record(
        sample_id=sample_id,
        image_rel_path=image_rel_path,
        user_text=PATCH_ONLY_PROMPT_TEMPLATE,
        assistant_payload={"lines": list(target_lines)},
        system_prompt=system_prompt,
    )
