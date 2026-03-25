"""Fixed16 阶段 helper。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .geometry import (
    canonicalize_line_direction,
    clamp_points,
    clip_polyline_to_rect,
    line_length_xy,
    point_boundary_side,
    resample_polyline,
    simplify_for_json,
    sort_lines,
)
from .viz import draw_endpoint


FIXED16_PROMPT_TEMPLATE = """<image>
Reconstruct the road-structure line map inside target box [{box_x_min},{box_y_min},{box_x_max},{box_y_max}].
Keep all coordinates in the patch-local coordinate system."""


def build_grid_boxes(patch_size: int, grid_size: int) -> list[Dict[str, int]]:
    edges = [int(round(i * patch_size / grid_size)) for i in range(grid_size + 1)]
    boxes: list[Dict[str, int]] = []
    for grid_row in range(grid_size):
        for grid_col in range(grid_size):
            boxes.append(
                {
                    "grid_row": int(grid_row),
                    "grid_col": int(grid_col),
                    "x_min": int(edges[grid_col]),
                    "y_min": int(edges[grid_row]),
                    "x_max": int(edges[grid_col + 1] - 1),
                    "y_max": int(edges[grid_row + 1] - 1),
                }
            )
    return boxes


def longest_piece_in_box(lines: Sequence[Dict], rect: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    best_piece: Optional[np.ndarray] = None
    best_len = -1.0
    for line in lines:
        points = np.asarray(line.get("points", []), dtype=np.float32)
        for piece in clip_polyline_to_rect(points, rect):
            current_len = line_length_xy(piece.tolist())
            if current_len > best_len:
                best_len = current_len
                best_piece = piece
    return best_piece


def build_prompt_endpoints(target_lines: Sequence[Dict], target_box: Dict[str, int], patch_size: int) -> Dict[str, object]:
    rect = (float(target_box["x_min"]), float(target_box["y_min"]), float(target_box["x_max"]), float(target_box["y_max"]))
    piece = longest_piece_in_box(target_lines, rect)
    if piece is None or piece.shape[0] < 2:
        center_x = int(round((int(target_box["x_min"]) + int(target_box["x_max"])) / 2.0))
        center_y = int(round((int(target_box["y_min"]) + int(target_box["y_max"])) / 2.0))
        return {
            "start_x": center_x,
            "start_y": center_y,
            "end_x": center_x,
            "end_y": center_y,
            "anchor_piece_points": [[center_x, center_y]],
            "anchor_source": "box_center",
        }
    piece = clamp_points(piece, patch_size=patch_size)
    piece_json = simplify_for_json(piece, patch_size=patch_size)
    if len(piece_json) < 2:
        center_x = int(round((int(target_box["x_min"]) + int(target_box["x_max"])) / 2.0))
        center_y = int(round((int(target_box["y_min"]) + int(target_box["y_max"])) / 2.0))
        return {
            "start_x": center_x,
            "start_y": center_y,
            "end_x": center_x,
            "end_y": center_y,
            "anchor_piece_points": [[center_x, center_y]],
            "anchor_source": "box_center",
        }
    return {
        "start_x": int(piece_json[0][0]),
        "start_y": int(piece_json[0][1]),
        "end_x": int(piece_json[-1][0]),
        "end_y": int(piece_json[-1][1]),
        "anchor_piece_points": piece_json,
        "anchor_source": "longest_clipped_piece",
    }


def build_target_lines_for_box(full_patch_target_lines: Sequence[Dict], target_box: Dict[str, int], patch_size: int, boundary_tol_px: float, resample_step_px: float) -> list[Dict]:
    rect = (float(target_box["x_min"]), float(target_box["y_min"]), float(target_box["x_max"]), float(target_box["y_max"]))
    output: list[Dict] = []
    for segment in full_patch_target_lines:
        points = np.asarray(segment.get("points", []), dtype=np.float32)
        for piece in clip_polyline_to_rect(points, rect):
            piece = clamp_points(piece, patch_size=patch_size)
            if resample_step_px > 0:
                piece = resample_polyline(piece, step_px=resample_step_px)
            if piece.ndim != 2 or piece.shape[0] < 2:
                continue
            start_side = point_boundary_side(piece[0], rect, boundary_tol_px)
            end_side = point_boundary_side(piece[-1], rect, boundary_tol_px)
            start_type = "cut" if start_side is not None else "start"
            end_type = "cut" if end_side is not None else "end"
            piece, start_type, end_type = canonicalize_line_direction(piece, start_type=start_type, end_type=end_type)
            points_json = simplify_for_json(piece, patch_size=patch_size)
            if len(points_json) < 2:
                continue
            output.append(
                {
                    "category": str(segment.get("category", "lane_line")),
                    "start_type": str(start_type),
                    "end_type": str(end_type),
                    "points": points_json,
                }
            )
    return sort_lines(output)


def format_fixed16_prompt(prompt_fields: Dict[str, int], prompt_template: str = "") -> str:
    template = str(prompt_template).strip() or FIXED16_PROMPT_TEMPLATE
    try:
        return template.format(**prompt_fields)
    except KeyError as exc:
        missing_key = str(exc).strip("'\"")
        raise ValueError(f"Fixed16 prompt template contains an unknown placeholder: {missing_key}") from exc


def save_fixed16_visualization(*, patch_image: Image.Image, target_lines: Sequence[Dict], target_box: Dict[str, int], anchor_piece_points: Sequence[Sequence[int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = patch_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    patch_size = int(image.size[0])
    draw.rectangle((0, 0, patch_size - 1, patch_size - 1), outline=(255, 0, 180), width=2)
    draw.rectangle((int(target_box["x_min"]), int(target_box["y_min"]), int(target_box["x_max"]), int(target_box["y_max"])), outline=(255, 210, 0), width=3)
    anchor_pts = [tuple(int(value) for value in point) for point in anchor_piece_points]
    if len(anchor_pts) >= 2:
        draw.line(anchor_pts, fill=(255, 140, 0), width=4)
        draw_endpoint(draw, anchor_pts[0], (0, 255, 80), 4)
        draw_endpoint(draw, anchor_pts[-1], (255, 40, 40), 4)
    elif len(anchor_pts) == 1:
        draw_endpoint(draw, anchor_pts[0], (255, 140, 0), 4)
    for line in target_lines:
        pts = [tuple(int(value) for value in point) for point in line.get("points", [])]
        if len(pts) >= 2:
            draw.line(pts, fill=(40, 220, 255), width=3)
            draw_endpoint(draw, pts[0], (0, 180, 220), 3)
            draw_endpoint(draw, pts[-1], (0, 180, 220), 3)
    image.save(out_path)
