"""Stage B 阶段 helper。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .geometry import clamp_points
from .viz import draw_endpoint, draw_polyline


STAGEB_TRACE_PROMPT_TEMPLATE = """<image>
Reconstruct the target road-structure line map inside target box [{box_x_min},{box_y_min},{box_x_max},{box_y_max}].
Incoming trace hints: {trace_points_json}
Keep all coordinates in the patch-local coordinate system."""

STAGEB_NO_STATE_PROMPT_TEMPLATE = """<image>
Reconstruct the target road-structure line map inside target box [{box_x_min},{box_y_min},{box_x_max},{box_y_max}].
Keep all coordinates in the patch-local coordinate system."""


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def simplify_points(points_xy: np.ndarray, patch_size: int) -> List[List[int]]:
    array = clamp_points(np.asarray(points_xy, dtype=np.float32), patch_size=patch_size)
    if array.ndim != 2 or array.shape[0] == 0:
        return []
    rounded = np.rint(array).astype(np.int32)
    output: List[List[int]] = []
    for point_xy in rounded.tolist():
        current = [int(point_xy[0]), int(point_xy[1])]
        if not output or output[-1] != current:
            output.append(current)
    return output


def point_sort_key(point_xy: Sequence[float]) -> Tuple[float, float, float]:
    x = float(point_xy[0])
    y = float(point_xy[1])
    return (x * x + y * y, y, x)


def extract_trace_points_for_endpoint(pts: np.ndarray, endpoint_idx: int, patch_size: int, max_points: int) -> List[List[int]]:
    array = np.asarray(pts, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 2:
        return []
    keep = max(2, int(max_points))
    trace = array[:keep] if int(endpoint_idx) == 0 else array[::-1][:keep]
    return simplify_points(trace, patch_size=patch_size)


def endpoint_matches_neighbor_boundary(point_xy: np.ndarray, neighbor_box: Dict[str, int], side_to_current: str, tol_px: float) -> bool:
    x = float(point_xy[0])
    y = float(point_xy[1])
    if str(side_to_current) == "left":
        return abs(x - float(neighbor_box["x_max"])) <= float(tol_px)
    if str(side_to_current) == "top":
        return abs(y - float(neighbor_box["y_max"])) <= float(tol_px)
    return False


def extract_state_points_from_neighbor(*, neighbor_meta: Dict, side_to_current: str, patch_size: int, boundary_tol_px: float, trace_points_per_hint: int) -> List[Dict]:
    neighbor_box = dict(neighbor_meta.get("target_box", {}) or {})
    if not neighbor_box:
        return []
    neighbor_subpatch_id = safe_int(neighbor_meta.get("subpatch_id", -1), default=-1)
    output: List[Dict] = []
    for line_index, line in enumerate(neighbor_meta.get("target_lines", [])):
        pts = np.asarray(line.get("points", []), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        endpoints = [(0, str(line.get("start_type", "start")).strip().lower()), (-1, str(line.get("end_type", "end")).strip().lower())]
        for endpoint_idx, endpoint_type in endpoints:
            if endpoint_type != "cut":
                continue
            point_xy = pts[0] if int(endpoint_idx) == 0 else pts[-1]
            if not endpoint_matches_neighbor_boundary(point_xy=point_xy, neighbor_box=neighbor_box, side_to_current=side_to_current, tol_px=float(boundary_tol_px)):
                continue
            trace_points = extract_trace_points_for_endpoint(pts=pts, endpoint_idx=int(endpoint_idx), patch_size=patch_size, max_points=int(trace_points_per_hint))
            if len(trace_points) < 2:
                continue
            output.append({"source_patch": int(neighbor_subpatch_id), "points": trace_points, "boundary_side": str(side_to_current), "line_index": int(line_index)})
    return output


def sort_state_points(points: List[Dict]) -> List[Dict]:
    return sorted(points, key=lambda item: (int(item.get("source_patch", 1_000_000_000)), *point_sort_key((item.get("points") or [[1e9, 1e9]])[0])))


def extract_state_points(*, source_group_meta: Dict[int, Dict], grid_size: int, grid_row: int, grid_col: int, patch_size: int, boundary_tol_px: float, trace_points_per_hint: int) -> List[Dict]:
    subpatch_id = int(grid_row) * int(grid_size) + int(grid_col)
    left_neighbor = subpatch_id - 1 if int(grid_col) > 0 else None
    top_neighbor = subpatch_id - int(grid_size) if int(grid_row) > 0 else None
    state_points: List[Dict] = []
    if left_neighbor is not None:
        state_points.extend(extract_state_points_from_neighbor(neighbor_meta=source_group_meta.get(int(left_neighbor), {}), side_to_current="left", patch_size=patch_size, boundary_tol_px=float(boundary_tol_px), trace_points_per_hint=int(trace_points_per_hint)))
    if top_neighbor is not None:
        state_points.extend(extract_state_points_from_neighbor(neighbor_meta=source_group_meta.get(int(top_neighbor), {}), side_to_current="top", patch_size=patch_size, boundary_tol_px=float(boundary_tol_px), trace_points_per_hint=int(trace_points_per_hint)))
    seen = set()
    deduped: List[Dict] = []
    for item in sort_state_points(state_points):
        key = (int(item["source_patch"]), tuple((int(point[0]), int(point[1])) for point in item["points"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def format_stageb_trace_prompt(*, target_box: Dict[str, int], state_points: Sequence[Dict], prompt_template: str = "", state_mode: str = "gt") -> str:
    trace_points_json = json.dumps([{"points": [list(point) for point in item["points"]]} for item in state_points], ensure_ascii=False, separators=(",", ":"))
    normalized_state_mode = str(state_mode).strip().lower()
    if str(prompt_template).strip():
        template = str(prompt_template).strip()
    elif normalized_state_mode == "none":
        template = STAGEB_NO_STATE_PROMPT_TEMPLATE
    else:
        template = STAGEB_TRACE_PROMPT_TEMPLATE
    try:
        return template.format(
            box_x_min=int(target_box["x_min"]),
            box_y_min=int(target_box["y_min"]),
            box_x_max=int(target_box["x_max"]),
            box_y_max=int(target_box["y_max"]),
            trace_points_json=trace_points_json,
        )
    except KeyError as exc:
        missing_key = str(exc).strip("'\"")
        raise ValueError(f"Stage B prompt template contains an unknown placeholder: {missing_key}") from exc


def save_stageb_visualization(*, patch_image: Image.Image, target_lines: Sequence[Dict], target_box: Dict[str, int], state_points: Sequence[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = patch_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    patch_w, patch_h = image.size
    draw.rectangle((0, 0, patch_w - 1, patch_h - 1), outline=(255, 0, 180), width=2)
    draw.rectangle((int(target_box["x_min"]), int(target_box["y_min"]), int(target_box["x_max"]), int(target_box["y_max"])), outline=(255, 210, 0), width=3)
    for line in target_lines:
        pts = [tuple(int(value) for value in point) for point in line.get("points", [])]
        if len(pts) >= 2:
            draw.line(pts, fill=(40, 220, 255), width=3)
            draw_endpoint(draw, pts[0], (0, 180, 220), 3)
            draw_endpoint(draw, pts[-1], (0, 180, 220), 3)
    for item in state_points:
        trace_points = [list(map(int, point[:2])) for point in item.get("points", [])]
        if len(trace_points) < 2:
            continue
        color = (255, 140, 0) if str(item.get("boundary_side")) == "left" else (120, 255, 80)
        draw_polyline(draw, trace_points, color=color, width=3)
        draw_endpoint(draw, trace_points[0], color, 4)
        draw_endpoint(draw, trace_points[-1], color, 3)
    image.save(out_path)