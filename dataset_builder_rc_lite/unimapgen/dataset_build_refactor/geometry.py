"""RC 方案用到的最小几何工具集合。"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def dedup_points(points: Sequence[np.ndarray], eps: float = 1e-3) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    output = [array[0]]
    for index in range(1, array.shape[0]):
        if float(np.linalg.norm(array[index] - output[-1])) > float(eps):
            output.append(array[index])
    return np.asarray(output, dtype=np.float32)


def clamp_points(points_xy: np.ndarray, patch_size: int) -> np.ndarray:
    array = np.asarray(points_xy, dtype=np.float32).copy()
    if array.ndim != 2 or array.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    max_coord = float(max(1, int(patch_size) - 1))
    array[:, 0] = np.clip(array[:, 0], 0.0, max_coord)
    array[:, 1] = np.clip(array[:, 1], 0.0, max_coord)
    return array


def clamp_points_float_rect(points_xy: np.ndarray, patch_width: float, patch_height: float) -> np.ndarray:
    array = np.asarray(points_xy, dtype=np.float32).copy()
    if array.ndim != 2 or array.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    array[:, 0] = np.clip(array[:, 0], 0.0, max(0.0, float(patch_width)))
    array[:, 1] = np.clip(array[:, 1], 0.0, max(0.0, float(patch_height)))
    return array


def simplify_for_json(points_xy: np.ndarray, patch_size: int) -> List[List[int]]:
    array = clamp_points(points_xy, patch_size=patch_size)
    if array.ndim != 2 or array.shape[0] == 0:
        return []
    rounded = np.rint(array).astype(np.int32)
    deduped = dedup_points(rounded.astype(np.float32)).astype(np.int32)
    return [[int(x), int(y)] for x, y in deduped.tolist()]


def resample_polyline(points_xy: np.ndarray, step_px: float, max_points: Optional[int] = None) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2:
        return points
    segment_lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
    total_length = float(np.sum(segment_lengths))
    if total_length < 1e-6:
        return points[:1]
    step = max(float(step_px), 1.0)
    point_count = max(2, int(math.floor(total_length / step)) + 1)
    if max_points is not None:
        point_count = min(point_count, int(max_points))
    targets = np.linspace(0.0, total_length, point_count, dtype=np.float32)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    sampled: List[np.ndarray] = []
    for target in targets:
        segment_index = int(np.searchsorted(cumulative, target, side="right") - 1)
        segment_index = min(max(segment_index, 0), len(segment_lengths) - 1)
        start_distance = float(cumulative[segment_index])
        end_distance = float(cumulative[segment_index + 1])
        ratio = 0.0 if end_distance <= start_distance else (float(target) - start_distance) / (end_distance - start_distance)
        sampled.append(points[segment_index] * (1.0 - ratio) + points[segment_index + 1] * ratio)
    return dedup_points(sampled)


def resample_polyline_keep_tail(points_xy: np.ndarray, step_px: float, max_points: Optional[int] = None) -> np.ndarray:
    points = dedup_points(np.asarray(points_xy, dtype=np.float32))
    if points.ndim != 2 or points.shape[0] < 2:
        return points.astype(np.float32)
    step = float(step_px)
    if step <= 0.0:
        return points.astype(np.float32)
    segment_lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
    total_length = float(np.sum(segment_lengths))
    if total_length < 1e-6:
        return points[:1].astype(np.float32)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    targets: List[float] = [0.0]
    distance = float(step)
    while distance < total_length:
        targets.append(float(distance))
        distance += float(step)
    if targets[-1] != float(total_length):
        targets.append(float(total_length))
    if max_points is not None and int(max_points) > 0 and len(targets) > int(max_points):
        targets = targets[: max(1, int(max_points) - 1)] + [float(total_length)]
    sampled: List[np.ndarray] = []
    for target in targets:
        if target >= total_length:
            sampled.append(points[-1].astype(np.float32))
            continue
        segment_index = int(np.searchsorted(cumulative, target, side="right") - 1)
        segment_index = min(max(segment_index, 0), len(segment_lengths) - 1)
        start_distance = float(cumulative[segment_index])
        end_distance = float(cumulative[segment_index + 1])
        ratio = 0.0 if end_distance <= start_distance else (float(target) - start_distance) / (end_distance - start_distance)
        sampled.append((points[segment_index] * (1.0 - ratio) + points[segment_index + 1] * ratio).astype(np.float32))
    return dedup_points(sampled).astype(np.float32)


def point_in_rect(point_xy: np.ndarray, rect: Tuple[float, float, float, float], eps: float = 1e-6) -> bool:
    x_min, y_min, x_max, y_max = rect
    x = float(point_xy[0])
    y = float(point_xy[1])
    return (x_min - eps) <= x <= (x_max + eps) and (y_min - eps) <= y <= (y_max + eps)


def clip_segment_liang_barsky(start_point: np.ndarray, end_point: np.ndarray, rect: Tuple[float, float, float, float]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    x_min, y_min, x_max, y_max = rect
    delta_x = float(end_point[0] - start_point[0])
    delta_y = float(end_point[1] - start_point[1])
    p_values = [-delta_x, delta_x, -delta_y, delta_y]
    q_values = [
        float(start_point[0] - x_min),
        float(x_max - start_point[0]),
        float(start_point[1] - y_min),
        float(y_max - start_point[1]),
    ]
    low = 0.0
    high = 1.0
    for p_value, q_value in zip(p_values, q_values):
        if abs(p_value) < 1e-8:
            if q_value < 0.0:
                return None
            continue
        ratio = q_value / p_value
        if p_value < 0.0:
            if ratio > high:
                return None
            if ratio > low:
                low = ratio
        else:
            if ratio < low:
                return None
            if ratio < high:
                high = ratio
    clipped_start = np.asarray([start_point[0] + low * delta_x, start_point[1] + low * delta_y], dtype=np.float32)
    clipped_end = np.asarray([start_point[0] + high * delta_x, start_point[1] + high * delta_y], dtype=np.float32)
    return clipped_start, clipped_end


def clip_polyline_to_rect(points_xy: np.ndarray, rect: Tuple[float, float, float, float]) -> List[np.ndarray]:
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2:
        return []
    pieces: List[np.ndarray] = []
    current_piece: List[np.ndarray] = []
    for index in range(points.shape[0] - 1):
        clipped = clip_segment_liang_barsky(points[index], points[index + 1], rect)
        if clipped is None:
            if len(current_piece) >= 2:
                pieces.append(dedup_points(current_piece))
            current_piece = []
            continue
        clipped_start, clipped_end = clipped
        if not current_piece:
            current_piece = [clipped_start, clipped_end]
        elif float(np.linalg.norm(current_piece[-1] - clipped_start)) <= 1e-3:
            current_piece.append(clipped_end)
        else:
            if len(current_piece) >= 2:
                pieces.append(dedup_points(current_piece))
            current_piece = [clipped_start, clipped_end]
        if not point_in_rect(points[index + 1], rect):
            if len(current_piece) >= 2:
                pieces.append(dedup_points(current_piece))
            current_piece = []
    if len(current_piece) >= 2:
        pieces.append(dedup_points(current_piece))
    return [piece for piece in pieces if piece.shape[0] >= 2]


def point_boundary_side(point_xy: np.ndarray, rect: Tuple[float, float, float, float], tol_px: float) -> Optional[str]:
    x_min, y_min, x_max, y_max = rect
    x = float(point_xy[0])
    y = float(point_xy[1])
    if abs(x - x_min) <= tol_px:
        return "left"
    if abs(y - y_min) <= tol_px:
        return "top"
    if abs(x - x_max) <= tol_px:
        return "right"
    if abs(y - y_max) <= tol_px:
        return "bottom"
    return None


def point_origin_sort_key(point_xy: Sequence[float]) -> Tuple[float, float, float]:
    x = float(point_xy[0])
    y = float(point_xy[1])
    return (x * x + y * y, y, x)


def canonicalize_line_direction(points_xy: np.ndarray, start_type: str, end_type: str) -> Tuple[np.ndarray, str, str]:
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2:
        return points, start_type, end_type
    reverse = False
    start_is_cut = str(start_type) == "cut"
    end_is_cut = str(end_type) == "cut"
    if end_is_cut and not start_is_cut:
        reverse = True
    elif not start_is_cut and not end_is_cut and point_origin_sort_key(points[-1]) < point_origin_sort_key(points[0]):
        reverse = True
    if not reverse:
        return points, start_type, end_type
    return points[::-1].copy(), end_type, start_type


def sort_lines(lines: List[Dict]) -> List[Dict]:
    def first_point(item: Dict) -> Sequence[float]:
        points = item.get("points")
        if points is None or len(points) == 0:
            points = item.get("points_global")
        if points is None or len(points) == 0:
            points = [[1e9, 1e9]]
        return points[0]

    return sorted(lines, key=lambda item: (*point_origin_sort_key(first_point(item)), int(item.get("source_patch", 1_000_000_000))))


def ensure_closed_ring(points_xy: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    points = dedup_points(np.asarray(points_xy, dtype=np.float32), eps=eps)
    if points.ndim != 2 or points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if points.shape[0] == 1:
        return np.repeat(points, 2, axis=0)
    if float(np.linalg.norm(points[0] - points[-1])) <= float(eps):
        points[-1] = points[0]
        return points.astype(np.float32)
    return np.concatenate([points, points[:1]], axis=0).astype(np.float32)


def clip_polygon_ring_to_rect(points_xy: np.ndarray, rect: Tuple[float, float, float, float]) -> List[np.ndarray]:
    ring = ensure_closed_ring(np.asarray(points_xy, dtype=np.float32))
    if ring.ndim != 2 or ring.shape[0] < 4:
        return []
    polygon = ring[:-1].astype(np.float32)
    x_min, y_min, x_max, y_max = [float(value) for value in rect]

    def inside_left(point: np.ndarray) -> bool:
        return float(point[0]) >= x_min

    def inside_right(point: np.ndarray) -> bool:
        return float(point[0]) <= x_max

    def inside_top(point: np.ndarray) -> bool:
        return float(point[1]) >= y_min

    def inside_bottom(point: np.ndarray) -> bool:
        return float(point[1]) <= y_max

    def intersect_vertical(start: np.ndarray, end: np.ndarray, x_edge: float) -> np.ndarray:
        delta_x = float(end[0] - start[0])
        if abs(delta_x) <= 1e-6:
            return np.asarray([float(x_edge), float(start[1])], dtype=np.float32)
        ratio = (float(x_edge) - float(start[0])) / delta_x
        return np.asarray([float(x_edge), float(start[1] + ratio * (end[1] - start[1]))], dtype=np.float32)

    def intersect_horizontal(start: np.ndarray, end: np.ndarray, y_edge: float) -> np.ndarray:
        delta_y = float(end[1] - start[1])
        if abs(delta_y) <= 1e-6:
            return np.asarray([float(start[0]), float(y_edge)], dtype=np.float32)
        ratio = (float(y_edge) - float(start[1])) / delta_y
        return np.asarray([float(start[0] + ratio * (end[0] - start[0])), float(y_edge)], dtype=np.float32)

    def clip_against(subject: List[np.ndarray], inside_fn, intersect_fn) -> List[np.ndarray]:
        if not subject:
            return []
        output: List[np.ndarray] = []
        previous = subject[-1]
        previous_inside = bool(inside_fn(previous))
        for current in subject:
            current_inside = bool(inside_fn(current))
            if current_inside:
                if not previous_inside:
                    output.append(np.asarray(intersect_fn(previous, current), dtype=np.float32))
                output.append(np.asarray(current, dtype=np.float32))
            elif previous_inside:
                output.append(np.asarray(intersect_fn(previous, current), dtype=np.float32))
            previous = current
            previous_inside = current_inside
        return output

    subject = [point.astype(np.float32) for point in polygon]
    subject = clip_against(subject, inside_left, lambda s, e: intersect_vertical(s, e, x_min))
    subject = clip_against(subject, inside_right, lambda s, e: intersect_vertical(s, e, x_max))
    subject = clip_against(subject, inside_top, lambda s, e: intersect_horizontal(s, e, y_min))
    subject = clip_against(subject, inside_bottom, lambda s, e: intersect_horizontal(s, e, y_max))
    if len(subject) < 3:
        return []
    clipped_ring = ensure_closed_ring(np.asarray(subject, dtype=np.float32))
    return [clipped_ring] if clipped_ring.shape[0] >= 4 else []


def line_length_xy(points_xy: Sequence[Sequence[float]]) -> float:
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(points[1:] - points[:-1], axis=1).sum())
