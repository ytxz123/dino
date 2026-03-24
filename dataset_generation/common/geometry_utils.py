from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def deduplicate_points(points: Sequence[np.ndarray], eps: float = 1e-3) -> np.ndarray:
    """去掉相邻重复点，避免裁切和重采样后产生零长度段。"""
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    output = [array[0]]
    for index in range(1, array.shape[0]):
        if float(np.linalg.norm(array[index] - output[-1])) > float(eps):
            output.append(array[index])
    return np.asarray(output, dtype=np.float32)


def clamp_points(points_xy: np.ndarray, patch_size: int) -> np.ndarray:
    """把坐标限制在 patch 边界内。"""
    array = np.asarray(points_xy, dtype=np.float32).copy()
    if array.ndim != 2 or array.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    array[:, 0] = np.clip(array[:, 0], 0.0, float(patch_size - 1))
    array[:, 1] = np.clip(array[:, 1], 0.0, float(patch_size - 1))
    return array


def clamp_points_float_rect(points_xy: np.ndarray, patch_width: float, patch_height: float) -> np.ndarray:
    """把浮点坐标限制到给定宽高范围。"""
    array = np.asarray(points_xy, dtype=np.float32).copy()
    if array.ndim != 2 or array.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    array[:, 0] = np.clip(array[:, 0], 0.0, max(0.0, float(patch_width)))
    array[:, 1] = np.clip(array[:, 1], 0.0, max(0.0, float(patch_height)))
    return array


def simplify_points_for_json(points_xy: np.ndarray, patch_size: int) -> List[List[int]]:
    """把浮点点列整理成训练 JSON 使用的整数点列。"""
    array = clamp_points(points_xy, patch_size=patch_size)
    if array.ndim != 2 or array.shape[0] == 0:
        return []
    rounded = np.rint(array).astype(np.int32)
    deduped = deduplicate_points(rounded.astype(np.float32)).astype(np.int32)
    return [[int(x), int(y)] for x, y in deduped.tolist()]


def resample_polyline(points_xy: np.ndarray, step_px: float, max_points: Optional[int] = None) -> np.ndarray:
    """按固定步长重采样 polyline。"""
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
    return deduplicate_points(sampled)


def resample_polyline_keep_tail(points_xy: np.ndarray, step_px: float, max_points: Optional[int] = None) -> np.ndarray:
    """按固定步长重采样，但始终保留终点对应的余段。"""
    points = deduplicate_points(np.asarray(points_xy, dtype=np.float32))
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
    return deduplicate_points(sampled).astype(np.float32)


def point_in_rect(point_xy: np.ndarray, rect: Tuple[float, float, float, float], eps: float = 1e-6) -> bool:
    """判断点是否在矩形内。"""
    x_min, y_min, x_max, y_max = rect
    x = float(point_xy[0])
    y = float(point_xy[1])
    return (x_min - eps) <= x <= (x_max + eps) and (y_min - eps) <= y <= (y_max + eps)


def clip_segment_liang_barsky(
    start_point: np.ndarray,
    end_point: np.ndarray,
    rect: Tuple[float, float, float, float],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """用 Liang-Barsky 算法把线段裁到矩形内。"""
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
    """把 polyline 裁成一个或多个位于矩形内的片段。"""
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2:
        return []
    pieces: List[np.ndarray] = []
    current_piece: List[np.ndarray] = []
    for index in range(points.shape[0] - 1):
        clipped = clip_segment_liang_barsky(points[index], points[index + 1], rect)
        if clipped is None:
            if len(current_piece) >= 2:
                pieces.append(deduplicate_points(current_piece))
            current_piece = []
            continue
        clipped_start, clipped_end = clipped
        if not current_piece:
            current_piece = [clipped_start, clipped_end]
        elif float(np.linalg.norm(current_piece[-1] - clipped_start)) <= 1e-3:
            current_piece.append(clipped_end)
        else:
            if len(current_piece) >= 2:
                pieces.append(deduplicate_points(current_piece))
            current_piece = [clipped_start, clipped_end]
        if not point_in_rect(points[index + 1], rect):
            if len(current_piece) >= 2:
                pieces.append(deduplicate_points(current_piece))
            current_piece = []
    if len(current_piece) >= 2:
        pieces.append(deduplicate_points(current_piece))
    return [piece for piece in pieces if piece.shape[0] >= 2]


def point_boundary_side(point_xy: np.ndarray, rect: Tuple[float, float, float, float], tol_px: float) -> Optional[str]:
    """判断点贴近矩形哪一条边，用于 cut 标记。"""
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
    """给线段排序时使用的稳定原点序。"""
    x = float(point_xy[0])
    y = float(point_xy[1])
    return (x * x + y * y, y, x)


def canonicalize_line_direction(points_xy: np.ndarray, start_type: str, end_type: str) -> Tuple[np.ndarray, str, str]:
    """统一线方向，优先让 cut 端落在起点侧。"""
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
    """稳定排序 lines，保证输出可复现。"""
    return sorted(lines, key=lambda item: (*point_origin_sort_key(item.get("points", [[1e9, 1e9]])[0]), int(item.get("source_patch", 1_000_000_000))))


def ensure_closed_ring(points_xy: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """保证 polygon ring 首尾闭合。"""
    points = deduplicate_points(np.asarray(points_xy, dtype=np.float32), eps=eps)
    if points.ndim != 2 or points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if points.shape[0] == 1:
        return np.repeat(points, 2, axis=0)
    if float(np.linalg.norm(points[0] - points[-1])) <= float(eps):
        points[-1] = points[0]
        return points.astype(np.float32)
    return np.concatenate([points, points[:1]], axis=0).astype(np.float32)


def clip_polygon_ring_to_rect(points_xy: np.ndarray, rect: Tuple[float, float, float, float]) -> List[np.ndarray]:
    """用 Sutherland-Hodgman 算法裁切 polygon ring。"""
    ring = ensure_closed_ring(np.asarray(points_xy, dtype=np.float32))
    if ring.ndim != 2 or ring.shape[0] < 4:
        return []
    polygon = ring[:-1].astype(np.float32)
    x_min, y_min, x_max, y_max = [float(value) for value in rect]

    def inside_left(point):
        return float(point[0]) >= x_min

    def inside_right(point):
        return float(point[0]) <= x_max

    def inside_top(point):
        return float(point[1]) >= y_min

    def inside_bottom(point):
        return float(point[1]) <= y_max

    def intersect_vertical(start, end, x_edge):
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)
        delta_x = float(end[0] - start[0])
        if abs(delta_x) <= 1e-6:
            return np.asarray([float(x_edge), float(start[1])], dtype=np.float32)
        ratio = (float(x_edge) - float(start[0])) / delta_x
        return np.asarray([float(x_edge), float(start[1] + ratio * (end[1] - start[1]))], dtype=np.float32)

    def intersect_horizontal(start, end, y_edge):
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)
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
    """计算折线长度。"""
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(points[1:] - points[:-1], axis=1).sum())