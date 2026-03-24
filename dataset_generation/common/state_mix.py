from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence

import numpy as np

from .geometry_utils import clamp_points, simplify_points_for_json, sort_lines


def build_sample_rng(sample_key: str) -> np.random.Generator:
    """为每个样本生成稳定随机数种子，保证下采样和弱 state 可复现。"""
    digest = hashlib.sha256(sample_key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return np.random.default_rng(seed)


def choose_state_mode(
    rng: np.random.Generator,
    mixture_mode: str,
    raw_state_lines: Sequence[Dict],
    no_state_ratio: float,
    weak_ratio: float,
    full_ratio: float,
) -> str:
    """按照旧逻辑选择 empty / no_state / weak_state / full_state。"""
    if len(raw_state_lines) == 0:
        return "empty"
    if str(mixture_mode) != "mixed":
        return "full"
    total = max(float(no_state_ratio) + float(weak_ratio) + float(full_ratio), 1e-8)
    no_state_limit = max(float(no_state_ratio), 0.0) / total
    weak_limit = no_state_limit + max(float(weak_ratio), 0.0) / total
    score = float(rng.random())
    if score < no_state_limit:
        return "no_state"
    if score < weak_limit:
        return "weak_state"
    return "full_state"


def apply_point_jitter(points: Sequence[Sequence[int]], jitter_px: float, patch_size: int, rng: np.random.Generator) -> List[List[int]]:
    """对弱 state 点列施加随机抖动。"""
    if float(jitter_px) <= 0.0:
        return [[int(point[0]), int(point[1])] for point in points]
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 2:
        return []
    noise = rng.uniform(low=-float(jitter_px), high=float(jitter_px), size=array.shape).astype(np.float32)
    jittered = clamp_points(array + noise, patch_size=patch_size)
    return simplify_points_for_json(jittered, patch_size=patch_size)


def build_state_lines_by_mode(
    raw_state_lines: Sequence[Dict],
    state_mode: str,
    patch_size: int,
    weak_trace_points: int,
    state_line_dropout: float,
    state_point_jitter_px: float,
    state_truncate_prob: float,
    rng: np.random.Generator,
) -> List[Dict]:
    """按旧规则构造 full 或 weak state。"""
    if state_mode in {"empty", "no_state"}:
        return []
    if state_mode in {"full", "full_state"}:
        return [dict(line) for line in raw_state_lines]
    weak_lines: List[Dict] = []
    keep_prob = 1.0 - max(0.0, min(1.0, float(state_line_dropout)))
    max_trace_points = max(2, int(weak_trace_points))
    truncate_prob = max(0.0, min(1.0, float(state_truncate_prob)))
    for line in raw_state_lines:
        if float(rng.random()) > keep_prob:
            continue
        points = [list(map(int, point)) for point in line.get("points", [])]
        if len(points) < 2:
            continue
        if float(rng.random()) < truncate_prob and len(points) > 2:
            new_length = int(rng.integers(2, min(len(points), max_trace_points) + 1))
        else:
            new_length = min(len(points), max_trace_points)
        truncated = points[:new_length]
        jittered = apply_point_jitter(
            points=truncated,
            jitter_px=state_point_jitter_px,
            patch_size=patch_size,
            rng=rng,
        )
        if len(jittered) < 2:
            continue
        weak_lines.append(
            {
                "source_patch": int(line.get("source_patch", -1)),
                "category": str(line.get("category", "road")),
                "start_type": str(line.get("start_type", "cut")),
                "end_type": str(line.get("end_type", "cut")),
                "points": jittered,
            }
        )
    if weak_lines:
        return sort_lines(weak_lines)
    fallback = next((line for line in raw_state_lines if len(line.get("points", [])) >= 2), None)
    if fallback is None:
        return []
    points = [list(map(int, point)) for point in fallback["points"][:max_trace_points]]
    jittered = apply_point_jitter(points=points, jitter_px=state_point_jitter_px, patch_size=patch_size, rng=rng)
    if len(jittered) < 2:
        return []
    return [
        {
            "source_patch": int(fallback.get("source_patch", -1)),
            "category": str(fallback.get("category", "road")),
            "start_type": str(fallback.get("start_type", "cut")),
            "end_type": str(fallback.get("end_type", "cut")),
            "points": jittered,
        }
    ]