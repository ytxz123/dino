from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from dataset_generation.common.geometry_utils import (
    canonicalize_line_direction,
    clamp_points,
    clip_polyline_to_rect,
    line_length_xy,
    point_boundary_side,
    resample_polyline,
    simplify_points_for_json,
    sort_lines,
)
from dataset_generation.common.io_utils import build_sharegpt_dataset_info, ensure_directory, load_jsonl, write_json, write_jsonl
from dataset_generation.common.progress import log_progress


DEFAULT_FIXED16_PROMPT_TEMPLATE = """<image>
Please construct the road map from ({start_x},{start_y}) to ({end_x},{end_y}) in the satellite image.
Only predict road segments inside the target box [{box_x_min},{box_y_min},{box_x_max},{box_y_max}].
Keep all coordinates in the patch-local coordinate system."""

DEFAULT_FIXED16_STATE_PROMPT_TEMPLATE = """<image>
Please construct the road map from ({start_x},{start_y}) to ({end_x},{end_y}) in the satellite image.
Only predict road segments inside the target box [{box_x_min},{box_y_min},{box_x_max},{box_y_max}].
Keep all coordinates in the patch-local coordinate system.
Previous state:
{state_json}"""


def extract_message_content(row: Dict, role: str) -> str:
    """从 ShareGPT messages 中取指定角色文本。"""
    wanted_role = str(role).strip().lower()
    for message in row.get("messages", []):
        if str(message.get("role", "")).strip().lower() == wanted_role:
            return str(message.get("content", ""))
    return ""


def build_grid_boxes(patch_size: int, grid_size: int) -> List[Dict[str, int]]:
    """把整张 patch 均匀切成 grid_size x grid_size 个 box。"""
    edges = [int(round(index * patch_size / grid_size)) for index in range(grid_size + 1)]
    boxes: List[Dict[str, int]] = []
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
    """在当前 box 中找最长的线段片段，用它生成 prompt 锚点。"""
    best_piece: Optional[np.ndarray] = None
    best_length = -1.0
    for line in lines:
        points = np.asarray(line.get("points", []), dtype=np.float32)
        for piece in clip_polyline_to_rect(points, rect):
            length = line_length_xy(piece.tolist())
            if length > best_length:
                best_length = length
                best_piece = piece
    return best_piece


def build_prompt_endpoints(target_lines: Sequence[Dict], target_box: Dict[str, int], patch_size: int) -> Dict[str, object]:
    """优先用最长片段两端作为 prompt 锚点，空 box 时退化到 box 中心。"""
    rect = (
        float(target_box["x_min"]),
        float(target_box["y_min"]),
        float(target_box["x_max"]),
        float(target_box["y_max"]),
    )
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
    points_json = simplify_points_for_json(piece, patch_size=patch_size)
    if len(points_json) < 2:
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
        "start_x": int(points_json[0][0]),
        "start_y": int(points_json[0][1]),
        "end_x": int(points_json[-1][0]),
        "end_y": int(points_json[-1][1]),
        "anchor_piece_points": points_json,
        "anchor_source": "longest_clipped_piece",
    }


def build_target_lines_for_box(
    full_patch_target_lines: Sequence[Dict],
    target_box: Dict[str, int],
    patch_size: int,
    boundary_tol_px: float,
    resample_step_px: float,
) -> List[Dict]:
    """从 patch 级 target_lines 中裁出当前 box 的 box 级真值。"""
    rect = (
        float(target_box["x_min"]),
        float(target_box["y_min"]),
        float(target_box["x_max"]),
        float(target_box["y_max"]),
    )
    output: List[Dict] = []
    for segment in full_patch_target_lines:
        points = np.asarray(segment.get("points", []), dtype=np.float32)
        for piece in clip_polyline_to_rect(points, rect):
            piece = clamp_points(piece, patch_size=patch_size)
            if float(resample_step_px) > 0.0:
                piece = resample_polyline(piece, step_px=float(resample_step_px))
            if piece.ndim != 2 or piece.shape[0] < 2:
                continue
            start_side = point_boundary_side(piece[0], rect, boundary_tol_px)
            end_side = point_boundary_side(piece[-1], rect, boundary_tol_px)
            start_type = "cut" if start_side is not None else "start"
            end_type = "cut" if end_side is not None else "end"
            piece, start_type, end_type = canonicalize_line_direction(piece, start_type=start_type, end_type=end_type)
            points_json = simplify_points_for_json(piece, patch_size=patch_size)
            if len(points_json) < 2:
                continue
            output.append(
                {
                    "category": str(segment.get("category", "road")),
                    "start_type": str(start_type),
                    "end_type": str(end_type),
                    "points": points_json,
                }
            )
    return sort_lines(output)


def format_prompt_text(prompt_fields: Dict[str, int], state_json: Optional[str] = None) -> str:
    """构造 fixed16 user prompt。"""
    if state_json is None:
        return DEFAULT_FIXED16_PROMPT_TEMPLATE.format(**prompt_fields)
    fields = dict(prompt_fields)
    fields["state_json"] = str(state_json)
    return DEFAULT_FIXED16_STATE_PROMPT_TEMPLATE.format(**fields)


def make_patch_only_record(sample_id: str, image_rel_path: str, prompt_text: str, target_lines: Sequence[Dict], system_prompt: str) -> Dict:
    """生成 patch-only fixed16 训练记录。"""
    target_json = json.dumps({"lines": list(target_lines)}, ensure_ascii=False, separators=(",", ":"))
    messages: List[Dict] = []
    if str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    messages.append({"role": "user", "content": str(prompt_text)})
    messages.append({"role": "assistant", "content": target_json})
    return {"id": sample_id, "messages": messages, "images": [image_rel_path]}


def make_state_record(sample_id: str, image_rel_path: str, prompt_text: str, target_lines: Sequence[Dict], state_lines: Sequence[Dict], system_prompt: str) -> Dict:
    """生成 state-aware fixed16 训练记录。"""
    state_json = json.dumps({"lines": list(state_lines)}, ensure_ascii=False, separators=(",", ":"))
    target_json = json.dumps({"lines": list(target_lines)}, ensure_ascii=False, separators=(",", ":"))
    messages: List[Dict] = []
    if str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    user_text = str(prompt_text).format(state_json=state_json) if "{state_json}" in str(prompt_text) else str(prompt_text)
    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": target_json})
    return {"id": sample_id, "messages": messages, "images": [image_rel_path]}


def sanitize_name(name: str) -> str:
    """把目录名整理成 dataset_info 的安全前缀。"""
    output = []
    for character in str(name):
        output.append(character if character.isalnum() or character in ("_", "-") else "_")
    return "".join(output).strip("_") or "dataset"


def expose_source_images(input_root: Path, output_root: Path, mode: str) -> str:
    """通过软链或复制复用上游 stage 数据集图片。"""
    source_images = input_root / "images"
    target_images = output_root / "images"
    if not source_images.exists() or str(mode) == "none":
        return "none"
    if target_images.exists() or target_images.is_symlink():
        return "existing"
    if str(mode) == "symlink":
        try:
            target_images.symlink_to(source_images, target_is_directory=True)
            return "symlink"
        except OSError:
            shutil.copytree(source_images, target_images)
            return "copy_fallback"
    shutil.copytree(source_images, target_images)
    return "copy"


def filter_pairs_to_empty_ratio(pairs: Sequence[Tuple[Dict, Dict]], target_empty_ratio: float, rng: random.Random) -> Tuple[List[Tuple[Dict, Dict]], Dict[str, float]]:
    """按目标空样本比例过滤 box 级空样本。"""
    non_empty_pairs = [pair for pair in pairs if int(pair[1].get("num_target_lines", 0)) > 0]
    empty_pairs = [pair for pair in pairs if int(pair[1].get("num_target_lines", 0)) <= 0]
    if not non_empty_pairs:
        raise ValueError("Split contains no non-empty samples; cannot enforce empty-ratio target.")
    if float(target_empty_ratio) >= 1.0:
        kept_pairs = list(pairs)
        kept_empty = len(empty_pairs)
        kept_total = len(kept_pairs)
        return kept_pairs, {
            "generated_total": len(pairs),
            "generated_non_empty": len(non_empty_pairs),
            "generated_empty": len(empty_pairs),
            "kept_total": kept_total,
            "kept_non_empty": kept_total - kept_empty,
            "kept_empty": kept_empty,
            "kept_empty_ratio": (kept_empty / kept_total if kept_total else 0.0),
        }
    max_empty = math.floor(len(non_empty_pairs) * float(target_empty_ratio) / (1.0 - float(target_empty_ratio)))
    keep_empty = min(len(empty_pairs), max_empty)
    kept_empty_ids = set()
    if keep_empty > 0:
        chosen = rng.sample(empty_pairs, keep_empty) if keep_empty < len(empty_pairs) else list(empty_pairs)
        kept_empty_ids = {str(meta["id"]) for _, meta in chosen}
    kept_pairs: List[Tuple[Dict, Dict]] = []
    for row, meta in pairs:
        if int(meta.get("num_target_lines", 0)) > 0 or str(meta["id"]) in kept_empty_ids:
            kept_pairs.append((row, meta))
    kept_empty = sum(1 for _, meta in kept_pairs if int(meta.get("num_target_lines", 0)) <= 0)
    kept_total = len(kept_pairs)
    return kept_pairs, {
        "generated_total": len(pairs),
        "generated_non_empty": len(non_empty_pairs),
        "generated_empty": len(empty_pairs),
        "kept_total": kept_total,
        "kept_non_empty": kept_total - kept_empty,
        "kept_empty": kept_empty,
        "kept_empty_ratio": (kept_empty / kept_total if kept_total else 0.0),
    }


def draw_endpoint(draw: ImageDraw.ImageDraw, point: Sequence[int], color: Tuple[int, int, int], radius: int = 3) -> None:
    """在调试图里画端点。"""
    x = int(point[0])
    y = int(point[1])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def save_visualization(patch_image: Image.Image, target_lines: Sequence[Dict], target_box: Dict[str, int], anchor_piece_points: Sequence[Sequence[int]], output_path: Path) -> None:
    """输出 fixed16 QA 图。"""
    ensure_directory(output_path.parent)
    image = patch_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    patch_size = int(image.size[0])
    draw.rectangle((0, 0, patch_size - 1, patch_size - 1), outline=(255, 0, 180), width=2)
    draw.rectangle((int(target_box["x_min"]), int(target_box["y_min"]), int(target_box["x_max"]), int(target_box["y_max"])), outline=(255, 210, 0), width=3)
    anchor_points = [tuple(int(value) for value in point) for point in anchor_piece_points]
    if len(anchor_points) >= 2:
        draw.line(anchor_points, fill=(255, 140, 0), width=4)
        draw_endpoint(draw, anchor_points[0], (0, 255, 80), 4)
        draw_endpoint(draw, anchor_points[-1], (255, 40, 40), 4)
    elif len(anchor_points) == 1:
        draw_endpoint(draw, anchor_points[0], (255, 140, 0), 4)
    for line in target_lines:
        points = [tuple(int(value) for value in point) for point in line.get("points", [])]
        if len(points) >= 2:
            draw.line(points, fill=(40, 220, 255), width=3)
            draw_endpoint(draw, points[0], (0, 180, 220), 3)
            draw_endpoint(draw, points[-1], (0, 180, 220), 3)
    image.save(output_path)


def build_split_dataset(
    split: str,
    input_root: Path,
    output_root: Path,
    grid_size: int,
    target_empty_ratio: float,
    rng: random.Random,
    max_source_samples_per_split: int,
    boundary_tol_px: float,
    resample_step_px: float,
    reuse_system_prompt: bool,
    export_visualizations: bool,
    max_visualizations_per_split: int,
) -> Dict[str, object]:
    """构建单个 split 的 fixed16 数据。"""
    split_jsonl = input_root / f"{split}.jsonl"
    split_meta_jsonl = input_root / f"meta_{split}.jsonl"
    if not split_jsonl.exists() or not split_meta_jsonl.exists():
        return {"missing_split": True, "split_jsonl": str(split_jsonl), "split_meta_jsonl": str(split_meta_jsonl)}
    rows = load_jsonl(split_jsonl)
    meta_rows = load_jsonl(split_meta_jsonl)
    log_progress("Fixed16", f"开始处理 split={split} source_rows={len(rows)} source_meta={len(meta_rows)}")
    row_by_id = {str(row.get("id")): row for row in rows}
    generated_pairs: List[Tuple[Dict, Dict]] = []
    visualization_count = 0
    used_source = 0
    total_source = len(meta_rows) if int(max_source_samples_per_split) <= 0 else min(len(meta_rows), int(max_source_samples_per_split))
    progress_step = max(1, total_source // 20) if total_source > 0 else 1
    for source_meta in meta_rows:
        source_id = str(source_meta.get("id"))
        source_row = row_by_id.get(source_id)
        if source_row is None:
            continue
        used_source += 1
        if int(max_source_samples_per_split) > 0 and used_source > int(max_source_samples_per_split):
            break
        if used_source == 1 or used_source == total_source or used_source % progress_step == 0:
            log_progress("Fixed16", f"split={split} source_progress={used_source}/{total_source} source_id={source_id}")
        crop_box = source_meta.get("crop_box", {})
        patch_size = int(crop_box.get("x_max", 0)) - int(crop_box.get("x_min", 0))
        if patch_size <= 1:
            continue
        full_target_lines = list(source_meta.get("target_lines", []))
        has_state = ("state_lines" in source_meta) or ("state_mode" in source_meta)
        full_state_lines = list(source_meta.get("state_lines", [])) if has_state else []
        image_rel_path = str(source_meta.get("image") or source_row.get("images", [""])[0])
        system_prompt = extract_message_content(source_row, "system") if reuse_system_prompt else ""
        boxes = build_grid_boxes(patch_size=patch_size, grid_size=grid_size)
        patch_image: Optional[Image.Image] = None
        if export_visualizations:
            patch_image = Image.open(input_root / image_rel_path).convert("RGB")
        for box in boxes:
            prompt_info = build_prompt_endpoints(target_lines=full_target_lines, target_box=box, patch_size=patch_size)
            target_lines = build_target_lines_for_box(
                full_patch_target_lines=full_target_lines,
                target_box=box,
                patch_size=patch_size,
                boundary_tol_px=float(boundary_tol_px),
                resample_step_px=float(resample_step_px),
            )
            sample_id = f"{source_id}_g{int(box['grid_row'])}{int(box['grid_col'])}"
            prompt_fields = {
                "start_x": int(prompt_info["start_x"]),
                "start_y": int(prompt_info["start_y"]),
                "end_x": int(prompt_info["end_x"]),
                "end_y": int(prompt_info["end_y"]),
                "box_x_min": int(box["x_min"]),
                "box_y_min": int(box["y_min"]),
                "box_x_max": int(box["x_max"]),
                "box_y_max": int(box["y_max"]),
            }
            if has_state:
                state_json = json.dumps({"lines": list(full_state_lines)}, ensure_ascii=False, separators=(",", ":"))
                prompt_text = format_prompt_text(prompt_fields, state_json=state_json)
                row = make_state_record(
                    sample_id=sample_id,
                    image_rel_path=image_rel_path,
                    prompt_text=prompt_text,
                    target_lines=target_lines,
                    state_lines=full_state_lines,
                    system_prompt=system_prompt,
                )
            else:
                prompt_text = format_prompt_text(prompt_fields)
                row = make_patch_only_record(
                    sample_id=sample_id,
                    image_rel_path=image_rel_path,
                    prompt_text=prompt_text,
                    target_lines=target_lines,
                    system_prompt=system_prompt,
                )
            meta = {
                "id": sample_id,
                "source_id": source_id,
                "split": split,
                "family_id": source_meta.get("family_id"),
                "source_sample_id": source_meta.get("source_sample_id"),
                "source_image": source_meta.get("source_image"),
                "source_image_path": source_meta.get("source_image_path", ""),
                "source_mask_path": source_meta.get("source_mask_path", ""),
                "source_lane_path": source_meta.get("source_lane_path", ""),
                "source_intersection_path": source_meta.get("source_intersection_path", ""),
                "image_size": source_meta.get("image_size", []),
                "patch_id": source_meta.get("patch_id"),
                "row": source_meta.get("row"),
                "col": source_meta.get("col"),
                "scan_index": source_meta.get("scan_index"),
                "image": image_rel_path,
                "crop_box": crop_box,
                "keep_box": source_meta.get("keep_box", {}),
                "target_mode": "fixed_grid_target_box_map",
                "coord_system": source_meta.get("coord_system", "patch_local_896"),
                "source_dataset_type": "state" if has_state else "patch_only",
                "serialization_mode": source_meta.get("serialization_mode", "paper_structured"),
                "line_direction_mode": source_meta.get("line_direction_mode", "canonical_cut_then_origin"),
                "line_sort_mode": source_meta.get("line_sort_mode", "first_point_distance_to_patch_origin"),
                "resample_mode": "equal_distance" if float(resample_step_px) > 0 else "inherit_source_spacing",
                "resample_step_px": float(resample_step_px),
                "has_system_prompt": bool(system_prompt.strip()),
                "grid_size": int(grid_size),
                "grid_row": int(box["grid_row"]),
                "grid_col": int(box["grid_col"]),
                "target_box": {"x_min": int(box["x_min"]), "y_min": int(box["y_min"]), "x_max": int(box["x_max"]), "y_max": int(box["y_max"])},
                "target_box_area": int((int(box["x_max"]) - int(box["x_min"]) + 1) * (int(box["y_max"]) - int(box["y_min"]) + 1)),
                "anchor_source": str(prompt_info["anchor_source"]),
                "anchor_start_xy": [int(prompt_info["start_x"]), int(prompt_info["start_y"])],
                "anchor_end_xy": [int(prompt_info["end_x"]), int(prompt_info["end_y"])],
                "anchor_piece_points": prompt_info["anchor_piece_points"],
                "num_state_lines": int(len(full_state_lines)),
                "state_lines": full_state_lines,
                "num_target_lines": len(target_lines),
                "num_target_points": int(sum(len(item.get("points", [])) for item in target_lines)),
                "prompt_text": prompt_text,
                "target_lines": target_lines,
            }
            generated_pairs.append((row, meta))
            if export_visualizations and patch_image is not None:
                if int(max_visualizations_per_split) <= 0 or visualization_count < int(max_visualizations_per_split):
                    output_path = output_root / "visualizations" / split / str(source_meta.get("family_id")) / f"p{int(source_meta.get('patch_id', 0)):02d}_g{int(box['grid_row'])}{int(box['grid_col'])}.png"
                    save_visualization(
                        patch_image=patch_image,
                        target_lines=target_lines,
                        target_box=box,
                        anchor_piece_points=prompt_info["anchor_piece_points"],
                        output_path=output_path,
                    )
                    visualization_count += 1
        if patch_image is not None:
            patch_image.close()
    kept_pairs, summary = filter_pairs_to_empty_ratio(pairs=generated_pairs, target_empty_ratio=float(target_empty_ratio), rng=rng)
    output_rows = [row for row, _ in kept_pairs]
    output_meta = [meta for _, meta in kept_pairs]
    written_rows = write_jsonl(output_root / f"{split}.jsonl", output_rows)
    written_meta = write_jsonl(output_root / f"meta_{split}.jsonl", output_meta)
    summary.update({
        "used_source_samples": used_source if int(max_source_samples_per_split) <= 0 else min(used_source, int(max_source_samples_per_split)),
        "written_rows": written_rows,
        "written_meta_rows": written_meta,
        "visualizations": visualization_count,
    })
    log_progress("Fixed16", f"split={split} 完成 written_rows={written_rows} written_meta={written_meta} visualizations={visualization_count}")
    return summary


def build_fixed16_dataset(
    input_root: Path,
    output_root: Path,
    splits: Sequence[str],
    grid_size: int,
    target_empty_ratio: float,
    target_empty_ratio_by_split: Optional[Dict[str, float]],
    seed: int,
    max_source_samples_per_split: int,
    boundary_tol_px: float,
    resample_step_px: float,
    reuse_system_prompt: bool,
    image_root_mode: str,
    export_visualizations: bool,
    max_visualizations_per_split: int,
) -> Dict[str, object]:
    """构建 fixed16 数据集总入口。"""
    if int(grid_size) <= 0:
        raise ValueError("grid_size must be positive.")
    if not (0.0 <= float(target_empty_ratio) <= 1.0):
        raise ValueError("target_empty_ratio must be in [0, 1].")
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    ensure_directory(output_root)
    log_progress("Fixed16", f"开始构建 fixed16 input_root={input_root} output_root={output_root}")
    image_mode = expose_source_images(input_root=input_root, output_root=output_root, mode=str(image_root_mode))
    rng = random.Random(int(seed))
    split_list = [str(split) for split in splits]
    summary: Dict[str, object] = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "grid_size": int(grid_size),
        "num_boxes_per_patch": int(grid_size) * int(grid_size),
        "target_empty_ratio": float(target_empty_ratio),
        "seed": int(seed),
        "image_root_mode": image_mode,
        "splits": {},
    }
    for split in split_list:
        split_ratio = float(target_empty_ratio)
        if target_empty_ratio_by_split is not None and split in target_empty_ratio_by_split:
            split_ratio = float(target_empty_ratio_by_split[split])
        summary["splits"][split] = build_split_dataset(
            split=split,
            input_root=input_root,
            output_root=output_root,
            grid_size=int(grid_size),
            target_empty_ratio=float(split_ratio),
            rng=rng,
            max_source_samples_per_split=int(max_source_samples_per_split),
            boundary_tol_px=float(boundary_tol_px),
            resample_step_px=float(resample_step_px),
            reuse_system_prompt=bool(reuse_system_prompt),
            export_visualizations=bool(export_visualizations),
            max_visualizations_per_split=int(max_visualizations_per_split),
        )
    dataset_prefix = f"unimapgen_{sanitize_name(output_root.name)}"
    dataset_info = build_sharegpt_dataset_info(dataset_root=output_root, prefix=dataset_prefix, splits=split_list)
    write_json(output_root / "dataset_info.json", dataset_info)
    write_json(output_root / "build_summary.json", summary)
    log_progress("Fixed16", f"构建完成 output_root={output_root} image_root_mode={image_mode}")
    return summary