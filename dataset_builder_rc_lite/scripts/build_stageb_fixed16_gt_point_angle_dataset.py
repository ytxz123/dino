"""从 fixed16 Stage A 数据集构建 Stage B。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unimapgen.dataset_build_refactor.common import build_sharegpt_dataset_info, ensure_dir, extract_message_content, format_progress, link_or_copy_images, load_jsonl, log_event, require_existing_path, make_sharegpt_record, resolve_optional_text, write_jsonl
from unimapgen.dataset_build_refactor.stageb import STAGEB_TRACE_PROMPT_TEMPLATE, extract_state_points, format_stageb_trace_prompt, safe_int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage B dataset from fixed16 Stage A dataset.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--boundary-tol-px", type=float, default=2.5)
    parser.add_argument("--trace-points-per-hint", type=int, default=3)
    parser.add_argument("--use-system-prompt-from-source", action="store_true")
    parser.add_argument("--system-prompt", type=str, default="")
    parser.add_argument("--system-prompt-file", type=str, default="")
    parser.add_argument("--image-root-mode", type=str, default="symlink", choices=["symlink", "copy", "none"])
    return parser.parse_args()


def build_split(*, split: str, input_root: Path, output_root: Path, default_grid_size: int, boundary_tol_px: float, trace_points_per_hint: int, explicit_system_prompt: str, reuse_system_prompt: bool) -> Dict[str, object]:
    split_jsonl = input_root / f"{split}.jsonl"
    split_meta_jsonl = input_root / f"meta_{split}.jsonl"
    if not split_jsonl.exists() or not split_meta_jsonl.exists():
        return {"missing_split": True, "split_jsonl": str(split_jsonl), "split_meta_jsonl": str(split_meta_jsonl)}
    rows = load_jsonl(split_jsonl)
    meta_rows = load_jsonl(split_meta_jsonl)
    log_event("StageB", f"split={split} loaded rows={len(rows)} meta_rows={len(meta_rows)}")
    if not meta_rows:
        raise RuntimeError(f"Split meta jsonl is empty: {split_meta_jsonl}")
    row_by_id = {str(row.get("id")): row for row in rows}
    source_group_meta: Dict[str, Dict[int, Dict]] = {}
    for src_meta in meta_rows:
        source_group_id = str(src_meta.get("source_id") or src_meta.get("id"))
        grid_size = safe_int(src_meta.get("grid_size", default_grid_size), default=default_grid_size)
        grid_row = safe_int(src_meta.get("grid_row", -1), default=-1)
        grid_col = safe_int(src_meta.get("grid_col", -1), default=-1)
        if grid_row < 0 or grid_col < 0:
            continue
        subpatch_id = int(grid_row) * int(grid_size) + int(grid_col)
        source_group_meta.setdefault(source_group_id, {})[int(subpatch_id)] = src_meta
    out_rows: List[Dict] = []
    out_meta: List[Dict] = []
    with_state = 0
    without_state = 0
    total_state_traces = 0
    total_state_trace_points = 0
    total_rows = len(meta_rows)
    for index, src_meta in enumerate(meta_rows, start=1):
        if index == 1 or index == total_rows or index % 100 == 0:
            log_event("StageB", f"split={split} source_progress={format_progress(index, total_rows)} row_id={src_meta.get('id', '')}")
        row_id = str(src_meta.get("id"))
        src_row = row_by_id.get(row_id)
        if src_row is None:
            log_event("StageB", f"split={split} skip missing stagea row id={row_id}")
            continue
        source_group_id = str(src_meta.get("source_id") or row_id)
        crop_box = src_meta.get("crop_box", {})
        patch_size = safe_int(crop_box.get("x_max", 0)) - safe_int(crop_box.get("x_min", 0))
        if patch_size <= 1:
            continue
        target_box = src_meta.get("target_box", {})
        if not isinstance(target_box, dict) or not target_box:
            continue
        target_lines = list(src_meta.get("target_lines", []))
        grid_size = safe_int(src_meta.get("grid_size", default_grid_size), default=default_grid_size)
        grid_row = safe_int(src_meta.get("grid_row", -1), default=-1)
        grid_col = safe_int(src_meta.get("grid_col", -1), default=-1)
        if grid_row < 0 or grid_col < 0:
            continue
        image_rel_path = str(src_meta.get("image") or src_row.get("images", [""])[0])
        system_prompt = explicit_system_prompt if explicit_system_prompt else (extract_message_content(src_row, "system") if reuse_system_prompt else "")
        state_points = extract_state_points(source_group_meta=source_group_meta.get(source_group_id, {}), grid_size=grid_size, grid_row=grid_row, grid_col=grid_col, patch_size=patch_size, boundary_tol_px=boundary_tol_px, trace_points_per_hint=trace_points_per_hint)
        prompt_text = format_stageb_trace_prompt(target_box=target_box, state_points=state_points)
        row = make_sharegpt_record(sample_id=row_id, image_rel_path=image_rel_path, user_text=prompt_text, assistant_payload={"lines": list(target_lines)}, system_prompt=system_prompt)
        subpatch_id = int(grid_row) * int(grid_size) + int(grid_col)
        state_source_patch_ids = sorted({int(item["source_patch"]) for item in state_points})
        meta = {
            "id": row_id,
            "source_stagea_row_id": row_id,
            "source_id": src_meta.get("source_id"),
            "split": split,
            "family_id": src_meta.get("family_id"),
            "source_image": src_meta.get("source_image"),
            "patch_id": src_meta.get("patch_id"),
            "row": src_meta.get("row"),
            "col": src_meta.get("col"),
            "scan_index": src_meta.get("scan_index"),
            "image": image_rel_path,
            "crop_box": crop_box,
            "target_mode": str(src_meta.get("target_mode", "fixed_grid_target_box_map")),
            "state_mode": "gt_neighbor_handoff_trace_points",
            "grid_size": int(grid_size),
            "grid_row": int(grid_row),
            "grid_col": int(grid_col),
            "subpatch_id": int(subpatch_id),
            "history_subpatch_ids": list(range(subpatch_id)),
            "state_source_patch_ids": state_source_patch_ids,
            "num_state_traces": len(state_points),
            "num_state_trace_points": int(sum(len(item.get("points", [])) for item in state_points)),
            "trace_points_per_hint": int(trace_points_per_hint),
            "target_box": {"x_min": safe_int(target_box["x_min"]), "y_min": safe_int(target_box["y_min"]), "x_max": safe_int(target_box["x_max"]), "y_max": safe_int(target_box["y_max"])},
            "target_box_area": safe_int(src_meta.get("target_box_area", 0)),
            "anchor_source": src_meta.get("anchor_source"),
            "anchor_start_xy": src_meta.get("anchor_start_xy"),
            "anchor_end_xy": src_meta.get("anchor_end_xy"),
            "anchor_piece_points": src_meta.get("anchor_piece_points"),
            "num_target_lines": len(target_lines),
            "prompt_text": prompt_text,
            "state_points": state_points,
            "target_lines": list(target_lines),
        }
        out_rows.append(row)
        out_meta.append(meta)
        total_state_traces += len(state_points)
        total_state_trace_points += int(sum(len(item.get("points", [])) for item in state_points))
        if state_points:
            with_state += 1
        else:
            without_state += 1
    count_rows = write_jsonl(output_root / f"{split}.jsonl", out_rows)
    count_meta = write_jsonl(output_root / f"meta_{split}.jsonl", out_meta)
    log_event("StageB", f"split={split} done written_rows={count_rows} with_state={with_state} without_state={without_state}")
    return {
        "source_groups": len(source_group_meta),
        "used_stagea_rows": len(out_rows),
        "written_rows": count_rows,
        "written_meta_rows": count_meta,
        "samples_with_state": int(with_state),
        "samples_without_state": int(without_state),
        "total_state_traces": int(total_state_traces),
        "avg_state_traces_per_sample": (float(total_state_traces) / float(len(out_rows)) if out_rows else 0.0),
        "total_state_trace_points": int(total_state_trace_points),
        "avg_state_trace_points_per_sample": (float(total_state_trace_points) / float(len(out_rows)) if out_rows else 0.0),
    }


def main() -> None:
    args = parse_args()
    if int(args.grid_size) <= 0:
        raise ValueError("--grid-size must be positive.")
    if int(args.trace_points_per_hint) < 2:
        raise ValueError("--trace-points-per-hint must be >= 2.")
    input_root = require_existing_path(args.input_root, kind="dir")
    output_root = args.output_root.resolve()
    ensure_dir(output_root)
    explicit_system_prompt = resolve_optional_text(inline_text=str(args.system_prompt), file_path=str(args.system_prompt_file))
    image_mode = link_or_copy_images(input_root=input_root, output_root=output_root, mode=str(args.image_root_mode))
    log_event("StageB", f"start input_root={input_root} output_root={output_root} grid_size={args.grid_size}")
    summary: Dict[str, object] = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "grid_size": int(args.grid_size),
        "num_boxes_per_patch": int(args.grid_size) * int(args.grid_size),
        "state_mode": "gt_neighbor_handoff_trace_points",
        "state_neighbors": ["left", "top"],
        "trace_points_per_hint": int(args.trace_points_per_hint),
        "boundary_tol_px": float(args.boundary_tol_px),
        "prompt_template": STAGEB_TRACE_PROMPT_TEMPLATE,
        "image_root_mode": image_mode,
        "splits": {},
    }
    splits = [str(item) for item in args.splits]
    for split in splits:
        summary["splits"][split] = build_split(split=split, input_root=input_root, output_root=output_root, default_grid_size=int(args.grid_size), boundary_tol_px=float(args.boundary_tol_px), trace_points_per_hint=int(args.trace_points_per_hint), explicit_system_prompt=explicit_system_prompt, reuse_system_prompt=bool(args.use_system_prompt_from_source))
        log_event("StageB", f"split={split} summary={summary['splits'][split]}")
    (output_root / "dataset_info.json").write_text(json.dumps(build_sharegpt_dataset_info(output_root=output_root, splits=splits), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log_event("StageB", f"done summary={output_root / 'build_summary.json'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()