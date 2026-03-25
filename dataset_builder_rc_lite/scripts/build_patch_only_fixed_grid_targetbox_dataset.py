"""从 patch-only 数据集构建 fixed16 Stage A。"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, IO, List, Optional, Sequence, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unimapgen.dataset_build_refactor.common import build_sharegpt_dataset_info, ensure_dir, extract_message_content, format_progress, link_or_copy_images, load_jsonl, log_event, require_existing_path, validate_ratio, make_sharegpt_record
from unimapgen.dataset_build_refactor.fixed16 import build_grid_boxes, build_prompt_endpoints, build_target_lines_for_box, format_fixed16_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed16 Stage A dataset from patch-only dataset.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--target-empty-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resample-step-px", type=float, default=4.0)
    parser.add_argument("--boundary-tol-px", type=float, default=2.5)
    parser.add_argument("--use-system-prompt-from-source", action="store_true")
    parser.add_argument("--image-root-mode", type=str, default="symlink", choices=["symlink", "copy", "none"])
    return parser.parse_args()


def write_jsonl_line(handle: IO[str], row: Dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def choose_empty_indices(empty_count: int, keep_empty: int, rng: random.Random) -> Optional[Set[int]]:
    if keep_empty <= 0:
        return set()
    if keep_empty >= empty_count:
        return None
    return set(rng.sample(range(empty_count), keep_empty))


def append_sampled_jsonl_pairs(*, rows_src: Path, meta_src: Path, rows_dst: Path, meta_dst: Path, keep_indices: Optional[Set[int]]) -> int:
    written = 0
    with rows_dst.open("a", encoding="utf-8") as row_out, meta_dst.open("a", encoding="utf-8") as meta_out:
        with rows_src.open("r", encoding="utf-8") as row_in, meta_src.open("r", encoding="utf-8") as meta_in:
            for index, (row_line, meta_line) in enumerate(zip(row_in, meta_in)):
                if keep_indices is not None and index not in keep_indices:
                    continue
                row_out.write(row_line)
                meta_out.write(meta_line)
                written += 1
    return written


def build_fixed16_record(*, sample_id: str, image_rel_path: str, prompt_text: str, target_lines: Sequence[Dict], system_prompt: str) -> Dict:
    return make_sharegpt_record(sample_id=sample_id, image_rel_path=image_rel_path, user_text=prompt_text, assistant_payload={"lines": list(target_lines)}, system_prompt=system_prompt)


def build_fixed16_meta_row(*, row_id: str, source_meta: Dict, box: Dict[str, int], prompt_info: Dict[str, object], prompt_text: str, target_lines: Sequence[Dict], grid_size: int, resample_step_px: float, system_prompt: str) -> Dict:
    return {
        "id": row_id,
        "source_id": str(source_meta.get("id")),
        "split": source_meta.get("split"),
        "family_id": source_meta.get("family_id"),
        "source_image": source_meta.get("source_image"),
        "patch_id": source_meta.get("patch_id"),
        "row": source_meta.get("row"),
        "col": source_meta.get("col"),
        "scan_index": source_meta.get("scan_index"),
        "image": source_meta.get("image"),
        "crop_box": source_meta.get("crop_box"),
        "target_mode": "fixed_grid_target_box_map",
        "coord_system": "patch_local",
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
        "num_target_lines": len(target_lines),
        "prompt_text": prompt_text,
        "target_lines": list(target_lines),
    }


def build_split(*, split: str, input_root: Path, output_root: Path, grid_size: int, target_empty_ratio: float, rng: random.Random, boundary_tol_px: float, resample_step_px: float, reuse_system_prompt: bool) -> Dict[str, object]:
    split_jsonl = input_root / f"{split}.jsonl"
    split_meta_jsonl = input_root / f"meta_{split}.jsonl"
    if not split_jsonl.exists() or not split_meta_jsonl.exists():
        return {"missing_split": True, "split_jsonl": str(split_jsonl), "split_meta_jsonl": str(split_meta_jsonl)}
    rows = load_jsonl(split_jsonl)
    meta_rows = load_jsonl(split_meta_jsonl)
    log_event("Fixed16", f"split={split} loaded rows={len(rows)} meta_rows={len(meta_rows)}")
    if not meta_rows:
        raise RuntimeError(f"Split meta jsonl is empty: {split_meta_jsonl}")
    row_by_id = {str(row.get("id")): row for row in rows}
    temp_root = output_root / ".tmp_fixed16_build" / split
    if temp_root.exists():
        shutil.rmtree(temp_root)
    ensure_dir(temp_root)
    non_empty_rows_path = temp_root / "non_empty_rows.jsonl"
    non_empty_meta_path = temp_root / "non_empty_meta.jsonl"
    empty_rows_path = temp_root / "empty_rows.jsonl"
    empty_meta_path = temp_root / "empty_meta.jsonl"
    generated_total = 0
    generated_non_empty = 0
    generated_empty = 0
    try:
        with (
            non_empty_rows_path.open("w", encoding="utf-8") as non_empty_rows_out,
            non_empty_meta_path.open("w", encoding="utf-8") as non_empty_meta_out,
            empty_rows_path.open("w", encoding="utf-8") as empty_rows_out,
            empty_meta_path.open("w", encoding="utf-8") as empty_meta_out,
        ):
            total_source = len(meta_rows)
            for source_index, src_meta in enumerate(meta_rows, start=1):
                if source_index == 1 or source_index == total_source or source_index % 100 == 0:
                    log_event("Fixed16", f"split={split} source_progress={format_progress(source_index, total_source)} source_id={src_meta.get('id', '')}")
                row_id = str(src_meta.get("id", ""))
                src_row = row_by_id.get(row_id)
                if src_row is None:
                    log_event("Fixed16", f"split={split} skip missing source row id={row_id}")
                    continue
                image_rel_path = str(src_meta.get("image") or src_row.get("images", [""])[0])
                crop_box = dict(src_meta.get("crop_box", {}) or {})
                patch_size = int(crop_box.get("x_max", 0)) - int(crop_box.get("x_min", 0))
                if patch_size <= 1:
                    continue
                full_patch_target_lines = list(src_meta.get("target_lines", []))
                system_prompt = extract_message_content(src_row, "system") if reuse_system_prompt else ""
                boxes = build_grid_boxes(patch_size=patch_size, grid_size=int(grid_size))
                for box in boxes:
                    prompt_info = build_prompt_endpoints(target_lines=full_patch_target_lines, target_box=box, patch_size=patch_size)
                    target_lines = build_target_lines_for_box(full_patch_target_lines=full_patch_target_lines, target_box=box, patch_size=patch_size, boundary_tol_px=boundary_tol_px, resample_step_px=resample_step_px)
                    prompt_text = format_fixed16_prompt({
                        "start_x": int(prompt_info["start_x"]),
                        "start_y": int(prompt_info["start_y"]),
                        "end_x": int(prompt_info["end_x"]),
                        "end_y": int(prompt_info["end_y"]),
                        "box_x_min": int(box["x_min"]),
                        "box_y_min": int(box["y_min"]),
                        "box_x_max": int(box["x_max"]),
                        "box_y_max": int(box["y_max"]),
                    })
                    new_row_id = f"{row_id}_g{int(box['grid_row'])}{int(box['grid_col'])}"
                    row = build_fixed16_record(sample_id=new_row_id, image_rel_path=image_rel_path, prompt_text=prompt_text, target_lines=target_lines, system_prompt=system_prompt)
                    meta = build_fixed16_meta_row(row_id=new_row_id, source_meta=src_meta, box=box, prompt_info=prompt_info, prompt_text=prompt_text, target_lines=target_lines, grid_size=grid_size, resample_step_px=resample_step_px, system_prompt=system_prompt)
                    if meta["num_target_lines"] > 0:
                        write_jsonl_line(non_empty_rows_out, row)
                        write_jsonl_line(non_empty_meta_out, meta)
                        generated_non_empty += 1
                    else:
                        write_jsonl_line(empty_rows_out, row)
                        write_jsonl_line(empty_meta_out, meta)
                        generated_empty += 1
                    generated_total += 1
        if generated_non_empty <= 0 and float(target_empty_ratio) < 1.0:
            raise ValueError("Split contains no non-empty samples; cannot enforce empty-ratio target.")
        keep_empty = generated_empty if float(target_empty_ratio) >= 1.0 else min(generated_empty, math.floor(generated_non_empty * target_empty_ratio / (1.0 - target_empty_ratio)))
        keep_empty_indices = choose_empty_indices(empty_count=generated_empty, keep_empty=keep_empty, rng=rng)
        final_rows_path = output_root / f"{split}.jsonl"
        final_meta_path = output_root / f"meta_{split}.jsonl"
        shutil.copyfile(non_empty_rows_path, final_rows_path)
        shutil.copyfile(non_empty_meta_path, final_meta_path)
        kept_empty = append_sampled_jsonl_pairs(rows_src=empty_rows_path, meta_src=empty_meta_path, rows_dst=final_rows_path, meta_dst=final_meta_path, keep_indices=keep_empty_indices)
        kept_total = generated_non_empty + kept_empty
        log_event("Fixed16", f"split={split} done generated={generated_total} kept={kept_total} kept_empty={kept_empty}")
        return {
            "generated_total": generated_total,
            "generated_non_empty": generated_non_empty,
            "generated_empty": generated_empty,
            "kept_total": kept_total,
            "kept_non_empty": generated_non_empty,
            "kept_empty": kept_empty,
            "kept_empty_ratio": (kept_empty / kept_total if kept_total else 0.0),
            "written_rows": kept_total,
            "written_meta_rows": kept_total,
        }
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)


def main() -> None:
    args = parse_args()
    if int(args.grid_size) <= 0:
        raise ValueError("--grid-size must be positive.")
    validate_ratio("--target-empty-ratio", float(args.target_empty_ratio))
    input_root = require_existing_path(args.input_root, kind="dir")
    output_root = args.output_root.resolve()
    ensure_dir(output_root)
    image_mode = link_or_copy_images(input_root=input_root, output_root=output_root, mode=str(args.image_root_mode))
    rng = random.Random(int(args.seed))
    log_event("Fixed16", f"start input_root={input_root} output_root={output_root} grid_size={args.grid_size}")
    summary: Dict[str, object] = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "grid_size": int(args.grid_size),
        "num_boxes_per_patch": int(args.grid_size) * int(args.grid_size),
        "target_empty_ratio": float(args.target_empty_ratio),
        "seed": int(args.seed),
        "image_root_mode": image_mode,
        "splits": {},
    }
    splits = [str(item) for item in args.splits]
    for split in splits:
        summary["splits"][split] = build_split(split=split, input_root=input_root, output_root=output_root, grid_size=int(args.grid_size), target_empty_ratio=float(args.target_empty_ratio), rng=rng, boundary_tol_px=float(args.boundary_tol_px), resample_step_px=float(args.resample_step_px), reuse_system_prompt=bool(args.use_system_prompt_from_source))
        log_event("Fixed16", f"split={split} summary={summary['splits'][split]}")
    (output_root / "dataset_info.json").write_text(json.dumps(build_sharegpt_dataset_info(output_root=output_root, splits=splits), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log_event("Fixed16", f"done summary={output_root / 'build_summary.json'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
