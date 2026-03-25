"""从 RC family manifest 导出 patch-only 数据集。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unimapgen.dataset_build_refactor.common import ensure_dir, format_progress, load_jsonl, log_error, log_event, log_warning, require_existing_path, resolve_optional_text, validate_ratio, write_json, write_jsonl
from unimapgen.dataset_build_refactor.patch_only import PATCH_ONLY_PROMPT_TEMPLATE, PATCH_ONLY_SYSTEM_PROMPT, build_patch_segments_global, build_patch_target_lines, make_patch_only_record
from unimapgen.dataset_build_refactor.rc_dataset import load_family_raster_and_mask, load_sample_global_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RC patch-only dataset from family manifest.")
    parser.add_argument("--family-manifest", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    parser.add_argument("--band-indices", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--resample-step-px", type=float, default=4.0)
    parser.add_argument("--boundary-tol-px", type=float, default=2.5)
    parser.add_argument("--lane-only", action="store_true")
    parser.add_argument("--include-lane", action="store_true")
    parser.add_argument("--include-intersection-boundary", action="store_true")
    parser.add_argument("--max-families-per-split", type=int, default=0)
    parser.add_argument("--empty-patch-drop-ratio", type=float, default=0.95)
    parser.add_argument("--empty-patch-seed", type=int, default=42)
    parser.add_argument("--use-system-prompt", action="store_true")
    parser.add_argument("--system-prompt", type=str, default="")
    parser.add_argument("--system-prompt-file", type=str, default="")
    parser.add_argument("--user-prompt", type=str, default="")
    parser.add_argument("--user-prompt-file", type=str, default="")
    return parser.parse_args()


def build_patch_only_meta_row(*, sample_id: str, split: str, family: Dict, patch: Dict, image_rel_path: str, target_lines: Sequence[Dict], resample_step_px: float, system_prompt: str) -> Dict:
    return {
        "id": sample_id,
        "split": split,
        "family_id": family["family_id"],
        "source_sample_id": family.get("source_sample_id", ""),
        "source_image": family.get("source_image", ""),
        "source_image_path": family.get("source_image_path", ""),
        "source_mask_path": family.get("source_mask_path", ""),
        "source_lane_path": family.get("source_lane_path", ""),
        "source_intersection_path": family.get("source_intersection_path", ""),
        "patch_id": int(patch["patch_id"]),
        "row": int(patch["row"]),
        "col": int(patch["col"]),
        "scan_index": int(patch["patch_id"]),
        "image": image_rel_path,
        "crop_box": patch["crop_box"],
        "keep_box": patch["keep_box"],
        "mask_ratio": float(patch.get("mask_ratio", 0.0)),
        "mask_pixels": int(patch.get("mask_pixels", 0)),
        "num_target_lines": len(target_lines),
        "resample_step_px": float(resample_step_px),
        "has_system_prompt": bool(str(system_prompt).strip()),
        "target_lines": list(target_lines),
    }


def downsample_empty_records(rows: Sequence[Dict], meta_rows: Sequence[Dict], drop_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict], Dict[str, int | float]]:
    import random

    safe_ratio = validate_ratio("empty patch drop ratio", float(drop_ratio))
    paired = list(zip(rows, meta_rows))
    non_empty = [pair for pair in paired if int(pair[1].get("num_target_lines", 0)) > 0]
    empty = [pair for pair in paired if int(pair[1].get("num_target_lines", 0)) <= 0]
    keep_empty = int(round(len(empty) * (1.0 - safe_ratio)))
    keep_empty = max(0, min(len(empty), keep_empty))
    rng = random.Random(int(seed))
    chosen_empty = empty if keep_empty >= len(empty) else rng.sample(empty, keep_empty)
    keep_ids = {id(pair) for pair in non_empty}
    keep_ids.update(id(pair) for pair in chosen_empty)
    kept_rows: List[Dict] = []
    kept_meta: List[Dict] = []
    for pair in paired:
        if id(pair) not in keep_ids:
            continue
        kept_rows.append(pair[0])
        kept_meta.append(pair[1])
    summary = {
        "generated_total": int(len(paired)),
        "generated_non_empty": int(len(non_empty)),
        "generated_empty": int(len(empty)),
        "kept_total": int(len(kept_rows)),
        "kept_non_empty": int(len(non_empty)),
        "kept_empty": int(len(kept_rows) - len(non_empty)),
        "drop_ratio": float(safe_ratio),
    }
    return kept_rows, kept_meta, summary


def export_split(*, split: str, families: Sequence[Dict], output_root: Path, band_indices: Sequence[int], mask_threshold: int, resample_step_px: float, boundary_tol_px: float, include_lane: bool, include_intersection_boundary: bool, max_families_per_split: int, empty_patch_drop_ratio: float, empty_patch_seed: int, system_prompt: str, user_prompt_text: str) -> Dict[str, object]:
    rows: List[Dict] = []
    meta_rows: List[Dict] = []
    family_count = 0
    split_families = [family for family in families if str(family.get("split", "")) == split]
    if int(max_families_per_split) > 0:
        split_families = split_families[: int(max_families_per_split)]
    log_event("PatchOnly", f"split={split} family_count={len(split_families)}")
    for index, family in enumerate(split_families, start=1):
        if index == 1 or index == len(split_families) or index % 10 == 0:
            log_event("PatchOnly", f"split={split} family_progress={format_progress(index, len(split_families))} family_id={family.get('family_id', '')}")
        try:
            raw_image_hwc, raster_meta, _ = load_family_raster_and_mask(family, band_indices=[int(index) for index in band_indices], mask_threshold=int(mask_threshold))
            global_features = load_sample_global_features(
                lane_path=Path(str(family.get("source_lane_path", ""))),
                intersection_path=Path(str(family.get("source_intersection_path", ""))),
                raster_meta=raster_meta,
                include_lane=bool(include_lane),
                include_intersection=bool(include_intersection_boundary),
            )
            if not global_features:
                log_warning("PatchOnly", f"split={split} family_id={family.get('family_id', '')} has no valid GeoJSON features")
            patches = sorted(list(family.get("patches", [])), key=lambda item: int(item["patch_id"]))
            for patch in patches:
                crop_box = patch["crop_box"]
                keep_box = patch["keep_box"]
                patch_image = Image.fromarray(raw_image_hwc[int(crop_box["y_min"]):int(crop_box["y_max"]), int(crop_box["x_min"]):int(crop_box["x_max"]), :])
                segments_global = build_patch_segments_global(
                    global_features=global_features,
                    rect_global=(float(keep_box["x_min"]), float(keep_box["y_min"]), float(keep_box["x_max"]), float(keep_box["y_max"])),
                    resample_step_px=float(resample_step_px),
                    boundary_tol_px=float(boundary_tol_px),
                )
                target_lines = build_patch_target_lines(segments_global=segments_global, patch=patch, quantize=True)
                patch_id = int(patch["patch_id"])
                image_rel = Path("images") / split / str(family["family_id"]) / f"p{patch_id:04d}.png"
                out_image = output_root / image_rel
                ensure_dir(out_image.parent)
                patch_image.save(out_image)
                patch_image.close()
                sample_id = f"{family['family_id']}_p{patch_id:04d}"
                rows.append(make_patch_only_record(sample_id=sample_id, image_rel_path=image_rel.as_posix(), target_lines=target_lines, system_prompt=system_prompt, user_prompt_text=user_prompt_text))
                meta_rows.append(build_patch_only_meta_row(sample_id=sample_id, split=split, family=family, patch=patch, image_rel_path=image_rel.as_posix(), target_lines=target_lines, resample_step_px=float(resample_step_px), system_prompt=system_prompt))
        except Exception as exc:
            log_error("PatchOnly", f"split={split} family_id={family.get('family_id', '')} failed: {exc}")
            raise
        family_count += 1
    kept_rows, kept_meta, filter_summary = downsample_empty_records(rows, meta_rows, drop_ratio=float(empty_patch_drop_ratio), seed=int(empty_patch_seed))
    count_rows = write_jsonl(output_root / f"{split}.jsonl", kept_rows)
    count_meta = write_jsonl(output_root / f"meta_{split}.jsonl", kept_meta)
    log_event("PatchOnly", f"split={split} done generated={len(rows)} kept={count_rows} empty_kept={filter_summary['kept_empty']}")
    return {"families": int(family_count), "samples": int(count_rows), "meta_samples": int(count_meta), "empty_patch_filter": filter_summary}


def main() -> None:
    args = parse_args()
    validate_ratio("--empty-patch-drop-ratio", float(args.empty_patch_drop_ratio))
    manifest_path = require_existing_path(Path(args.family_manifest), kind="file")
    families = load_jsonl(manifest_path)
    if not families:
        raise RuntimeError(f"Family manifest is empty: {manifest_path}")
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)
    if bool(args.lane_only):
        include_lane = True
        include_intersection_boundary = False
    else:
        include_lane = True if (not bool(args.include_lane) and not bool(args.include_intersection_boundary)) else bool(args.include_lane)
        include_intersection_boundary = True if (not bool(args.include_lane) and not bool(args.include_intersection_boundary)) else bool(args.include_intersection_boundary)
    system_prompt = resolve_optional_text(inline_text=str(args.system_prompt), file_path=str(args.system_prompt_file), fallback=PATCH_ONLY_SYSTEM_PROMPT if bool(args.use_system_prompt) else "")
    user_prompt_text = resolve_optional_text(inline_text=str(args.user_prompt), file_path=str(args.user_prompt_file), fallback=PATCH_ONLY_PROMPT_TEMPLATE)
    log_event("PatchOnly", f"start manifest={manifest_path} output_root={output_root} family_count={len(families)}")
    summary: Dict[str, object] = {
        "source_family_manifest": str(manifest_path),
        "output_root": str(output_root),
        "include_lane": bool(include_lane),
        "include_intersection_boundary": bool(include_intersection_boundary),
        "user_prompt_template": user_prompt_text,
        "band_indices": [int(index) for index in args.band_indices],
        "splits": {},
    }
    for split in [str(item) for item in args.splits]:
        summary["splits"][split] = export_split(
            split=split,
            families=families,
            output_root=output_root,
            band_indices=[int(index) for index in args.band_indices],
            mask_threshold=int(args.mask_threshold),
            resample_step_px=float(args.resample_step_px),
            boundary_tol_px=float(args.boundary_tol_px),
            include_lane=bool(include_lane),
            include_intersection_boundary=bool(include_intersection_boundary),
            max_families_per_split=int(args.max_families_per_split),
            empty_patch_drop_ratio=float(args.empty_patch_drop_ratio),
            empty_patch_seed=int(args.empty_patch_seed),
            system_prompt=system_prompt,
            user_prompt_text=user_prompt_text,
        )
        split_info = summary["splits"][split]
        log_event("PatchOnly", f"split={split} summary families={split_info['families']} samples={split_info['samples']}")
    write_json(
        output_root / "dataset_info.json",
        {
            "dataset_name": "unimapgen_rc_patch_only",
            "source_family_manifest": str(Path(args.family_manifest).resolve()),
            "summary": summary["splits"],
        },
    )
    write_json(output_root / "export_summary.json", summary)
    log_event("PatchOnly", f"done summary={output_root / 'export_summary.json'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
