from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from dataset_generation.common.defaults import (
    DEFAULT_STAGE_A_PROMPT_TEMPLATE,
    DEFAULT_STAGE_A_SYSTEM_PROMPT,
    DEFAULT_STAGE_B_PROMPT_TEMPLATE,
    DEFAULT_STAGE_B_SYSTEM_PROMPT,
)
from dataset_generation.common.io_utils import build_sharegpt_dataset_info, ensure_directory, load_jsonl, write_json, write_jsonl
from dataset_generation.common.progress import log_progress
from dataset_generation.common.state_mix import build_sample_rng, build_state_lines_by_mode, choose_state_mode
from dataset_generation.stages.features import (
    build_owned_segments_by_patch,
    build_patch_image,
    build_patch_target_lines,
    build_patch_target_lines_float,
    build_patch_target_lines_quantized,
    extract_state_lines,
    family_global_lines,
    load_family_raster_and_mask,
    local_lines_to_uv,
)
from dataset_generation.stages.records import build_patch_only_record, build_state_record



def downsample_empty_patch_records(records: Sequence[Dict], drop_ratio: float, seed: int, split: str) -> Tuple[List[Dict], Dict[str, float]]:
    """对空 patch 做下采样，非空样本全部保留。"""
    safe_ratio = max(0.0, min(1.0, float(drop_ratio)))
    non_empty = [record for record in records if int(record["stagea_meta"].get("num_target_lines", 0)) > 0]
    empty = [record for record in records if int(record["stagea_meta"].get("num_target_lines", 0)) <= 0]
    rng = build_sample_rng(f"empty::{split}::{int(seed)}")
    keep_empty = int(round(len(empty) * (1.0 - safe_ratio)))
    keep_empty = max(0, min(len(empty), keep_empty))
    if keep_empty <= 0:
        kept_empty = []
    elif keep_empty >= len(empty):
        kept_empty = list(empty)
    else:
        indices = rng.choice(len(empty), size=keep_empty, replace=False)
        kept_empty = [empty[int(index)] for index in sorted(indices.tolist())]
    keep_ids = {id(record) for record in non_empty}
    keep_ids.update(id(record) for record in kept_empty)
    kept = [record for record in records if id(record) in keep_ids]
    return kept, {
        "generated_total": int(len(records)),
        "generated_non_empty": int(len(non_empty)),
        "generated_empty": int(len(empty)),
        "kept_total": int(len(kept)),
        "kept_non_empty": int(len(non_empty)),
        "kept_empty": int(len(kept_empty)),
        "dropped_empty": int(len(empty) - len(kept_empty)),
        "drop_ratio": float(safe_ratio),
    }



def export_stage_datasets(
    families: Sequence[Dict],
    output_root: Path,
    splits: Sequence[str],
    band_indices: Sequence[int],
    mask_threshold: int,
    resample_step_px: float,
    boundary_tol_px: float,
    trace_points: int,
    state_mixture_mode: str,
    state_no_state_ratio: float,
    state_weak_ratio: float,
    state_full_ratio: float,
    state_weak_trace_points: int,
    state_line_dropout: float,
    state_point_jitter_px: float,
    state_truncate_prob: float,
    include_lane: bool,
    include_intersection_boundary: bool,
    max_families_per_split: int,
    empty_patch_drop_ratio: float,
    empty_patch_seed: int,
    empty_patch_drop_ratio_by_split: Optional[Dict[str, float]],
    stagea_system_prompt: str,
    stagea_prompt_template: str,
    stageb_system_prompt: str,
    stageb_prompt_template: str,
) -> Dict[str, object]:
    """从 family manifest 导出 stage_a 和 stage_b 两套数据集。"""
    output_root = Path(output_root).resolve()
    stage_a_root = output_root / "stage_a" / "dataset"
    stage_b_root = output_root / "stage_b" / "dataset"
    ensure_directory(stage_a_root)
    ensure_directory(stage_b_root)
    split_set = {str(split) for split in splits}
    split_records: Dict[str, List[Dict]] = {str(split): [] for split in splits}
    family_seen: Dict[str, int] = {str(split): 0 for split in splits}
    family_exported: Dict[str, int] = {str(split): 0 for split in splits}
    split_family_targets = {
        split_name: sum(1 for family in families if str(family.get("split", "")) == split_name)
        for split_name in split_set
    }
    for split_name in splits:
        split_text = str(split_name)
        log_progress("Stage", f"开始处理 split={split_text} family_count={split_family_targets.get(split_text, 0)}")
    for family in families:
        split = str(family.get("split", ""))
        if split not in split_set:
            continue
        family_seen[split] += 1
        if int(max_families_per_split) > 0 and family_seen[split] > int(max_families_per_split):
            continue
        family_exported[split] += 1
        log_progress(
            "Stage",
            f"split={split} family={family_exported[split]}/{min(split_family_targets.get(split, 0), int(max_families_per_split)) if int(max_families_per_split) > 0 else split_family_targets.get(split, 0)} family_id={family.get('family_id', '')}",
        )
        raw_image_hwc, raster_meta, _ = load_family_raster_and_mask(
            family=family,
            band_indices=[int(index) for index in band_indices],
            mask_threshold=int(mask_threshold),
        )
        global_lines = family_global_lines(
            family=family,
            raster_meta=raster_meta,
            include_lane=bool(include_lane),
            include_intersection=bool(include_intersection_boundary),
        )
        owned_segments_by_patch = build_owned_segments_by_patch(
            family=family,
            global_lines=global_lines,
            resample_step_px=float(resample_step_px),
            boundary_tol_px=float(boundary_tol_px),
        )
        for patch in sorted(list(family.get("patches", [])), key=lambda item: int(item["patch_id"])):
            patch_id = int(patch["patch_id"])
            patch_image = build_patch_image(raw_image_hwc=raw_image_hwc, patch=patch)
            owned_segments = owned_segments_by_patch.get(patch_id, [])
            target_lines_float = build_patch_target_lines_float(owned_segments, patch=patch)
            target_lines_quantized = build_patch_target_lines_quantized(owned_segments, patch=patch)
            target_lines = build_patch_target_lines(owned_segments, patch=patch)
            image_rel = Path("images") / split / str(family["family_id"]) / f"p{patch_id:04d}.png"
            stagea_image_path = stage_a_root / image_rel
            stageb_image_path = stage_b_root / image_rel
            ensure_directory(stagea_image_path.parent)
            ensure_directory(stageb_image_path.parent)
            patch_image.save(stagea_image_path)
            patch_image.save(stageb_image_path)
            patch_image.close()
            sample_id = f"{family['family_id']}_p{patch_id:04d}"
            stagea_row = build_patch_only_record(
                image_rel_path=image_rel.as_posix(),
                target_lines=target_lines,
                sample_id=sample_id,
                system_prompt=stagea_system_prompt,
                prompt_template=stagea_prompt_template,
            )
            stagea_meta = {
                "id": sample_id,
                "split": split,
                "family_id": family["family_id"],
                "source_sample_id": family.get("source_sample_id", ""),
                "source_image": family.get("source_image", ""),
                "source_image_path": family.get("source_image_path", ""),
                "source_mask_path": family.get("source_mask_path", ""),
                "source_lane_path": family.get("source_lane_path", ""),
                "source_intersection_path": family.get("source_intersection_path", ""),
                "image": image_rel.as_posix(),
                "image_size": family.get("image_size", []),
                "patch_id": patch_id,
                "row": int(patch["row"]),
                "col": int(patch["col"]),
                "crop_box": patch["crop_box"],
                "keep_box": patch["keep_box"],
                "mask_ratio": float(patch.get("mask_ratio", 0.0)),
                "mask_pixels": int(patch.get("mask_pixels", 0)),
                "num_target_lines": int(len(target_lines)),
                "target_lines": target_lines,
                "target_lines_quantized": target_lines_quantized,
                "target_lines_float": target_lines_float,
            }
            raw_state_lines = extract_state_lines(
                patch=patch,
                family=family,
                owned_segments_by_patch=owned_segments_by_patch,
                trace_points=int(trace_points),
                boundary_tol_px=float(boundary_tol_px),
            )
            sample_rng = build_sample_rng(sample_id)
            state_mode = choose_state_mode(
                rng=sample_rng,
                mixture_mode=str(state_mixture_mode),
                raw_state_lines=raw_state_lines,
                no_state_ratio=float(state_no_state_ratio),
                weak_ratio=float(state_weak_ratio),
                full_ratio=float(state_full_ratio),
            )
            patch_size = int(patch["crop_box"]["x_max"]) - int(patch["crop_box"]["x_min"])
            state_lines_float = build_state_lines_by_mode(
                raw_state_lines=raw_state_lines,
                state_mode=state_mode,
                patch_size=patch_size,
                weak_trace_points=int(state_weak_trace_points),
                state_line_dropout=float(state_line_dropout),
                state_point_jitter_px=float(state_point_jitter_px),
                state_truncate_prob=float(state_truncate_prob),
                rng=sample_rng,
            )
            state_lines = local_lines_to_uv(state_lines_float, patch=patch, quantize=True)
            stageb_row = build_state_record(
                image_rel_path=image_rel.as_posix(),
                state_lines=state_lines,
                target_lines=target_lines,
                sample_id=sample_id,
                system_prompt=stageb_system_prompt,
                prompt_template=stageb_prompt_template,
            )
            stageb_meta = {
                "id": sample_id,
                "split": split,
                "family_id": family["family_id"],
                "source_sample_id": family.get("source_sample_id", ""),
                "source_image": family.get("source_image", ""),
                "source_image_path": family.get("source_image_path", ""),
                "source_mask_path": family.get("source_mask_path", ""),
                "source_lane_path": family.get("source_lane_path", ""),
                "source_intersection_path": family.get("source_intersection_path", ""),
                "image": image_rel.as_posix(),
                "image_size": family.get("image_size", []),
                "patch_id": patch_id,
                "row": int(patch["row"]),
                "col": int(patch["col"]),
                "crop_box": patch["crop_box"],
                "keep_box": patch["keep_box"],
                "state_mode": str(state_mode),
                "num_state_lines": int(len(state_lines)),
                "num_target_lines": int(len(target_lines)),
                "state_lines": state_lines,
                "state_lines_float": state_lines_float,
                "target_lines": target_lines,
                "target_lines_quantized": target_lines_quantized,
                "target_lines_float": target_lines_float,
            }
            split_records[split].append(
                {
                    "stagea_row": stagea_row,
                    "stagea_meta": stagea_meta,
                    "stageb_row": stageb_row,
                    "stageb_meta": stageb_meta,
                }
            )
        log_progress(
            "Stage",
            f"split={split} family_id={family.get('family_id', '')} patch_count={len(family.get('patches', []))} 已完成导出",
        )
    filter_summary: Dict[str, Dict[str, float]] = {}
    stagea_summary: Dict[str, Dict[str, int]] = {}
    stageb_summary: Dict[str, Dict[str, int]] = {}
    for split in splits:
        split_name = str(split)
        split_drop_ratio = float(empty_patch_drop_ratio)
        if empty_patch_drop_ratio_by_split is not None and split_name in empty_patch_drop_ratio_by_split:
            split_drop_ratio = float(empty_patch_drop_ratio_by_split[split_name])
        kept_records, filter_summary[split_name] = downsample_empty_patch_records(
            records=split_records[split_name],
            drop_ratio=split_drop_ratio,
            seed=int(empty_patch_seed),
            split=split_name,
        )
        stagea_rows = [record["stagea_row"] for record in kept_records]
        stagea_meta = [record["stagea_meta"] for record in kept_records]
        stageb_rows = [record["stageb_row"] for record in kept_records]
        stageb_meta = [record["stageb_meta"] for record in kept_records]
        stagea_summary[split_name] = {
            "families": int(family_exported.get(split_name, 0)),
            "samples": int(write_jsonl(stage_a_root / f"{split_name}.jsonl", stagea_rows)),
            "meta_samples": int(write_jsonl(stage_a_root / f"meta_{split_name}.jsonl", stagea_meta)),
        }
        stageb_summary[split_name] = {
            "families": int(family_exported.get(split_name, 0)),
            "samples": int(write_jsonl(stage_b_root / f"{split_name}.jsonl", stageb_rows)),
            "meta_samples": int(write_jsonl(stage_b_root / f"meta_{split_name}.jsonl", stageb_meta)),
        }
        log_progress(
            "Stage",
            f"split={split_name} stage_a_samples={stagea_summary[split_name]['samples']} stage_b_samples={stageb_summary[split_name]['samples']} empty_kept={filter_summary[split_name]['kept_empty']}",
        )
    write_json(stage_a_root / "dataset_info.json", build_sharegpt_dataset_info(stage_a_root, prefix="unimapgen_geo_current_patch_only", splits=splits))
    write_json(stage_b_root / "dataset_info.json", build_sharegpt_dataset_info(stage_b_root, prefix="unimapgen_geo_current_state", splits=splits))
    write_json(
        stage_a_root / "export_summary.json",
        {
            "dataset_name_prefix": "unimapgen_geo_current_patch_only",
            "source_family_manifest": "",
            "summary": stagea_summary,
            "empty_patch_filter": filter_summary,
        },
    )
    write_json(
        stage_b_root / "export_summary.json",
        {
            "dataset_name_prefix": "unimapgen_geo_current_state",
            "source_family_manifest": "",
            "summary": stageb_summary,
            "empty_patch_filter": filter_summary,
        },
    )
    return {
        "stage_a_root": stage_a_root,
        "stage_b_root": stage_b_root,
        "stage_a_summary": stagea_summary,
        "stage_b_summary": stageb_summary,
        "empty_patch_filter": filter_summary,
    }



def export_stage_datasets_from_manifest(
    family_manifest_path: Path,
    output_root: Path,
    splits: Sequence[str],
    band_indices: Sequence[int],
    mask_threshold: int,
    resample_step_px: float,
    boundary_tol_px: float,
    trace_points: int,
    state_mixture_mode: str,
    state_no_state_ratio: float,
    state_weak_ratio: float,
    state_full_ratio: float,
    state_weak_trace_points: int,
    state_line_dropout: float,
    state_point_jitter_px: float,
    state_truncate_prob: float,
    include_lane: bool,
    include_intersection_boundary: bool,
    max_families_per_split: int,
    empty_patch_drop_ratio: float,
    empty_patch_seed: int,
    empty_patch_drop_ratio_by_split: Optional[Dict[str, float]],
    stagea_system_prompt: str = DEFAULT_STAGE_A_SYSTEM_PROMPT,
    stagea_prompt_template: str = DEFAULT_STAGE_A_PROMPT_TEMPLATE,
    stageb_system_prompt: str = DEFAULT_STAGE_B_SYSTEM_PROMPT,
    stageb_prompt_template: str = DEFAULT_STAGE_B_PROMPT_TEMPLATE,
) -> Dict[str, object]:
    """从 family_manifest.jsonl 读取并导出 stage 数据集。"""
    manifest_path = Path(family_manifest_path).resolve()
    families = load_jsonl(manifest_path)
    result = export_stage_datasets(
        families=families,
        output_root=output_root,
        splits=splits,
        band_indices=band_indices,
        mask_threshold=mask_threshold,
        resample_step_px=resample_step_px,
        boundary_tol_px=boundary_tol_px,
        trace_points=trace_points,
        state_mixture_mode=state_mixture_mode,
        state_no_state_ratio=state_no_state_ratio,
        state_weak_ratio=state_weak_ratio,
        state_full_ratio=state_full_ratio,
        state_weak_trace_points=state_weak_trace_points,
        state_line_dropout=state_line_dropout,
        state_point_jitter_px=state_point_jitter_px,
        state_truncate_prob=state_truncate_prob,
        include_lane=include_lane,
        include_intersection_boundary=include_intersection_boundary,
        max_families_per_split=max_families_per_split,
        empty_patch_drop_ratio=empty_patch_drop_ratio,
        empty_patch_seed=empty_patch_seed,
        empty_patch_drop_ratio_by_split=empty_patch_drop_ratio_by_split,
        stagea_system_prompt=stagea_system_prompt,
        stagea_prompt_template=stagea_prompt_template,
        stageb_system_prompt=stageb_system_prompt,
        stageb_prompt_template=stageb_prompt_template,
    )
    for stage_root in (result["stage_a_root"], result["stage_b_root"]):
        summary_path = Path(stage_root) / "export_summary.json"
        with summary_path.open("r", encoding="utf-8") as file:
            current = json.load(file)
        current["source_family_manifest"] = str(manifest_path)
        write_json(summary_path, current)
    return result
