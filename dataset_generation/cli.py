from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset_generation.common.defaults import (
    DEFAULT_IMAGE_RELPATH,
    DEFAULT_INTERSECTION_RELPATH,
    DEFAULT_LANE_RELPATH,
    DEFAULT_MASK_RELPATH,
    DEFAULT_STAGE_A_PROMPT_TEMPLATE,
    DEFAULT_STAGE_A_SYSTEM_PROMPT,
    DEFAULT_STAGE_B_PROMPT_TEMPLATE,
    DEFAULT_STAGE_B_SYSTEM_PROMPT,
)
from dataset_generation.common.io_utils import load_jsonl, write_json
from dataset_generation.fixed16.builder import build_fixed16_dataset
from dataset_generation.manifest.builder import build_family_manifest, save_family_manifest
from dataset_generation.stages.exporter import export_stage_datasets


DEFAULT_OUTPUT_ROOT = Path("dataset_generation_output") / "processed_all"
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_ROOT / "family_manifest.jsonl"


def build_root_parser() -> argparse.ArgumentParser:
    """构建统一 CLI 根命令。"""
    parser = argparse.ArgumentParser(description="独立版 Geo 数据集生成工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("build-manifest", help="生成 family_manifest.jsonl")
    manifest_parser.add_argument("--dataset-root", type=str, default="/dataset/zsy/dataset-extracted")
    manifest_parser.add_argument("--train-root", type=str, default="")
    manifest_parser.add_argument("--val-root", type=str, default="")
    manifest_parser.add_argument("--output-manifest", type=str, required=True)
    manifest_parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    manifest_parser.add_argument("--image-relpath", type=str, default=DEFAULT_IMAGE_RELPATH)
    manifest_parser.add_argument("--mask-relpath", type=str, default=DEFAULT_MASK_RELPATH)
    manifest_parser.add_argument("--lane-relpath", type=str, default=DEFAULT_LANE_RELPATH)
    manifest_parser.add_argument("--intersection-relpath", type=str, default=DEFAULT_INTERSECTION_RELPATH)
    manifest_parser.add_argument("--mask-threshold", type=int, default=127)
    manifest_parser.add_argument("--tile-size-px", type=int, default=896)
    manifest_parser.add_argument("--overlap-px", type=int, default=232)
    manifest_parser.add_argument("--keep-margin-px", type=int, default=116)
    manifest_parser.add_argument("--review-crop-pad-px", type=int, default=64)
    manifest_parser.add_argument("--tile-min-mask-ratio", type=float, default=0.02)
    manifest_parser.add_argument("--tile-min-mask-pixels", type=int, default=256)
    manifest_parser.add_argument("--tile-max-per-sample", type=int, default=0)
    manifest_parser.add_argument("--search-within-review-bbox", action="store_true")
    manifest_parser.add_argument("--fallback-to-all-if-empty", action="store_true")
    manifest_parser.add_argument("--max-samples-per-split", type=int, default=0)
    manifest_parser.add_argument("--shard-index", type=int, default=0)
    manifest_parser.add_argument("--num-shards", type=int, default=1)

    fixed16_parser = subparsers.add_parser("build-fixed16", help="从 stage_a/stage_b 母集生成 fixed16 数据集")
    fixed16_parser.add_argument("--input-root", type=str, required=True)
    fixed16_parser.add_argument("--output-root", type=str, required=True)
    fixed16_parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    fixed16_parser.add_argument("--grid-size", type=int, default=4)
    fixed16_parser.add_argument("--target-empty-ratio", type=float, default=0.10)
    fixed16_parser.add_argument("--seed", type=int, default=42)
    fixed16_parser.add_argument("--max-source-samples-per-split", type=int, default=0)
    fixed16_parser.add_argument("--resample-step-px", type=float, default=4.0)
    fixed16_parser.add_argument("--boundary-tol-px", type=float, default=2.5)
    fixed16_parser.add_argument("--use-system-prompt-from-source", action="store_true")
    fixed16_parser.add_argument("--image-root-mode", type=str, default="symlink", choices=["symlink", "copy", "none"])
    fixed16_parser.add_argument("--export-visualizations", action="store_true")
    fixed16_parser.add_argument("--max-visualizations-per-split", type=int, default=0)

    export_parser = subparsers.add_parser("export-stages", help="从 family manifest 导出 stage_a / stage_b 数据集")
    export_parser.add_argument("--family-manifest", type=str, required=True)
    export_parser.add_argument("--output-root", type=str, required=True)
    export_parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    export_parser.add_argument("--band-indices", type=int, nargs="+", default=[1, 2, 3])
    export_parser.add_argument("--mask-threshold", type=int, default=127)
    export_parser.add_argument("--resample-step-px", type=float, default=4.0)
    export_parser.add_argument("--boundary-tol-px", type=float, default=2.5)
    export_parser.add_argument("--trace-points", type=int, default=8)
    export_parser.add_argument("--state-mixture-mode", type=str, default="full", choices=["full", "mixed"])
    export_parser.add_argument("--state-no-state-ratio", type=float, default=0.30)
    export_parser.add_argument("--state-weak-ratio", type=float, default=0.40)
    export_parser.add_argument("--state-full-ratio", type=float, default=0.30)
    export_parser.add_argument("--state-weak-trace-points", type=int, default=3)
    export_parser.add_argument("--state-line-dropout", type=float, default=0.40)
    export_parser.add_argument("--state-point-jitter-px", type=float, default=2.0)
    export_parser.add_argument("--state-truncate-prob", type=float, default=0.30)
    export_parser.add_argument("--lane-only", action="store_true", help="只导出 lane_line，忽略 intersection 标注")
    export_parser.add_argument("--include-lane", action="store_true")
    export_parser.add_argument("--include-intersection-boundary", action="store_true")
    export_parser.add_argument("--max-families-per-split", type=int, default=0)
    export_parser.add_argument("--empty-patch-drop-ratio", type=float, default=0.95)
    export_parser.add_argument("--empty-patch-seed", type=int, default=42)
    export_parser.add_argument("--use-system-prompt", action="store_true")
    export_parser.add_argument("--stagea-system-prompt", type=str, default=DEFAULT_STAGE_A_SYSTEM_PROMPT)
    export_parser.add_argument("--stagea-prompt-template", type=str, default=DEFAULT_STAGE_A_PROMPT_TEMPLATE)
    export_parser.add_argument("--stageb-system-prompt", type=str, default=DEFAULT_STAGE_B_SYSTEM_PROMPT)
    export_parser.add_argument("--stageb-prompt-template", type=str, default=DEFAULT_STAGE_B_PROMPT_TEMPLATE)

    build_all_parser = subparsers.add_parser("build-all", help="一键构建 manifest + stage_a/stage_b + fixed16")
    build_all_parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    build_all_parser.add_argument("--manifest-path", type=str, default="")
    build_all_parser.add_argument("--dataset-root", type=str, default="/dataset/zsy/dataset-extracted")
    build_all_parser.add_argument("--train-root", type=str, default="")
    build_all_parser.add_argument("--val-root", type=str, default="")
    build_all_parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    build_all_parser.add_argument("--image-relpath", type=str, default=DEFAULT_IMAGE_RELPATH)
    build_all_parser.add_argument("--mask-relpath", type=str, default=DEFAULT_MASK_RELPATH)
    build_all_parser.add_argument("--lane-relpath", type=str, default=DEFAULT_LANE_RELPATH)
    build_all_parser.add_argument("--intersection-relpath", type=str, default=DEFAULT_INTERSECTION_RELPATH)
    build_all_parser.add_argument("--mask-threshold", type=int, default=127)
    build_all_parser.add_argument("--tile-size-px", type=int, default=896)
    build_all_parser.add_argument("--overlap-px", type=int, default=232)
    build_all_parser.add_argument("--keep-margin-px", type=int, default=116)
    build_all_parser.add_argument("--review-crop-pad-px", type=int, default=64)
    build_all_parser.add_argument("--tile-min-mask-ratio", type=float, default=0.02)
    build_all_parser.add_argument("--tile-min-mask-pixels", type=int, default=256)
    build_all_parser.add_argument("--tile-max-per-sample", type=int, default=0)
    build_all_parser.add_argument("--search-within-review-bbox", action="store_true")
    build_all_parser.add_argument("--fallback-to-all-if-empty", action="store_true")
    build_all_parser.add_argument("--max-samples-per-split", type=int, default=0)
    build_all_parser.add_argument("--shard-index", type=int, default=0)
    build_all_parser.add_argument("--num-shards", type=int, default=1)
    build_all_parser.add_argument("--band-indices", type=int, nargs="+", default=[1, 2, 3])
    build_all_parser.add_argument("--resample-step-px", type=float, default=4.0)
    build_all_parser.add_argument("--boundary-tol-px", type=float, default=2.5)
    build_all_parser.add_argument("--trace-points", type=int, default=8)
    build_all_parser.add_argument("--state-mixture-mode", type=str, default="full", choices=["full", "mixed"])
    build_all_parser.add_argument("--state-no-state-ratio", type=float, default=0.30)
    build_all_parser.add_argument("--state-weak-ratio", type=float, default=0.40)
    build_all_parser.add_argument("--state-full-ratio", type=float, default=0.30)
    build_all_parser.add_argument("--state-weak-trace-points", type=int, default=3)
    build_all_parser.add_argument("--state-line-dropout", type=float, default=0.40)
    build_all_parser.add_argument("--state-point-jitter-px", type=float, default=2.0)
    build_all_parser.add_argument("--state-truncate-prob", type=float, default=0.30)
    build_all_parser.add_argument("--lane-only", action="store_true", help="只导出 lane_line，忽略 intersection 标注")
    build_all_parser.add_argument("--include-lane", action="store_true")
    build_all_parser.add_argument("--include-intersection-boundary", action="store_true")
    build_all_parser.add_argument("--max-families-per-split", type=int, default=0)
    build_all_parser.add_argument("--empty-patch-drop-ratio", type=float, default=0.95)
    build_all_parser.add_argument("--empty-patch-seed", type=int, default=42)
    build_all_parser.add_argument("--use-system-prompt", action="store_true")
    build_all_parser.add_argument("--stagea-system-prompt", type=str, default=DEFAULT_STAGE_A_SYSTEM_PROMPT)
    build_all_parser.add_argument("--stagea-prompt-template", type=str, default=DEFAULT_STAGE_A_PROMPT_TEMPLATE)
    build_all_parser.add_argument("--stageb-system-prompt", type=str, default=DEFAULT_STAGE_B_SYSTEM_PROMPT)
    build_all_parser.add_argument("--stageb-prompt-template", type=str, default=DEFAULT_STAGE_B_PROMPT_TEMPLATE)
    build_all_parser.add_argument("--skip-fixed16-build", action="store_true")
    build_all_parser.add_argument("--fixed16-output-name", type=str, default="fixed16_stage_a")
    build_all_parser.add_argument("--fixed16-stageb-output-name", type=str, default="fixed16_stage_b")
    build_all_parser.add_argument("--fixed16-grid-size", type=int, default=4)
    build_all_parser.add_argument("--fixed16-target-empty-ratio", type=float, default=0.10)
    build_all_parser.add_argument("--fixed16-seed", type=int, default=42)
    build_all_parser.add_argument("--fixed16-max-source-samples-per-split", type=int, default=0)
    build_all_parser.add_argument("--fixed16-resample-step-px", type=float, default=4.0)
    build_all_parser.add_argument("--fixed16-boundary-tol-px", type=float, default=2.5)
    build_all_parser.add_argument("--fixed16-image-root-mode", type=str, default="symlink", choices=["symlink", "copy", "none"])
    build_all_parser.add_argument("--fixed16-export-visualizations", action="store_true")
    build_all_parser.add_argument("--fixed16-max-visualizations-per-split", type=int, default=0)
    return parser


def run_build_manifest(args: argparse.Namespace) -> None:
    """执行 family manifest 构建。"""
    split_roots = {}
    if str(args.train_root).strip():
        split_roots["train"] = Path(str(args.train_root).strip()).resolve()
    if str(args.val_root).strip():
        split_roots["val"] = Path(str(args.val_root).strip()).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    families = build_family_manifest(
        dataset_root=Path(args.dataset_root).resolve(),
        splits=[str(split) for split in args.splits],
        image_relpath=str(args.image_relpath),
        mask_relpath=str(args.mask_relpath),
        lane_relpath=str(args.lane_relpath),
        intersection_relpath=str(args.intersection_relpath),
        mask_threshold=int(args.mask_threshold),
        tile_size_px=int(args.tile_size_px),
        overlap_px=int(args.overlap_px),
        keep_margin_px=int(args.keep_margin_px),
        review_crop_pad_px=int(args.review_crop_pad_px),
        tile_min_mask_ratio=float(args.tile_min_mask_ratio),
        tile_min_mask_pixels=int(args.tile_min_mask_pixels),
        tile_max_per_sample=int(args.tile_max_per_sample),
        search_within_review_bbox=bool(args.search_within_review_bbox),
        fallback_to_all_if_empty=bool(args.fallback_to_all_if_empty),
        max_samples_per_split=int(args.max_samples_per_split),
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
        split_roots=split_roots or None,
    )
    save_family_manifest(
        output_manifest=output_manifest,
        families=families,
        dataset_root=Path(args.dataset_root).resolve(),
        splits=[str(split) for split in args.splits],
        tile_size_px=int(args.tile_size_px),
        overlap_px=int(args.overlap_px),
        keep_margin_px=int(args.keep_margin_px),
        review_crop_pad_px=int(args.review_crop_pad_px),
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
    )
    print(f"Built {len(families)} families", flush=True)
    print(f"Manifest: {output_manifest}", flush=True)


def run_build_fixed16(args: argparse.Namespace) -> None:
    """执行 fixed16 数据集构建。"""
    summary = build_fixed16_dataset(
        input_root=Path(args.input_root).resolve(),
        output_root=Path(args.output_root).resolve(),
        splits=[str(split) for split in args.splits],
        grid_size=int(args.grid_size),
        target_empty_ratio=float(args.target_empty_ratio),
        target_empty_ratio_by_split=None,
        seed=int(args.seed),
        max_source_samples_per_split=int(args.max_source_samples_per_split),
        boundary_tol_px=float(args.boundary_tol_px),
        resample_step_px=float(args.resample_step_px),
        reuse_system_prompt=bool(args.use_system_prompt_from_source),
        image_root_mode=str(args.image_root_mode),
        export_visualizations=bool(args.export_visualizations),
        max_visualizations_per_split=int(args.max_visualizations_per_split),
    )
    print(f"Built fixed16 dataset: {summary['output_root']}", flush=True)


def run_export_stages(args: argparse.Namespace) -> None:
    """执行 stage_a / stage_b 导出。"""
    families = load_jsonl(Path(args.family_manifest).resolve())
    if bool(args.lane_only):
        include_lane = True
        include_intersection_boundary = False
    else:
        include_lane = True if (not bool(args.include_lane) and not bool(args.include_intersection_boundary)) else bool(args.include_lane)
        include_intersection_boundary = True if (not bool(args.include_lane) and not bool(args.include_intersection_boundary)) else bool(args.include_intersection_boundary)
    stagea_system_prompt = str(args.stagea_system_prompt).strip() if bool(args.use_system_prompt) else ""
    stageb_system_prompt = str(args.stageb_system_prompt).strip() if bool(args.use_system_prompt) else ""
    result = export_stage_datasets(
        families=families,
        output_root=Path(args.output_root).resolve(),
        splits=[str(split) for split in args.splits],
        band_indices=[int(index) for index in args.band_indices],
        mask_threshold=int(args.mask_threshold),
        resample_step_px=float(args.resample_step_px),
        boundary_tol_px=float(args.boundary_tol_px),
        trace_points=int(args.trace_points),
        state_mixture_mode=str(args.state_mixture_mode),
        state_no_state_ratio=float(args.state_no_state_ratio),
        state_weak_ratio=float(args.state_weak_ratio),
        state_full_ratio=float(args.state_full_ratio),
        state_weak_trace_points=int(args.state_weak_trace_points),
        state_line_dropout=float(args.state_line_dropout),
        state_point_jitter_px=float(args.state_point_jitter_px),
        state_truncate_prob=float(args.state_truncate_prob),
        include_lane=include_lane,
        include_intersection_boundary=include_intersection_boundary,
        max_families_per_split=int(args.max_families_per_split),
        empty_patch_drop_ratio=float(args.empty_patch_drop_ratio),
        empty_patch_seed=int(args.empty_patch_seed),
        empty_patch_drop_ratio_by_split=None,
        stagea_system_prompt=stagea_system_prompt,
        stagea_prompt_template=str(args.stagea_prompt_template),
        stageb_system_prompt=stageb_system_prompt,
        stageb_prompt_template=str(args.stageb_prompt_template),
    )
    manifest_path = str(Path(args.family_manifest).resolve())
    for stage_root in (Path(result["stage_a_root"]), Path(result["stage_b_root"])):
        summary_path = stage_root / "export_summary.json"
        with summary_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        payload["source_family_manifest"] = manifest_path
        write_json(summary_path, payload)
    print(f"Built stage datasets: {result['stage_a_root']} and {result['stage_b_root']}", flush=True)


def run_build_all(args: argparse.Namespace) -> None:
    """执行 manifest + stage_a/stage_b + fixed16 的一键流程。"""
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_path).resolve() if str(args.manifest_path).strip() else output_root / DEFAULT_MANIFEST_PATH.name
    split_roots = {}
    if str(args.train_root).strip():
        split_roots["train"] = Path(str(args.train_root).strip()).resolve()
    if str(args.val_root).strip():
        split_roots["val"] = Path(str(args.val_root).strip()).resolve()
    split_list = [str(split) for split in args.splits]
    families = []
    dataset_root = Path(args.dataset_root).resolve()
    if "train" in split_list:
        families.extend(
            build_family_manifest(
                dataset_root=dataset_root,
                splits=["train"],
                image_relpath=str(args.image_relpath),
                mask_relpath=str(args.mask_relpath),
                lane_relpath=str(args.lane_relpath),
                intersection_relpath=str(args.intersection_relpath),
                mask_threshold=int(args.mask_threshold),
                tile_size_px=int(args.tile_size_px),
                overlap_px=int(args.overlap_px),
                keep_margin_px=int(args.keep_margin_px),
                review_crop_pad_px=int(args.review_crop_pad_px),
                tile_min_mask_ratio=float(args.tile_min_mask_ratio),
                tile_min_mask_pixels=int(args.tile_min_mask_pixels),
                tile_max_per_sample=int(args.tile_max_per_sample),
                search_within_review_bbox=bool(args.search_within_review_bbox),
                fallback_to_all_if_empty=bool(args.fallback_to_all_if_empty),
                max_samples_per_split=int(args.max_samples_per_split),
                shard_index=int(args.shard_index),
                num_shards=int(args.num_shards),
                split_roots=split_roots or None,
            )
        )
    if "val" in split_list:
        families.extend(
            build_family_manifest(
                dataset_root=dataset_root,
                splits=["val"],
                image_relpath=str(args.image_relpath),
                mask_relpath=str(args.mask_relpath),
                lane_relpath=str(args.lane_relpath),
                intersection_relpath=str(args.intersection_relpath),
                mask_threshold=int(args.mask_threshold),
                tile_size_px=int(args.tile_size_px),
                overlap_px=int(args.overlap_px),
                keep_margin_px=int(args.keep_margin_px),
                review_crop_pad_px=int(args.review_crop_pad_px),
                tile_min_mask_ratio=0.0,
                tile_min_mask_pixels=0,
                tile_max_per_sample=0,
                search_within_review_bbox=False,
                fallback_to_all_if_empty=True,
                max_samples_per_split=int(args.max_samples_per_split),
                shard_index=int(args.shard_index),
                num_shards=int(args.num_shards),
                split_roots=split_roots or None,
            )
        )
    save_family_manifest(
        output_manifest=manifest_path,
        families=families,
        dataset_root=dataset_root,
        splits=split_list,
        tile_size_px=int(args.tile_size_px),
        overlap_px=int(args.overlap_px),
        keep_margin_px=int(args.keep_margin_px),
        review_crop_pad_px=int(args.review_crop_pad_px),
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
    )
    if bool(args.lane_only):
        include_lane = True
        include_intersection_boundary = False
    else:
        include_lane = True if (not bool(args.include_lane) and not bool(args.include_intersection_boundary)) else bool(args.include_lane)
        include_intersection_boundary = True if (not bool(args.include_lane) and not bool(args.include_intersection_boundary)) else bool(args.include_intersection_boundary)
    stagea_system_prompt = str(args.stagea_system_prompt).strip() if bool(args.use_system_prompt) else ""
    stageb_system_prompt = str(args.stageb_system_prompt).strip() if bool(args.use_system_prompt) else ""
    export_result = export_stage_datasets(
        families=families,
        output_root=output_root,
        splits=split_list,
        band_indices=[int(index) for index in args.band_indices],
        mask_threshold=int(args.mask_threshold),
        resample_step_px=float(args.resample_step_px),
        boundary_tol_px=float(args.boundary_tol_px),
        trace_points=int(args.trace_points),
        state_mixture_mode=str(args.state_mixture_mode),
        state_no_state_ratio=float(args.state_no_state_ratio),
        state_weak_ratio=float(args.state_weak_ratio),
        state_full_ratio=float(args.state_full_ratio),
        state_weak_trace_points=int(args.state_weak_trace_points),
        state_line_dropout=float(args.state_line_dropout),
        state_point_jitter_px=float(args.state_point_jitter_px),
        state_truncate_prob=float(args.state_truncate_prob),
        include_lane=include_lane,
        include_intersection_boundary=include_intersection_boundary,
        max_families_per_split=int(args.max_families_per_split),
        empty_patch_drop_ratio=float(args.empty_patch_drop_ratio),
        empty_patch_seed=int(args.empty_patch_seed),
        empty_patch_drop_ratio_by_split={"val": 0.0},
        stagea_system_prompt=stagea_system_prompt,
        stagea_prompt_template=str(args.stagea_prompt_template),
        stageb_system_prompt=stageb_system_prompt,
        stageb_prompt_template=str(args.stageb_prompt_template),
    )
    if not bool(args.skip_fixed16_build):
        build_fixed16_dataset(
            input_root=Path(export_result["stage_a_root"]),
            output_root=output_root / str(args.fixed16_output_name).strip(),
            splits=split_list,
            grid_size=int(args.fixed16_grid_size),
            target_empty_ratio=float(args.fixed16_target_empty_ratio),
            target_empty_ratio_by_split={"val": 1.0},
            seed=int(args.fixed16_seed),
            max_source_samples_per_split=int(args.fixed16_max_source_samples_per_split),
            boundary_tol_px=float(args.fixed16_boundary_tol_px),
            resample_step_px=float(args.fixed16_resample_step_px),
            reuse_system_prompt=True,
            image_root_mode=str(args.fixed16_image_root_mode),
            export_visualizations=bool(args.fixed16_export_visualizations),
            max_visualizations_per_split=int(args.fixed16_max_visualizations_per_split),
        )
        build_fixed16_dataset(
            input_root=Path(export_result["stage_b_root"]),
            output_root=output_root / str(args.fixed16_stageb_output_name).strip(),
            splits=split_list,
            grid_size=int(args.fixed16_grid_size),
            target_empty_ratio=float(args.fixed16_target_empty_ratio),
            target_empty_ratio_by_split={"val": 1.0},
            seed=int(args.fixed16_seed),
            max_source_samples_per_split=int(args.fixed16_max_source_samples_per_split),
            boundary_tol_px=float(args.fixed16_boundary_tol_px),
            resample_step_px=float(args.fixed16_resample_step_px),
            reuse_system_prompt=True,
            image_root_mode=str(args.fixed16_image_root_mode),
            export_visualizations=bool(args.fixed16_export_visualizations),
            max_visualizations_per_split=int(args.fixed16_max_visualizations_per_split),
        )
    print(f"Built full pipeline under: {output_root}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


def main() -> None:
    """统一 CLI 入口。"""
    parser = build_root_parser()
    args = parser.parse_args()
    if args.command == "build-manifest":
        run_build_manifest(args)
        return
    if args.command == "build-fixed16":
        run_build_fixed16(args)
        return
    if args.command == "export-stages":
        run_export_stages(args)
        return
    if args.command == "build-all":
        run_build_all(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()