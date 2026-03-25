"""带默认参数的一键构建入口。

用途：
- 用一个 Python 脚本串联执行 4 个主链脚本
- 把常用参数集中到顶部配置区，方便直接修改
- 在运行时打印每一步命令和输出目录，便于排查

使用方式：
1. 直接修改本文件顶部的 CONFIG
2. 运行：python dataset_builder_rc_lite/scripts/run_build_all_default.py
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


@dataclass
class BuildConfig:
    # 数据根目录。若 train / val 分开，可以改成分别填写 train_root / val_root。
    dataset_root: str = "/path/to/rc_dataset"
    train_root: str = ""
    val_root: str = ""

    # 主输出目录。
    output_root: str = str(PROJECT_ROOT / "dataset_builder_rc_lite_output")

    # 是否只导出 lane_line。
    lane_only: bool = False

    # 通用 split。
    splits: List[str] = field(default_factory=lambda: ["train", "val"])

    # manifest 参数。
    image_relpath: str = ""
    mask_relpath: str = ""
    image_dir_relpath: str = "patch_tif"
    image_glob: str = "*.tif"
    mask_suffix: str = "_edit_poly.tif"
    lane_relpath: str = "label_check_crop/Lane.geojson"
    intersection_relpath: str = "label_check_crop/Intersection.geojson"
    mask_threshold: int = 127
    tile_size_px: int = 896
    overlap_px: int = 232
    keep_margin_px: int = 116
    review_crop_pad_px: int = 64
    tile_min_mask_ratio: float = 0.02
    tile_min_mask_pixels: int = 256
    tile_max_per_sample: int = 0
    search_within_review_bbox: bool = False
    fallback_to_all_if_empty: bool = False
    max_samples_per_split: int = 0

    # patch-only 参数。
    band_indices: List[int] = field(default_factory=lambda: [1, 2, 3])
    patch_resample_step_px: float = 4.0
    patch_boundary_tol_px: float = 2.5
    max_families_per_split: int = 0
    empty_patch_drop_ratio: float = 0.95
    empty_patch_seed: int = 42
    use_patch_system_prompt: bool = False

    # fixed16 参数。
    fixed16_grid_size: int = 4
    fixed16_target_empty_ratio: float = 0.10
    fixed16_seed: int = 42
    fixed16_resample_step_px: float = 4.0
    fixed16_boundary_tol_px: float = 2.5
    fixed16_use_system_prompt_from_source: bool = False
    fixed16_image_root_mode: str = "symlink"

    # Stage B 参数。
    stageb_grid_size: int = 4
    stageb_boundary_tol_px: float = 2.5
    stageb_trace_points_per_hint: int = 3
    stageb_use_system_prompt_from_source: bool = False
    stageb_image_root_mode: str = "symlink"


CONFIG = BuildConfig()


def log(message: str) -> None:
    print(f"[run-build-all] {message}", flush=True)


def run_command(command: List[str], step_name: str) -> None:
    log(f"start step={step_name}")
    log("command=" + " ".join(command))
    try:
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Step failed: {step_name}") from exc
    log(f"done step={step_name}")


def main() -> None:
    python_executable = sys.executable
    config = CONFIG

    if config.dataset_root == "/path/to/rc_dataset" and not config.train_root and not config.val_root:
        raise ValueError("Please edit CONFIG.dataset_root or CONFIG.train_root / CONFIG.val_root before running.")

    output_root = Path(config.output_root).resolve()
    manifest_path = output_root / "family_manifest.jsonl"
    patch_only_root = output_root / ("patch_only_rc_lane_only" if config.lane_only else "patch_only_rc")
    fixed16_root = output_root / ("fixed16_stage_a_lane_only" if config.lane_only else "fixed16_stage_a_rc")
    stageb_root = output_root / ("fixed16_stage_b_lane_only" if config.lane_only else "fixed16_stage_b_rc")

    log(f"project_root={PROJECT_ROOT}")
    log(f"output_root={output_root}")
    log(f"manifest_path={manifest_path}")
    log(f"patch_only_root={patch_only_root}")
    log(f"fixed16_root={fixed16_root}")
    log(f"stageb_root={stageb_root}")

    manifest_cmd = [
        python_executable,
        str(SCRIPT_DIR / "build_rc_family_manifest.py"),
        "--output-manifest",
        str(manifest_path),
        "--splits",
        *config.splits,
        "--image-dir-relpath",
        config.image_dir_relpath,
        "--image-glob",
        config.image_glob,
        "--mask-suffix",
        config.mask_suffix,
        "--lane-relpath",
        config.lane_relpath,
        "--intersection-relpath",
        config.intersection_relpath,
        "--mask-threshold",
        str(config.mask_threshold),
        "--tile-size-px",
        str(config.tile_size_px),
        "--overlap-px",
        str(config.overlap_px),
        "--keep-margin-px",
        str(config.keep_margin_px),
        "--review-crop-pad-px",
        str(config.review_crop_pad_px),
        "--tile-min-mask-ratio",
        str(config.tile_min_mask_ratio),
        "--tile-min-mask-pixels",
        str(config.tile_min_mask_pixels),
        "--tile-max-per-sample",
        str(config.tile_max_per_sample),
        "--max-samples-per-split",
        str(config.max_samples_per_split),
    ]
    if config.image_relpath:
        manifest_cmd.extend(["--image-relpath", config.image_relpath])
    if config.mask_relpath:
        manifest_cmd.extend(["--mask-relpath", config.mask_relpath])
    if config.train_root:
        manifest_cmd.extend(["--train-root", config.train_root])
    if config.val_root:
        manifest_cmd.extend(["--val-root", config.val_root])
    if (not config.train_root) and (not config.val_root):
        manifest_cmd.extend(["--dataset-root", config.dataset_root])
    if config.search_within_review_bbox:
        manifest_cmd.append("--search-within-review-bbox")
    if config.fallback_to_all_if_empty:
        manifest_cmd.append("--fallback-to-all-if-empty")

    patch_only_cmd = [
        python_executable,
        str(SCRIPT_DIR / "export_llamafactory_patch_only_from_raw_family_manifest.py"),
        "--family-manifest",
        str(manifest_path),
        "--output-root",
        str(patch_only_root),
        "--splits",
        *config.splits,
        "--band-indices",
        *[str(index) for index in config.band_indices],
        "--mask-threshold",
        str(config.mask_threshold),
        "--resample-step-px",
        str(config.patch_resample_step_px),
        "--boundary-tol-px",
        str(config.patch_boundary_tol_px),
        "--max-families-per-split",
        str(config.max_families_per_split),
        "--empty-patch-drop-ratio",
        str(config.empty_patch_drop_ratio),
        "--empty-patch-seed",
        str(config.empty_patch_seed),
    ]
    if config.lane_only:
        patch_only_cmd.append("--lane-only")
    if config.use_patch_system_prompt:
        patch_only_cmd.append("--use-system-prompt")

    fixed16_cmd = [
        python_executable,
        str(SCRIPT_DIR / "build_patch_only_fixed_grid_targetbox_dataset.py"),
        "--input-root",
        str(patch_only_root),
        "--output-root",
        str(fixed16_root),
        "--splits",
        *config.splits,
        "--grid-size",
        str(config.fixed16_grid_size),
        "--target-empty-ratio",
        str(config.fixed16_target_empty_ratio),
        "--seed",
        str(config.fixed16_seed),
        "--resample-step-px",
        str(config.fixed16_resample_step_px),
        "--boundary-tol-px",
        str(config.fixed16_boundary_tol_px),
        "--image-root-mode",
        config.fixed16_image_root_mode,
    ]
    if config.fixed16_use_system_prompt_from_source:
        fixed16_cmd.append("--use-system-prompt-from-source")

    stageb_cmd = [
        python_executable,
        str(SCRIPT_DIR / "build_stageb_fixed16_gt_point_angle_dataset.py"),
        "--input-root",
        str(fixed16_root),
        "--output-root",
        str(stageb_root),
        "--splits",
        *config.splits,
        "--grid-size",
        str(config.stageb_grid_size),
        "--boundary-tol-px",
        str(config.stageb_boundary_tol_px),
        "--trace-points-per-hint",
        str(config.stageb_trace_points_per_hint),
        "--image-root-mode",
        config.stageb_image_root_mode,
    ]
    if config.stageb_use_system_prompt_from_source:
        stageb_cmd.append("--use-system-prompt-from-source")

    run_command(manifest_cmd, "build-manifest")
    run_command(patch_only_cmd, "export-patch-only")
    run_command(fixed16_cmd, "build-fixed16-stage-a")
    run_command(stageb_cmd, "build-stage-b")

    log("all steps completed")


if __name__ == "__main__":
    main()