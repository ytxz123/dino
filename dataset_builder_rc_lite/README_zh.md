# dataset_builder_rc_lite

基于原始 dataset_builder 的四段主链，按 dataset_generation 中已经验证过的 RC 适配规则重新整理出的轻量版 RC 数据集构建方案。

这个目录的目标不是复刻 dataset_generation 的工程结构，而是把其中已经证明有效的 RC 适配能力压回原始 dataset_builder 的脚本式工作流中，得到一套更容易理解、更容易运行、更容易定位错误的最小方案。

## 方案特点

1. 保留原始 dataset_builder 的四段脚本主链。
2. 保留 RC 数据集适配能力。
3. 删除统一 CLI 和过度工程化的分层。
4. 增加运行时日志、进度提示和更清晰的报错上下文。
5. 保持每一步输入输出明确，便于单步排查。

## 适用数据

本方案适用于 RC 风格样本目录。默认每个样本目录结构如下：

```text
sample_xxx/
├── patch_tif/
│   ├── 0.tif
│   └── 0_edit_poly.tif
└── label_check_crop/
    ├── Lane.geojson
    └── Intersection.geojson
```

默认相对路径：

- image-relpath: patch_tif/0.tif
- mask-relpath: patch_tif/0_edit_poly.tif
- lane-relpath: label_check_crop/Lane.geojson
- intersection-relpath: label_check_crop/Intersection.geojson

如果你的目录不同，可以通过脚本参数覆盖。

## 核心改造说明

### 保留的原始主链能力

1. family manifest 构建
2. patch-only 数据集导出
3. fixed16 Stage A 数据集构建
4. Stage B trace 数据集构建

### 从 RC 改造方案中保留的必要规则

1. 使用 GeoTIFF 作为影像输入。
2. 使用 Lane.geojson 和 Intersection.geojson 作为矢量标注输入。
3. 使用 review mask 对候选 patch 做筛选。
4. 使用 overlap 和 keep_box 解决重叠 patch 的监督归属问题。
5. 支持 lane-only 数据口径。
6. 支持 band_indices 指定读取波段。

### 明确删除的复杂逻辑

1. 不提供统一 CLI。
2. 不提供 build-all。
3. 不提供 state mixture 的 weak/no-state/full 多模式混合。
4. 不提供多层冗余 meta 表达。
5. 不提供 dataset_generation 里的额外工具子系统。

## 目录结构

```text
dataset_builder_rc_lite/
├── README_zh.md
├── scripts/
│   ├── build_rc_family_manifest.py
│   ├── export_llamafactory_patch_only_from_raw_family_manifest.py
│   ├── build_patch_only_fixed_grid_targetbox_dataset.py
│   └── build_stageb_fixed16_gt_point_angle_dataset.py
└── unimapgen/
    └── dataset_build_refactor/
        ├── __init__.py
        ├── common.py
        ├── geometry.py
        ├── rc_dataset.py
        ├── tiling.py
        ├── patch_only.py
        ├── fixed16.py
        ├── stageb.py
        └── viz.py
```

## 文件职责

### 脚本层

1. [scripts/build_rc_family_manifest.py](scripts/build_rc_family_manifest.py)

- 扫描 train/val 样本目录。
- 读取 GeoTIFF 尺寸和 review mask。
- 生成 RC 滑窗 patch。
- 输出 family_manifest.jsonl。

2. [scripts/export_llamafactory_patch_only_from_raw_family_manifest.py](scripts/export_llamafactory_patch_only_from_raw_family_manifest.py)

- 读取 family manifest。
- 从 GeoJSON 解析 lane 和 intersection。
- 生成 patch 图像与 patch-local target_lines。
- 输出 patch-only ShareGPT 数据集。

3. [scripts/build_patch_only_fixed_grid_targetbox_dataset.py](scripts/build_patch_only_fixed_grid_targetbox_dataset.py)

- 从 patch-only 样本展开 fixed16 box 任务。
- 为每个 box 生成 prompt anchor 和 box 内 target_lines。
- 按空样本比例控制 empty 样本保留量。

4. [scripts/build_stageb_fixed16_gt_point_angle_dataset.py](scripts/build_stageb_fixed16_gt_point_angle_dataset.py)

- 从 fixed16 Stage A 样本重建左邻 / 上邻 trace。
- 输出带 incoming trace prompt 的 Stage B 样本。

### helper 层

1. [unimapgen/dataset_build_refactor/common.py](unimapgen/dataset_build_refactor/common.py)

- JSON/JSONL 读写
- ShareGPT 样本组装
- 日志输出
- 基础输入校验

2. [unimapgen/dataset_build_refactor/rc_dataset.py](unimapgen/dataset_build_refactor/rc_dataset.py)

- GeoTIFF 元信息读取
- review mask 读取
- GeoJSON CRS 投影
- 世界坐标转像素坐标

3. [unimapgen/dataset_build_refactor/tiling.py](unimapgen/dataset_build_refactor/tiling.py)

- RC 滑窗生成
- keep_box 计算
- mask 统计与候选 patch 筛选

4. [unimapgen/dataset_build_refactor/patch_only.py](unimapgen/dataset_build_refactor/patch_only.py)

- keep_box 内几何裁剪
- lane / polygon patch-local target_lines 构造

5. [unimapgen/dataset_build_refactor/fixed16.py](unimapgen/dataset_build_refactor/fixed16.py)

- fixed16 网格 box 切分
- anchor 选择
- box 内 GT 裁切

6. [unimapgen/dataset_build_refactor/stageb.py](unimapgen/dataset_build_refactor/stageb.py)

- 左邻 / 上邻 trace 提取
- Stage B prompt 构造

## 依赖环境

至少需要这些依赖：

- numpy
- pillow
- rasterio
- pyproj

如果你使用虚拟环境，建议先激活环境再运行。

## 数据处理主链

本方案严格按以下顺序执行：

1. 构建 family manifest
2. 导出 patch-only 数据集
3. 构建 fixed16 Stage A
4. 构建 Stage B

任何一步失败，都应先修复该步输入，再继续后续步骤。

## 运行方法

### 一键运行默认参数脚本

如果你希望只改少量默认配置，然后一键串联执行四步主链，可以直接使用：

[scripts/run_build_all_default.py](scripts/run_build_all_default.py)

使用方法：

1. 打开 [scripts/run_build_all_default.py](scripts/run_build_all_default.py)
2. 修改顶部的 CONFIG，至少填写：
  - dataset_root
  - 或 train_root / val_root
3. 运行：

```bash
python dataset_builder_rc_lite/scripts/run_build_all_default.py
```

这个脚本会依次执行：

1. build_rc_family_manifest.py
2. export_llamafactory_patch_only_from_raw_family_manifest.py
3. build_patch_only_fixed_grid_targetbox_dataset.py
4. build_stageb_fixed16_gt_point_angle_dataset.py

默认输出目录结构：

```text
dataset_builder_rc_lite_output/
├── family_manifest.jsonl
├── family_manifest.summary.json
├── patch_only_rc/ 或 patch_only_rc_lane_only/
├── fixed16_stage_a_rc/ 或 fixed16_stage_a_lane_only/
└── fixed16_stage_b_rc/ 或 fixed16_stage_b_lane_only/
```

脚本顶部已经预置了常用参数，包括：

- tile_size_px
- overlap_px
- keep_margin_px
- band_indices
- lane_only
- fixed16_grid_size
- stageb_trace_points_per_hint

如果只是常规跑数，通常只需要改数据路径和 lane_only。

### 1. 构建 family manifest

最常见写法：

```bash
python dataset_builder_rc_lite/scripts/build_rc_family_manifest.py \
  --dataset-root /path/to/rc_dataset \
  --output-manifest /path/to/output/family_manifest.jsonl
```

当 train 和 val 分开存放时：

```bash
python dataset_builder_rc_lite/scripts/build_rc_family_manifest.py \
  --train-root /path/to/train \
  --val-root /path/to/val \
  --output-manifest /path/to/output/family_manifest.jsonl
```

### 2. 导出 patch-only 数据集

```bash
python dataset_builder_rc_lite/scripts/export_llamafactory_patch_only_from_raw_family_manifest.py \
  --family-manifest /path/to/output/family_manifest.jsonl \
  --output-root /path/to/output/patch_only_rc
```

只导出 lane_line：

```bash
python dataset_builder_rc_lite/scripts/export_llamafactory_patch_only_from_raw_family_manifest.py \
  --family-manifest /path/to/output/family_manifest.jsonl \
  --output-root /path/to/output/patch_only_rc_lane_only \
  --lane-only
```

### 3. 构建 fixed16 Stage A

```bash
python dataset_builder_rc_lite/scripts/build_patch_only_fixed_grid_targetbox_dataset.py \
  --input-root /path/to/output/patch_only_rc \
  --output-root /path/to/output/fixed16_stage_a_rc
```

### 4. 构建 Stage B

```bash
python dataset_builder_rc_lite/scripts/build_stageb_fixed16_gt_point_angle_dataset.py \
  --input-root /path/to/output/fixed16_stage_a_rc \
  --output-root /path/to/output/fixed16_stage_b_rc
```

## 输出文件组织

### 第一步输出

manifest 阶段会输出：

- family_manifest.jsonl
- family_manifest.summary.json

其中：

- family_manifest.jsonl 是后续 patch-only 导出的唯一输入。
- summary.json 用于核对 split 样本数、family 数和滑窗参数。

### 第二步输出

patch-only 阶段会输出：

```text
patch_only_rc/
├── images/
├── train.jsonl
├── val.jsonl
├── meta_train.jsonl
├── meta_val.jsonl
├── dataset_info.json
└── export_summary.json
```

说明：

- images/ 保存裁剪后的 patch 图片。
- train.jsonl 和 val.jsonl 是训练主文件。
- meta_*.jsonl 是附带元信息，便于排查样本问题。

### 第三步输出

fixed16 Stage A 会输出：

```text
fixed16_stage_a_rc/
├── images/
├── train.jsonl
├── val.jsonl
├── meta_train.jsonl
├── meta_val.jsonl
├── dataset_info.json
└── build_summary.json
```

### 第四步输出

Stage B 会输出：

```text
fixed16_stage_b_rc/
├── images/
├── train.jsonl
├── val.jsonl
├── meta_train.jsonl
├── meta_val.jsonl
├── dataset_info.json
└── build_summary.json
```

## 运行时日志说明

四个脚本都已经补充了统一日志输出。日志格式如下：

```text
[2026-03-25 12:34:56] [Manifest] split=train sample_progress=20/100 (20.0%) sample_id=xxx
```

日志包含三类信息：

1. 阶段开始和结束

- start
- done
- summary 文件位置

2. 处理进度

- split 级进度
- family 级进度
- source sample 级进度

3. 错误和警告

- WARNING: 某个样本缺图、无有效窗口、无有效 GeoJSON 几何
- ERROR: 某个 split、family、sample 在具体步骤失败

## 参数详解

### build_rc_family_manifest.py

输入相关：

- --dataset-root
  - 数据根目录，默认按 root/train 和 root/val 查找。
- --train-root
  - 单独指定 train 目录。
- --val-root
  - 单独指定 val 目录。
- --output-manifest
  - 输出 manifest 路径，必填。
- --splits
  - 要扫描的 split，默认 train val。

路径相关：

- --image-relpath
  - 样本内影像相对路径。
- --mask-relpath
  - 样本内审核 mask 相对路径。
- --lane-relpath
  - 样本内 lane GeoJSON 相对路径。
- --intersection-relpath
  - 样本内 intersection GeoJSON 相对路径。

滑窗相关：

- --tile-size-px
  - patch 尺寸。
- --overlap-px
  - patch 间重叠像素。
- --keep-margin-px
  - keep_box 收缩边距。
- --review-crop-pad-px
  - review mask bbox 外扩像素。

筛选相关：

- --mask-threshold
  - 二值化 mask 的阈值。
- --tile-min-mask-ratio
  - patch 内最小 mask 占比。
- --tile-min-mask-pixels
  - patch 内最小 mask 像素数。
- --tile-max-per-sample
  - 每个样本最多保留多少 patch，0 表示不限制。
- --search-within-review-bbox
  - 先按 review mask bbox 缩小搜索范围。
- --fallback-to-all-if-empty
  - 如果筛选后一个 patch 都没有，退回全部候选窗口。
- --max-samples-per-split
  - 每个 split 最多扫描多少个样本，0 表示不限制。

### export_llamafactory_patch_only_from_raw_family_manifest.py

输入输出：

- --family-manifest
  - 上一步生成的 family_manifest.jsonl。
- --output-root
  - patch-only 输出目录。
- --splits
  - 导出的 split。

影像和几何处理：

- --band-indices
  - GeoTIFF 读取波段，默认 1 2 3。
- --mask-threshold
  - review mask 阈值。
- --resample-step-px
  - 线段重采样步长。
- --boundary-tol-px
  - 边界 cut 判定容差。

类别控制：

- --lane-only
  - 只导出 lane_line。
- --include-lane
  - 显式启用 lane。
- --include-intersection-boundary
  - 显式启用 intersection_polygon。

采样控制：

- --max-families-per-split
  - 每个 split 最多处理多少个 family。
- --empty-patch-drop-ratio
  - 空 patch 丢弃比例。
- --empty-patch-seed
  - 空 patch 采样随机种子。

Prompt：

- --use-system-prompt
  - 使用内置 system prompt。
- --system-prompt
  - 直接传入 system prompt 文本。
- --system-prompt-file
  - 从文件读取 system prompt。

### build_patch_only_fixed_grid_targetbox_dataset.py

输入输出：

- --input-root
  - patch-only 数据集目录。
- --output-root
  - fixed16 输出目录。
- --splits
  - 处理哪些 split。

box 任务：

- --grid-size
  - patch 每边切多少格，默认 4，得到 16 个 box。
- --resample-step-px
  - box 内目标线重采样步长。
- --boundary-tol-px
  - box 边界 cut 判定容差。

采样：

- --target-empty-ratio
  - 最终空 box 占比上限。
- --seed
  - 空 box 采样随机种子。

其他：

- --use-system-prompt-from-source
  - 继承 patch-only 中的 system prompt。
- --image-root-mode
  - 输出 images 目录的暴露方式，支持 symlink、copy、none。

### build_stageb_fixed16_gt_point_angle_dataset.py

输入输出：

- --input-root
  - fixed16 Stage A 数据集目录。
- --output-root
  - Stage B 输出目录。
- --splits
  - 处理哪些 split。

trace 控制：

- --grid-size
  - 用于重建 box 索引的网格大小。
- --boundary-tol-px
  - 判断邻居 box cut 点是否接到共享边界的容差。
- --trace-points-per-hint
  - 每条 incoming trace 最多保留多少点。

Prompt：

- --use-system-prompt-from-source
  - 优先沿用 Stage A 的 system prompt。
- --system-prompt
  - 手动覆盖 system prompt 文本。
- --system-prompt-file
  - 从文件读取 system prompt。

其他：

- --image-root-mode
  - images 目录暴露方式。

## 常见错误与排查

### 1. manifest 为空

常见原因：

- --dataset-root、--train-root、--val-root 路径不对。
- --image-relpath 配置错误。
- mask 过滤阈值过严。

排查方法：

1. 查看 split 根目录是否真的存在。
2. 查看日志里是否出现 skip missing image。
3. 降低 --tile-min-mask-ratio 或 --tile-min-mask-pixels。
4. 打开 --fallback-to-all-if-empty 试一次。

### 2. patch-only 导出时没有有效几何

常见原因：

- Lane.geojson 或 Intersection.geojson 路径错误。
- GeoJSON CRS 和影像 CRS 不匹配。
- lane-only 配置导致 intersection 被关闭。

排查方法：

1. 看日志是否有 has no valid GeoJSON features。
2. 检查 manifest 里的 source_lane_path 和 source_intersection_path。
3. 先只保留 lane-only 做最小验证。

### 3. fixed16 报 no non-empty samples

常见原因：

- patch-only 阶段 target_lines 全为空。
- grid-size 太细，box 被切得太小。

排查方法：

1. 先检查 patch-only 的 meta_*.jsonl 是否有非空样本。
2. 先试更大的 box，比如减少切分密度。

### 4. Stage B 样本全部没有 state

这不一定是错误，通常说明：

- 当前数据在左邻 / 上邻边界没有 cut trace。
- 或 Stage A 的 target_lines 本身较少。

如果你确认不合理，优先检查 fixed16 的 meta_*.jsonl 中 target_lines 的 cut 标记是否正确。

## 推荐调试方式

如果第一次跑，建议这样做：

1. 先用 --max-samples-per-split 只跑 1 到 3 个样本。
2. 再用 --max-families-per-split 只跑少量 family。
3. 先确认 family_manifest.summary.json、export_summary.json、build_summary.json 是否符合预期。
4. 再放开到完整数据集。

## 当前验证状态

本目录已经完成：

1. 脚本级问题检查。
2. Python 编译级检查。
3. 运行时日志与进度补充。

当前还没有在真实 RC 全量数据上完成端到端数据验证，因此如果你的数据目录、GeoJSON CRS 或影像波段口径与默认假设不同，仍需要按日志提示做一次实跑校验。