# dataset_generation

独立版 Geo 数据集生成重构目录。

这个目录的目标是把原来散落在 scripts 下的数据集生成与配套处理逻辑，按更清晰的分层重新组织到一个可独立运行的新目录中。

当前重构遵循以下原则：

- 不依赖原项目包或原 scripts 模块
- 保留原有数据集生成业务语义，不故意改动算法口径
- 文件命名更规范，目录分层更清晰
- 代码尽量简洁，关键函数和核心逻辑块补中文注释
- 统一通过一个 CLI 入口调度

## 项目说明

当前原项目中的数据集主链可以概括为三段：

1. 从原始 train/val 目录生成 family_manifest.jsonl
2. 从 family manifest 导出 stage_a 和 stage_b 数据集
3. 从 stage_a 或 stage_b 母集继续生成 fixed16 数据集

此外，还有几类配套工具：

- 处理结果重建
- fixed16 预测回拼
- patch 可视化
- ShareGPT 缺图清洗

本目录就是为这些能力提供一个新的、独立的代码组织方式。

## 当前状态

当前已经落地的模块：

- 通用基础模块
- manifest 子系统
- fixed16 子系统
- 统一 CLI 的部分子命令
- 本 README

当前已经可以使用的命令：

- build-manifest
- export-stages
- build-fixed16
- build-all

当前还未补齐的子系统：

- reconstruct-processed
- reconstruct-fixed16
- visualize-processed
- clean-missing-images

也就是说，这个目录当前是一个进行中的重构版本，不是最终全量替代版。

## 文件结构

当前目录结构如下：

```text
dataset_generation/
├── __init__.py
├── cli.py
├── README.md
├── common/
│   ├── defaults.py
│   ├── geometry_utils.py
│   ├── io_utils.py
│   ├── raster_utils.py
│   └── state_mix.py
├── manifest/
│   ├── builder.py
│   └── tiling.py
├── fixed16/
│   └── builder.py
├── stages/
└── tools/
```

各目录职责：

- common
  放通用基础能力，包括默认路径、几何工具、JSONL 读写、栅格读取、state 混合辅助逻辑。

- manifest
  放 family manifest 的构建逻辑，包括滑窗切 patch、mask 覆盖率统计、ownership keep_box 修正等。

- fixed16
  放从 stage_a 或 stage_b 母集继续构建 fixed16 数据集的逻辑。

- stages
  预留给 stage_a / stage_b 导出逻辑。

- tools
  预留给重建、拼接、可视化、清洗等配套工具。

## 运行环境

默认运行环境是已经配置完成的：

```bash
conda activate data
```

本目录不额外要求新的环境说明，默认沿用原项目里已经安装好的依赖。

如果你的终端当前不在这个环境中，先切换：

```bash
conda activate data
```

## 输入数据要求

默认输入数据仍遵循原项目的数据组织方式。

一个典型样本目录通常类似：

```text
dataset_root/
├── train/
│   └── sample_xxx/
│       ├── patch_tif/
│       │   ├── 0.tif
│       │   └── 0_edit_poly.tif
│       └── label_check_crop/
│           ├── Lane.geojson
│           └── Intersection.geojson
└── val/
    └── sample_xxx/
        ├── patch_tif/
        │   ├── 0.tif
        │   └── 0_edit_poly.tif
        └── label_check_crop/
            ├── Lane.geojson
            └── Intersection.geojson
```

默认相对路径如下：

- image-relpath: patch_tif/0.tif
- mask-relpath: patch_tif/0_edit_poly.tif
- lane-relpath: label_check_crop/Lane.geojson
- intersection-relpath: label_check_crop/Intersection.geojson

如果你的数据结构不同，可以通过命令行参数覆盖这些相对路径。

如果你只想生成 lane 数据集，不想把 Intersection.geojson 纳入训练目标，也不需要改原始目录结构，只需要在导出阶段或一键构建阶段传 lane-only。

## lane-only 数据集模式

lane-only 模式用于构造“只包含 lane_line，不包含 intersection_polygon”的数据集。

这个模式的目标是：

- 保留原始影像和 Lane.geojson 的处理逻辑
- 不把 Intersection.geojson 写入生成数据集
- 让 stage_a、stage_b、fixed16 都只围绕 lane 目标构造样本

### 适用场景

适合以下情况：

- 你当前只想训练 lane 生成任务
- 你不希望路口 polygon 混入 target_lines
- 你想对比 lane-only 与 lane+intersection 两种数据口径

### 行为说明

打开 lane-only 后：

- 会继续读取原始样本目录
- 仍然允许原始数据里存在 Intersection.geojson
- 但生成数据集时不会把 intersection_polygon 写入 target_lines
- stage_a / stage_b 的 target_lines 会只保留 lane_line
- 由 stage 数据集继续生成的 fixed16 也会自动变成 lane-only

没有打开 lane-only 时：

- 默认仍按当前主链逻辑，同时导出 lane 和 intersection

### 如何使用

#### 1. 只导出 lane-only 的 stage_a / stage_b

```bash
python -m dataset_generation.cli export-stages \
  --family-manifest /path/to/output/family_manifest.jsonl \
  --output-root /path/to/output/processed_all \
  --splits train val \
  --lane-only
```

#### 2. 一键构建完整 lane-only 主链

```bash
python -m dataset_generation.cli build-all \
  --train-root /path/to/raw/train \
  --val-root /path/to/raw/val \
  --lane-only
```

#### 3. 使用默认一键脚本构建 lane-only 数据集

```bash
LANE_ONLY=1 bash dataset_generation/run_build_all_default.sh
```

### 与原始数据目录的关系

lane-only 模式不会要求你删除原始数据中的 Intersection.geojson。

也就是说：

- 原始样本目录可以保持不变
- 影像、mask、Lane.geojson、Intersection.geojson 都可以继续保留
- lane-only 只影响“生成数据集时是否把 intersection 写入样本”

### 注意事项

1. lane-only 改变的是导出数据口径，不改变 manifest 扫描路径

manifest 里仍可能保留 source_intersection_path 这类源路径字段，因为它们描述的是原始样本来源，而不是最终训练目标内容。

2. lane-only 会影响后续整个主链

如果你是通过 build-all 开启 lane-only，那么：

- stage_a 是 lane-only
- stage_b 是 lane-only
- fixed16_stage_a 也是 lane-only
- fixed16_stage_b 也是 lane-only

3. 如果你要做口径对比实验

建议把 lane-only 输出到单独目录，避免和 lane+intersection 的产物混在一起。

## 统一 CLI

统一入口文件是：

- cli.py

如果你希望最少操作直接跑默认全流程，也可以直接运行：

- run_build_all_default.sh

推荐从仓库根目录执行：

```bash
python -m dataset_generation.cli --help
```

最短的一键运行方式：

```bash
bash dataset_generation/run_build_all_default.sh
```

查看当前已实现的子命令：

```bash
python -m dataset_generation.cli build-manifest --help
python -m dataset_generation.cli export-stages --help
python -m dataset_generation.cli build-fixed16 --help
python -m dataset_generation.cli build-all --help
```

## 运行步骤

### 1. 生成 family manifest

这个步骤用于把原始 train/val 根目录切成 family + patch 索引，并写出 family_manifest.jsonl。

示例命令：

```bash
python -m dataset_generation.cli build-manifest \
  --train-root /path/to/raw/train \
  --val-root /path/to/raw/val \
  --output-manifest /path/to/output/family_manifest.jsonl \
  --splits train val \
  --image-relpath patch_tif/0.tif \
  --mask-relpath patch_tif/0_edit_poly.tif \
  --lane-relpath label_check_crop/Lane.geojson \
  --intersection-relpath label_check_crop/Intersection.geojson \
  --tile-size-px 896 \
  --overlap-px 232 \
  --keep-margin-px 116 \
  --review-crop-pad-px 64 \
  --tile-min-mask-ratio 0.02 \
  --tile-min-mask-pixels 256 \
  --search-within-review-bbox \
  --fallback-to-all-if-empty
```

输出文件：

- family_manifest.jsonl
- family_manifest.summary.json

### 2. 从 stage 母集生成 fixed16

这个步骤要求输入目录已经是一个现有 stage 数据集根目录，通常包含：

- train.jsonl / val.jsonl
- meta_train.jsonl / meta_val.jsonl
- images/

示例命令：

```bash
python -m dataset_generation.cli build-fixed16 \
  --input-root /path/to/stage_a/dataset \
  --output-root /path/to/fixed16_stage_a \
  --splits train val \
  --grid-size 4 \
  --target-empty-ratio 0.10 \
  --seed 42 \
  --resample-step-px 4.0 \
  --boundary-tol-px 2.5 \
  --use-system-prompt-from-source \
  --image-root-mode symlink
```

输出目录通常包括：

- train.jsonl / val.jsonl
- meta_train.jsonl / meta_val.jsonl
- dataset_info.json
- build_summary.json
- images/

### 2. 从 family manifest 导出 stage_a / stage_b

这个步骤会读取 family_manifest.jsonl，并生成两套 ShareGPT 风格数据集：

- stage_a: patch-only
- stage_b: state-aware

示例命令：

```bash
python -m dataset_generation.cli export-stages \
  --family-manifest /path/to/output/family_manifest.jsonl \
  --output-root /path/to/output/processed_all \
  --splits train val \
  --band-indices 1 2 3 \
  --mask-threshold 127 \
  --resample-step-px 4.0 \
  --boundary-tol-px 2.5 \
  --trace-points 8 \
  --state-mixture-mode full \
  --empty-patch-drop-ratio 0.95 \
  --empty-patch-seed 42 \
  --use-system-prompt
```

如果你只想导出 lane，不要 intersection：

```bash
python -m dataset_generation.cli export-stages \
  --family-manifest /path/to/output/family_manifest.jsonl \
  --output-root /path/to/output/processed_all \
  --splits train val \
  --lane-only
```

输出目录：

- stage_a/dataset
- stage_b/dataset

每个 stage 目录下通常包含：

- train.jsonl / val.jsonl
- meta_train.jsonl / meta_val.jsonl
- dataset_info.json
- export_summary.json
- images/

### 3. 一键构建全流程

这个命令会顺序执行：

1. 生成 family manifest
2. 导出 stage_a / stage_b
3. 基于 stage_a 和 stage_b 各自继续生成 fixed16

示例命令：

```bash
python -m dataset_generation.cli build-all \
  --train-root /path/to/raw/train \
  --val-root /path/to/raw/val \
  --output-root /path/to/output/processed_all \
  --splits train val \
  --image-relpath patch_tif/0.tif \
  --mask-relpath patch_tif/0_edit_poly.tif \
  --lane-relpath label_check_crop/Lane.geojson \
  --intersection-relpath label_check_crop/Intersection.geojson \
  --tile-size-px 896 \
  --overlap-px 232 \
  --keep-margin-px 116 \
  --review-crop-pad-px 64 \
  --tile-min-mask-ratio 0.02 \
  --tile-min-mask-pixels 256 \
  --search-within-review-bbox \
  --fallback-to-all-if-empty \
  --resample-step-px 4.0 \
  --boundary-tol-px 2.5 \
  --trace-points 8 \
  --state-mixture-mode full \
  --empty-patch-drop-ratio 0.95 \
  --fixed16-grid-size 4 \
  --fixed16-target-empty-ratio 0.10 \
  --fixed16-image-root-mode symlink \
  --use-system-prompt
```

如果你要整条主链都走 lane-only：

```bash
python -m dataset_generation.cli build-all \
  --train-root /path/to/raw/train \
  --val-root /path/to/raw/val \
  --lane-only
```

如果你希望完全不传参数，当前也支持直接运行：

```bash
python -m dataset_generation.cli build-all
```

此时会使用以下默认值：

- dataset-root: /dataset/zsy/dataset-extracted
- output-root: ./dataset_generation_output/processed_all
- manifest-path: ./dataset_generation_output/processed_all/family_manifest.jsonl
- splits: train val
- image-relpath: patch_tif/0.tif
- mask-relpath: patch_tif/0_edit_poly.tif
- lane-relpath: label_check_crop/Lane.geojson
- intersection-relpath: label_check_crop/Intersection.geojson
- tile-size-px: 896
- overlap-px: 232
- keep-margin-px: 116
- review-crop-pad-px: 64
- tile-min-mask-ratio: 0.02
- tile-min-mask-pixels: 256
- resample-step-px: 4.0
- boundary-tol-px: 2.5
- trace-points: 8
- state-mixture-mode: full
- empty-patch-drop-ratio: 0.95
- fixed16-grid-size: 4
- fixed16-target-empty-ratio: 0.10
- fixed16-image-root-mode: symlink

如果你本机原始数据不在默认目录，可以只改最少几个参数：

```bash
python -m dataset_generation.cli build-all \
  --train-root /path/to/raw/train \
  --val-root /path/to/raw/val
```

或者直接用一键脚本配环境变量：

```bash
DATASET_ROOT=/path/to/raw_dataset \
OUTPUT_ROOT=/path/to/output/processed_all \
bash dataset_generation/run_build_all_default.sh
```

这个一键入口已经内建与旧主链一致的两类覆盖：

- stage 导出时，val 不丢空 patch
- fixed16 构建时，val 保留全部 box

## 参数说明

先看这一节：

- build-manifest：原始 train/val -> family_manifest.jsonl
- export-stages：family_manifest.jsonl -> stage_a / stage_b
- build-fixed16：stage_a 或 stage_b -> fixed16
- build-all：一键串起前三段

### 常用参数速查

| 参数 | 适用命令 | 作用 | 常见值 |
|---|---|---|---|
| --train-root | build-manifest, build-all | 训练集根目录 | /data/raw/train |
| --val-root | build-manifest, build-all | 验证集根目录 | /data/raw/val |
| --image-relpath | build-manifest, build-all | 每个样本里的影像相对路径 | patch_tif/0.tif |
| --mask-relpath | build-manifest, build-all | 每个样本里的 mask 相对路径 | patch_tif/0_edit_poly.tif |
| --lane-relpath | build-manifest, build-all | 每个样本里的 lane 标注相对路径 | label_check_crop/Lane.geojson |
| --intersection-relpath | build-manifest, build-all | 每个样本里的 intersection 标注相对路径 | label_check_crop/Intersection.geojson |
| --tile-size-px | build-manifest, build-all | patch 边长 | 896 |
| --overlap-px | build-manifest, build-all | patch 重叠像素 | 232 |
| --keep-margin-px | build-manifest, build-all | keep_box 收缩边距 | 116 |
| --resample-step-px | export-stages, build-all, build-fixed16 | 线段重采样步长 | 4.0 |
| --boundary-tol-px | export-stages, build-all, build-fixed16 | cut 判定容差 | 2.5 |
| --trace-points | export-stages, build-all | state trace 点数 | 8 |
| --empty-patch-drop-ratio | export-stages, build-all | 空 patch 下采样比例 | 0.95 |
| --fixed16-grid-size | build-all | fixed16 每边网格数 | 4 |
| --fixed16-target-empty-ratio | build-all | fixed16 空 box 保留比例 | 0.10 |
| --lane-only | export-stages, build-all | 只导出 lane，不导出 intersection | 开关 |

### build-manifest 参数表

| 参数 | 默认值 | 说明 |
|---|---|---|
| --dataset-root | /dataset/zsy/dataset-extracted | 总数据根目录；未单独传 train-root 和 val-root 时，会自动查找 dataset-root/train 与 dataset-root/val。 |
| --train-root | 空 | 显式指定 train 根目录。 |
| --val-root | 空 | 显式指定 val 根目录。 |
| --output-manifest | 无 | 输出 family_manifest.jsonl 的路径。 |
| --splits | train val | 要处理的 split 列表。 |
| --image-relpath | patch_tif/0.tif | 每个 sample 目录下影像 tif 的相对路径。 |
| --mask-relpath | patch_tif/0_edit_poly.tif | 每个 sample 目录下 review mask 的相对路径。 |
| --lane-relpath | label_check_crop/Lane.geojson | 每个 sample 目录下 lane 标注的相对路径。 |
| --intersection-relpath | label_check_crop/Intersection.geojson | 每个 sample 目录下 intersection 标注的相对路径。 |
| --mask-threshold | 127 | mask 二值化阈值。 |
| --tile-size-px | 896 | patch 裁切边长。 |
| --overlap-px | 232 | 相邻 patch 的重叠像素。 |
| --keep-margin-px | 116 | keep_box 相对 crop_box 的收缩边距，用于 ownership 分配。 |
| --review-crop-pad-px | 64 | review bbox 的额外扩张像素。 |
| --tile-min-mask-ratio | 0.02 | patch 最低 mask 覆盖率。 |
| --tile-min-mask-pixels | 256 | patch 最低 mask 像素数。 |
| --tile-max-per-sample | 0 | 单样本最多保留多少个 patch；0 表示不限制。 |
| --search-within-review-bbox | 关闭 | 只在 review mask 外接框附近切 patch。 |
| --fallback-to-all-if-empty | 关闭 | 若筛选后无 patch，回退到整图范围重新切。 |
| --max-samples-per-split | 0 | 每个 split 最多处理多少个样本；0 表示不限制。 |
| --shard-index | 0 | 当前分片编号。 |
| --num-shards | 1 | 总分片数。 |

### export-stages 参数表

| 参数 | 默认值 | 说明 |
|---|---|---|
| --family-manifest | 无 | 输入的 family_manifest.jsonl 路径。 |
| --output-root | 无 | stage_a / stage_b 输出根目录。 |
| --splits | train val | 要导出的 split 列表。 |
| --band-indices | 1 2 3 | 从原始 tif 读取的波段编号。 |
| --mask-threshold | 127 | review mask 二值化阈值。 |
| --resample-step-px | 4.0 | patch GT 线段重采样步长。 |
| --boundary-tol-px | 2.5 | patch 边界 cut 判定容差。 |
| --trace-points | 8 | stage_b state trace 默认保留点数。 |
| --state-mixture-mode | full | state 混合模式，支持 full 和 mixed。 |
| --state-no-state-ratio | 0.30 | mixed 模式下 no_state 比例。 |
| --state-weak-ratio | 0.40 | mixed 模式下 weak_state 比例。 |
| --state-full-ratio | 0.30 | mixed 模式下 full_state 比例。 |
| --state-weak-trace-points | 3 | weak state 最多保留多少个 trace 点。 |
| --state-line-dropout | 0.40 | weak state 的线级 dropout 比例。 |
| --state-point-jitter-px | 2.0 | weak state 点抖动幅度。 |
| --state-truncate-prob | 0.30 | weak state 线段截断概率。 |
| --lane-only | 关闭 | 只导出 lane_line，忽略 intersection_polygon。 |
| --include-lane | 关闭 | 手动启用 lane 导出。若 include-lane 和 include-intersection-boundary 都不传，则默认两者都导出。 |
| --include-intersection-boundary | 关闭 | 手动启用 intersection 导出。 |
| --max-families-per-split | 0 | 每个 split 最多导出多少个 family；0 表示不限制。 |
| --empty-patch-drop-ratio | 0.95 | 空 patch 下采样比例。 |
| --empty-patch-seed | 42 | 空 patch 下采样随机种子。 |
| --use-system-prompt | 关闭 | 是否启用 system prompt。 |
| --stagea-system-prompt | 内置默认文案 | stage_a 的 system prompt 文本。 |
| --stagea-prompt-template | 内置默认文案 | stage_a 的 user prompt 模板。 |
| --stageb-system-prompt | 内置默认文案 | stage_b 的 system prompt 文本。 |
| --stageb-prompt-template | 内置默认文案 | stage_b 的 user prompt 模板。 |

### build-fixed16 参数表

| 参数 | 默认值 | 说明 |
|---|---|---|
| --input-root | 无 | stage_a 或 stage_b 母集根目录。 |
| --output-root | 无 | fixed16 输出根目录。 |
| --splits | train val | 要处理的 split 列表。 |
| --grid-size | 4 | 每边划分多少格；4 表示 4x4，共 16 个 box。 |
| --target-empty-ratio | 0.10 | 空 box 的目标保留比例。 |
| --seed | 42 | 空 box 过滤随机种子。 |
| --max-source-samples-per-split | 0 | 最多处理多少个上游 patch；0 表示不限制。 |
| --resample-step-px | 4.0 | box 内 target line 重采样步长。 |
| --boundary-tol-px | 2.5 | box 内 cut 端点判定容差。 |
| --use-system-prompt-from-source | 关闭 | 复用上游 stage 数据集中的 system prompt。 |
| --image-root-mode | symlink | images 目录暴露方式，支持 symlink、copy、none。 |
| --export-visualizations | 关闭 | 是否导出 fixed16 QA 可视化图。 |
| --max-visualizations-per-split | 0 | 每个 split 最多导出多少张 QA 图；0 表示不限制。 |

### build-all 参数表

build-all 是总入口，基本上等于把 build-manifest、export-stages 和 build-fixed16 的参数并到同一个命令里。

| 参数 | 默认值 | 说明 |
|---|---|---|
| --output-root | dataset_generation_output/processed_all | 全流程输出根目录。 |
| --manifest-path | 空 | 单独指定 manifest 输出路径；不传时默认写到 output-root/family_manifest.jsonl。 |
| --dataset-root | /dataset/zsy/dataset-extracted | 总数据根目录。 |
| --train-root | 空 | 单独指定 train 根目录。 |
| --val-root | 空 | 单独指定 val 根目录。 |
| --splits | train val | 要处理的 split 列表。 |
| --image-relpath | patch_tif/0.tif | 影像相对路径。 |
| --mask-relpath | patch_tif/0_edit_poly.tif | mask 相对路径。 |
| --lane-relpath | label_check_crop/Lane.geojson | lane 标注相对路径。 |
| --intersection-relpath | label_check_crop/Intersection.geojson | intersection 标注相对路径。 |
| --mask-threshold | 127 | mask 二值化阈值。 |
| --tile-size-px | 896 | patch 边长。 |
| --overlap-px | 232 | patch 重叠像素。 |
| --keep-margin-px | 116 | keep_box 边距。 |
| --review-crop-pad-px | 64 | review bbox 扩张像素。 |
| --tile-min-mask-ratio | 0.02 | train patch 最低 mask 覆盖率。 |
| --tile-min-mask-pixels | 256 | train patch 最低 mask 像素数。 |
| --tile-max-per-sample | 0 | 单样本最多 patch 数；0 表示不限制。 |
| --search-within-review-bbox | 关闭 | train 阶段是否收缩到 review bbox。 |
| --fallback-to-all-if-empty | 关闭 | 无 patch 时是否回退整图。 |
| --max-samples-per-split | 0 | 每个 split 最多处理多少个样本。 |
| --shard-index | 0 | 当前分片编号。 |
| --num-shards | 1 | 总分片数。 |
| --band-indices | 1 2 3 | stage 图像使用的波段。 |
| --resample-step-px | 4.0 | stage GT 重采样步长。 |
| --boundary-tol-px | 2.5 | stage cut 判定容差。 |
| --trace-points | 8 | stage_b trace 点数。 |
| --state-mixture-mode | full | state 混合模式。 |
| --state-no-state-ratio | 0.30 | mixed 模式 no_state 比例。 |
| --state-weak-ratio | 0.40 | mixed 模式 weak_state 比例。 |
| --state-full-ratio | 0.30 | mixed 模式 full_state 比例。 |
| --state-weak-trace-points | 3 | weak state trace 点数上限。 |
| --state-line-dropout | 0.40 | weak state 线 dropout 比例。 |
| --state-point-jitter-px | 2.0 | weak state 点抖动幅度。 |
| --state-truncate-prob | 0.30 | weak state 截断概率。 |
| --lane-only | 关闭 | 整条主链只保留 lane，不写入 intersection。 |
| --include-lane | 关闭 | 手动启用 lane 导出。 |
| --include-intersection-boundary | 关闭 | 手动启用 intersection 导出。 |
| --max-families-per-split | 0 | 每个 split 最多导出多少个 family。 |
| --empty-patch-drop-ratio | 0.95 | train 空 patch 下采样比例。 |
| --empty-patch-seed | 42 | train 空 patch 下采样随机种子。 |
| --use-system-prompt | 关闭 | 是否启用 system prompt。 |
| --stagea-system-prompt | 内置默认文案 | stage_a system prompt。 |
| --stagea-prompt-template | 内置默认文案 | stage_a prompt 模板。 |
| --stageb-system-prompt | 内置默认文案 | stage_b system prompt。 |
| --stageb-prompt-template | 内置默认文案 | stage_b prompt 模板。 |
| --skip-fixed16-build | 关闭 | 只生成 manifest 和 stage_a / stage_b，不继续做 fixed16。 |
| --fixed16-output-name | fixed16_stage_a | stage_a 母集对应的 fixed16 输出目录名。 |
| --fixed16-stageb-output-name | fixed16_stage_b | stage_b 母集对应的 fixed16 输出目录名。 |
| --fixed16-grid-size | 4 | fixed16 每边网格数。 |
| --fixed16-target-empty-ratio | 0.10 | train 阶段空 box 保留比例。 |
| --fixed16-seed | 42 | fixed16 空 box 过滤种子。 |
| --fixed16-max-source-samples-per-split | 0 | 每个 split 最多处理多少个上游 patch。 |
| --fixed16-resample-step-px | 4.0 | fixed16 线段重采样步长。 |
| --fixed16-boundary-tol-px | 2.5 | fixed16 cut 判定容差。 |
| --fixed16-image-root-mode | symlink | fixed16 images 暴露方式。 |
| --fixed16-export-visualizations | 关闭 | 是否导出 fixed16 QA 图。 |
| --fixed16-max-visualizations-per-split | 0 | 每个 split 最多导出多少张 fixed16 QA 图。 |

build-all 的额外行为：

- val 会自动走完整输出口径
- stage 导出时，val 的空 patch 不会被丢弃
- fixed16 导出时，val 的空 box 会全部保留
- 不传任何参数时，也可以直接跑默认全流程

## 使用示例

### 示例 1：只构建 manifest

```bash
python -m dataset_generation.cli build-manifest \
  --train-root /data/geo/train \
  --val-root /data/geo/val \
  --output-manifest /data/output/family_manifest.jsonl \
  --splits train val \
  --tile-size-px 896 \
  --overlap-px 232 \
  --keep-margin-px 116 \
  --review-crop-pad-px 64 \
  --tile-min-mask-ratio 0.02 \
  --tile-min-mask-pixels 256 \
  --search-within-review-bbox \
  --fallback-to-all-if-empty
```

### 示例 2：从 stage_a 生成 fixed16_stage_a

```bash
python -m dataset_generation.cli build-fixed16 \
  --input-root /data/output/stage_a/dataset \
  --output-root /data/output/fixed16_stage_a \
  --splits train val \
  --grid-size 4 \
  --target-empty-ratio 0.10 \
  --seed 42 \
  --resample-step-px 4.0 \
  --boundary-tol-px 2.5 \
  --use-system-prompt-from-source \
  --image-root-mode symlink
```

### 示例 3：从 family manifest 导出 stage_a / stage_b

```bash
python -m dataset_generation.cli export-stages \
  --family-manifest /data/output/family_manifest.jsonl \
  --output-root /data/output/processed_all \
  --splits train val \
  --resample-step-px 4.0 \
  --boundary-tol-px 2.5 \
  --trace-points 8 \
  --state-mixture-mode mixed \
  --state-no-state-ratio 0.30 \
  --state-weak-ratio 0.40 \
  --state-full-ratio 0.30 \
  --empty-patch-drop-ratio 0.95 \
  --use-system-prompt
```

### 示例 4：一键跑完整主链

```bash
python -m dataset_generation.cli build-all \
  --train-root /data/geo/train \
  --val-root /data/geo/val \
  --output-root /data/output/processed_all \
  --splits train val \
  --tile-size-px 896 \
  --overlap-px 232 \
  --keep-margin-px 116 \
  --review-crop-pad-px 64 \
  --tile-min-mask-ratio 0.02 \
  --tile-min-mask-pixels 256 \
  --search-within-review-bbox \
  --fallback-to-all-if-empty \
  --resample-step-px 4.0 \
  --boundary-tol-px 2.5 \
  --trace-points 8 \
  --empty-patch-drop-ratio 0.95 \
  --fixed16-grid-size 4 \
  --fixed16-target-empty-ratio 0.10 \
  --fixed16-image-root-mode symlink
```

### 示例 5：只处理 val 或只做 smoke test

```bash
python -m dataset_generation.cli build-manifest \
  --val-root /data/geo/val \
  --output-manifest /data/output/val_manifest.jsonl \
  --splits val \
  --max-samples-per-split 8
```

```bash
python -m dataset_generation.cli build-fixed16 \
  --input-root /data/output/stage_a/dataset \
  --output-root /data/output/fixed16_stage_a_smoke \
  --splits val \
  --max-source-samples-per-split 16
```

## train / val 使用注意事项

当前目录已经有两种口径：

- build-manifest / export-stages / build-fixed16：手工分步口径，行为取决于你自己传的参数
- build-all：一键总入口口径，内建了旧主链对 val 的特殊覆盖

旧主链里 val 常见的策略是：

- 不按 review bbox 缩小搜索区域
- tile-min-mask-ratio 设为 0
- tile-min-mask-pixels 设为 0
- tile-max-per-sample 设为 0
- fallback-to-all-if-empty 打开

fixed16 侧同理：

- train 可以丢空 box
- val 通常应保留全部 box

如果你要和旧的一键构建行为完全对齐，优先使用 build-all。

## 输出结构说明

### family manifest 输出

manifest 阶段的主要输出：

```text
/path/to/output/
├── family_manifest.jsonl
└── family_manifest.summary.json
```

family_manifest.jsonl 中每条记录通常包含：

- family_id
- split
- source_sample_id
- source_image_path
- source_mask_path
- source_lane_path
- source_intersection_path
- image_size
- crop_size
- tiling
- crop_bbox
- patches
- tile_audits

### fixed16 输出

fixed16 阶段的输出通常类似：

```text
/path/to/fixed16_output/
├── train.jsonl
├── val.jsonl
├── meta_train.jsonl
├── meta_val.jsonl
├── dataset_info.json
├── build_summary.json
└── images/
```

其中：

- train.jsonl / val.jsonl
  是训练样本本体

- meta_train.jsonl / meta_val.jsonl
  是 box 级元数据

- dataset_info.json
  用于数据集注册

- build_summary.json
  记录构建统计

## 注意事项

1. 当前目录已经可以独立跑通 manifest、stage_a / stage_b、fixed16 这三段主链

其中最省事的入口是：

```bash
bash dataset_generation/run_build_all_default.sh
```

如果你想用一键脚本直接生成 lane-only 数据集：

```bash
LANE_ONLY=1 bash dataset_generation/run_build_all_default.sh
```

2. 当前 README 是按实际已落地代码编写的

没有实现的命令没有写成“可直接使用”，避免文档和代码不一致。

3. fixed16 输入必须是一个已经存在的 stage 数据集

也就是说，build-fixed16 不是从原始 tif/geojson 直接构建，而是从 stage 母集继续加工。

4. 推荐从仓库根目录运行

这样可以直接使用：

```bash
python -m dataset_generation.cli ...
```

5. 如果你追求和旧脚本一模一样的产物

不要只看命令名是否相似，还要重点核对：

- train / val 参数覆盖
- empty patch / empty box 过滤
- resample-step-px
- boundary-tol-px
- keep-margin-px
- review bbox 搜索策略

## 后续计划

后续会继续补齐以下模块：

- tools: 重建、拼接、可视化、清洗
- cli.py: tools 相关子命令和更多统一入口

等这些模块补齐后，这个目录会成为完整的独立版数据集生成工具集。
