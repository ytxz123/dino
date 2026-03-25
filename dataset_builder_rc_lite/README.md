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
│   ├── 0_edit_poly.tif
│   ├── 1.tif
│   ├── 1_edit_poly.tif
│   └── ...
└── label_check_crop/
    ├── Lane.geojson
    └── Intersection.geojson
```

默认处理逻辑：

- 默认扫描 patch_tif 目录下所有原始 tif。
- 会自动忽略 *_edit_poly.tif 这类 mask 文件本身。
- 对每个原始 tif，会自动寻找同名 mask，例如：
  - 0.tif 对应 0_edit_poly.tif
  - 1.tif 对应 1_edit_poly.tif
- lane-relpath: label_check_crop/Lane.geojson
- intersection-relpath: label_check_crop/Intersection.geojson

如果你的目录不同，或者你只想强制处理单个 tif，也可以通过脚本参数覆盖。

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
├── README.md
├── scripts/
│   ├── build_manifest.py
│   ├── build_patch_only.py
│   ├── build_fixed16.py
│   ├── build_stageb.py
│   └── run_all.py
├── templates/
│   ├── fixed16.txt
│   ├── patch_only.txt
│   └── stageb.txt
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

1. [scripts/build_manifest.py](scripts/build_manifest.py)

- 扫描 train/val 样本目录。
- 读取 GeoTIFF 尺寸和 review mask。
- 生成 RC 滑窗 patch。
- 输出 family_manifest.jsonl。

2. [scripts/build_patch_only.py](scripts/build_patch_only.py)

- 读取 family manifest。
- 从 GeoJSON 解析 lane 和 intersection。
- 生成 patch 图像与 patch-local target_lines。
- 输出 patch-only ShareGPT 数据集。

3. [scripts/build_fixed16.py](scripts/build_fixed16.py)

- 从 patch-only 样本展开 fixed16 box 任务。
- 为每个 box 生成 prompt anchor 和 box 内 target_lines。
- 按空样本比例控制 empty 样本保留量。

4. [scripts/build_stageb.py](scripts/build_stageb.py)

- 从 fixed16 Stage A 样本构建 Stage B 数据集。
- 默认输出 no-state 版本；只有显式开启 gt 模式时才会额外构建邻块 trace 提示。

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

[scripts/run_all.py](scripts/run_all.py)

使用方法：

1. 打开 [scripts/run_all.py](scripts/run_all.py)
2. 修改顶部的 CONFIG，至少填写：
  - dataset_root
  - 或 train_root / val_root
3. 运行：

```bash
python dataset_builder_rc_lite/scripts/run_all.py
```

这个脚本会依次执行：

1. build_manifest.py
2. build_patch_only.py
3. build_fixed16.py
4. build_stageb.py

在真正启动四步主链之前，一键脚本还会先做一次模板预检查：

- 检查模板文件是否存在
- 检查模板文件是否为空
- 检查 fixed16 和 Stage B 模板中的占位符是否完整且合法
- 检查 patch-only 模板里是否误写了不会被替换的格式化占位符

默认模板文件已经放在 [dataset_builder_rc_lite/templates/patch_only.txt](dataset_builder_rc_lite/templates/patch_only.txt)、[dataset_builder_rc_lite/templates/fixed16.txt](dataset_builder_rc_lite/templates/fixed16.txt)、[dataset_builder_rc_lite/templates/stageb.txt](dataset_builder_rc_lite/templates/stageb.txt)。其中 Stage B 这份文件对应推荐默认的 no-state 写法；如果你不填写 stageb_user_prompt_file，脚本会直接按 stageb_state_mode 选择内置默认模板。

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

其中与 prompt 相关的新增配置有：

- patch_user_prompt_file
- fixed16_user_prompt_file
- stageb_user_prompt_file

其中 patch_user_prompt_file 和 fixed16_user_prompt_file 默认已经指向项目内置模板文件；Stage B 推荐默认留空，让脚本按 stageb_state_mode 自动选择内置模板。

如果只是常规跑数，通常只需要改数据路径和 lane_only。

## 一键脚本参数说明

只需要修改 [scripts/run_all.py](scripts/run_all.py) 顶部的 CONFIG。

常用参数含义如下。

- dataset_root: 统一数据根目录，内部应包含 train 和 val。
- train_root: 单独指定 train 目录；填写后可覆盖 dataset_root 的 train 路径推导。
- val_root: 单独指定 val 目录；填写后可覆盖 dataset_root 的 val 路径推导。
- output_root: 四步主链的总输出目录。
- lane_only: 为 true 时只导出 lane_line，不导出 intersection_polygon。
- splits: 需要处理的 split 列表，默认是 train 和 val。

- image_dir_relpath: 每个样本内原始 tif 所在目录，默认是 patch_tif。
- image_glob: 在 image_dir_relpath 下扫描原始 tif 的匹配规则，默认是 *.tif。
- mask_suffix: 原始 tif 对应 review mask 的文件名后缀，默认是 _edit_poly.tif。
- lane_relpath: 车道 GeoJSON 相对路径。
- intersection_relpath: 路口 GeoJSON 相对路径。

- tile_size_px: manifest 阶段滑窗大小。
- overlap_px: 相邻滑窗重叠像素。
- keep_margin_px: keep_box 的边缘收缩宽度。
- review_crop_pad_px: review 区域外扩像素。
- tile_min_mask_ratio: patch 至少需要满足的 mask 覆盖比例。
- tile_min_mask_pixels: patch 至少需要满足的 mask 像素数。

- patch_resample_step_px: patch-only 阶段折线重采样步长。
- patch_boundary_tol_px: patch-only 阶段边界 cut 判定容差。
- empty_patch_drop_ratio: patch-only 阶段空样本丢弃比例。

- fixed16_grid_size: fixed16 阶段网格边长，4 表示切成 4x4 共 16 个 box。
- fixed16_target_empty_ratio: fixed16 阶段最终保留的空样本比例目标。
- fixed16_resample_step_px: fixed16 阶段 box 内折线重采样步长。
- fixed16_boundary_tol_px: fixed16 阶段 box 边界 cut 判定容差。

- stageb_grid_size: Stage B 默认网格大小，通常与 fixed16_grid_size 保持一致。
- stageb_state_mode: Stage B 状态模式。推荐默认使用 none；只有在你明确要做“带邻块提示的条件续接任务”时才改成 gt。
- stageb_boundary_tol_px: Stage B 邻块 trace 边界匹配容差。
- stageb_trace_points_per_hint: 每条 incoming trace 最多保留多少个提示点。

- patch_user_prompt_file: patch-only user prompt 模板文件路径。留空时使用内置默认模板。
- fixed16_user_prompt_file: fixed16 user prompt 模板文件路径。留空时使用内置默认模板。
- stageb_user_prompt_file: Stage B user prompt 模板文件路径。推荐默认留空，由脚本按 stageb_state_mode 自动选择内置模板；只有在你明确要自定义 Stage B prompt 时再填写。

其中：

- patch-only 模板是纯文本模板，不需要占位符。
- fixed16 模板如果自定义，应只保留这些占位符：{box_x_min}、{box_y_min}、{box_x_max}、{box_y_max}。
- 当 stageb_state_mode=none 时，Stage B 模板应只保留这些占位符：{box_x_min}、{box_y_min}、{box_x_max}、{box_y_max}。
- 当 stageb_state_mode=gt 时，Stage B 模板应保留这些占位符：{box_x_min}、{box_y_min}、{box_x_max}、{box_y_max}、{trace_points_json}。
- fixed16 模板不再允许引用 start_x、start_y、end_x、end_y，避免把由 GT 提取的锚点端点泄露给模型。
- 如果模板里写了不存在的占位符，脚本会直接报错，便于及时定位模板问题。

示例：

```text
<image>
Reconstruct the road-structure line map inside target box [{box_x_min},{box_y_min},{box_x_max},{box_y_max}].
Keep all coordinates in the patch-local coordinate system.
```

```text
<image>
Reconstruct the target road-structure line map inside target box [{box_x_min},{box_y_min},{box_x_max},{box_y_max}].
Keep all coordinates in the patch-local coordinate system.
```

### 1. 构建 family manifest

最常见写法：

```bash
python dataset_builder_rc_lite/scripts/build_manifest.py \
  --dataset-root /path/to/rc_dataset \
  --output-manifest /path/to/output/family_manifest.jsonl
```

当 train 和 val 分开存放时：

```bash
python dataset_builder_rc_lite/scripts/build_manifest.py \
  --train-root /path/to/train \
  --val-root /path/to/val \
  --output-manifest /path/to/output/family_manifest.jsonl
```

### 2. 导出 patch-only 数据集

```bash
python dataset_builder_rc_lite/scripts/build_patch_only.py \
  --family-manifest /path/to/output/family_manifest.jsonl \
  --output-root /path/to/output/patch_only_rc
```

只导出 lane_line：

```bash
python dataset_builder_rc_lite/scripts/build_patch_only.py \
  --family-manifest /path/to/output/family_manifest.jsonl \
  --output-root /path/to/output/patch_only_rc_lane_only \
  --lane-only
```

### 3. 构建 fixed16 Stage A

```bash
python dataset_builder_rc_lite/scripts/build_fixed16.py \
  --input-root /path/to/output/patch_only_rc \
  --output-root /path/to/output/fixed16_stage_a_rc
```

### 4. 构建 Stage B

```bash
python dataset_builder_rc_lite/scripts/build_stageb.py \
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

这里只解释一键运行脚本 [scripts/run_all.py](scripts/run_all.py) 顶部的 CONFIG 参数。

### 数据输入参数

- dataset_root
  - 数据总根目录。
  - 当 train 和 val 都在同一个根目录下时使用。
  - 目录通常应满足 dataset_root/train 和 dataset_root/val。

- train_root
  - 单独指定 train 目录。
  - 如果填写了它，就不必依赖 dataset_root/train。

- val_root
  - 单独指定 val 目录。
  - 如果填写了它，就不必依赖 dataset_root/val。

### 输出参数

- output_root
  - 一键脚本的总输出目录。
  - manifest、patch-only、fixed16 Stage A、Stage B 都会输出到这个目录下。

### 数据口径参数

- lane_only
  - 是否只导出 lane_line。
  - True 时不会把 intersection_polygon 写入数据集。
  - False 时同时导出 lane_line 和 intersection_polygon。

- splits
  - 要处理的 split 列表。
  - 默认是 train 和 val。

### 样本路径参数

- image_relpath
  - 可选。
  - 如果填写，表示强制只处理单个 tif 文件。
  - 适用于你想只处理某一个固定影像的场景。

- mask_relpath
  - 可选。
  - 当 image_relpath 被填写时，可同时指定对应 mask 的相对路径。

- image_dir_relpath
  - 默认影像目录相对路径。
  - 默认值是 patch_tif。
  - 一键脚本会在这个目录下扫描多个 tif。

- image_glob
  - 扫描原始 tif 时使用的文件匹配模式。
  - 默认是 *.tif。

- mask_suffix
  - 原始 tif 对应 mask 的文件名后缀。
  - 默认是 _edit_poly.tif。
  - 例如 1.tif 会自动匹配 1_edit_poly.tif。

- lane_relpath
  - 每个样本目录中 Lane.geojson 的相对路径。

- intersection_relpath
  - 每个样本目录中 Intersection.geojson 的相对路径。

### manifest 构建参数

- mask_threshold
  - review mask 二值化阈值。

- tile_size_px
  - manifest 阶段 patch 的尺寸。

- overlap_px
  - 相邻 patch 之间的重叠像素。

- keep_margin_px
  - keep_box 相对 crop_box 向内收缩的边距。

- review_crop_pad_px
  - 如果按 review bbox 缩小搜索区域，会在 bbox 基础上额外外扩的像素。

- tile_min_mask_ratio
  - patch 内 mask 占比最低阈值。

- tile_min_mask_pixels
  - patch 内 mask 像素数最低阈值。

- tile_max_per_sample
  - 每个样本最多保留多少个 patch。
  - 0 表示不限制。

- search_within_review_bbox
  - 是否先根据 review mask 的最小外接框缩小滑窗搜索区域。

- fallback_to_all_if_empty
  - 如果按 mask 条件筛选后一个 patch 都没有，是否退回全部候选窗口。

- max_samples_per_split
  - 每个 split 最多处理多少个样本。
  - 0 表示不限制。

### patch-only 参数

- band_indices
  - 读取 GeoTIFF 时使用的波段顺序。
  - 默认 1, 2, 3。

- patch_resample_step_px
  - patch-only 阶段线段重采样步长。

- patch_boundary_tol_px
  - patch-only 阶段边界 cut 判定容差。

- max_families_per_split
  - 每个 split 最多处理多少个 family。
  - 0 表示不限制。

- empty_patch_drop_ratio
  - patch-only 阶段空 patch 丢弃比例。

- empty_patch_seed
  - patch-only 阶段空 patch 随机采样种子。

- use_patch_system_prompt
  - 是否在 patch-only 阶段启用内置 system prompt。

### fixed16 Stage A 参数

- fixed16_grid_size
  - patch 每边切分成多少格。
  - 默认 4，对应 16 个 box。

- fixed16_target_empty_ratio
  - fixed16 阶段最终允许保留的空 box 占比上限。

- fixed16_seed
  - fixed16 空 box 采样种子。

- fixed16_resample_step_px
  - fixed16 阶段 box 内目标线重采样步长。

- fixed16_boundary_tol_px
  - fixed16 阶段 box 边界 cut 判定容差。

- fixed16_use_system_prompt_from_source
  - 是否继承 patch-only 阶段的 system prompt。

- fixed16_image_root_mode
  - fixed16 输出目录如何暴露 images。
  - 可选值：symlink、copy、none。

### Stage B 参数

- stageb_grid_size
  - Stage B 重建子 patch 邻接关系时使用的网格大小。

- stageb_boundary_tol_px
  - Stage B 判断邻接 trace 是否触边的容差。

- stageb_trace_points_per_hint
  - 每条 incoming trace 最多保留的点数。

- stageb_use_system_prompt_from_source
  - 是否继承 fixed16 Stage A 的 system prompt。

- stageb_image_root_mode
  - Stage B 输出目录如何暴露 images。
  - 可选值：symlink、copy、none。

### 最常修改的参数

如果你只是正常跑一套数据，通常优先改这些：

1. dataset_root 或 train_root / val_root
2. output_root
3. lane_only
4. band_indices
5. max_samples_per_split
6. max_families_per_split

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