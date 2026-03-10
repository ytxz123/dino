# 基于 DINOv3 浅层特征的 TIF 语义分割

本文档对应 [dinov3/segmentation/model.py](dinov3/segmentation/model.py)、[dinov3/segmentation/train_tif.py](dinov3/segmentation/train_tif.py)、[dinov3/segmentation/eval_tif.py](dinov3/segmentation/eval_tif.py)、[dinov3/segmentation/predict_tif.py](dinov3/segmentation/predict_tif.py)、[dinov3/segmentation/metrics.py](dinov3/segmentation/metrics.py) 和 [dinov3/segmentation/datasets.py](dinov3/segmentation/datasets.py) 中实现的自定义语义分割方案。

这套方案面向以下需求：

- 输入是 tif 或 tiff 图像。
- 真值标注固定为 png。
- 可以从 DINOv3 backbone 中提取任意单层浅层特征，也可以组合多层浅层特征。
- 再接一个轻量分割解码器输出像素级类别图。
- 支持冻结 backbone，只训练分割头。
- 支持不定尺寸 tif 样本的直接批量训练与评估。
- 支持在训练后通过主流分割指标独立评估 checkpoint。
- 支持使用本地 DINOv3 权重或默认预训练权重。

## 1. 方法概述

这个分割网络不是直接复用仓库中现有的评估式 segmentation adapter，而是单独设计了一套更适合你的场景的结构：

1. DINOv3 浅层特征提取器。
从 DINOv3 前半段 block 中提取若干中间层特征，默认自动均匀选取浅层 block，也可以手动指定层号。

2. 细节分支。
原始输入图像同时经过一个轻量 CNN，得到 1/2、1/4、1/8 三个尺度的局部细节特征，用来补足 ViT patch 特征在边缘恢复方面的不足。

3. 融合解码器。
浅层 DINOv3 特征先在 1/16 尺度做融合，再逐级上采样，并和细节分支的特征进行 skip fusion，最终恢复到原图分辨率输出类别 logits。

这种设计更适合以下情况：

- 你更关注局部纹理、边缘和浅层空间结构。
- 你的数据是遥感、医学、工业检测等 tif 图像。
- 你的训练数据规模不大，希望先冻结 backbone，稳定训练一个轻量分割头。

## 2. 代码位置

- 数据读取与 tif 预处理: [dinov3/segmentation/datasets.py](dinov3/segmentation/datasets.py)
- 分割网络定义: [dinov3/segmentation/model.py](dinov3/segmentation/model.py)
- 指标与评估公共逻辑: [dinov3/segmentation/metrics.py](dinov3/segmentation/metrics.py)
- 训练脚本: [dinov3/segmentation/train_tif.py](dinov3/segmentation/train_tif.py)
- 独立评估脚本: [dinov3/segmentation/eval_tif.py](dinov3/segmentation/eval_tif.py)
- 推理脚本: [dinov3/segmentation/predict_tif.py](dinov3/segmentation/predict_tif.py)

## 3. 环境准备

建议先安装仓库依赖，再安装新增的 tif 相关依赖。

### 3.1 安装依赖

如果你已经在仓库根目录下：

```bash
pip install -r requirements.txt
pip install -e .
```

如果你使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 3.2 GPU 建议

- 推荐使用 CUDA 环境训练。
- 如果显存较小，优先尝试以下方式降负载：
  - 减小 `--batch-size`
  - 减小 `--image-size`
  - 使用更小的 backbone，例如 `dinov3_vits16`
  - 加上 `--freeze-backbone`
  - 打开 `--amp`

### 3.3 ViT-L/16 distilled SAT-493M 权重放置方式

如果你已经把 ViT-L/16 distilled 300M 的 SAT-493M 权重放到了仓库本地，默认就按下面这个路径读取：

```text
checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
```

训练脚本当前默认等价于：

```bash
--backbone-weights checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
```

如果你之后想切回自动下载，也仍然可以手动改成：

```bash
--backbone-weights SAT493M
```

### 3.4 默认轻量训练配置

为了降低训练成本，训练脚本已经切成一组更保守的默认值：

- backbone: `dinov3_vitl16`
- 权重: `checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`
- 输入统计量: 自动切到 SAT-493M 的 mean/std
- 冻结 backbone: 开启
- 浅层数: `3`
- decoder_dim: `128`
- detail_dims: `32,64,128`
- 输入尺寸: `384 x 384`
- batch size: `1`
- epochs: `20`
- AMP: 开启
- 输出目录默认值: `outputs/shallow_seg_sat_lite`

这组默认值更适合先验证流程是否跑通，尤其适合小数据集、虚拟数据集和单卡显存有限的场景。

### 3.5 从零到一完整流程

如果你第一次跑这套方案，可以直接按下面顺序执行：

1. 准备 Python 环境并安装依赖。
2. 整理数据目录，确保 `tif` 图像和 `png` 标注按相对路径同名对应。
3. 先检查类别编号是否连续，忽略区统一使用 `255`。
4. 先用默认轻量配置起步，必要时再覆盖个别参数。
5. 训练过程中关注 `mIoU`、`dice`、`aAcc`，优先看 `best_metrics.json` 和 `metrics_history.jsonl`。
6. 训练结束后，用 `python -m dinov3.segmentation.eval_tif` 对最佳 checkpoint 在验证集或测试集上独立复评。
7. 如果指标达到预期，再用 `python -m dinov3.segmentation.predict_tif` 对整批 tif 做推理导出。
8. 如果效果不理想，再回头调整 `--layer-indices`、`--num-shallow-layers`、`--decoder-dim`、`--image-size` 和损失权重。

## 4. 数据集组织方式

训练和验证需要分别提供图像目录与 mask 目录，并通过相对路径和同名文件 stem 一一对应。

推荐目录结构如下：

```text
your_dataset/
  images/
    train/
      sample_001.tif
      sample_002.tif
    val/
      sample_101.tif
  masks/
    train/
      sample_001.png
      sample_002.png
    val/
      sample_101.png
```

也支持带子目录的结构，例如：

```text
your_dataset/
  images/
    train/
      city_a/
        tile_001.tif
      city_b/
        tile_002.tif
  masks/
    train/
      city_a/
        tile_001.png
      city_b/
        tile_002.png
```

要求如下：

- 图像和 mask 的相对路径以及文件 stem 必须一致。
- 例如 `images/train/a/sample_001.tif` 可以对应 `masks/train/a/sample_001.png`。
- mask 必须是单通道类别图。
- mask 中每个像素值表示类别编号。
- 默认忽略标签为 `255`，可以通过 `--ignore-index` 调整。
- 如果 mask 中存在负值，读取后会被自动映射到 `ignore_index`。

当前训练/验证标注只支持 `.png`。

如果你希望直接使用训练脚本默认路径，请把数据整理成下面这种结构，默认根目录就是 [dinov3/segmentation/dataset](dinov3/segmentation/dataset)：

```text
dinov3/segmentation/dataset/
  images/
    train/
    val/
  masks/
    train/
    val/
```

这样训练时只需要显式提供 `--num-classes`。

### 4.1 不定尺寸样本如何处理

这套实现支持两种训练/评估方式：

1. 固定尺寸模式。
  - 通过 `--image-size 512 512` 之类的参数，先把输入图像和 mask 统一 resize。
  - 这更稳定，也更容易控制显存。

2. 保持原始尺寸模式。
  - 通过 `--keep-size` 关闭 resize。
  - dataloader 会在同一个 batch 内自动把样本 pad 到最大高宽。
  - mask 的 pad 区域会自动填成 `ignore_index`，不会参与 loss 和指标计算。

如果你的 tif 图像尺寸差异很大，建议这样起步：

- 先固定尺寸训练，快速验证方案可行性。
- 再改成 `--keep-size` 做更贴近真实部署的数据流。

## 5. tif 输入约束与波段说明

当前实现最终会将图像整理成 3 通道送入 DINOv3，因此适用于以下情况：

1. 单通道 tif。
会自动复制成 3 通道。

2. 双通道 tif。
会自动补成 3 通道。

3. 三通道 tif。
直接使用。

4. 多通道 tif。
默认截取前 3 个通道，或者你可以通过 `--bands` 指定波段，例如：

```bash
--bands 3,2,1
```

表示用第 4、3、2 个波段组装 3 通道输入。

注意：

- 当前版本不是“原生多光谱 backbone”，而是“从多波段中选 3 个通道输入 DINOv3”。
- 如果你后续希望保留 4 波段、8 波段或更多波段，可以继续扩展一个 spectral adapter。

## 6. 归一化方式

支持两种输入归一化策略。

### 6.1 百分位归一化

默认值：

```bash
--normalize-mode percentile --percentile-range 2,98
```

含义：

- 每个通道独立统计 2% 和 98% 分位数。
- 把数值裁剪到这个区间内。
- 再线性映射到 `[0, 1]`。

这种方式适合遥感、医学、工业图像中存在极亮或极暗异常值的场景。

### 6.2 按 dtype 满量程归一化

```bash
--normalize-mode dtype
```

含义：

- 对整型 tif，根据原始 dtype 的数值范围进行归一化。
- 例如 `uint16` 会按 `0 ~ 65535` 映射到 `[0, 1]`。

如果你的数据分布比较稳定，且像元值本身有明确物理含义，可以优先试这个模式。

## 7. 训练脚本使用说明

训练入口是：

```bash
python -m dinov3.segmentation.train_tif
```

### 7.1 最小训练示例

```bash
python -m dinov3.segmentation.train_tif \
  --dataset-root your_dataset \
  --output-dir outputs/shallow_seg_sat_lite \
  --num-classes 6
```

### 7.2 按默认配置直接试跑

如果你把数据整理到了默认目录 [dinov3/segmentation/dataset](dinov3/segmentation/dataset)，可以直接运行：

```bash
python -m dinov3.segmentation.train_tif \
  --num-classes 2
```

上面这条命令默认会使用：

```bash
python -m dinov3.segmentation.train_tif \
  --dataset-root dinov3/segmentation/dataset \
  --backbone-name dinov3_vitl16 \
  --backbone-weights checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth \
  --freeze-backbone \
  --input-stats auto \
  --image-size 384 384 \
  --batch-size 1 \
  --epochs 20 \
  --amp \
  --output-dir outputs/shallow_seg_sat_lite \
  --num-classes 2
```

如果你之后想切回自动下载，可以把上面的本地路径替换成：

```bash
--backbone-weights SAT493M
```

### 7.3 常用训练参数说明

#### 数据相关

- `--dataset-root`
默认数据集根目录。脚本会自动查找 `images/train`、`masks/train`、`images/val`、`masks/val`。

- `--train-image-dir`
训练图像目录。如果你不想使用默认目录结构，可以单独覆盖。

- `--train-mask-dir`
训练 mask 目录。

- `--val-image-dir`
验证图像目录。

- `--val-mask-dir`
验证 mask 目录。

- `--bands`
指定 tif 中使用的波段，例如 `0,1,2` 或 `3,2,1`。

- `--image-size`
训练前统一 resize 的尺寸，可写为 `512` 或 `512 512`。

- `--keep-size`
不对输入做固定 resize，保持原始 tif / png 尺寸，并在 batch 内自动 pad。

- `--normalize-mode`
输入归一化方式，取值为 `percentile` 或 `dtype`。

- `--input-stats`
输入标准化统计量，支持 `auto`、`imagenet`、`sat493m`。默认 `auto` 会在使用 SAT-493M 权重时自动切到卫星域统计量。

- `--percentile-range`
百分位归一化区间，例如 `2,98`。

- `--ignore-index`
忽略标签编号，默认 `255`。

#### 模型相关

- `--backbone-name`
可选 backbone 名称来自 [dinov3/hub/backbones.py](dinov3/hub/backbones.py)，常用包括：
  - `dinov3_vits16`
  - `dinov3_vitb16`
  - `dinov3_vitl16`
  - `dinov3_vit7b16`

- `--backbone-weights`
指定本地权重路径、URL，或者直接填写 `SAT493M` / `LVD1689M` 别名。

- `--disable-backbone-pretrained`
不加载默认预训练权重，从随机初始化 backbone 开始。

- `--freeze-backbone`
冻结 DINOv3 backbone，只训练细节分支和分割头。数据量小的时候建议优先开启。

- `--layer-indices`
手动指定浅层 block，例如：

```bash
--layer-indices 1,2,4,5
```

- `--num-shallow-layers`
如果没有手动指定 `--layer-indices`，则自动在前半段 block 中均匀选取若干层，默认是 `3`。

- `--decoder-dim`
浅层特征融合后的通道数，默认 `128`。

- `--detail-dims`
细节分支三个尺度的通道数，默认 `32,64,128`。

- `--dropout`
分类头前的 dropout 比例，默认 `0.1`。

#### 优化相关

- `--num-classes`
类别数，必须与你的 mask 标签数一致。

- `--batch-size`
训练 batch size。

- `--epochs`
训练轮数。

- `--lr`
学习率。

- `--weight-decay`
权重衰减。

- `--ce-weight`
交叉熵损失权重。

- `--dice-weight`
Dice 损失权重。

- `--grad-clip`
梯度裁剪阈值。

- `--amp`
开启自动混合精度，仅在 CUDA 上生效。

- `--resume`
从已有 checkpoint 恢复训练。

- `--save-every`
每隔多少个 epoch 保存一次常规 checkpoint。

### 7.3 训练输出内容

训练完成后，输出目录通常会包含：

- `train_config.json`
本次训练的参数配置快照。

- `checkpoint_epoch_001.pth`
- `checkpoint_epoch_002.pth`
- `...`
按 epoch 保存的训练断点。

- `best_model.pth`
按验证集 `mIoU` 最优保存的模型。

- `last_metrics.json`
最近一个 epoch 的训练/验证指标快照。

- `best_metrics.json`
当前最优 `mIoU` 对应的指标快照。

- `metrics_history.jsonl`
每个 epoch 一行 JSON，便于后处理画曲线或做实验对比。

### 7.4 几个推荐训练起点

#### 情况 A：数据量较小，优先稳定

```bash
python -m dinov3.segmentation.train_tif \
  --train-image-dir your_dataset/images/train \
  --train-mask-dir your_dataset/masks/train \
  --val-image-dir your_dataset/images/val \
  --val-mask-dir your_dataset/masks/val \
  --output-dir outputs/exp_small \
  --backbone-name dinov3_vits16 \
  --num-classes 6 \
  --freeze-backbone \
  --batch-size 8 \
  --image-size 384 384 \
  --amp
```

#### 情况 B：数据量中等，希望更强语义表达

```bash
python -m dinov3.segmentation.train_tif \
  --train-image-dir your_dataset/images/train \
  --train-mask-dir your_dataset/masks/train \
  --val-image-dir your_dataset/images/val \
  --val-mask-dir your_dataset/masks/val \
  --output-dir outputs/exp_base \
  --backbone-name dinov3_vitb16 \
  --num-classes 6 \
  --batch-size 4 \
  --image-size 512 512 \
  --lr 3e-4 \
  --ce-weight 1.0 \
  --dice-weight 0.5 \
  --amp
```

#### 情况 C：明确知道哪些浅层 block 更有用

```bash
python -m dinov3.segmentation.train_tif \
  --train-image-dir your_dataset/images/train \
  --train-mask-dir your_dataset/masks/train \
  --val-image-dir your_dataset/images/val \
  --val-mask-dir your_dataset/masks/val \
  --output-dir outputs/exp_layers \
  --backbone-name dinov3_vitb16 \
  --num-classes 6 \
  --layer-indices 1,2,4,5 \
  --freeze-backbone \
  --amp
```

#### 情况 D：保持原始不定尺寸训练

```bash
python -m dinov3.segmentation.train_tif \
  --train-image-dir your_dataset/images/train \
  --train-mask-dir your_dataset/masks/train \
  --val-image-dir your_dataset/images/val \
  --val-mask-dir your_dataset/masks/val \
  --output-dir outputs/exp_keep_size \
  --backbone-name dinov3_vitb16 \
  --num-classes 6 \
  --freeze-backbone \
  --keep-size \
  --batch-size 2 \
  --amp
```

## 8. 独立评估脚本使用说明

训练过程中会按 epoch 自动评估一次验证集，但更推荐在训练完成后再单独跑一次评估脚本，原因有两个：

1. 你可以把同一个 checkpoint 复用于验证集、测试集或其他 tif 数据目录。
2. 你可以在不重新训练的前提下，对评估尺寸、波段和归一化方式做独立实验。

评估入口是：

```bash
python -m dinov3.segmentation.eval_tif
```

### 8.1 最小评估示例

```bash
python -m dinov3.segmentation.eval_tif \
  --checkpoint outputs/shallow_seg/best_model.pth \
  --image-dir your_dataset/images/val \
  --mask-dir your_dataset/masks/val \
  --output-dir outputs/shallow_seg/eval_val \
  --bands 0,1,2 \
  --amp
```

### 8.2 保持原始尺寸评估

```bash
python -m dinov3.segmentation.eval_tif \
  --checkpoint outputs/shallow_seg/best_model.pth \
  --image-dir your_dataset/images/val \
  --mask-dir your_dataset/masks/val \
  --output-dir outputs/shallow_seg/eval_keep_size \
  --keep-size \
  --batch-size 2 \
  --amp
```

### 8.3 评估输出说明

评估完成后，输出目录至少会包含：

- `eval_metrics.json`

其中默认包含以下主流指标：

- `mIoU`
- `acc`
- `aAcc`
- `dice`
- `precision`
- `recall`
- `fscore`

这意味着训练结束后，你可以直接用同一套 checkpoint 对任意 tif + png 评测集做可复现实验。

## 9. 推理脚本使用说明

推理入口是：

```bash
python -m dinov3.segmentation.predict_tif
```

### 9.1 对整个目录推理

```bash
python -m dinov3.segmentation.predict_tif \
  --checkpoint outputs/shallow_seg/best_model.pth \
  --input-path your_dataset/images/val \
  --output-dir outputs/shallow_seg/predictions \
  --bands 0,1,2 \
  --amp
```

### 9.2 对单张 tif 推理

```bash
python -m dinov3.segmentation.predict_tif \
  --checkpoint outputs/shallow_seg/best_model.pth \
  --input-path your_dataset/images/val/sample_101.tif \
  --output-dir outputs/single_prediction \
  --bands 0,1,2
```

### 9.3 推理输出说明

- 输出仍然保存为 tif 文件。
- 输出文件的相对目录结构会尽量保持与输入目录一致。
- 输出数据类型为 `uint16`。
- 每个像素值表示预测类别编号。

例如：

```text
输入目录: your_dataset/images/val/city_a/tile_001.tif
输出目录: outputs/shallow_seg/predictions/city_a/tile_001.tif
```

### 9.4 推理时的参数覆盖逻辑

推理脚本会优先从 checkpoint 中读取以下信息：

- backbone 名称
- 浅层层号配置
- 解码器通道配置
- 类别数
- 是否冻结 backbone

如果你在命令行中重新传入对应参数，则会覆盖 checkpoint 中的设置。

## 10. checkpoint 结构说明

`best_model.pth` 和普通 checkpoint 内部主要包含：

- `model`
模型权重。

- `optimizer`
优化器状态。

- `scaler`
混合精度状态，如果训练时启用了 AMP。

- `epoch`
保存时所在 epoch。

- `best_miou`
当前最优 mIoU。

- `model_config`
模型构造参数，供恢复训练和推理时重建网络结构。

其中还会记录 `resolved_layer_indices`，表示本次训练真正使用的浅层 block 号，方便后续复现实验。

## 11. 类别标签建议

为了避免训练时标签歧义，建议遵循下面的规范：

- 有效类别从 `0` 开始连续编号。
- 如果有背景类，建议把背景作为 `0`。
- 如果有无效区域，统一使用 `255`，并保持 `--ignore-index 255`。

例如 6 类分割：

```text
0: background
1: class_1
2: class_2
3: class_3
4: class_4
5: class_5
255: ignore
```

## 12. 常见问题

### 12.1 为什么输入 tif 最后只用了 3 个通道？

因为当前方案复用了标准 DINOv3 预训练 backbone，它的输入接口是 3 通道。对于多波段 tif，本实现支持先选 3 个波段输入。如果你希望保留全部波段，需要额外增加光谱适配层。

### 12.2 为什么要额外加一个 CNN 细节分支？

因为 ViT 的 patch token 对语义抽象更强，但边界恢复通常不如卷积特征。细节分支提供了更高分辨率的局部纹理和轮廓信息，能提高分割边缘质量。

### 12.3 为什么默认建议先冻结 backbone？

当训练样本较少时，直接微调整个 DINOv3 容易不稳定，也更吃显存。冻结 backbone 可以先快速验证浅层特征是否对你的任务有效。

### 12.4 输入尺寸必须是 16 的倍数吗？

不必须。模型内部会自动把图像填充到 patch size 的倍数，再在输出时裁回原图大小。

### 12.5 不定尺寸样本能否直接 batch 训练？

可以。

- 如果你传 `--keep-size`，dataloader 会把同一 batch 内的样本 pad 到最大高宽。
- 图像 pad 区域填 0。
- mask pad 区域填 `ignore_index`，因此不会影响 loss 和指标。

### 12.6 推理输出为什么是 uint16？

因为很多分割任务的类别编号不止 255，使用 `uint16` 更稳妥，也兼容 tif 保存类别图的常见方式。

## 13. 调参建议

如果你的结果边界不够好，可以尝试：

- 增大 `--dice-weight`
- 提高 `--image-size`
- 解冻 backbone 做小学习率微调
- 手动调整 `--layer-indices`，更偏前层一些

如果你的结果类别混淆严重，可以尝试：

- 增大 `--decoder-dim`
- 使用更大的 backbone
- 增加训练轮数
- 检查 mask 类别编号是否连续且正确

如果训练不稳定或显存不足，可以尝试：

- 打开 `--freeze-backbone`
- 减小 `--batch-size`
- 使用 `dinov3_vits16`
- 打开 `--amp`

## 14. 当前实现的边界

当前版本已经可以直接完成基础训练与推理，但仍有几个明确边界：

- 训练脚本目前是单机单进程版本，不是分布式训练版本。
- 默认只支持单输出语义分割，不包含实例分割或全景分割。
- 多光谱 tif 目前是“选 3 个波段输入”，不是全波段原生编码。
- 数据增强目前只包含简单翻转，没有更复杂的裁剪、旋转、颜色扰动等策略。

## 15. 你接下来最可能会改的地方

如果你要把它真正落到自己的数据集上，通常会优先修改这些点：

1. [dinov3/segmentation/train_tif.py](dinov3/segmentation/train_tif.py)
把默认训练参数改成你自己的类别数、输入尺寸和学习率。

2. [dinov3/segmentation/datasets.py](dinov3/segmentation/datasets.py)
如果你的 tif 或 mask 有特殊格式，可以在这里改读取逻辑。

3. [dinov3/segmentation/model.py](dinov3/segmentation/model.py)
如果你想增强分割头、增加注意力模块或接入多光谱适配器，可以从这里扩展。

## 16. 快速命令汇总

### 训练

```bash
python -m dinov3.segmentation.train_tif \
  --train-image-dir your_dataset/images/train \
  --train-mask-dir your_dataset/masks/train \
  --val-image-dir your_dataset/images/val \
  --val-mask-dir your_dataset/masks/val \
  --output-dir outputs/shallow_seg \
  --backbone-name dinov3_vitb16 \
  --num-classes 6 \
  --freeze-backbone \
  --bands 0,1,2 \
  --image-size 512 512 \
  --amp
```

### 恢复训练

```bash
python -m dinov3.segmentation.train_tif \
  --train-image-dir your_dataset/images/train \
  --train-mask-dir your_dataset/masks/train \
  --val-image-dir your_dataset/images/val \
  --val-mask-dir your_dataset/masks/val \
  --output-dir outputs/shallow_seg_resume \
  --backbone-name dinov3_vitb16 \
  --num-classes 6 \
  --resume outputs/shallow_seg/checkpoint_epoch_010.pth \
  --amp
```

### 独立评估

```bash
python -m dinov3.segmentation.eval_tif \
  --checkpoint outputs/shallow_seg/best_model.pth \
  --image-dir your_dataset/images/val \
  --mask-dir your_dataset/masks/val \
  --output-dir outputs/shallow_seg/eval_val \
  --bands 0,1,2 \
  --amp
```

### 不定尺寸训练

```bash
python -m dinov3.segmentation.train_tif \
  --train-image-dir your_dataset/images/train \
  --train-mask-dir your_dataset/masks/train \
  --val-image-dir your_dataset/images/val \
  --val-mask-dir your_dataset/masks/val \
  --output-dir outputs/exp_keep_size \
  --backbone-name dinov3_vitb16 \
  --num-classes 6 \
  --freeze-backbone \
  --keep-size \
  --batch-size 2 \
  --amp
```

### 推理

```bash
python -m dinov3.segmentation.predict_tif \
  --checkpoint outputs/shallow_seg/best_model.pth \
  --input-path your_dataset/images/val \
  --output-dir outputs/shallow_seg/predictions \
  --bands 0,1,2 \
  --amp
```