# Frozen DINOv3 ViT-L/16 分割训练

这个目录是一套独立的小工程，用于在冻结 DINOv3 ViT-L/16 骨干权重的前提下，训练一个中等大小的语义分割头。

目标场景：

- 输入图像是 512x512 的 PNG。
- 真值掩码也是 512x512 的 PNG。
- 一共 4 个类别。
- 掩码像素值不是 0/1/2/3，而是 0、80、160、240。
- 只训练分割头，不更新 DINOv3 主干。

## 目录结构

当前目录包含：

- config.py：所有路径、层选择、训练超参、评估超参都放在这里。
- dataset.py：PNG 图像和 PNG 掩码的数据集读取。
- model.py：冻结 DINOv3 ViT-L/16 和轻量分割头。
- train.py：训练与评估入口。

## 模型设计

整体结构如下：

1. 用 dinov3_vitl16 加载预训练 ViT-L/16 骨干。
2. 冻结全部骨干参数，并固定为 eval 模式。
3. 通过 get_intermediate_layers 提取任意指定层的 patch 特征。
4. 将多层特征在 32x32 分辨率下做投影和拼接融合。
5. 经过一个适中大小的卷积分割头，逐步上采样到更高分辨率。
6. 最终上采样回原图大小，输出 4 类 logits。

这个分割头没有做很重的金字塔或大规模 Transformer 解码器，原因是你的需求更偏向：

- 代码精简
- 结构清楚
- 16G 显存可训
- 支持单层或多层融合

## 关于层选择

在 config.py 里通过 selected_layers 控制：

- [23]：只取最后一层，最简单，显存最低。
- [11, 17, 23]：三层融合，通常是比较稳的选择。
- [5, 11, 17, 23]：四层融合，语义层次更丰富，是默认配置。

这里的层号是 0-based，也就是：

- 0 表示第 1 个 Transformer block 输出
- 23 表示第 24 个 Transformer block 输出

ViT-L/16 在 512x512 输入下，patch 大小是 16，所以 patch 网格是 32x32。经过 reshape=True 后，每个被提取的中间层特征 shape 都是：

- B x 1024 x 32 x 32

如果你融合 4 层，那么先拿到 4 个这样的特征图，再进入分割头。

## 分割头结构

分割头分成三步：

1. 每一层先用 1x1 卷积把 1024 通道压到 projector_dim。
2. 把多层特征拼接后，再用 1x1 卷积融合到 decoder_dim。
3. 用两次逐步上采样和深度可分离卷积做细化，再输出分类图。

默认配置下：

- projector_dim = 128
- decoder_dim = 256

这是一个相对均衡的选择：

- 比单层线性头强很多
- 比大解码器轻很多
- 更适合冻结大骨干时训练

## 数据组织方式

默认要求训练集和验证集都按下面的形式组织：

```text
data/
  seg/
    train/
      images/
        000001.png
        000002.png
      masks/
        000001.png
        000002.png
    val/
      images/
        100001.png
        100002.png
      masks/
        100001.png
        100002.png
```

要求：

- 图像和掩码文件名一一对应。
- 图像是 RGB PNG。
- 掩码是单通道 PNG。
- 掩码像素值为 0、80、160、240。

dataset.py 会自动把它们解码成类别索引 0、1、2、3。

## 16G 显存推荐配置

默认 config.py 已经按 16G 单卡做了较稳妥的配置：

- batch_size = 4
- grad_accum_steps = 2
- val_batch_size = 8
- selected_layers = [5, 11, 17, 23]
- use_amp = True

如果你还是显存不足，优先按下面顺序收缩：

1. 把 selected_layers 改成 [11, 17, 23]。
2. 再把 batch_size 改成 2。
3. 再把 val_batch_size 改成 4。
4. 再把 decoder_dim 从 256 改成 192 或 128。

## 训练参数建议

当前默认值：

- max_epochs = 30
- lr = 3e-4
- weight_decay = 1e-4
- warmup_epochs = 1
- grad_clip_norm = 1.0

这是针对“冻结大骨干，只训练中等大小解码头”的常见稳定区间。

如果训练初期损失震荡，可以把学习率降到 1e-4。

## 评估指标

train.py 会在验证阶段输出：

- pixel_acc
- mIoU
- mDice
- 每个类别的 IoU
- 每个类别的 Dice

并将所有 epoch 指标写入：

- outputs/frozen_dinov3_vitl16_seg/metrics.csv

同时保存：

- checkpoints/best.pt
- checkpoints/last.pt

注意：checkpoint 只保存分割头和优化器状态，不保存冻结骨干，避免文件过大。

## 使用方法

先修改 config.py，至少改这些路径：

- train_images_dir
- train_masks_dir
- val_images_dir
- val_masks_dir
- work_dir
- backbone_weights

如果使用官方公开权重，backbone_weights 保持为 "LVD1689M" 即可。

### 训练

在仓库根目录执行：

```bash
python -m frozen_dinov3_seg.train
```

或者：

```bash
python frozen_dinov3_seg/train.py
```

### 纯评估

先在 config.py 中修改：

- run_mode = "eval"
- eval_checkpoint = "你的 best.pt 路径"

然后执行同样的命令：

```bash
python -m frozen_dinov3_seg.train
```

## 代码里的几个关键点

1. 骨干是冻结的

model.py 里对 backbone 的参数全部设置了 requires_grad=False，并且强制 backbone 始终保持 eval 模式。

1. 特征提取是任意层可配的

通过 get_intermediate_layers 提取指定 block 的 patch 特征，并直接返回 B x C x H x W 形式。

1. 训练时不保存整套 backbone checkpoint

因为 backbone 不训练，重复保存它没有意义，而且会让 checkpoint 很大。

1. 掩码值会自动从 0/80/160/240 映射成 0/1/2/3

你不用在外部额外预处理标签。

## 建议的第一轮实验

如果你要先快速跑一个稳妥基线，建议直接用下面这组配置：

- selected_layers = [11, 17, 23]
- batch_size = 4
- grad_accum_steps = 2
- lr = 3e-4
- max_epochs = 30

如果你更重视精度，再切到 4 层融合：

- selected_layers = [5, 11, 17, 23]
