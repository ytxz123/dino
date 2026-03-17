# DINOv3 ViT-L/16 三类语义分割

本目录提供一个独立、极简的三类语义分割工程，满足以下约束：

- 冻结 DINOv3 ViT-L/16 全部权重。
- 默认读取本地 SAT493M 预训练权重。
- 仅提取骨干最后一层特征。
- 面向 512×512 PNG 输入与三类标注掩码。
- 单卡 16G 显存可训练。

## 1. 模型原理

整体结构分成两段：

1. DINOv3 ViT-L/16 作为冻结骨干，只负责提取最后一层 patch 特征。
2. 线性分割头对最后一层特征做 1×1 卷积分类，并将输出上采样回 512×512。

在 512×512 输入下，ViT-L/16 的 patch 网格大小为 32×32，因此训练时真正参与反向传播的只有一个很轻的线性头，显存压力主要来自骨干前向。这里默认使用：

- BF16 自动混合精度
- train batch size = 4
- gradient accumulation = 4
- 有效 batch size = 16

这套默认参数是面向 16G 显存给出的保守配置，若显存还有余量，可以优先提高 train batch size，再相应降低 accumulation_steps。

## 2. 数据组织

默认按同名文件配对图像与标注：

```text
data/
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
    masks/
      100001.png
```

约定：

- 输入图像为 RGB PNG。
- 标注图为单通道 PNG。
- 类别像素值分别为 0、50、100，对应类别 0、1、2。
- 代码内部会直接将标注除以 50 还原为 0/1/2。

## 3. 文件结构

```text
three_class_segmentation/
  __init__.py
  config.py
  config.yaml
  dataset.py
  engine.py
  model.py
  run.py
  README.md
```

各文件职责：

- config.py：配置 dataclass 与配置加载/保存。
- config.yaml：所有可调参数统一入口。
- dataset.py：512×512 图像与掩码读取、归一化、轻量增强。
- model.py：冻结 ViT-L/16 与线性分割头封装。
- engine.py：训练、评估、checkpoint、指标统计。
- run.py：命令行入口。

## 4. 配置说明

所有参数都在 config.yaml 中维护，最关键的是以下几组：

### 4.1 dataset

- train_images：训练图像目录。
- train_masks：训练标注目录。
- val_images：验证图像目录。
- val_masks：验证标注目录。
- image_size：输入和标注统一缩放尺寸，默认 512。
- label_divisor：标注像素值还原因子，默认 50。

### 4.2 backbone

- weights_path：本地骨干权重路径，默认指向 SAT493M 版 ViT-L/16 文件名。
- out_layers：默认 LAST，对应只取最后一层特征。
- autocast_dtype：默认 bfloat16，适合 16G 显存场景。

### 4.3 head

- num_classes：类别数，默认 3。
- dropout：线性头 dropout。
- use_batchnorm：默认 false，单卡轻量训练更直接。

### 4.4 train

- epochs：训练轮数。
- batch_size：单卡微批大小。
- accumulation_steps：梯度累积步数。
- hflip_prob：训练时水平翻转概率。
- ce_weight：交叉熵损失权重。
- dice_weight：Dice 损失权重。
- resume_from：恢复训练 checkpoint 路径。

### 4.5 eval

- checkpoint_path：评估时加载的分割头 checkpoint。

### 4.6 runtime

- output_dir：输出目录。
- device：默认 cuda。
- seed：随机种子。

## 5. 训练步骤

### 5.1 准备本地权重

将 ViT-L/16 SAT493M 权重放到本地，例如：

```text
checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
```

然后在 config.yaml 中把 backbone.weights_path 改成你的实际本地路径。

### 5.2 启动训练

在仓库根目录执行：

```bash
python -m three_class_segmentation.run --config three_class_segmentation/config.yaml --mode train
```

训练输出默认写入 output_dir，包括：

- resolved_config.yaml：实际生效配置。
- best.pth：最佳 mIoU 对应的分割头权重。
- last.pth：最后一次保存的分割头权重。
- best_metrics.json：最佳验证指标。
- last_metrics.json：最近一次验证指标。

## 6. 评估步骤

先把 config.yaml 中的 eval.checkpoint_path 改成待评估 checkpoint，然后执行：

```bash
python -m three_class_segmentation.run --config three_class_segmentation/config.yaml --mode eval
```

评估结果会写入：

```text
outputs/three_class_segmentation/eval_metrics.json
```

## 7. 16G 显存建议

默认配置已经按 16G 单卡做了约束：

- 冻结骨干，只训练线性头。
- 只取最后一层特征，不做多层融合。
- 使用 BF16。
- 使用小微批加梯度累积。

如果仍然显存紧张，按下面顺序调整：

1. 先把 train.batch_size 从 4 降到 2。
2. 再把 accumulation_steps 从 4 提到 8。
3. 最后再把 eval.batch_size 从 4 降到 2。

## 8. 默认适用场景

该实现适合你当前描述的任务边界：

- 输入固定 512×512。
- 三类像素级语义分割。
- 训练集约 12000 张，评估集约 5000 张。
- 重点是快速验证冻结 DINOv3 ViT-L/16 的迁移效果。

如果后续需要更强表达能力，可以继续保留本目录的数据与训练框架，只替换 model.py 中的线性头为更复杂的解码器。