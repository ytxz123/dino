# 轻量级分割模块

这个目录提供了一套独立于 dinov3/eval/segmentation 的轻量级语义分割流程。它使用冻结的 DINOv3 backbone 提取特征，只训练一个较小的 decoder，适合快速实验、自定义小中型数据集训练，以及基于 checkpoint 的评估和推理。

## 目录结构

```text
dinov3/segmentation/
├── dataset.py
├── evaluate.py
├── inference.py
├── model.py
├── smoke_test_train.py
├── train.py
└── datasets/
    ├── images/
    │   ├── train/
    │   └── val/
    └── masks/
        ├── train/
        └── val/
```

图片和掩码文件名必须一一对应。也就是说，images/train 或 images/val 下的每一张图片，都必须在对应的 masks 目录下有同名文件。

## 数据格式

- 图片按 RGB PNG 读取。
- 掩码按单通道标签图读取。
- 图片和掩码都会在加载时统一缩放到 --size x --size。
- 训练阶段默认启用随机水平翻转增强。
- 掩码标签默认通过灰度值整除 80 得到类别编号。

默认掩码编码如下：

| 像素值 | 类别编号 |
| --- | --- |
| 0 | 0 |
| 80 | 1 |
| 160 | 2 |
| 240 | 3 |

如果你的掩码编码方式不同，需要先修改 [dinov3/segmentation/dataset.py](dinov3/segmentation/dataset.py) 里的 mask_to_tensor 逻辑，再进行训练或评估。

## 默认参数建议

当前默认参数按以下假设做了调整：

- 单卡 16G 显存。
- 默认 backbone 为 dinov3_vitb16。
- 输入分辨率为 512。
- 训练集约 5 万张 PNG，验证集约 1 万张 PNG。

在这个前提下，默认值侧重于三件事：

- 训练吞吐量不要太低。
- 评估和目录推理尽量批处理，不要逐张跑。
- 日志输出不要过密，否则一个 epoch 会产生大量重复信息。

如果你改用更大的 backbone，比如 vitl 或 vith，建议优先把 batch size 降到当前默认值的一半甚至四分之一。

## 分割模型详细介绍

这个分割模型定义在 [dinov3/segmentation/model.py](dinov3/segmentation/model.py)，整体上可以概括为：冻结的 DINOv3 backbone 加一个轻量级卷积 decoder。

它不是一个完整端到端微调的重型分割网络，而是一个偏工程实用的方案，核心目标是：

- 尽量复用 DINOv3 已有的强表征能力。
- 尽量减少训练参数量和显存压力。
- 在自定义数据集上快速完成语义分割适配。

### 1. Backbone 部分

backbone 使用的是 DINOv3 的视觉主干网络，支持以下几种名称：

- dinov3_vits16
- dinov3_vits16plus
- dinov3_vitb16
- dinov3_vitl16
- dinov3_vitl16plus
- dinov3_vith16plus
- dinov3_vit7b16

默认是 dinov3_vitb16。模型内部通过 FrozenDinov3Backbone 包装 backbone，并且做了两件关键事情：

- 将 backbone 的所有参数都设为 requires_grad = False。
- 无论外部是否调用 train()，backbone 都保持在 eval() 状态。

这意味着训练阶段不会反向更新 DINOv3 主干，优化器只负责更新 decoder。这样做的好处很直接：

- 显存占用更稳定。
- 训练速度更快。
- 对 5 万级别的数据集来说，更容易先拿到一个可用基线。

### 2. 中间层特征抽取

backbone 前向时不是只取最后一层特征，而是通过 get_intermediate_layers 提取指定层的中间特征图。这个行为由 --layers 参数控制，默认值是 12，也就是默认只取一个较深层特征。

如果你传入多层，比如 --layers 8 10 12，模型会：

- 从这几层分别取出特征。
- 将它们 reshape 成二维空间特征图。
- 在 decoder 中对这些特征做统一投影和融合。

这样做的意义是：

- 深层特征语义更强。
- 多层融合时，对边界和局部结构通常更友好。
- 但层数越多，decoder 输入越宽，显存和计算量也会增加。

所以这个实现默认采取较保守的单层深层特征方案，优先保证稳定和轻量。

### 3. Decoder 结构

decoder 的实现类是 LightweightSegmentationDecoder。它不是 Transformer decoder，而是一条纯卷积上采样路径，结构非常直接：

1. 先对每个输入特征图做 1x1 卷积投影，把通道数统一映射到 decoder_dim。
2. 如果用了多层特征，先把不同层的空间尺寸对齐，再在通道维拼接。
3. 经过一个 fuse 卷积块做初步融合。
4. 再经过一个 context 卷积块和 Dropout2d 增加上下文建模与正则化。
5. 接四个连续的上采样块，每个块都先双线性插值放大 2 倍，再做卷积。
6. 最后用 1x1 卷积输出 num_classes 个类别通道。

这里每个卷积块内部使用的是：

- 3x3 Conv
- GroupNorm
- GELU

选择 GroupNorm 而不是 BatchNorm，是因为这个模块默认 batch size 不算大，GroupNorm 对小批量更稳定。

### 4. 输出分辨率恢复

DINOv3 backbone 输出的特征图分辨率低于输入图像。decoder 内部连续做四次 2 倍上采样，相当于把低分辨率特征逐步恢复到接近输入尺度。

在 Dinov3SegmentationModel 的 forward 末尾，还会额外检查输出 logits 的空间尺寸。如果和原图不完全一致，会再做一次双线性插值，把 logits 对齐到输入图像大小。

因此从接口上看，这个分割模型始终输出与输入同尺寸的类别预测图，方便直接计算像素级交叉熵损失和 IoU 指标。

### 5. 为什么 checkpoint 只保存 decoder

训练脚本保存 checkpoint 时，只保存 decoder 权重和模型配置，不保存完整 backbone 参数。这是这个模块一个很重要的设计点。

原因是：

- backbone 本身是冻结的，不参与训练更新。
- backbone 权重通常已经作为独立预训练文件存在。
- 这样可以明显减小 checkpoint 大小，保存和加载都更轻量。

恢复模型时，会根据 checkpoint 里的 backbone_name、layers、num_classes、decoder_dim 等配置重新构建模型，再把 decoder 权重加载回去。

### 6. 这个模型适合什么场景

这个轻量分割模型更适合以下场景：

- 你已经有 DINOv3 权重，希望快速在自己的标注数据上做语义分割。
- 你需要一个训练和部署都相对简单的基线模型。
- 你更关注快速迭代，而不是一开始就追求最复杂的分割头结构。

它不太适合以下场景：

- 需要极致边界质量和多尺度细节恢复。
- 需要对 backbone 做充分联合微调。
- 需要专门为 dense prediction 设计的复杂解码器或金字塔结构。

如果后续你要追求更高上限，可以沿着三个方向继续改：

- 开启多层特征融合，而不是只用单层深层特征。
- 把 decoder 从单一路径改成带 skip connection 或金字塔融合的结构。
- 在显存允许的情况下，对 backbone 的后几层做部分解冻微调。

## 训练

训练入口是 [dinov3/segmentation/train.py](dinov3/segmentation/train.py)。当前更适合大规模数据默认值如下：

- epochs = 30
- batch_size = 8
- num_workers = 8
- lr = 5e-4
- log_interval = 100

示例命令：

```bash
python -m dinov3.segmentation.train \
  --train-images dinov3/segmentation/datasets/images/train \
  --train-masks dinov3/segmentation/datasets/masks/train \
  --val-images dinov3/segmentation/datasets/images/val \
  --val-masks dinov3/segmentation/datasets/masks/val \
  --output-dir output \
  --backbone dinov3_vitb16 \
  --weights dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

说明：

- backbone 始终冻结，训练时只优化 decoder。
- 如果 --weights 不是绝对路径，会自动按 dinov3/checkpointer 相对路径解析。
- 如果不传 --val-images 和 --val-masks，则不会执行验证，只会持续写出 last.pt。
- checkpoint 中保存的是 decoder 权重，以及重建 backbone 和层配置所需的元信息。

## 评估

评估入口是 [dinov3/segmentation/evaluate.py](dinov3/segmentation/evaluate.py)。默认值已经针对 1 万张验证集做过调整：

- batch_size = 16
- num_workers = 8

示例命令：

```bash
python -m dinov3.segmentation.evaluate \
  --images dinov3/segmentation/datasets/images/val \
  --masks dinov3/segmentation/datasets/masks/val \
  --checkpoint output/dinov3_vitb16/<weights_tag>/best.pt \
  --output output/dinov3_vitb16/<weights_tag>/val_metrics.json
```

脚本会输出以下指标：

- val_loss
- miou
- pixel_acc
- iou_per_class
- samples

结果始终会以 JSON 打印到标准输出；如果传入 --output，还会额外写盘。

如果 checkpoint 中记录的 backbone 权重路径在另一台机器上不可用，可以用 --backbone-weights 覆盖。

## 推理

推理入口是 [dinov3/segmentation/inference.py](dinov3/segmentation/inference.py)。现在目录推理默认支持批处理，适合成批处理大量 PNG：

- batch_size = 16
- num_workers = 8

示例命令：

```bash
python -m dinov3.segmentation.inference \
  --input dinov3/segmentation/datasets/images/val \
  --output output/dinov3_vitb16/<weights_tag>/predictions \
  --checkpoint output/dinov3_vitb16/<weights_tag>/best.pt
```

说明：

- 如果 --input 指向单张 PNG，脚本会单张推理。
- 如果 --input 指向目录，脚本会按 batch 读取和推理，提高大规模目录处理速度。

## 输出目录

真实训练的输出结构如下：

```text
output/<backbone_name>/<weights_tag>/
├── best.pt
├── config.json
├── last.pt
├── metrics.jsonl
├── summary.json
└── training.log
```

- config.json：训练时的完整命令行参数。
- metrics.jsonl：每个 epoch 一行 JSON。
- summary.json：最佳验证 mIoU 和运行目录信息。
- best.pt：验证集 mIoU 最优的 checkpoint。
- last.pt：最后一次写出的 checkpoint。

smoke test 输出在单独目录下：

```text
output/<backbone_name>/<weights_tag>/smoke_test/
```

## 快速冒烟测试

冒烟测试入口是 [dinov3/segmentation/smoke_test_train.py](dinov3/segmentation/smoke_test_train.py)。它不会调用真实 DINOv3 backbone，只是用随机特征和合成掩码验证 decoder、优化器和输出流程是否正常。

当前默认值：

- samples = 8
- batch_size = 4
- lr = 5e-4

示例命令：

```bash
python -m dinov3.segmentation.smoke_test_train \
  --weights dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

## 常见问题

- 如果 images/val 和 masks/val 为空，评估虽然能跑，但结果没有实际意义。
- 图片名和掩码名必须完全一致。
- 默认数据集加载器只扫描 *.png。如果你的数据是 JPG 或 TIFF，需要扩展 [dinov3/segmentation/dataset.py](dinov3/segmentation/dataset.py)。
- num_classes 必须和掩码在执行 // 80 之后的类别数一致。
- 即使 backbone 冻结，更大的模型仍然会显著增加显存占用。
- 当前默认参数是按 vitb16 和 16G 显存设定的，不适合直接照搬到更大的 backbone。