# 基于 DINOv3 浅层特征的固定尺寸 PNG 语义分割

本文档描述 [dinov3/segmentation/datasets.py](dinov3/segmentation/datasets.py)、[dinov3/segmentation/model.py](dinov3/segmentation/model.py)、[dinov3/segmentation/train_tif.py](dinov3/segmentation/train_tif.py)、[dinov3/segmentation/eval_tif.py](dinov3/segmentation/eval_tif.py) 和 [dinov3/segmentation/predict_tif.py](dinov3/segmentation/predict_tif.py) 当前采用的精简分割流程。

## 当前约束

- 输入图像固定为 512x512 RGB PNG。
- 标注固定为 512x512 单通道 PNG。
- 推理输出固定保存为 PNG。
- 不再支持 tif、多波段、动态尺寸、keep-size、resize 后恢复原图。

## 数据目录

推荐目录结构如下：

```text
dataset/
  images/
    train/
    val/
  masks/
    train/
    val/
```

要求如下：

- 图像与标注都必须是 PNG。
- 图像与标注必须同名同相对路径。
- 每张图和每张标注都必须是 512x512。
- 标注像素值表示类别编号。
- 忽略标签默认是 255。

## 模型流程

当前模型固定走下面这条路径：

1. 读取 512x512 PNG 并做标准化。
2. 用 DINOv3 提取浅层特征。
3. 用细节分支补充边缘和纹理信息。
4. 用逐级上采样解码器直接输出 512x512 logits。

因为输入尺寸已经固定且能被 patch size 整除，所以模型里不再做 padding、裁回原尺寸或额外插值恢复。

## 训练

最小训练命令：

```bash
python -m dinov3.segmentation.train_tif \
  --dataset-root dinov3/segmentation/dataset \
  --num-classes 2
```

常用参数：

- `--backbone-name`
- `--backbone-weights`
- `--freeze-backbone`
- `--num-classes`
- `--layer-indices`
- `--num-shallow-layers`
- `--decoder-dim`
- `--detail-dims`
- `--batch-size`
- `--epochs`
- `--lr`
- `--amp`

## 评估

```bash
python -m dinov3.segmentation.eval_tif \
  --checkpoint outputs/shallow_seg_sat_lite/best_model.pth \
  --image-dir your_dataset/images/val \
  --mask-dir your_dataset/masks/val \
  --output-dir outputs/shallow_seg_sat_lite_eval
```

## 推理

```bash
python -m dinov3.segmentation.predict_tif \
  --checkpoint outputs/shallow_seg_sat_lite/best_model.pth \
  --input-path your_dataset/images/val \
  --output-dir outputs/shallow_seg_predictions
```

输出会保持输入目录结构，并统一保存为 PNG。

## 备注

- 脚本文件名仍保留 `train_tif.py`、`eval_tif.py`、`predict_tif.py`，但内部逻辑已经切换为 PNG 固定尺寸流程。
- `datasets.py` 中保留了旧类名别名，只是为了兼容已有 import，实际行为已经是 PNG-only。