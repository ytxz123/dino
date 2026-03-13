"""分割模型评估入口。

评估阶段会重新构建冻结骨干，并加载保存好的分割头权重，
最终输出 loss、mIoU、像素准确率和逐类 IoU。
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dinov3_seg.config import get_config
from dinov3_seg.dataset import SegmentationDataset
from dinov3_seg.metrics import summarize_metrics, update_confusion_matrix
from dinov3_seg.model import FrozenDinoV3Segmenter
from dinov3_seg.train import autocast_context, configure_logging


def main():
    """执行一次完整的验证集评估。"""

    cfg = get_config()
    eval_log_dir = Path(cfg.paths.eval_checkpoint).resolve().parent
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(eval_log_dir, log_name="eval.log")

    # 评估也沿用训练脚本的设备选择规则，优先使用 CUDA。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("evaluation started: device=%s checkpoint=%s", device, cfg.paths.eval_checkpoint)

    # 评估阶段不做数据增强，只保留与训练一致的归一化流程。
    val_dataset = SegmentationDataset(
        image_dir=cfg.paths.val_image_dir,
        mask_dir=cfg.paths.val_mask_dir,
        mean=cfg.data.mean,
        std=cfg.data.std,
        mask_divisor=cfg.data.mask_divisor,
        training=False,
        hflip_prob=0.0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    logger.info("dataset summary: val_samples=%d val_batches=%d", len(val_dataset), len(val_loader))

    # 先重建完整模型，再只恢复训练时保存的分割头参数。
    model = FrozenDinoV3Segmenter(cfg).to(device)

    # checkpoint 中只存了分割头和优化器，不包含冻结的 backbone 完整权重。
    checkpoint = torch.load(cfg.paths.eval_checkpoint, map_location="cpu")
    model.head.load_state_dict(checkpoint["head"], strict=True)
    model.eval()
    logger.info("checkpoint loaded successfully")

    # 评估损失与训练保持一致，这样 loss 才具有可比较性。
    criterion = nn.CrossEntropyLoss()
    confusion_matrix = torch.zeros(cfg.data.num_classes, cfg.data.num_classes, dtype=torch.int64, device=device)
    loss_sum = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            # non_blocking=True 配合 pin_memory 可减少主机到显存拷贝等待。
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # autocast 与训练阶段保持一致，可以减少精度设置不一致带来的偏差。
            with autocast_context(device, cfg.train.amp_dtype):
                logits = model(images)
                loss = criterion(logits, masks)

            # loss_sum 按样本数做加权累计，最后再除以总样本数得到平均损失。
            loss_sum += loss.item() * images.size(0)
            update_confusion_matrix(confusion_matrix, logits, masks)

            # 统一通过混淆矩阵统计指标，避免训练/评估两套实现不一致。
    metrics = summarize_metrics(confusion_matrix)
    metrics["loss"] = loss_sum / len(val_loader.dataset)

    # 评估输出尽量保持简单直接，便于脚本化抓取日志。
    logger.info("evaluation finished: loss=%.4f mIoU=%.4f pixel_acc=%.4f", metrics["loss"], metrics["mIoU"], metrics["pixel_acc"])
    logger.info("class_iou=%s", metrics["class_iou"])


if __name__ == "__main__":
    main()
