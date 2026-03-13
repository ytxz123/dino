"""分割模型评估入口。

评估阶段会重新构建冻结骨干，并加载保存好的分割头权重，
最终输出 loss、mIoU、像素准确率和逐类 IoU。
"""

import torch
from torch import nn
from torch.utils.data import DataLoader

from dinov3_seg.config import get_config
from dinov3_seg.dataset import SegmentationDataset
from dinov3_seg.metrics import summarize_metrics, update_confusion_matrix
from dinov3_seg.model import FrozenDinoV3Segmenter
from dinov3_seg.train import autocast_context


def main():
    """执行一次完整的验证集评估。"""

    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # 先重建完整模型，再只恢复训练时保存的分割头参数。
    model = FrozenDinoV3Segmenter(cfg).to(device)
    checkpoint = torch.load(cfg.paths.eval_checkpoint, map_location="cpu")
    model.head.load_state_dict(checkpoint["head"], strict=True)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    confusion_matrix = torch.zeros(cfg.data.num_classes, cfg.data.num_classes, dtype=torch.int64, device=device)
    loss_sum = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            # non_blocking=True 配合 pin_memory 可减少主机到显存拷贝等待。
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with autocast_context(device, cfg.train.amp_dtype):
                logits = model(images)
                loss = criterion(logits, masks)
            loss_sum += loss.item() * images.size(0)
            update_confusion_matrix(confusion_matrix, logits, masks)

            # 统一通过混淆矩阵统计指标，避免训练/评估两套实现不一致。
    metrics = summarize_metrics(confusion_matrix)
    metrics["loss"] = loss_sum / len(val_loader.dataset)

    print(f"loss={metrics['loss']:.4f}")
    print(f"mIoU={metrics['mIoU']:.4f}")
    print(f"pixel_acc={metrics['pixel_acc']:.4f}")
    print(f"class_iou={metrics['class_iou']}")


if __name__ == "__main__":
    main()
