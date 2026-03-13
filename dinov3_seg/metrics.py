"""分割评估指标工具。

这里采用混淆矩阵做累计统计，便于在整个 epoch 内逐 batch 汇总结果。
"""

import torch


def update_confusion_matrix(confusion_matrix: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor) -> None:
    """把当前 batch 的预测结果累计到混淆矩阵中。"""

    num_classes = confusion_matrix.shape[0]

    # 对类别维取 argmax，得到每个像素的预测类别编号。
    preds = logits.argmax(dim=1)

    # 过滤非法标签，避免越界类别污染统计结果。
    valid = (targets >= 0) & (targets < num_classes)

    # 将二维的 (target, pred) 对编码成一维索引，再通过 bincount 高效计数。
    bins = num_classes * targets[valid] + preds[valid]
    confusion_matrix += torch.bincount(bins.view(-1), minlength=num_classes * num_classes).view(num_classes, num_classes)


def summarize_metrics(confusion_matrix: torch.Tensor) -> dict[str, float | list[float]]:
    """根据累计混淆矩阵计算常见分割指标。"""

    confusion_matrix = confusion_matrix.float()
    true_positive = confusion_matrix.diag()

    # IoU 分母 = 该类真值像素 + 该类预测像素 - 交集像素。
    denominator = confusion_matrix.sum(1) + confusion_matrix.sum(0) - true_positive
    iou = true_positive / denominator.clamp_min(1.0)
    miou = iou.mean().item()

    # 像素准确率衡量全局像素级命中率，通常比 mIoU 更乐观。
    pixel_acc = (true_positive.sum() / confusion_matrix.sum().clamp_min(1.0)).item()
    return {
        "mIoU": miou,
        "pixel_acc": pixel_acc,
        "class_iou": iou.tolist(),
    }
