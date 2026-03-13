import torch


def update_confusion_matrix(confusion_matrix: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor) -> None:
    num_classes = confusion_matrix.shape[0]
    preds = logits.argmax(dim=1)
    valid = (targets >= 0) & (targets < num_classes)
    bins = num_classes * targets[valid] + preds[valid]
    confusion_matrix += torch.bincount(bins.view(-1), minlength=num_classes * num_classes).view(num_classes, num_classes)


def summarize_metrics(confusion_matrix: torch.Tensor) -> dict[str, float | list[float]]:
    confusion_matrix = confusion_matrix.float()
    true_positive = confusion_matrix.diag()
    denominator = confusion_matrix.sum(1) + confusion_matrix.sum(0) - true_positive
    iou = true_positive / denominator.clamp_min(1.0)
    miou = iou.mean().item()
    pixel_acc = (true_positive.sum() / confusion_matrix.sum().clamp_min(1.0)).item()
    return {
        "mIoU": miou,
        "pixel_acc": pixel_acc,
        "class_iou": iou.tolist(),
    }
