from __future__ import annotations

"""分割训练与评估阶段共用的损失函数和指标计算工具。

这个文件专门负责两类事情：
1. 训练时的损失函数定义。
2. 验证/测试时的混淆矩阵累计与指标统计。

这样做的好处是训练脚本、评估脚本和后续可能新增的测试脚本
都可以复用同一套指标实现，避免不同入口的指标口径不一致。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedSegmentationLoss(nn.Module):
    """组合交叉熵与 Dice 的分割损失。

    设计意图：
    1. 交叉熵负责提供稳定的类别判别监督。
    2. Dice 负责缓解类别不均衡，并增强区域重叠优化。
    3. 两者按权重线性组合，便于针对不同数据集调整侧重点。
    """

    def __init__(self, ce_weight: float, dice_weight: float, ignore_index: int):
        """保存损失权重和忽略标签编号。"""
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算一个 batch 的总损失。

        参数：
        - logits: 模型输出的原始类别分数，形状为 [B, C, H, W]。
        - targets: 像素级标签图，形状为 [B, H, W]。

        返回：
        - 标量损失张量。
        """
        loss = logits.new_tensor(0.0)
        if self.ce_weight > 0:
            loss = loss + self.ce_weight * F.cross_entropy(logits, targets, ignore_index=self.ignore_index)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self._dice_loss(logits, targets)
        return loss

    def _dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """基于 soft Dice 计算区域重叠损失。

        实现细节：
        1. 先对 logits 做 softmax，得到每类概率图。
        2. 把标签图转成 one-hot 形式，方便逐类统计。
        3. 对 ignore_index 区域做 mask，避免无效像素参与训练。
        4. 只对验证集中真实出现过的类别求均值，减少空类别干扰。
        """
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        valid_mask = (targets != self.ignore_index).unsqueeze(1)
        clamped_targets = torch.clamp(targets, min=0, max=num_classes - 1)
        target_one_hot = F.one_hot(clamped_targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        target_one_hot = target_one_hot * valid_mask
        probs = probs * valid_mask
        dims = (0, 2, 3)
        intersection = (probs * target_one_hot).sum(dim=dims)
        denominator = probs.sum(dim=dims) + target_one_hot.sum(dim=dims)
        valid_classes = target_one_hot.sum(dim=dims) > 0
        dice = 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        if valid_classes.any():
            dice = dice[valid_classes]
        return dice.mean()


def update_confusion_matrix(
    confusion: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    """把一个 batch 的预测结果累计到混淆矩阵中。

    混淆矩阵的第 i 行、第 j 列表示：
    真实类别为 i、模型预测为 j 的像素数量。

    这里使用 bincount 做扁平化统计，速度通常比逐类循环更高。
    """
    valid = target != ignore_index
    if not torch.any(valid):
        return confusion
    prediction = prediction[valid].to(torch.int64)
    target = target[valid].to(torch.int64)
    bins = target * num_classes + prediction
    confusion += torch.bincount(bins, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return confusion


def compute_metrics(confusion: torch.Tensor) -> dict[str, float]:
    """根据混淆矩阵计算常见语义分割指标。

    返回的指标包括：
    - mIoU: 各类别 IoU 的平均值。
    - acc: 各类别像素准确率的平均值。
    - aAcc: 全局像素准确率。
    - dice / precision / recall / fscore: 常用区域重叠和检索式指标。
    """
    confusion = confusion.float()
    intersection = torch.diag(confusion)
    target_area = confusion.sum(dim=1)
    pred_area = confusion.sum(dim=0)
    union = target_area + pred_area - intersection
    valid = target_area > 0

    iou = intersection / union.clamp_min(1.0)
    cls_acc = intersection / target_area.clamp_min(1.0)
    dice = 2.0 * intersection / (target_area + pred_area).clamp_min(1.0)
    precision = intersection / pred_area.clamp_min(1.0)
    recall = intersection / target_area.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-6)

    return {
        "mIoU": iou[valid].mean().item() if valid.any() else 0.0,
        "acc": cls_acc[valid].mean().item() if valid.any() else 0.0,
        "aAcc": intersection.sum().item() / target_area.sum().clamp_min(1.0).item(),
        "dice": dice[valid].mean().item() if valid.any() else 0.0,
        "precision": precision[valid].mean().item() if valid.any() else 0.0,
        "recall": recall[valid].mean().item() if valid.any() else 0.0,
        "fscore": f1[valid].mean().item() if valid.any() else 0.0,
    }


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device,
    amp_enabled: bool,
    num_classes: int,
    ignore_index: int,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    """在给定数据集上完整跑一遍评估。

    评估流程：
    1. 切换模型到 eval 模式。
    2. 逐 batch 前向推理。
    3. 如有需要，同时统计验证损失。
    4. 累计混淆矩阵并在最后统一计算指标。
    """
    model.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total_loss = 0.0

    for batch in dataloader:
        # 图像和标签都放到目标设备上。
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, masks) if criterion is not None else None
        if loss is not None:
            total_loss += loss.item()
        # 指标统计放在 CPU 上做即可，节省显存并简化后处理。
        prediction = logits.argmax(dim=1).cpu()
        target = masks.cpu()
        confusion = update_confusion_matrix(confusion, prediction, target, num_classes, ignore_index)

    metrics = compute_metrics(confusion)
    if criterion is not None:
        metrics["val_loss"] = total_loss / max(len(dataloader), 1)
    return metrics