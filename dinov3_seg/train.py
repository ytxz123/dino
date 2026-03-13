"""分割模型训练入口。

脚本负责搭建数据集、模型、优化器和学习率调度，并在每轮结束后执行验证与保存权重。
"""

from pathlib import Path
import logging
import math
import random

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dinov3_seg.config import get_config
from dinov3_seg.dataset import SegmentationDataset
from dinov3_seg.metrics import summarize_metrics, update_confusion_matrix
from dinov3_seg.model import FrozenDinoV3Segmenter


def configure_logging(output_dir: Path, log_name: str = "training.log") -> logging.Logger:
    """配置同时输出到终端和文件的标准日志。"""

    logger = logging.getLogger("dinov3_seg.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(output_dir / log_name, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def set_seed(seed: int) -> None:
    """固定常见随机源，尽量提升实验可复现性。"""

    # Python、NumPy 和 PyTorch 的随机源都要同步设置，否则很难做到可复现。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, amp_dtype: str):
    """根据设备和配置返回合适的自动混合精度上下文。"""

    if device.type != "cuda":
        # CPU 路径直接关闭 autocast，避免引入无意义的上下文开销。
        return torch.autocast(device_type="cpu", enabled=False)

    # 训练脚本只区分 bf16 和 fp16 两条常见 CUDA 混合精度路径。
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def update_lr(optimizer: AdamW, step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> None:
    """执行线性 warmup + 余弦退火学习率更新。"""

    if step < warmup_steps:
        # warmup 阶段从很小的学习率线性抬升到 base_lr，减少训练早期震荡。
        lr = base_lr * float(step + 1) / float(max(1, warmup_steps))
    else:
        # warmup 结束后进入余弦退火，让学习率平滑下降到 min_lr。
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    # AdamW 可能存在多个参数组，因此要逐组同步更新学习率。
    for group in optimizer.param_groups:
        group["lr"] = lr


def evaluate(
    model: FrozenDinoV3Segmenter,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg,
    logger: logging.Logger | None = None,
):
    """在验证集上跑完整轮评估，并返回汇总指标。"""

    model.eval()
    confusion_matrix = torch.zeros(cfg.data.num_classes, cfg.data.num_classes, dtype=torch.int64, device=device)
    loss_sum = 0.0
    sample_count = 0

    with torch.no_grad():
        for images, masks in loader:
            # 验证阶段不保留梯度，但仍复用训练时的 AMP 设置，保证吞吐一致。
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with autocast_context(device, cfg.train.amp_dtype):
                logits = model(images)
                loss = criterion(logits, masks)

            # 按 batch 内样本数累计，最终换算成整个验证集的平均损失。
            loss_sum += loss.item() * images.size(0)
            sample_count += images.size(0)
            update_confusion_matrix(confusion_matrix, logits, masks)

    if logger is not None:
        logger.info("validation finished: samples=%d avg_loss=%.4f", sample_count, loss_sum / sample_count)

    metrics = summarize_metrics(confusion_matrix)
    metrics["loss"] = loss_sum / sample_count
    return metrics


def save_checkpoint(output_dir: Path, epoch: int, model: FrozenDinoV3Segmenter, optimizer: AdamW, metrics: dict) -> dict:
    """保存当前分割头和优化器状态。"""

    # 只保存 head，避免把冻结 backbone 反复写入磁盘造成 checkpoint 过大。
    checkpoint = {
        "head": model.head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "feature_layers": model.feature_layers,
    }
    torch.save(checkpoint, output_dir / "last_head.pth")
    return checkpoint


def main():
    """执行完整训练流程。"""

    cfg = get_config()

    # 训练前先确保输出目录存在，避免第一次保存 checkpoint 时因目录缺失失败。
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(cfg.paths.output_dir)

    # 先创建输出目录，再初始化随机种子和算子精度策略。
    set_seed(cfg.train.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("training started: device=%s output_dir=%s", device, cfg.paths.output_dir)
    logger.info(
        "config summary: train_batch=%d eval_batch=%d grad_accum=%d epochs=%d lr=%.6e amp=%s",
        cfg.train.batch_size,
        cfg.train.eval_batch_size,
        cfg.train.grad_accum_steps,
        cfg.train.epochs,
        cfg.train.lr,
        cfg.train.amp_dtype,
    )

    # 训练集开启翻转增强，验证集保持纯净评估。
    train_dataset = SegmentationDataset(
        image_dir=cfg.paths.train_image_dir,
        mask_dir=cfg.paths.train_mask_dir,
        mean=cfg.data.mean,
        std=cfg.data.std,
        mask_divisor=cfg.data.mask_divisor,
        training=True,
        hflip_prob=cfg.data.train_hflip_prob,
    )
    val_dataset = SegmentationDataset(
        image_dir=cfg.paths.val_image_dir,
        mask_dir=cfg.paths.val_mask_dir,
        mean=cfg.data.mean,
        std=cfg.data.std,
        mask_divisor=cfg.data.mask_divisor,
        training=False,
        hflip_prob=0.0,
    )

    if len(train_dataset) == 0:
        raise ValueError(f"训练集为空: {cfg.paths.train_image_dir}")

    logger.info(
        "dataset summary: train_samples=%d val_samples=%d train_image_dir=%s val_image_dir=%s",
        len(train_dataset),
        len(val_dataset),
        cfg.paths.train_image_dir,
        cfg.paths.val_image_dir,
    )

    # 单图或极小数据集调试时，允许把训练集临时当作验证集，便于快速检查链路是否跑通。
    if len(val_dataset) == 0 and cfg.train.allow_train_as_val_when_val_empty:
        logger.warning("验证集为空，回退为使用训练集做验证，仅适合单图 smoke test。")
        val_dataset = SegmentationDataset(
            image_dir=cfg.paths.train_image_dir,
            mask_dir=cfg.paths.train_mask_dir,
            mean=cfg.data.mean,
            std=cfg.data.std,
            mask_divisor=cfg.data.mask_divisor,
            training=False,
            hflip_prob=0.0,
        )

    train_batch_size = min(cfg.train.batch_size, len(train_dataset))

    # 只有在“数据集长度明显大于 batch size”时，drop_last 才可能真正丢掉尾部样本。
    train_drop_last = cfg.train.drop_last and len(train_dataset) > train_batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=train_drop_last,
    )
    val_loader = None
    if len(val_dataset) > 0:
        # 验证 batch size 同样不应超过验证样本数，避免无意义的配置值。
        eval_batch_size = min(cfg.train.eval_batch_size, len(val_dataset))
        val_loader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )
    else:
        logger.warning("验证集为空，将跳过验证，只保存 last_head.pth。")

    # 模型只优化分割头，冻结骨干保持预训练表征稳定。
    model = FrozenDinoV3Segmenter(cfg).to(device)
    criterion = nn.CrossEntropyLoss()

    # 只把分割头参数交给优化器，确保冻结骨干不会被误更新。
    optimizer = AdamW(model.head.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # GradScaler 只在 fp16 模式下启用；bf16 一般不需要额外缩放。
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and cfg.train.amp_dtype == "fp16")

    # 学习率按全局 step 更新，因此这里先计算完整训练过程的步数。
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError(
            "训练 DataLoader 没有产生任何 batch。请减小 batch_size，或把 train.drop_last 设为 False。"
        )

    # 极小数据集下，累积步数不应超过一个 epoch 内真实存在的 batch 数。
    grad_accum_steps = min(cfg.train.grad_accum_steps, steps_per_epoch)
    total_steps = cfg.train.epochs * steps_per_epoch
    warmup_steps = int(cfg.train.warmup_epochs * steps_per_epoch)
    best_miou = -1.0

    logger.info(
        "loader summary: train_batches=%d val_batches=%s effective_train_batch=%d warmup_steps=%d total_steps=%d",
        steps_per_epoch,
        len(val_loader) if val_loader is not None else 0,
        train_batch_size * grad_accum_steps,
        warmup_steps,
        total_steps,
    )

    if grad_accum_steps != cfg.train.grad_accum_steps:
        logger.warning(
            "grad_accum_steps 从 %d 自动调整为 %d，以匹配当前极小数据集。",
            cfg.train.grad_accum_steps,
            grad_accum_steps,
        )

    for epoch in range(cfg.train.epochs):
        # head 参与训练，backbone 保持 eval，避免冻结骨干中的状态发生漂移。
        model.train()
        model.backbone.eval()
        logger.info("epoch started: %d/%d", epoch + 1, cfg.train.epochs)

        # 即使整个模型切到 train，骨干仍强制保持 eval，避免归一化/随机层状态变化。

        confusion_matrix = torch.zeros(cfg.data.num_classes, cfg.data.num_classes, dtype=torch.int64, device=device)
        loss_sum = 0.0
        sample_count = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (images, masks) in enumerate(train_loader):
            # global_step 用于统一驱动学习率调度器，而不是每个 epoch 单独重置。
            global_step = epoch * steps_per_epoch + step
            update_lr(optimizer, global_step, total_steps, warmup_steps, cfg.train.lr, cfg.train.min_lr)

            # 采用异步拷贝把数据搬到设备，减小数据准备带来的停顿。
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast_context(device, cfg.train.amp_dtype):
                logits = model(images)
                loss = criterion(logits, masks)

                # 梯度累积时，把 loss 按累积步数缩放，保证等效梯度大小不变。
                scaled_loss = loss / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # 只有达到累积步数，或者已经走到 epoch 最后一个 batch，才真正更新一次参数。
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == steps_per_epoch:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)

                # 只裁剪分割头梯度，因为骨干并不参与反向传播。
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), cfg.train.grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            loss_sum += loss.item() * images.size(0)
            sample_count += images.size(0)

            # 训练指标同样基于整轮累计混淆矩阵，避免只看最后一个 batch 产生误判。
            update_confusion_matrix(confusion_matrix, logits.detach(), masks)

            if (step + 1) % cfg.train.log_interval == 0:
                # 日志使用“当前 epoch 已累计”的混淆矩阵，更能反映整体训练趋势。
                metrics = summarize_metrics(confusion_matrix)
                avg_loss = loss_sum / sample_count
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "train step: epoch=%d/%d step=%d/%d lr=%.6e loss=%.4f miou=%.4f pixacc=%.4f",
                    epoch + 1,
                    cfg.train.epochs,
                    step + 1,
                    steps_per_epoch,
                    lr,
                    avg_loss,
                    metrics["mIoU"],
                    metrics["pixel_acc"],
                )

        train_metrics = summarize_metrics(confusion_matrix)
        train_metrics["loss"] = loss_sum / sample_count
        logger.info(
            "train epoch finished: epoch=%d loss=%.4f miou=%.4f pixacc=%.4f",
            epoch + 1,
            train_metrics["loss"],
            train_metrics["mIoU"],
            train_metrics["pixel_acc"],
        )

        checkpoint_metrics = train_metrics

        # 即使本轮不做验证，也会保存最新训练得到的分割头，便于中断恢复或 smoke test 检查。
        checkpoint = save_checkpoint(cfg.paths.output_dir, epoch + 1, model, optimizer, checkpoint_metrics)
        logger.info("checkpoint saved: %s", cfg.paths.output_dir / "last_head.pth")

        if val_loader is not None and (epoch + 1) % cfg.train.eval_interval == 0:
            val_metrics = evaluate(model, val_loader, criterion, device, cfg, logger=logger)
            logger.info(
                "validation epoch finished: epoch=%d loss=%.4f miou=%.4f pixacc=%.4f",
                epoch + 1,
                val_metrics["loss"],
                val_metrics["mIoU"],
                val_metrics["pixel_acc"],
            )

            checkpoint = save_checkpoint(cfg.paths.output_dir, epoch + 1, model, optimizer, val_metrics)
            logger.info("checkpoint updated after validation: %s", cfg.paths.output_dir / "last_head.pth")

            # best checkpoint 只由验证集 mIoU 决定，避免被训练集指标误导。
            if val_metrics["mIoU"] > best_miou:
                best_miou = val_metrics["mIoU"]
                torch.save(checkpoint, cfg.paths.output_dir / "best_head.pth")
                logger.info("best checkpoint updated: path=%s mIoU=%.4f", cfg.paths.output_dir / "best_head.pth", best_miou)

    logger.info("training completed")


if __name__ == "__main__":
    main()
