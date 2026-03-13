from pathlib import Path
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def update_lr(optimizer: AdamW, step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> None:
    if step < warmup_steps:
        lr = base_lr * float(step + 1) / float(max(1, warmup_steps))
    else:
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr


def evaluate(model: FrozenDinoV3Segmenter, loader: DataLoader, criterion: nn.Module, device: torch.device, cfg):
    model.eval()
    confusion_matrix = torch.zeros(cfg.data.num_classes, cfg.data.num_classes, dtype=torch.int64, device=device)
    loss_sum = 0.0
    sample_count = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with autocast_context(device, cfg.train.amp_dtype):
                logits = model(images)
                loss = criterion(logits, masks)
            loss_sum += loss.item() * images.size(0)
            sample_count += images.size(0)
            update_confusion_matrix(confusion_matrix, logits, masks)

    metrics = summarize_metrics(confusion_matrix)
    metrics["loss"] = loss_sum / sample_count
    return metrics


def main():
    cfg = get_config()
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.train.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    model = FrozenDinoV3Segmenter(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.head.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and cfg.train.amp_dtype == "fp16")

    steps_per_epoch = len(train_loader)
    total_steps = cfg.train.epochs * steps_per_epoch
    warmup_steps = int(cfg.train.warmup_epochs * steps_per_epoch)
    best_miou = -1.0

    for epoch in range(cfg.train.epochs):
        model.train()
        model.backbone.eval()

        confusion_matrix = torch.zeros(cfg.data.num_classes, cfg.data.num_classes, dtype=torch.int64, device=device)
        loss_sum = 0.0
        sample_count = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (images, masks) in enumerate(train_loader):
            global_step = epoch * steps_per_epoch + step
            update_lr(optimizer, global_step, total_steps, warmup_steps, cfg.train.lr, cfg.train.min_lr)

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast_context(device, cfg.train.amp_dtype):
                logits = model(images)
                loss = criterion(logits, masks)
                scaled_loss = loss / cfg.train.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if (step + 1) % cfg.train.grad_accum_steps == 0 or (step + 1) == steps_per_epoch:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), cfg.train.grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            loss_sum += loss.item() * images.size(0)
            sample_count += images.size(0)
            update_confusion_matrix(confusion_matrix, logits.detach(), masks)

            if (step + 1) % cfg.train.log_interval == 0:
                metrics = summarize_metrics(confusion_matrix)
                avg_loss = loss_sum / sample_count
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch={epoch + 1}/{cfg.train.epochs} step={step + 1}/{steps_per_epoch} "
                    f"lr={lr:.6e} loss={avg_loss:.4f} miou={metrics['mIoU']:.4f} pixacc={metrics['pixel_acc']:.4f}"
                )

        train_metrics = summarize_metrics(confusion_matrix)
        train_metrics["loss"] = loss_sum / sample_count
        print(
            f"train epoch={epoch + 1} loss={train_metrics['loss']:.4f} "
            f"miou={train_metrics['mIoU']:.4f} pixacc={train_metrics['pixel_acc']:.4f}"
        )

        if (epoch + 1) % cfg.train.eval_interval == 0:
            val_metrics = evaluate(model, val_loader, criterion, device, cfg)
            print(
                f"valid epoch={epoch + 1} loss={val_metrics['loss']:.4f} "
                f"miou={val_metrics['mIoU']:.4f} pixacc={val_metrics['pixel_acc']:.4f}"
            )

            checkpoint = {
                "head": model.head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "metrics": val_metrics,
                "feature_layers": model.feature_layers,
            }
            torch.save(checkpoint, cfg.paths.output_dir / "last_head.pth")

            if val_metrics["mIoU"] > best_miou:
                best_miou = val_metrics["mIoU"]
                torch.save(checkpoint, cfg.paths.output_dir / "best_head.pth")
                print(f"best checkpoint updated: mIoU={best_miou:.4f}")


if __name__ == "__main__":
    main()
