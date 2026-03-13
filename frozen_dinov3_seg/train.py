from __future__ import annotations

import csv
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from frozen_dinov3_seg.config import get_config
    from frozen_dinov3_seg.dataset import build_dataloaders
    from frozen_dinov3_seg.model import FrozenDinoV3Segmenter
else:
    from .config import get_config
    from .dataset import build_dataloaders
    from .model import FrozenDinoV3Segmenter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_autocast_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def build_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    pred = pred.view(-1)
    target = target.view(-1)
    encoded = target * num_classes + pred
    hist = torch.bincount(encoded, minlength=num_classes * num_classes)
    return hist.view(num_classes, num_classes)


def compute_metrics(hist: torch.Tensor, class_names: tuple[str, ...]) -> dict:
    hist = hist.float()
    true_positive = hist.diag()
    pred_area = hist.sum(dim=0)
    target_area = hist.sum(dim=1)
    union = pred_area + target_area - true_positive

    iou = true_positive / union.clamp_min(1.0)
    dice = (2.0 * true_positive) / (pred_area + target_area).clamp_min(1.0)
    pixel_acc = true_positive.sum() / hist.sum().clamp_min(1.0)

    metrics = {
        "pixel_acc": pixel_acc.item(),
        "mIoU": iou.mean().item(),
        "mDice": dice.mean().item(),
    }
    for idx, class_name in enumerate(class_names):
        metrics[f"IoU_{class_name}"] = iou[idx].item()
        metrics[f"Dice_{class_name}"] = dice[idx].item()
    return metrics


def evaluate(model, data_loader, criterion, device, config) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    hist = torch.zeros(config.dataset.num_classes, config.dataset.num_classes, dtype=torch.int64)

    amp_enabled = config.train.use_amp and device.type == "cuda"
    autocast_dtype = get_autocast_dtype(config.train.amp_dtype)

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if config.train.use_channels_last and device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)

            autocast_ctx = torch.autocast(device_type=device.type, dtype=autocast_dtype) if amp_enabled else nullcontext()
            with autocast_ctx:
                logits = model(images)
                loss = criterion(logits, masks)

            preds = logits.argmax(dim=1).cpu()
            hist += confusion_matrix(preds, masks.cpu(), config.dataset.num_classes)
            total_loss += loss.item()
            total_steps += 1

    metrics = compute_metrics(hist, config.dataset.class_names)
    return total_loss / max(1, total_steps), metrics


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch: int, best_miou: float) -> None:
    checkpoint = {
        "epoch": epoch,
        "best_miou": best_miou,
        "decoder": model.decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location="cpu")
    model.decoder.load_state_dict(checkpoint["decoder"], strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint.get("epoch", 0), checkpoint.get("best_miou", 0.0)


def append_metrics(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train() -> None:
    config = get_config()
    work_dir = Path(config.paths.work_dir)
    checkpoint_dir = work_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with (work_dir / "config_snapshot.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(config), file, ensure_ascii=False, indent=2)

    set_seed(config.train.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    if config.train.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        device = torch.device("cpu")

    train_loader, val_loader = build_dataloaders(config)
    model = FrozenDinoV3Segmenter(config).to(device)
    if config.train.use_channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)

    updates_per_epoch = math.ceil(len(train_loader) / config.train.grad_accum_steps)
    total_updates = updates_per_epoch * config.train.max_epochs
    warmup_updates = updates_per_epoch * config.train.warmup_epochs
    scheduler = build_scheduler(optimizer, warmup_updates, total_updates, config.train.min_lr_ratio)

    scaler = torch.cuda.amp.GradScaler(enabled=config.train.use_amp and device.type == "cuda")
    start_epoch = 1
    best_miou = 0.0

    if config.paths.resume_checkpoint:
        last_epoch, best_miou = load_checkpoint(
            config.paths.resume_checkpoint,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        start_epoch = last_epoch + 1

    if config.train.run_mode == "eval":
        if config.paths.eval_checkpoint:
            load_checkpoint(config.paths.eval_checkpoint, model)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, config)
        print(f"eval_loss={val_loss:.4f} mIoU={val_metrics['mIoU']:.4f} mDice={val_metrics['mDice']:.4f}")
        print(json.dumps(val_metrics, ensure_ascii=False, indent=2))
        return

    amp_enabled = config.train.use_amp and device.type == "cuda"
    autocast_dtype = get_autocast_dtype(config.train.amp_dtype)
    metrics_csv = work_dir / "metrics.csv"

    for epoch in range(start_epoch, config.train.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        epoch_start = time.time()

        for step, (images, masks) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if config.train.use_channels_last and device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)

            autocast_ctx = torch.autocast(device_type=device.type, dtype=autocast_dtype) if amp_enabled else nullcontext()
            with autocast_ctx:
                logits = model(images)
                loss = criterion(logits, masks)
                loss = loss / config.train.grad_accum_steps

            scaler.scale(loss).backward()

            should_step = step % config.train.grad_accum_steps == 0 or step == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.decoder.parameters(), config.train.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running_loss += loss.item() * config.train.grad_accum_steps

            if step % config.train.log_every == 0 or step == len(train_loader):
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"epoch={epoch:02d} step={step:05d}/{len(train_loader):05d} "
                    f"loss={running_loss / step:.4f} lr={current_lr:.6e}"
                )

        train_loss = running_loss / max(1, len(train_loader))
        epoch_time = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "time_sec": round(epoch_time, 2),
            "lr": scheduler.get_last_lr()[0],
        }

        if epoch % config.train.eval_every == 0:
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device, config)
            row["val_loss"] = round(val_loss, 6)
            row.update({key: round(value, 6) for key, value in val_metrics.items()})

            print(
                f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"mIoU={val_metrics['mIoU']:.4f} mDice={val_metrics['mDice']:.4f}"
            )

            if val_metrics["mIoU"] >= best_miou:
                best_miou = val_metrics["mIoU"]
                save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, best_miou)

        append_metrics(metrics_csv, row)

        if epoch % config.train.save_every == 0:
            save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, best_miou)


if __name__ == "__main__":
    train()
