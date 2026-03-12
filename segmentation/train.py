import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dinov3.segmentation.dataset import SegmentationDataset
from dinov3.segmentation.model import BACKBONE_NAMES, Dinov3SegmentationModel, get_weights_tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-images", type=str, required=True)
    parser.add_argument("--train-masks", type=str, required=True)
    parser.add_argument("--val-images", type=str, default=None)
    parser.add_argument("--val-masks", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="dinov3_vitb16", choices=BACKBONE_NAMES)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[12])
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--decoder-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def build_run_dir(output_root: str | Path, backbone: str, weights: str) -> Path:
    return Path(output_root) / backbone / get_weights_tag(weights)


def configure_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("dinov3.segmentation.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(run_dir / "training.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def append_metrics(run_dir: Path, metrics: dict) -> None:
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, ensure_ascii=True) + "\n")


def save_run_config(run_dir: Path, args: argparse.Namespace) -> None:
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to_device(batch, device: torch.device):
    images, masks = batch
    return images.to(device, non_blocking=True), masks.to(device, non_blocking=True)


def update_confusion_matrix(confusion_matrix: torch.Tensor, logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> None:
    prediction = logits.argmax(dim=1)
    valid = (target >= 0) & (target < num_classes)
    hist = torch.bincount(
        num_classes * target[valid].reshape(-1) + prediction[valid].reshape(-1),
        minlength=num_classes * num_classes,
    )
    confusion_matrix += hist.view(num_classes, num_classes)


def compute_metrics(confusion_matrix: torch.Tensor) -> dict:
    confusion_matrix = confusion_matrix.float()
    intersection = confusion_matrix.diag()
    union = confusion_matrix.sum(0) + confusion_matrix.sum(1) - intersection
    iou = intersection / union.clamp_min(1.0)
    pixel_acc = intersection.sum() / confusion_matrix.sum().clamp_min(1.0)
    return {
        "miou": iou.mean().item(),
        "pixel_acc": pixel_acc.item(),
        "iou_per_class": iou.tolist(),
    }


def train_one_epoch(
    model: Dinov3SegmentationModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int,
    logger: logging.Logger,
) -> float:
    model.train()
    running_loss = 0.0
    for step, batch in enumerate(dataloader, start=1):
        images, masks = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

        if step % log_interval == 0 or step == len(dataloader):
            logger.info("epoch=%s step=%s/%s loss=%.4f", epoch, step, len(dataloader), running_loss / step)
    return running_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(
    model: Dinov3SegmentationModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)
    for batch in dataloader:
        images, masks = move_to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)
        total_loss += loss.item()
        update_confusion_matrix(confusion_matrix, logits, masks, num_classes)

    metrics = compute_metrics(confusion_matrix)
    return total_loss / max(len(dataloader), 1), metrics


def save_checkpoint(
    checkpoint_path: Path,
    model: Dinov3SegmentationModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_miou: float,
) -> None:
    checkpoint = {
        "config": model.export_config(),
        "decoder": model.decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_miou": best_miou,
    }
    torch.save(checkpoint, checkpoint_path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    run_dir = build_run_dir(args.output_dir, args.backbone, args.weights)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(run_dir)
    save_run_config(run_dir, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("run_dir=%s", run_dir)
    logger.info("device=%s", device)
    train_dataset = SegmentationDataset(args.train_images, args.train_masks, size=args.size, augment=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    val_loader = None
    if args.val_images and args.val_masks:
        val_dataset = SegmentationDataset(args.val_images, args.val_masks, size=args.size, augment=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

    model = Dinov3SegmentationModel(
        backbone_name=args.backbone,
        backbone_weights=args.weights,
        layers=args.layers,
        num_classes=args.num_classes,
        decoder_dim=args.decoder_dim,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 1
    best_miou = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.decoder.load_state_dict(checkpoint["decoder"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_miou = checkpoint.get("best_miou", 0.0)
        logger.info("resume=%s start_epoch=%s best_miou=%.4f", args.resume, start_epoch, best_miou)

    logger.info("train_samples=%s val_samples=%s", len(train_dataset), len(val_loader.dataset) if val_loader else 0)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            device,
            epoch,
            args.log_interval,
            logger,
        )
        logger.info("epoch=%s train_loss=%.4f", epoch, train_loss)

        epoch_metrics = {"epoch": epoch, "train_loss": train_loss}

        if val_loader is None:
            append_metrics(run_dir, epoch_metrics)
            save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, best_miou)
            continue

        val_loss, metrics = evaluate(model, val_loader, criterion, device, args.num_classes)
        logger.info(
            "epoch=%s val_loss=%.4f miou=%.4f pixel_acc=%.4f",
            epoch,
            val_loss,
            metrics["miou"],
            metrics["pixel_acc"],
        )
        epoch_metrics.update({"val_loss": val_loss, **metrics})
        append_metrics(run_dir, epoch_metrics)
        if metrics["miou"] >= best_miou:
            best_miou = metrics["miou"]
            save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, best_miou)
            logger.info("new_best_miou=%.4f epoch=%s", best_miou, epoch)
        save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, best_miou)

    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"best_miou": best_miou, "run_dir": str(run_dir)}, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()