import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dinov3.eval.segmentation.loss import MultiSegmentationLoss
from dinov3.eval.segmentation.metrics import calculate_intersect_and_union, calculate_segmentation_metrics
from dinov3.eval.segmentation.schedulers import build_scheduler

from .config import ThreeClassSegConfig, save_config
from .dataset import build_dataloader
from .model import build_model, load_checkpoint


logger = logging.getLogger("three_class_segmentation")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _checkpoint_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_miou: float,
) -> dict:
    return {
        "epoch": epoch,
        "best_miou": best_miou,
        "head": model.head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_miou: float,
    output_path: Path,
) -> None:
    torch.save(_checkpoint_state(model, optimizer, scheduler, epoch, best_miou), output_path)


def save_metrics(metrics: dict[str, float], output_path: Path) -> None:
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")


def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    model.eval()
    pre_eval_results = []
    with torch.inference_mode():
        for images, masks in dataloader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model.predict(images, rescale_to=tuple(masks.shape[-2:]))
            preds = logits.argmax(dim=1)
            for pred, mask in zip(preds, masks):
                pre_eval_results.append(
                    calculate_intersect_and_union(
                        pred,
                        mask,
                        num_classes=num_classes,
                        reduce_zero_label=False,
                    )
                )
    metrics = calculate_segmentation_metrics(pre_eval_results, metrics=["mIoU", "dice", "fscore"])
    return {name: round(float(value.cpu().item() * 100), 2) for name, value in metrics.items()}


def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: torch.nn.Module,
    device: torch.device,
    accumulation_steps: int,
    grad_clip: float,
    log_interval: int,
) -> float:
    model.train()
    model.backbone.eval()
    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    update_steps = 0
    head_parameters = [parameter for parameter in model.head.parameters() if parameter.requires_grad]

    for step, (images, masks) in enumerate(dataloader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        loss = criterion(logits, masks)
        (loss / accumulation_steps).backward()
        running_loss += loss.item()

        if step % accumulation_steps == 0 or step == len(dataloader):
            # 用梯度累积把 16G 显存下的单卡有效 batch 拉到更稳定的范围。
            torch.nn.utils.clip_grad_norm_(head_parameters, grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_steps += 1

            if update_steps % log_interval == 0:
                logger.info(
                    "update=%s/%s loss=%.4f lr=%.6f",
                    update_steps,
                    math.ceil(len(dataloader) / accumulation_steps),
                    running_loss / step,
                    optimizer.param_groups[0]["lr"],
                )

    return running_loss / len(dataloader)


def train(config: ThreeClassSegConfig) -> dict[str, float]:
    setup_logging()
    set_seed(config.runtime.seed)
    torch.set_float32_matmul_precision("high")

    output_dir = Path(config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, str(output_dir / "resolved_config.yaml"))

    device = resolve_device(config.runtime.device)
    train_loader = build_dataloader(
        config.dataset.train_images,
        config.dataset.train_masks,
        config.dataset,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        shuffle=True,
        pin_memory=config.runtime.pin_memory,
        hflip_prob=config.train.hflip_prob,
    )
    val_loader = build_dataloader(
        config.dataset.val_images,
        config.dataset.val_masks,
        config.dataset,
        batch_size=config.eval.batch_size,
        num_workers=config.eval.num_workers,
        shuffle=False,
        pin_memory=config.runtime.pin_memory,
    )

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.optimizer.lr,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        weight_decay=config.optimizer.weight_decay,
    )

    total_steps = math.ceil(len(train_loader) / config.train.accumulation_steps) * config.train.epochs
    scheduler = build_scheduler(
        config.scheduler.name,
        optimizer=optimizer,
        lr=config.optimizer.lr,
        total_iter=total_steps,
        constructor_kwargs={
            "warmup_iters": config.scheduler.warmup_iters,
            "pct_start": config.scheduler.pct_start,
            "div_factor": config.scheduler.div_factor,
            "final_div_factor": config.scheduler.final_div_factor,
        },
    )
    criterion = MultiSegmentationLoss(
        diceloss_weight=config.train.dice_weight,
        celoss_weight=config.train.ce_weight,
    )

    start_epoch = 0
    best_miou = -1.0
    if config.train.resume_from:
        checkpoint = load_checkpoint(model, config.train.resume_from, device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_miou = float(checkpoint["best_miou"])

    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    logger.info("device=%s train_samples=%s val_samples=%s trainable_params=%s", device, len(train_loader.dataset), len(val_loader.dataset), trainable_params)

    best_metrics = {}
    for epoch in range(start_epoch, config.train.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            config.train.accumulation_steps,
            config.optimizer.grad_clip,
            config.train.log_interval,
        )
        logger.info("epoch=%s/%s train_loss=%.4f", epoch + 1, config.train.epochs, train_loss)

        if (epoch + 1) % config.train.eval_interval == 0:
            metrics = evaluate_model(model, val_loader, device, config.head.num_classes)
            logger.info("epoch=%s/%s val_metrics=%s", epoch + 1, config.train.epochs, metrics)
            save_metrics(metrics, output_dir / "last_metrics.json")
            if metrics["mIoU"] > best_miou:
                best_miou = metrics["mIoU"]
                best_metrics = metrics
                save_checkpoint(model, optimizer, scheduler, epoch, best_miou, output_dir / "best.pth")
                save_metrics(metrics, output_dir / "best_metrics.json")

        if (epoch + 1) % config.train.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_miou, output_dir / "last.pth")

    if not best_metrics:
        best_metrics = evaluate_model(model, val_loader, device, config.head.num_classes)
        save_metrics(best_metrics, output_dir / "best_metrics.json")
    save_checkpoint(model, optimizer, scheduler, config.train.epochs - 1, best_miou, output_dir / "last.pth")
    return best_metrics


def evaluate(config: ThreeClassSegConfig) -> dict[str, float]:
    setup_logging()
    torch.set_float32_matmul_precision("high")

    output_dir = Path(config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.runtime.device)

    dataloader = build_dataloader(
        config.dataset.val_images,
        config.dataset.val_masks,
        config.dataset,
        batch_size=config.eval.batch_size,
        num_workers=config.eval.num_workers,
        shuffle=False,
        pin_memory=config.runtime.pin_memory,
    )
    model = build_model(config).to(device)
    checkpoint = config.eval.checkpoint_path or config.train.resume_from
    load_checkpoint(model, checkpoint, device)
    metrics = evaluate_model(model, dataloader, device, config.head.num_classes)
    save_metrics(metrics, output_dir / "eval_metrics.json")
    logger.info("eval_metrics=%s", metrics)
    return metrics