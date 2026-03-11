from __future__ import annotations

"""浅层 DINOv3 语义分割训练入口。

这个脚本负责把整个训练流程串起来：
1. 解析命令行参数。
2. 构建数据集与 dataloader。
3. 构建分割模型、损失函数和优化器。
4. 执行 epoch 级训练与验证。
5. 保存 checkpoint 和指标文件。

它面向的是当前这个固定 512x512 PNG 分割场景，尽量保持单机、直接、可读。
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dinov3.segmentation.datasets import PngSegmentationDataset
from dinov3.segmentation.metrics import CombinedSegmentationLoss, evaluate_model
from dinov3.segmentation.model import build_shallow_dinov3_segmentor, parse_layer_indices


logger = logging.getLogger("dinov3.segmentation")

DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent / "dataset"
DEFAULT_OUTPUT_DIR = Path("outputs/shallow_seg_sat_lite")
DEFAULT_BACKBONE_WEIGHTS = "checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"


def parse_args():
    """定义训练脚本可用的全部命令行参数。"""
    parser = argparse.ArgumentParser(description="Train a shallow-feature semantic segmentation head on fixed-size PNG images.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--train-image-dir", type=Path, default=None)
    parser.add_argument("--train-mask-dir", type=Path, default=None)
    parser.add_argument("--val-image-dir", type=Path, default=None)
    parser.add_argument("--val-mask-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backbone-name", type=str, default="dinov3_vitl16")
    parser.add_argument("--backbone-weights", type=str, default=DEFAULT_BACKBONE_WEIGHTS)
    parser.add_argument("--disable-backbone-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--layer-indices", type=str, default=None)
    parser.add_argument("--num-shallow-layers", type=int, default=3)
    parser.add_argument("--decoder-dim", type=int, default=128)
    parser.add_argument("--detail-dims", type=str, default="32,64,128")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--input-stats", choices=["auto", "imagenet", "sat493m"], default="auto")
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=5)
    return parser.parse_args()


def set_seed(seed: int):
    """固定随机种子，尽量减少重复实验之间的波动。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(args, dataset_dirs: dict[str, Path]):
    """根据当前参数构建训练集和验证集的 dataloader。"""
    train_dataset = PngSegmentationDataset(
        image_dir=dataset_dirs["train_image_dir"],
        mask_dir=dataset_dirs["train_mask_dir"],
        input_stats=args.input_stats,
        backbone_weights=args.backbone_weights,
        train=True,
        ignore_index=args.ignore_index,
    )
    val_dataset = PngSegmentationDataset(
        image_dir=dataset_dirs["val_image_dir"],
        mask_dir=dataset_dirs["val_mask_dir"],
        input_stats=args.input_stats,
        backbone_weights=args.backbone_weights,
        train=False,
        hflip_prob=0.0,
        vflip_prob=0.0,
        ignore_index=args.ignore_index,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def save_checkpoint(path: Path, model, optimizer, scaler, epoch: int, best_miou: float, model_config: dict):
    """把当前训练状态保存成 checkpoint 文件。"""
    # 除了模型参数，还要保存优化器、AMP 状态和模型构造参数，
    # 这样恢复训练、独立评估和单独推理都能复用同一套结构定义。
    checkpoint = {
        "epoch": epoch,
        "best_miou": best_miou,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": model_config,
    }
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    torch.save(checkpoint, path)


def train_one_epoch(model, dataloader, optimizer, scaler, criterion, device, amp_enabled: bool, grad_clip: float):
    """执行一个完整 epoch 的训练并返回平均损失。"""
    # 这里默认 dataloader 已经输出标准化后的图像和同尺寸 mask。
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        # batch 内部字段由自定义数据集和 collate_fn 统一组织。
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        # AMP 只在 CUDA 上开启，CPU 环境下会自动退化为普通 fp32。
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, masks)

        if scaler is not None:
            # 混合精度训练时，需要先 scale，再 backward。
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(dataloader), 1)


def main():
    """训练脚本主入口。"""
    # 整体流程：
    # 1. 解析参数并准备训练环境。
    # 2. 构建分割模型与数据集。
    # 3. 执行 epoch 循环。
    # 4. 保存最优权重与训练日志。
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = args.amp and device.type == "cuda"
    dataset_dirs = {
        "train_image_dir": args.train_image_dir or args.dataset_root / "images" / "train",
        "train_mask_dir": args.train_mask_dir or args.dataset_root / "masks" / "train",
        "val_image_dir": args.val_image_dir or args.dataset_root / "images" / "val",
        "val_mask_dir": args.val_mask_dir or args.dataset_root / "masks" / "val",
    }

    # detail_dims 控制细节分支在 1/2、1/4、1/8 三个尺度的通道数。
    # 这三个数字越大，局部细节建模能力越强，但显存和参数量也越高。
    detail_dims = tuple(int(item.strip()) for item in args.detail_dims.split(","))
    model_config = {
        "backbone_name": args.backbone_name,
        "backbone_weights": args.backbone_weights,
        "pretrained": not args.disable_backbone_pretrained,
        "layer_indices": parse_layer_indices(args.layer_indices),
        "num_shallow_layers": args.num_shallow_layers,
        "num_classes": args.num_classes,
        "decoder_dim": args.decoder_dim,
        "detail_dims": detail_dims,
        "dropout": args.dropout,
        "freeze_backbone": args.freeze_backbone,
        "input_size": (512, 512),
    }

    model = build_shallow_dinov3_segmentor(**model_config).to(device)
    model_config["resolved_layer_indices"] = list(model.layer_indices)
    train_loader, val_loader = build_dataloaders(args, dataset_dirs)
    criterion = CombinedSegmentationLoss(args.ce_weight, args.dice_weight, args.ignore_index)
    # 仅优化 requires_grad=True 的参数。
    # 当 backbone 被冻结时，这里自然只会训练细节分支和解码头。
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # 这里用简单稳定的余弦退火调度器，便于中小规模数据集直接起跑。
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None

    start_epoch = 0
    best_miou = 0.0
    if args.resume is not None:
        # 从断点恢复时，不只恢复模型权重，
        # 还要把优化器和 AMP 状态一起接回来，保证训练轨迹连续。
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_miou = checkpoint.get("best_miou", 0.0)

    with (args.output_dir / "train_config.json").open("w", encoding="utf-8") as file:
        serialized_args = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
        serialized_args.update({key: str(value) for key, value in dataset_dirs.items()})
        json.dump(serialized_args, file, indent=2)

    logger.info("Training samples: %s | Validation samples: %s", len(train_loader.dataset), len(val_loader.dataset))
    logger.info("Using shallow layers: %s", model.layer_indices)
    logger.info("Fixed input size: 512x512 PNG")
    history_path = args.output_dir / "metrics_history.jsonl"

    for epoch in range(start_epoch, args.epochs):
        # 每一轮都遵循统一顺序：训练 -> 验证 -> 记录指标 -> 保存断点。
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, amp_enabled, args.grad_clip)
        metrics = evaluate_model(
            model,
            val_loader,
            device=device,
            amp_enabled=amp_enabled,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            criterion=criterion,
        )
        scheduler.step()
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **metrics,
        }
        with history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(epoch_record) + "\n")
        with (args.output_dir / "last_metrics.json").open("w", encoding="utf-8") as file:
            json.dump(epoch_record, file, indent=2)

        logger.info(
            "Epoch %s/%s | train_loss=%.4f | val_loss=%.4f | mIoU=%.4f | dice=%.4f | aAcc=%.4f",
            epoch + 1,
            args.epochs,
            train_loss,
            metrics["val_loss"],
            metrics["mIoU"],
            metrics["dice"],
            metrics["aAcc"],
        )

        if (epoch + 1) % args.save_every == 0:
            # 常规 checkpoint 方便回溯训练过程或中断后恢复。
            save_checkpoint(args.output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pth", model, optimizer, scaler, epoch, best_miou, model_config)

        if metrics["mIoU"] >= best_miou:
            # 以 mIoU 作为主指标保存最优模型。
            best_miou = metrics["mIoU"]
            save_checkpoint(args.output_dir / "best_model.pth", model, optimizer, scaler, epoch, best_miou, model_config)
            with (args.output_dir / "best_metrics.json").open("w", encoding="utf-8") as file:
                json.dump(epoch_record, file, indent=2)


if __name__ == "__main__":
    main()