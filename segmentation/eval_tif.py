from __future__ import annotations

"""独立评估入口。

这个脚本用于在训练结束后，单独对某个 checkpoint 做可复现评估。
它与训练阶段内嵌的验证逻辑使用同一套指标实现，但把执行入口独立出来，
更适合做正式对比实验、测试集复评和不同输入参数的离线试验。
"""

import argparse
import json
import logging
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dinov3.segmentation.datasets import TifSegmentationDataset, segmentation_collate_fn
from dinov3.segmentation.metrics import evaluate_model
from dinov3.segmentation.model import load_shallow_dinov3_segmentor_from_checkpoint, parse_layer_indices


logger = logging.getLogger("dinov3.segmentation")

DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent / "dataset"
DEFAULT_OUTPUT_DIR = Path("outputs/shallow_seg_sat_lite_eval")


def parse_args():
    """定义独立评估脚本所需的命令行参数。"""
    parser = argparse.ArgumentParser(description="Evaluate a shallow-feature semantic segmentation checkpoint on tif images.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backbone-name", type=str, default=None)
    parser.add_argument("--backbone-weights", type=str, default=None)
    parser.add_argument("--disable-backbone-pretrained", action="store_true")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--layer-indices", type=str, default=None)
    parser.add_argument("--num-shallow-layers", type=int, default=None)
    parser.add_argument("--decoder-dim", type=int, default=None)
    parser.add_argument("--detail-dims", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--freeze-backbone", action="store_true", default=False)
    parser.add_argument("--bands", type=str, default=None)
    parser.add_argument("--normalize-mode", choices=["percentile", "dtype"], default="percentile")
    parser.add_argument("--input-stats", choices=["auto", "imagenet", "sat493m"], default="auto")
    parser.add_argument("--percentile-range", type=str, default="2,98")
    parser.add_argument("--image-size", type=int, nargs="+", default=[384, 384])
    parser.add_argument("--keep-size", action="store_true")
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_image_size(values, keep_size: bool = False):
    """把评估阶段的输入尺寸参数转换成统一格式。"""
    if keep_size:
        return None
    if values is None:
        return None
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise ValueError("--image-size expects one integer or two integers")


def parse_int_list(value: str | None) -> list[int] | None:
    """解析波段参数，例如 0,1,2。"""
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main():
    """独立评估主入口。"""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 先从 checkpoint 恢复模型结构和参数，保证评估时的网络与训练时一致。
    model, checkpoint = load_shallow_dinov3_segmentor_from_checkpoint(
        args.checkpoint,
        backbone_name=args.backbone_name,
        backbone_weights=args.backbone_weights,
        pretrained=False if args.disable_backbone_pretrained else None,
        layer_indices=parse_layer_indices(args.layer_indices) if args.layer_indices is not None else None,
        num_shallow_layers=args.num_shallow_layers,
        num_classes=args.num_classes,
        decoder_dim=args.decoder_dim,
        detail_dims=args.detail_dims,
        dropout=args.dropout,
        freeze_backbone=True if args.freeze_backbone else None,
    )
    image_dir = args.image_dir or args.dataset_root / "images" / "val"
    mask_dir = args.mask_dir or args.dataset_root / "masks" / "val"
    saved_model_config = checkpoint.get("model_config", {})
    dataset_backbone_weights = args.backbone_weights or saved_model_config.get("backbone_weights")

    # 评估数据集默认读取验证集目录。
    dataset = TifSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_size=parse_image_size(args.image_size, keep_size=args.keep_size),
        band_indices=parse_int_list(args.bands),
        normalize_mode=args.normalize_mode,
        input_stats=args.input_stats,
        backbone_weights=dataset_backbone_weights,
        percentile_range=tuple(float(item.strip()) for item in args.percentile_range.split(",")),
        train=False,
        hflip_prob=0.0,
        vflip_prob=0.0,
        ignore_index=args.ignore_index,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(segmentation_collate_fn, ignore_index=args.ignore_index),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = args.amp and device.type == "cuda"
    model = model.to(device)

    # 类别数优先使用命令行覆盖值，否则回退到 checkpoint 中保存的训练配置。
    if args.num_classes is None:
        saved_model_config = checkpoint.get("model_config", {})
        num_classes = int(saved_model_config["num_classes"])
    else:
        num_classes = args.num_classes

    metrics = evaluate_model(
        model,
        dataloader,
        device=device,
        amp_enabled=amp_enabled,
        num_classes=num_classes,
        ignore_index=args.ignore_index,
        criterion=None,
    )
    metrics_path = args.output_dir / "eval_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    logger.info(
        "Evaluation finished | mIoU=%.4f | dice=%.4f | fscore=%.4f | aAcc=%.4f | precision=%.4f | recall=%.4f",
        metrics["mIoU"],
        metrics["dice"],
        metrics["fscore"],
        metrics["aAcc"],
        metrics["precision"],
        metrics["recall"],
    )
    logger.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()