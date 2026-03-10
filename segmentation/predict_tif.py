from __future__ import annotations

"""分割推理入口。

这个脚本负责把训练好的 checkpoint 应用到单张 tif 或一整个 tif 目录上，
并把预测类别图重新保存成 tif 文件。
"""

import argparse
import logging
from pathlib import Path

import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dinov3.segmentation.datasets import TifInferenceDataset, collect_tif_files
from dinov3.segmentation.model import load_shallow_dinov3_segmentor_from_checkpoint, parse_layer_indices


logger = logging.getLogger("dinov3.segmentation")


def parse_args():
    """定义推理脚本可接受的命令行参数。"""
    # 推理脚本支持从 checkpoint 自动恢复大部分建模配置，
    # 这里只保留必要的覆盖参数，方便临时改输入目录或波段设置。
    parser = argparse.ArgumentParser(description="Run semantic segmentation inference on tif images.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-path", type=Path, required=True, help="Single tif file or a directory containing tif files.")
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--image-size", type=int, nargs="+", default=None)
    parser.add_argument("--keep-size", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_image_size(values, keep_size: bool = False):
    """把推理阶段的输入尺寸参数转换成统一格式。"""
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
    """解析波段参数，例如 3,2,1。"""
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


@torch.no_grad()
def main():
    """推理脚本主入口。"""
    # 推理流程：收集 tif -> 恢复模型 -> 构建数据集 -> 批量推理 -> 保存预测图。
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = args.amp and device.type == "cuda"
    image_paths = collect_tif_files(args.input_path)
    root_dir = args.input_path if args.input_path.is_dir() else args.input_path.parent

    # 模型结构优先从 checkpoint 恢复，命令行参数只做必要覆盖。
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
    saved_model_config = checkpoint.get("model_config", {})
    dataset_backbone_weights = args.backbone_weights or saved_model_config.get("backbone_weights")

    # 推理数据集不需要标签，只保留图像张量和路径信息即可。
    dataset = TifInferenceDataset(
        image_paths=image_paths,
        image_size=parse_image_size(args.image_size, keep_size=args.keep_size),
        band_indices=parse_int_list(args.bands),
        normalize_mode=args.normalize_mode,
        input_stats=args.input_stats,
        backbone_weights=dataset_backbone_weights,
        percentile_range=tuple(float(item.strip()) for item in args.percentile_range.split(",")),
        root_dir=root_dir,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = model.to(device)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for batch in dataloader:
        # 一个 batch 里可能包含多张图，因此后面需要逐张恢复原始尺寸并保存。
        images = batch["image"].to(device, non_blocking=True)
        original_sizes = batch["original_size"]
        relative_paths = batch["relative_path"]
        # 与训练阶段一致，推理时也可选择开启 AMP 提升吞吐。
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)

        for index in range(logits.shape[0]):
            logits_i = logits[index : index + 1]
            original_size = tuple(int(value) for value in original_sizes[index].tolist())
            if logits_i.shape[-2:] != original_size:
                # 如果推理前做过 resize，这里需要把预测结果插值回原图大小再保存。
                logits_i = F.interpolate(logits_i, size=original_size, mode="bilinear", align_corners=False)
            prediction = logits_i.argmax(dim=1).squeeze(0).cpu().numpy()
            relative_path = Path(relative_paths[index])
            output_path = args.output_dir / relative_path.with_suffix(".tif")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # 用 uint16 保存类别图，兼容大多数多类别分割场景。
            tifffile.imwrite(output_path, prediction.astype("uint16"))
            logger.info("Saved prediction to %s", output_path)


if __name__ == "__main__":
    main()