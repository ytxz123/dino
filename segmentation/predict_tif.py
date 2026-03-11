from __future__ import annotations

"""固定 512x512 PNG 分割推理入口。"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dinov3.segmentation.datasets import PngInferenceDataset, collect_png_files
from dinov3.segmentation.model import load_shallow_dinov3_segmentor_from_checkpoint, parse_layer_indices


logger = logging.getLogger("dinov3.segmentation")


def parse_args():
    """定义推理脚本可接受的命令行参数。"""
    parser = argparse.ArgumentParser(description="Run semantic segmentation inference on fixed-size PNG images.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-path", type=Path, required=True, help="Single PNG file or a directory containing PNG files.")
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
    parser.add_argument("--input-stats", choices=["auto", "imagenet", "sat493m"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def save_prediction_png(prediction: np.ndarray, output_path: Path):
    """把类别图保存为 PNG。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if int(prediction.max()) <= 255:
        Image.fromarray(prediction.astype(np.uint8), mode="L").save(output_path)
        return
    Image.fromarray(prediction.astype(np.uint16), mode="I;16").save(output_path)


@torch.no_grad()
def main():
    """推理脚本主入口。"""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = args.amp and device.type == "cuda"
    image_paths = collect_png_files(args.input_path)
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

    dataset = PngInferenceDataset(
        image_paths=image_paths,
        input_stats=args.input_stats,
        backbone_weights=dataset_backbone_weights,
        root_dir=root_dir,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = model.to(device)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        relative_paths = batch["relative_path"]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)

        for index in range(logits.shape[0]):
            prediction = logits[index].argmax(dim=0).cpu().numpy()
            relative_path = Path(relative_paths[index])
            output_path = args.output_dir / relative_path.with_suffix(".png")
            save_prediction_png(prediction, output_path)
            logger.info("Saved prediction to %s", output_path)


if __name__ == "__main__":
    main()