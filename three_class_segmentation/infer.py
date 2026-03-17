import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def _bootstrap_import_path() -> None:
    if __package__:
        return

    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_import_path()

from three_class_segmentation.config import DatasetConfig, load_config
from three_class_segmentation.engine import resolve_device, setup_logging
from three_class_segmentation.model import build_model, load_checkpoint


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_PALETTE = np.array(
    [
        [0, 0, 0],
        [220, 20, 60],
        [0, 158, 96],
        [255, 184, 0],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DINOv3 ViT-L/16 语义分割推理")
    parser.add_argument("--input", required=True, help="输入图像路径，支持单张图或目录")
    parser.add_argument("--output-dir", default="", help="推理结果保存目录，默认输出到 runtime.output_dir/inference")
    parser.add_argument("--checkpoint", default="", help="分割头 checkpoint 路径，默认取 eval.checkpoint_path")
    parser.add_argument("--device", default="", help="设备覆盖，例如 cpu 或 cuda")
    return parser.parse_args()


def collect_image_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    image_paths = sorted(path for path in input_path.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found under: {input_path}")
    return image_paths


def preprocess_image(image: Image.Image, dataset_config: DatasetConfig) -> torch.Tensor:
    resized = TF.resize(
        image,
        [dataset_config.image_size, dataset_config.image_size],
        interpolation=InterpolationMode.BILINEAR,
    )
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(tensor, dataset_config.mean, dataset_config.std)
    return tensor.unsqueeze(0)


def build_palette(num_classes: int) -> np.ndarray:
    if num_classes <= len(DEFAULT_PALETTE):
        return DEFAULT_PALETTE[:num_classes]

    palette = [DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)] for i in range(num_classes)]
    return np.stack(palette, axis=0).astype(np.uint8)


def predict_mask(model: torch.nn.Module, image: Image.Image, dataset_config: DatasetConfig, device: torch.device) -> np.ndarray:
    image_tensor = preprocess_image(image, dataset_config).to(device, non_blocking=True)
    logits = model.predict(image_tensor, rescale_to=(image.height, image.width))
    pred_mask = logits.argmax(dim=1).squeeze(0).to(torch.uint8).cpu().numpy()
    return pred_mask


def save_prediction_outputs(
    image: Image.Image,
    pred_mask: np.ndarray,
    output_stem: Path,
    num_classes: int,
    label_divisor: int,
) -> None:
    palette = build_palette(num_classes)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    index_mask = Image.fromarray(pred_mask, mode="L")
    encoded_mask = Image.fromarray((pred_mask.astype(np.uint16) * label_divisor).astype(np.uint8), mode="L")
    color_mask = Image.fromarray(palette[pred_mask], mode="RGB")

    image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
    color_array = np.asarray(color_mask, dtype=np.float32)
    overlay = Image.fromarray(np.clip(image_array * 0.6 + color_array * 0.4, 0, 255).astype(np.uint8), mode="RGB")

    index_mask.save(output_stem.with_name(f"{output_stem.name}_index.png"))
    encoded_mask.save(output_stem.with_name(f"{output_stem.name}_mask.png"))
    color_mask.save(output_stem.with_name(f"{output_stem.name}_color.png"))
    overlay.save(output_stem.with_name(f"{output_stem.name}_overlay.png"))


def main() -> None:
    args = parse_args()
    setup_logging()

    config = load_config()
    if args.device:
        config.runtime.device = args.device
    if args.checkpoint:
        config.eval.checkpoint_path = args.checkpoint

    checkpoint_path = config.eval.checkpoint_path or config.train.resume_from
    if not checkpoint_path:
        raise ValueError("Checkpoint path is required. Set eval.checkpoint_path or pass --checkpoint.")

    input_path = Path(args.input).expanduser().resolve()
    image_paths = collect_image_paths(input_path)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else Path(config.runtime.output_dir) / "inference"
    output_dir = output_dir.resolve()

    device = resolve_device(config.runtime.device)
    model = build_model(config).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        pred_mask = predict_mask(model, image, config.dataset, device)
        if input_path.is_dir():
            relative_stem = image_path.relative_to(input_path).with_suffix("")
        else:
            relative_stem = Path(image_path.stem)
        save_prediction_outputs(
            image=image,
            pred_mask=pred_mask,
            output_stem=output_dir / relative_stem,
            num_classes=config.head.num_classes,
            label_divisor=config.dataset.label_divisor,
        )


if __name__ == "__main__":
    main()