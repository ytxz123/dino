import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dinov3.segmentation.model import BACKBONE_NAMES, LightweightSegmentationDecoder, get_weights_tag


EMBED_DIMS = {
    "dinov3_vits16": 384,
    "dinov3_vits16plus": 384,
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
    "dinov3_vitl16plus": 1024,
    "dinov3_vith16plus": 1280,
    "dinov3_vit7b16": 4096,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--backbone", type=str, default="dinov3_vitb16", choices=BACKBONE_NAMES)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--decoder-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_run_dir(output_root: str | Path, backbone: str, weights: str) -> Path:
    return Path(output_root) / backbone / get_weights_tag(weights) / "smoke_test"


def configure_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("dinov3.segmentation.smoke")
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


def create_random_mask_pngs(mask_dir: Path, num_samples: int, size: int, num_classes: int, seed: int) -> None:
    generator = np.random.default_rng(seed)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for index in range(num_samples):
        mask = generator.integers(0, num_classes, size=(size, size), dtype=np.uint8) * 80
        Image.fromarray(mask, mode="L").save(mask_dir / f"sample_{index:03d}.png")


class RandomFeatureMaskDataset(Dataset):
    def __init__(self, mask_dir: Path, embed_dim: int, feature_size: int = 32):
        self.mask_paths = sorted(mask_dir.glob("*.png"))
        self.embed_dim = embed_dim
        self.feature_size = feature_size

    def __len__(self) -> int:
        return len(self.mask_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        mask = np.asarray(Image.open(self.mask_paths[index]), dtype=np.uint8)
        target = torch.from_numpy(mask // 80).long()
        feature = torch.randn(self.embed_dim, self.feature_size, self.feature_size)
        return feature, target


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


def get_device_info(device: torch.device) -> dict:
    device_info = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        device_info.update(
            {
                "cuda_device_index": index,
                "cuda_device_name": props.name,
                "cuda_total_memory_gb": round(props.total_memory / 1024**3, 2),
                "cuda_capability": f"{props.major}.{props.minor}",
            }
        )
    return device_info


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_dir = build_run_dir(args.output_dir, args.backbone, args.weights)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(run_dir)

    embed_dim = EMBED_DIMS[args.backbone]
    feature_size = args.size // 16
    mask_dir = run_dir / "random_masks"
    create_random_mask_pngs(mask_dir, args.samples, args.size, args.num_classes, args.seed)

    dataset = RandomFeatureMaskDataset(mask_dir=mask_dir, embed_dim=embed_dim, feature_size=feature_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_info = get_device_info(device)
    logger.info("device_info=%s", json.dumps(device_info, ensure_ascii=True))
    decoder = LightweightSegmentationDecoder(
        in_channels=embed_dim,
        feature_count=1,
        num_classes=args.num_classes,
        decoder_dim=args.decoder_dim,
        dropout=0.0,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr)

    total_loss = 0.0
    confusion_matrix = torch.zeros(args.num_classes, args.num_classes, dtype=torch.int64, device=device)
    decoder.train()
    for step, batch in enumerate(dataloader, start=1):
        features, targets = batch
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = decoder([features])
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        update_confusion_matrix(confusion_matrix, logits.detach(), targets, args.num_classes)
        logger.info("step=%s/%s loss=%.4f", step, len(dataloader), loss.item())

    metrics = compute_metrics(confusion_matrix)
    results = {
        "backbone": args.backbone,
        "weights_tag": get_weights_tag(args.weights),
        "embed_dim": embed_dim,
        "feature_size": feature_size,
        "samples": args.samples,
        "device_info": device_info,
        "train_loss": total_loss / max(len(dataloader), 1),
        **metrics,
    }

    torch.save(
        {
            "decoder": decoder.state_dict(),
            "results": results,
            "config": vars(args),
        },
        run_dir / "smoke_last.pt",
    )
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=True)
    logger.info("train_loss=%.4f miou=%.4f pixel_acc=%.4f", results["train_loss"], results["miou"], results["pixel_acc"])


if __name__ == "__main__":
    main()