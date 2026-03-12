import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dinov3.segmentation.dataset import SegmentationDataset
from dinov3.segmentation.model import build_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--masks", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--backbone-weights", type=str, default=None)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


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


@torch.no_grad()
def evaluate(
    model,
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


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model_from_checkpoint(checkpoint, override_backbone_weights=args.backbone_weights).to(device)
    criterion = nn.CrossEntropyLoss()

    dataset = SegmentationDataset(args.images, args.masks, size=args.size, augment=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    val_loss, metrics = evaluate(model, dataloader, criterion, device, model.num_classes)
    results = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "images": str(Path(args.images).resolve()),
        "masks": str(Path(args.masks).resolve()),
        "samples": len(dataset),
        "val_loss": val_loss,
        **metrics,
    }

    print(json.dumps(results, indent=2, ensure_ascii=True))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
