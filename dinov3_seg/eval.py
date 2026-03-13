import torch
from torch import nn
from torch.utils.data import DataLoader

from dinov3_seg.config import get_config
from dinov3_seg.dataset import SegmentationDataset
from dinov3_seg.metrics import summarize_metrics, update_confusion_matrix
from dinov3_seg.model import FrozenDinoV3Segmenter
from dinov3_seg.train import autocast_context


def main():
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = SegmentationDataset(
        image_dir=cfg.paths.val_image_dir,
        mask_dir=cfg.paths.val_mask_dir,
        mean=cfg.data.mean,
        std=cfg.data.std,
        mask_divisor=cfg.data.mask_divisor,
        training=False,
        hflip_prob=0.0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    model = FrozenDinoV3Segmenter(cfg).to(device)
    checkpoint = torch.load(cfg.paths.eval_checkpoint, map_location="cpu")
    model.head.load_state_dict(checkpoint["head"], strict=True)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    confusion_matrix = torch.zeros(cfg.data.num_classes, cfg.data.num_classes, dtype=torch.int64, device=device)
    loss_sum = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with autocast_context(device, cfg.train.amp_dtype):
                logits = model(images)
                loss = criterion(logits, masks)
            loss_sum += loss.item() * images.size(0)
            update_confusion_matrix(confusion_matrix, logits, masks)

    metrics = summarize_metrics(confusion_matrix)
    metrics["loss"] = loss_sum / len(val_loader.dataset)

    print(f"loss={metrics['loss']:.4f}")
    print(f"mIoU={metrics['mIoU']:.4f}")
    print(f"pixel_acc={metrics['pixel_acc']:.4f}")
    print(f"class_iou={metrics['class_iou']}")


if __name__ == "__main__":
    main()
