from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class PNGSegmentationDataset(Dataset):
    """读取 512x512 PNG 图像与 PNG 掩码。"""

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        image_size: int,
        mask_values: tuple[int, ...],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        train: bool,
        hflip_prob: float,
        vflip_prob: float,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.train = train
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob

        self.image_paths = sorted(self.images_dir.glob("*.png"))
        self.mask_paths = [self.masks_dir / path.name for path in self.image_paths]

        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

        decode_table = np.zeros(256, dtype=np.uint8)
        for class_index, pixel_value in enumerate(mask_values):
            decode_table[pixel_value] = class_index
        self.decode_table = decode_table

    def __len__(self) -> int:
        return len(self.image_paths)

    def _resize_if_needed(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        target_size = (self.image_size, self.image_size)
        if image.size != target_size:
            image = image.resize(target_size, Image.BILINEAR)
        if mask.size != target_size:
            mask = mask.resize(target_size, Image.NEAREST)
        return image, mask

    def _augment(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if self.train and random.random() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if self.train and random.random() < self.vflip_prob:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]).convert("L")

        image, mask = self._resize_if_needed(image, mask)
        image, mask = self._augment(image, mask)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
        image_tensor = (image_tensor - self.mean) / self.std

        mask_np = np.asarray(mask, dtype=np.uint8)
        mask_tensor = torch.from_numpy(self.decode_table[mask_np].astype(np.int64))
        return image_tensor, mask_tensor


def build_dataloaders(config):
    train_dataset = PNGSegmentationDataset(
        images_dir=config.paths.train_images_dir,
        masks_dir=config.paths.train_masks_dir,
        image_size=config.dataset.image_size,
        mask_values=config.dataset.mask_values,
        mean=config.dataset.mean,
        std=config.dataset.std,
        train=True,
        hflip_prob=config.dataset.hflip_prob,
        vflip_prob=config.dataset.vflip_prob,
    )
    val_dataset = PNGSegmentationDataset(
        images_dir=config.paths.val_images_dir,
        masks_dir=config.paths.val_masks_dir,
        image_size=config.dataset.image_size,
        mask_values=config.dataset.mask_values,
        mean=config.dataset.mean,
        std=config.dataset.std,
        train=False,
        hflip_prob=0.0,
        vflip_prob=0.0,
    )

    persistent_workers = config.train.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.val_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader
