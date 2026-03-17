from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .config import DatasetConfig


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        dataset_config: DatasetConfig,
        *,
        hflip_prob: float = 0.0,
    ) -> None:
        self.image_paths = sorted(Path(image_dir).glob("*.png"))
        self.mask_dir = Path(mask_dir)
        self.image_size = dataset_config.image_size
        self.label_divisor = dataset_config.label_divisor
        self.mean = dataset_config.mean
        self.std = dataset_config.std
        self.hflip_prob = hflip_prob

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        mask_path = self.mask_dir / image_path.name

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

        if self.hflip_prob > 0 and torch.rand(1).item() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image_tensor = TF.normalize(TF.to_tensor(image), self.mean, self.std)
        # 标注像素值固定为类别编号乘以 50，这里直接还原为 0/1/2。
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.uint8) // self.label_divisor).long()
        return image_tensor, mask_tensor


def build_dataloader(
    image_dir: str,
    mask_dir: str,
    dataset_config: DatasetConfig,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
    hflip_prob: float = 0.0,
) -> DataLoader:
    dataset = SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        dataset_config=dataset_config,
        hflip_prob=hflip_prob,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
    )