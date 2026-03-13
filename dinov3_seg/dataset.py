from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        mask_divisor: int,
        training: bool,
        hflip_prob: float,
    ):
        self.image_paths = sorted(Path(image_dir).glob("*.png"))
        self.mask_dir = Path(mask_dir)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.mask_divisor = mask_divisor
        self.training = training
        self.hflip_prob = hflip_prob

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        mask_path = self.mask_dir / image_path.name

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.training and random.random() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        image = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1) / 255.0
        image = (image - self.mean) / self.std

        mask = torch.from_numpy(np.array(mask, dtype=np.int64) // self.mask_divisor)
        return image, mask
