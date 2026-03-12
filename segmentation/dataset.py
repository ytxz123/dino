from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def load_rgb_image(image_path: str | Path, size: int = 512) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), resample=Image.BILINEAR)
    return image


def load_mask_image(mask_path: str | Path, size: int = 512) -> Image.Image:
    mask = Image.open(mask_path)
    if mask.size != (size, size):
        mask = mask.resize((size, size), resample=Image.NEAREST)
    return mask


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    mask_array = np.asarray(mask, dtype=np.uint8)
    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]
    return torch.from_numpy(mask_array // 80).long()


def prediction_to_png(prediction: torch.Tensor) -> Image.Image:
    prediction = prediction.to(torch.uint8).cpu().numpy() * 80
    return Image.fromarray(prediction, mode="L")


class SegmentationDataset(Dataset):
    def __init__(self, image_dir: str | Path, mask_dir: str | Path, size: int = 512, augment: bool = False):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.size = size
        self.augment = augment
        self.image_paths = sorted(self.image_dir.glob("*.png"))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        mask_path = self.mask_dir / image_path.name

        image = load_rgb_image(image_path, size=self.size)
        mask = load_mask_image(mask_path, size=self.size)

        if self.augment and torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        return image_to_tensor(image), mask_to_tensor(mask)