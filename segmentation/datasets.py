from __future__ import annotations

"""固定 512x512 PNG 分割数据读取模块。"""

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


IMAGE_SUFFIX = ".png"
MASK_SUFFIX = ".png"
FIXED_IMAGE_SIZE = (512, 512)
SAT493M_DEFAULT_MEAN = (0.430, 0.411, 0.296)
SAT493M_DEFAULT_STD = (0.213, 0.156, 0.143)


def _looks_like_sat493m_weights(backbone_weights) -> bool:
    """根据权重名或路径粗略判断是否使用了 SAT-493M 预训练权重。"""
    return backbone_weights is not None and any(
        token in str(backbone_weights).strip().lower() for token in ("sat493m", "sat-493m", "eadcf0ff")
    )


def resolve_input_stats(input_stats: str = "auto", backbone_weights=None) -> tuple[tuple[float, float, float], tuple[float, float, float], str]:
    """决定输入标准化所使用的 mean/std。"""
    if input_stats == "sat493m" or (input_stats == "auto" and _looks_like_sat493m_weights(backbone_weights)):
        return SAT493M_DEFAULT_MEAN, SAT493M_DEFAULT_STD, "sat493m"
    return IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, "imagenet"


def segmentation_collate_fn(batch: list[dict], ignore_index: int = 255, image_pad_value: float = 0.0) -> dict:
    """固定尺寸样本的轻量 collate。"""
    del ignore_index, image_pad_value
    collated = {
        "image": torch.stack([item["image"] for item in batch]),
        "image_path": [item["image_path"] for item in batch],
        "relative_path": [item["relative_path"] for item in batch],
    }
    if "mask" in batch[0]:
        collated["mask"] = torch.stack([item["mask"] for item in batch]).long()
        collated["mask_path"] = [item["mask_path"] for item in batch]
    return collated


def collect_png_files(input_path: str | Path) -> list[Path]:
    """收集单个 PNG 文件或目录下全部 PNG 文件。"""
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() != IMAGE_SUFFIX:
            raise ValueError(f"Expected a PNG file, got {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(file for file in path.rglob(f"*{IMAGE_SUFFIX}") if file.is_file())
    if not files:
        raise FileNotFoundError(f"No PNG files found under {path}")
    return files


def collect_tif_files(input_path: str | Path) -> list[Path]:
    """兼容旧入口名，实际只收集 PNG。"""
    return collect_png_files(input_path)


def _resolve_mask_path(mask_dir: Path, relative_image_path: Path) -> Path:
    """按照固定规则把图像相对路径映射为 PNG 标注路径。"""
    return mask_dir / relative_image_path.with_suffix(MASK_SUFFIX)


def _validate_image_size(path: Path, size: tuple[int, int]):
    if size != FIXED_IMAGE_SIZE:
        raise ValueError(f"Expected {path} to be {FIXED_IMAGE_SIZE[0]}x{FIXED_IMAGE_SIZE[1]}, got {size[0]}x{size[1]}")


def _load_image(path: Path) -> np.ndarray:
    """读取固定 512x512 RGB PNG 图像。"""
    with Image.open(path) as image:
        image = image.convert("RGB")
        _validate_image_size(path, image.size)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
    return image_array


def _load_mask(path: Path) -> np.ndarray:
    """读取固定 512x512 单通道 PNG 标注图。"""
    with Image.open(path) as mask_image:
        _validate_image_size(path, mask_image.size)
        mask = np.asarray(mask_image)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int64, copy=False)


class _BasePngDataset(Dataset):
    """PNG 数据集的公共基类，只负责标准化和张量化。"""

    def __init__(self, input_stats: str = "auto", backbone_weights=None):
        mean, std, _ = resolve_input_stats(input_stats, backbone_weights=backbone_weights)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        return (image_tensor - self.mean) / self.std


class PngSegmentationDataset(_BasePngDataset):
    """训练/验证用的固定尺寸 PNG 语义分割数据集。"""

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        input_stats: str = "auto",
        backbone_weights=None,
        train: bool = True,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.0,
        ignore_index: int = 255,
    ):
        super().__init__(input_stats=input_stats, backbone_weights=backbone_weights)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.train = train
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.ignore_index = ignore_index
        self.samples = self._build_samples()

    def _build_samples(self) -> list[tuple[Path, Path, Path]]:
        image_paths = collect_png_files(self.image_dir)
        samples = []
        for image_path in image_paths:
            relative_path = image_path.relative_to(self.image_dir)
            mask_path = _resolve_mask_path(self.mask_dir, relative_path)
            if not mask_path.is_file():
                raise FileNotFoundError(f"Mask not found for {image_path}: {mask_path}")
            samples.append((image_path, mask_path, relative_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path, relative_path = self.samples[index]
        image = _load_image(image_path)
        mask = _load_mask(mask_path)

        image_tensor = self._to_tensor(image)
        mask_tensor = torch.from_numpy(mask).long()

        if self.train and random.random() < self.hflip_prob:
            image_tensor = torch.flip(image_tensor, dims=[2])
            mask_tensor = torch.flip(mask_tensor, dims=[1])
        if self.train and random.random() < self.vflip_prob:
            image_tensor = torch.flip(image_tensor, dims=[1])
            mask_tensor = torch.flip(mask_tensor, dims=[0])

        mask_tensor = torch.where(mask_tensor < 0, torch.full_like(mask_tensor, self.ignore_index), mask_tensor)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "relative_path": str(relative_path),
        }


class PngInferenceDataset(_BasePngDataset):
    """推理阶段专用数据集，只返回图像及其路径信息。"""

    def __init__(
        self,
        image_paths: Iterable[str | Path],
        input_stats: str = "auto",
        backbone_weights=None,
        root_dir: str | Path | None = None,
    ):
        super().__init__(input_stats=input_stats, backbone_weights=backbone_weights)
        self.image_paths = [Path(path) for path in image_paths]
        self.root_dir = Path(root_dir) if root_dir is not None else None

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = _load_image(image_path)
        image_tensor = self._to_tensor(image)
        if self.root_dir is not None and image_path.is_relative_to(self.root_dir):
            relative_path = image_path.relative_to(self.root_dir)
        else:
            relative_path = Path(image_path.name)
        return {
            "image": image_tensor,
            "image_path": str(image_path),
            "relative_path": str(relative_path),
        }


TifSegmentationDataset = PngSegmentationDataset
TifInferenceDataset = PngInferenceDataset