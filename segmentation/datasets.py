from __future__ import annotations

"""tif/png 分割数据读取模块。

这个文件负责处理当前这个精简场景下的数据流：
1. 读取 tif 图像。
2. 读取对应的 png 标注。
3. 完成通道整理、归一化、resize 和张量化。
4. 在 keep-size 模式下做 batch 级 padding。

它假设训练/验证数据格式固定：图像是 tif 或 tiff，标注是同相对路径的 png。
"""

import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


VALID_TIF_SUFFIXES = {".tif", ".tiff"}
MASK_SUFFIX = ".png"
SAT493M_DEFAULT_MEAN = (0.430, 0.411, 0.296)
SAT493M_DEFAULT_STD = (0.213, 0.156, 0.143)


def _looks_like_sat493m_weights(backbone_weights) -> bool:
    """根据权重名或路径粗略判断是否使用了 SAT-493M 预训练权重。"""
    return backbone_weights is not None and any(
        token in str(backbone_weights).strip().lower() for token in ("sat493m", "sat-493m", "eadcf0ff")
    )


def resolve_input_stats(input_stats: str = "auto", backbone_weights=None) -> tuple[tuple[float, float, float], tuple[float, float, float], str]:
    """决定输入标准化所使用的 mean/std。

    当前支持两套统计量：
    - ImageNet 默认统计量。
    - SAT-493M 卫星图像统计量。

    当 input_stats=auto 时，会根据权重文件名自动选择。
    """
    if input_stats == "sat493m" or (input_stats == "auto" and _looks_like_sat493m_weights(backbone_weights)):
        return SAT493M_DEFAULT_MEAN, SAT493M_DEFAULT_STD, "sat493m"
    return IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, "imagenet"


def _pad_spatial_tensor(tensor: torch.Tensor, target_size: tuple[int, int], pad_value: float | int) -> torch.Tensor:
    """把单个张量补到指定空间尺寸。

    这里只做右侧和下侧 padding，保持左上角像素对齐，
    这样后续把预测裁回原始大小时逻辑最简单。
    """
    target_height, target_width = target_size
    pad_height = target_height - tensor.shape[-2]
    pad_width = target_width - tensor.shape[-1]
    return F.pad(tensor, (0, pad_width, 0, pad_height), value=pad_value)


def segmentation_collate_fn(batch: list[dict], ignore_index: int = 255, image_pad_value: float = 0.0) -> dict:
    """支持不定尺寸样本的自定义 batch 拼接函数。"""
    # 支持不定尺寸样本的批量训练/评估：
    # 先把同一 batch 内的 image/mask pad 到最大高宽，再整体 stack。
    max_height = max(item["image"].shape[-2] for item in batch)
    max_width = max(item["image"].shape[-1] for item in batch)
    target_size = (max_height, max_width)

    collated = {
        "image": torch.stack([_pad_spatial_tensor(item["image"], target_size, image_pad_value) for item in batch]),
        "image_path": [item["image_path"] for item in batch],
        "relative_path": [item["relative_path"] for item in batch],
        "original_size": torch.stack([item["original_size"] for item in batch]),
    }
    if "mask" in batch[0]:
        collated["mask"] = torch.stack(
            [_pad_spatial_tensor(item["mask"], target_size, ignore_index) for item in batch]
        ).long()
        collated["mask_path"] = [item["mask_path"] for item in batch]
    return collated


def collect_tif_files(input_path: str | Path) -> list[Path]:
    """收集单个 tif 文件或目录下全部 tif/tiff 文件。"""
    path = Path(input_path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(file for file in path.rglob("*") if file.is_file() and file.suffix.lower() in VALID_TIF_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No tif files found under {path}")
    return files


def _resolve_mask_path(mask_dir: Path, relative_image_path: Path) -> Path:
    """按照固定规则把图像相对路径映射为 png 标注路径。"""
    return mask_dir / relative_image_path.with_suffix(MASK_SUFFIX)


def _ensure_hwc(array: np.ndarray) -> np.ndarray:
    """把 tif 读出的数组统一整理为 HWC 格式。"""
    # tif 数据的通道布局并不统一：
    # 可能是 H x W、H x W x C，也可能是 C x H x W。
    # 这里统一整理成 H x W x C，减少后续处理分支。
    if array.ndim == 2:
        return array[..., None]
    if array.ndim != 3:
        raise ValueError(f"Expected 2D or 3D tif array, got shape {array.shape}")
    if array.shape[0] <= 16 and array.shape[-1] > 16:
        return np.moveaxis(array, 0, -1)
    return array


def _select_bands(array: np.ndarray, band_indices: Sequence[int] | None) -> np.ndarray:
    """把任意通道 tif 整理成 DINOv3 可接受的 3 通道输入。"""
    if band_indices is not None:
        array = array[..., list(band_indices)]

    if array.shape[-1] == 1:
        return np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 2:
        return np.concatenate([array, array[..., -1:]], axis=-1)
    return array[..., :3]


def _normalize_channel(
    channel: np.ndarray,
    mode: str,
    percentile_range: tuple[float, float],
    source_dtype: np.dtype,
) -> np.ndarray:
    """对单个通道做归一化。

    - percentile: 用分位数裁剪后缩放到 [0, 1]。
    - dtype: 用原始 dtype 的数值范围或浮点最值缩放到 [0, 1]。
    """
    finite = channel[np.isfinite(channel)]
    if finite.size == 0:
        return np.zeros_like(channel, dtype=np.float32)

    if mode == "percentile":
        low = float(np.percentile(finite, percentile_range[0]))
        high = float(np.percentile(finite, percentile_range[1]))
    else:
        if np.issubdtype(source_dtype, np.integer):
            dtype_info = np.iinfo(source_dtype)
            low = float(dtype_info.min)
            high = float(dtype_info.max)
        else:
            low = float(finite.min())
            high = float(finite.max())
    scale = max(high - low, 1e-6)
    channel = np.clip(channel, low, high)
    return (channel - low) / scale


def _load_tif_image(path: Path, band_indices: Sequence[int] | None, normalize_mode: str, percentile_range):
    """读取并预处理一张 tif 图像。"""
    image = tifffile.imread(path)
    image = _ensure_hwc(np.asarray(image))
    image = _select_bands(image, band_indices)
    source_dtype = image.dtype
    image = image.astype(np.float32, copy=False)
    image = np.stack(
        [
            _normalize_channel(
                image[..., channel_index],
                normalize_mode,
                percentile_range,
                source_dtype,
            )
            for channel_index in range(3)
        ],
        axis=-1,
    )
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    image = np.clip(image, 0.0, 1.0)
    return image


def _load_mask(path: Path) -> np.ndarray:
    """读取单通道 png 标注图，并转成 int64 类别图。"""
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int64, copy=False)


class _BaseTifDataset(Dataset):
    """tif 数据集的公共基类。

    负责：
    1. 保存归一化配置。
    2. 将图像 resize 到训练尺寸。
    3. 按对应预训练域的统计量做标准化，兼容 DINOv3 预训练分布。
    """

    def __init__(
        self,
        image_size: int | tuple[int, int] | None,
        band_indices: Sequence[int] | None,
        normalize_mode: str,
        percentile_range: tuple[float, float],
        input_stats: str = "auto",
        backbone_weights=None,
    ):
        self.image_size = image_size
        self.band_indices = tuple(band_indices) if band_indices is not None else None
        self.normalize_mode = normalize_mode
        self.percentile_range = percentile_range
        mean, std, _ = resolve_input_stats(input_stats, backbone_weights=backbone_weights)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def _resize_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """按设定尺寸对图像做双线性缩放。"""
        # 图像使用双线性插值，适合连续值数据。
        if self.image_size is None:
            return image_tensor
        return F.interpolate(
            image_tensor.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _resize_mask(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        """按设定尺寸对标签图做最近邻缩放。"""
        # mask 必须使用最近邻插值，避免类别编号被插值污染。
        if self.image_size is None:
            return mask_tensor
        return F.interpolate(mask_tensor[None, None].float(), size=self.image_size, mode="nearest").squeeze(0).squeeze(0).long()

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """把 numpy 图像转成标准化后的 CHW Tensor。"""
        # 转成 CHW Tensor，并按 DINOv3 预训练期望做标准化。
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        image_tensor = self._resize_image(image_tensor)
        return (image_tensor - self.mean) / self.std


class TifSegmentationDataset(_BaseTifDataset):
    """训练/验证用的 tif 语义分割数据集。

    图像固定从 tif 读取。
    标注固定为 png；匹配时按相对路径和同名 stem 查找。
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        image_size: int | tuple[int, int] | None = 512,
        band_indices: Sequence[int] | None = None,
        normalize_mode: str = "percentile",
        percentile_range: tuple[float, float] = (2.0, 98.0),
        input_stats: str = "auto",
        backbone_weights=None,
        train: bool = True,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.0,
        ignore_index: int = 255,
    ):
        super().__init__(image_size, band_indices, normalize_mode, percentile_range, input_stats=input_stats, backbone_weights=backbone_weights)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.train = train
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.ignore_index = ignore_index
        self.samples = self._build_samples()

    def _build_samples(self) -> list[tuple[Path, Path, Path]]:
        """扫描图像目录并建立图像与标注的一一对应关系。"""
        # 通过相对路径和同名 stem 匹配 image 与 png mask。
        image_paths = collect_tif_files(self.image_dir)
        samples = []
        for image_path in image_paths:
            relative_path = image_path.relative_to(self.image_dir)
            mask_path = _resolve_mask_path(self.mask_dir, relative_path)
            samples.append((image_path, mask_path, relative_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        """读取一条训练/验证样本并返回字典格式结果。"""
        # 返回 dict 而不是 tuple，方便后续脚本按字段名访问。
        image_path, mask_path, relative_path = self.samples[index]
        image = _load_tif_image(image_path, self.band_indices, self.normalize_mode, self.percentile_range)
        mask = _load_mask(mask_path)

        image_tensor = self._to_tensor(image)
        mask_tensor = self._resize_mask(torch.from_numpy(mask).long())

        # 仅在训练集启用简单增强，避免验证结果受到随机性影响。
        if self.train and random.random() < self.hflip_prob:
            image_tensor = torch.flip(image_tensor, dims=[2])
            mask_tensor = torch.flip(mask_tensor, dims=[1])
        if self.train and random.random() < self.vflip_prob:
            image_tensor = torch.flip(image_tensor, dims=[1])
            mask_tensor = torch.flip(mask_tensor, dims=[0])

        # 对负标签统一映射为 ignore_index，防止损失函数报错。
        mask_tensor = torch.where(mask_tensor < 0, torch.full_like(mask_tensor, self.ignore_index), mask_tensor)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "relative_path": str(relative_path),
            "original_size": torch.tensor(mask.shape, dtype=torch.int64),
        }


class TifInferenceDataset(_BaseTifDataset):
    """推理阶段专用数据集，只返回图像及其路径信息。"""

    def __init__(
        self,
        image_paths: Iterable[str | Path],
        image_size: int | tuple[int, int] | None = None,
        band_indices: Sequence[int] | None = None,
        normalize_mode: str = "percentile",
        percentile_range: tuple[float, float] = (2.0, 98.0),
        input_stats: str = "auto",
        backbone_weights=None,
        root_dir: str | Path | None = None,
    ):
        super().__init__(image_size, band_indices, normalize_mode, percentile_range, input_stats=input_stats, backbone_weights=backbone_weights)
        self.image_paths = [Path(path) for path in image_paths]
        self.root_dir = Path(root_dir) if root_dir is not None else None

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """读取一条推理样本，只返回图像与路径信息。"""
        image_path = self.image_paths[index]
        image = _load_tif_image(image_path, self.band_indices, self.normalize_mode, self.percentile_range)
        image_tensor = self._to_tensor(image)
        # original_size 用于把预测结果恢复回原始 tif 尺寸。
        original_size = torch.tensor(image.shape[:2], dtype=torch.int64)
        if self.root_dir is not None and image_path.is_relative_to(self.root_dir):
            relative_path = image_path.relative_to(self.root_dir)
        else:
            relative_path = Path(image_path.name)
        return {
            "image": image_tensor,
            "image_path": str(image_path),
            "relative_path": str(relative_path),
            "original_size": original_size,
        }