"""分割数据集定义。

负责读取 PNG 图像与标签，并完成基础增强、张量化和标签映射。
"""

from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """读取图像和掩码对的轻量分割数据集。"""

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
        # 通过排序保证图像顺序稳定，便于复现实验与排查数据问题。
        self.image_paths = sorted(Path(image_dir).glob("*.png"))
        self.mask_dir = Path(mask_dir)

        # 均值和方差预先整理成 [C, 1, 1]，便于后续直接做广播归一化。
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.mask_divisor = mask_divisor
        self.training = training
        self.hflip_prob = hflip_prob

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """返回一对已经预处理好的图像与分割标签。"""

        image_path = self.image_paths[index]
        mask_path = self.mask_dir / image_path.name

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        # 训练阶段只做最基础的水平翻转，保证图像与标签严格同步变换。
        if self.training and random.random() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # 把 HWC 的 uint8 图像转换为 CHW 的 float32，并缩放到 [0, 1]。
        image = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1) / 255.0
        image = (image - self.mean) / self.std

        # 标签原始值为 0/80/160/240，整除后映射到连续类别编号 0~3。
        mask = torch.from_numpy(np.array(mask, dtype=np.int64) // self.mask_divisor)
        return image, mask
