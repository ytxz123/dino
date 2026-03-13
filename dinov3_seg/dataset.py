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

    @staticmethod
    def _load_mask_array(mask_path: Path) -> np.ndarray:
        """把掩码文件统一解析成二维类别图。

        训练使用 CrossEntropyLoss，因此目标张量必须是 [H, W] 的整型类别索引。
        实际数据里有些 PNG 掩码虽然语义上是灰度标签图，但会被保存成 RGB 或 RGBA，
        这里统一压回单通道，避免后续堆叠成 [B, H, W, C] 触发 loss 报错。
        """

        mask = np.array(Image.open(mask_path), dtype=np.int64)

        if mask.ndim == 2:
            return mask

        if mask.ndim == 3 and mask.shape[2] == 1:
            return mask[..., 0]

        if mask.ndim == 3 and mask.shape[2] in (3, 4):
            rgb = mask[..., :3]

            # 常见情况是三个通道内容完全一致，只是文件被编码成 RGB。
            if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2]):
                return rgb[..., 0]

            raise ValueError(
                f"标签图 {mask_path} 是多通道彩色掩码，当前实现只支持灰度标签或三个通道数值完全一致的 RGB 标签。"
            )

        raise ValueError(f"标签图 {mask_path} 的形状异常: {mask.shape}")

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
        # 掩码目录单独保存，getitem 时按“同名文件”规则动态拼接路径。
        self.mask_dir = Path(mask_dir)

        # 均值和方差预先整理成 [C, 1, 1]，便于后续直接做广播归一化。
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        # 记录标签映射和训练增强的控制参数。
        self.mask_divisor = mask_divisor
        self.training = training
        self.hflip_prob = hflip_prob

    def __len__(self) -> int:
        # DataLoader 会通过这个长度决定 epoch 内有多少个样本可迭代。
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """返回一对已经预处理好的图像与分割标签。"""

        image_path = self.image_paths[index]
        mask_path = self.mask_dir / image_path.name

        # 图像显式转为 RGB，避免灰度图或调色板图引入通道不一致问题。
        image = Image.open(image_path).convert("RGB")
        # 掩码读成二维类别图；如果文件被错误保存成 RGB，也会在这里自动压回单通道。
        mask = self._load_mask_array(mask_path)

        # 训练阶段只做最基础的水平翻转，保证图像与标签严格同步变换。
        if self.training and random.random() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # 把 HWC 的 uint8 图像转换为 CHW 的 float32，并缩放到 [0, 1]。
        image = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1) / 255.0
        image = (image - self.mean) / self.std

        # 标签原始值为 0/80/160/240，整除后映射到连续类别编号 0~3。
        # 这里不做 one-hot，保持 CrossEntropyLoss 期望的整型类别索引格式。
        mask = torch.from_numpy(mask // self.mask_divisor)
        return image, mask
