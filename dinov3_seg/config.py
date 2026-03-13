"""语义分割实验的集中配置定义。

这个文件只负责声明“实验需要哪些配置”，不承担训练逻辑。
这样训练、评估、推理脚本都能共享同一套默认参数，避免多处硬编码。
"""

from dataclasses import dataclass, field
from pathlib import Path


# 以当前文件所在目录作为所有默认相对路径的锚点，便于整个子项目独立搬迁。
ROOT = Path(__file__).resolve().parent


@dataclass
class PathConfig:
    """管理数据、权重和输出文件的位置。"""

    train_image_dir: Path = ROOT / "data/train/images"
    train_mask_dir: Path = ROOT / "data/train/masks"
    val_image_dir: Path = ROOT / "data/val/images"
    val_mask_dir: Path = ROOT / "data/val/masks"
    backbone_weights: Path = ROOT / "weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
    output_dir: Path = ROOT / "outputs/vitl16_seg"
    eval_checkpoint: Path = ROOT / "outputs/vitl16_seg/best_head.pth"
    checkpoint_key: str | None = None


@dataclass
class DataConfig:
    """管理输入尺寸、类别数和预处理超参数。"""

    image_size: int = 512
    num_classes: int = 4
    mask_divisor: int = 80
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    train_hflip_prob: float = 0.5
    num_workers: int = 8
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """管理骨干提取层和分割头规模。"""

    backbone_profile: str = "SAT493M"
    feature_layers: tuple[int, ...] = (11, 17, 23)
    fusion_dim: int = 256
    decoder_channels: tuple[int, ...] = (256, 192, 128, 96, 64)
    norm_groups: int = 16
    use_normed_features: bool = True


@dataclass
class TrainConfig:
    """管理训练过程相关的优化和日志参数。"""

    seed: int = 3407
    epochs: int = 40
    batch_size: int = 4
    eval_batch_size: int = 6
    grad_accum_steps: int = 2
    lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_epochs: float = 1.0
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp_dtype: str = "bf16"
    log_interval: int = 50
    eval_interval: int = 1


@dataclass
class ExperimentConfig:
    """把路径、数据、模型和训练配置聚合成一个统一对象。"""

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def get_config() -> ExperimentConfig:
    """返回一份新的默认实验配置。

    这里不做全局单例，目的是让调用方可以安全地修改返回值，
    而不会污染其它脚本中的默认配置。
    """

    return ExperimentConfig()
