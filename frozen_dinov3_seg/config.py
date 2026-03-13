from dataclasses import dataclass, field


@dataclass
class PathConfig:
    # 数据路径
    train_images_dir: str = "data/seg/train/images"
    train_masks_dir: str = "data/seg/train/masks"
    val_images_dir: str = "data/seg/val/images"
    val_masks_dir: str = "data/seg/val/masks"

    # 输出目录
    work_dir: str = "outputs/frozen_dinov3_vitl16_seg"

    # 预训练骨干权重。
    # 可选值：
    # 1. "LVD1689M" 或 "SAT493M"
    # 2. 本地 .pth 权重路径
    # 3. 远程 URL
    backbone_weights: str = "LVD1689M"

    # 可选：继续训练或纯评估时加载的分割头 checkpoint。
    resume_checkpoint: str = ""
    eval_checkpoint: str = ""


@dataclass
class DatasetConfig:
    image_size: int = 512
    num_classes: int = 4

    # 真值 png 是把类别索引乘以 80 存进去的：0, 80, 160, 240。
    mask_values: tuple[int, ...] = (0, 80, 160, 240)
    class_names: tuple[str, ...] = ("class_0", "class_1", "class_2", "class_3")

    # DINOv3 视觉骨干常用的归一化统计量。
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # 简单且足够稳定的数据增强。
    hflip_prob: float = 0.5
    vflip_prob: float = 0.0


@dataclass
class ModelConfig:
    backbone_name: str = "dinov3_vitl16"

    # 这里使用 block 下标，0-based。
    # 例如 [5, 11, 17, 23] 表示融合第 6、12、18、24 个 block 的输出。
    selected_layers: list[int] = field(default_factory=lambda: [5, 11, 17, 23])

    # ViT-L/16 主干输出通道。
    feature_dim: int = 1024

    # 适中大小的分割头配置，兼顾 16G 显存和表达能力。
    projector_dim: int = 128
    decoder_dim: int = 256
    dropout: float = 0.1

    # 是否校验公开权重哈希。
    check_hash: bool = False


@dataclass
class TrainConfig:
    run_mode: str = "train"  # train 或 eval
    seed: int = 42
    device: str = "cuda"

    # 16G 显存下的保守默认值。
    batch_size: int = 4
    val_batch_size: int = 8
    grad_accum_steps: int = 2
    num_workers: int = 8
    max_epochs: int = 30

    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    grad_clip_norm: float = 1.0

    use_amp: bool = True
    amp_dtype: str = "float16"  # float16 或 bfloat16
    use_channels_last: bool = True

    log_every: int = 50
    eval_every: int = 1
    save_every: int = 1


@dataclass
class ExperimentConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def get_config() -> ExperimentConfig:
    return ExperimentConfig()
