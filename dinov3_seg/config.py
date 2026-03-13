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

    # 训练集图像目录，默认约定为 PNG 文件。
    train_image_dir: Path = ROOT / "data/train/images"
    # 训练集标签目录，要求与图像文件同名一一对应。
    train_mask_dir: Path = ROOT / "data/train/masks"
    # 验证集图像目录，正式实验建议与训练集严格分离。
    val_image_dir: Path = ROOT / "data/val/images"
    # 验证集标签目录。
    val_mask_dir: Path = ROOT / "data/val/masks"
    # DINOv3 骨干本地权重路径；训练与评估都会从这里恢复冻结 backbone。
    backbone_weights: Path = ROOT / "weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
    # 分割头输出目录，训练过程中的 checkpoint 都会写到这里。
    output_dir: Path = ROOT / "outputs/vitl16_seg"
    # 评估脚本默认读取的分割头权重路径。
    eval_checkpoint: Path = ROOT / "outputs/vitl16_seg/best_head.pth"
    # 某些 checkpoint 会把权重放在特定 key 下；没有时设为 None 直接读取顶层。
    checkpoint_key: str | None = None


@dataclass
class DataConfig:
    """管理输入尺寸、类别数和预处理超参数。"""

    # 这里保留输入尺寸配置，便于后续扩展到显式 resize 或断言检查。
    image_size: int = 512
    # 当前分割任务的类别数；需要与标签映射结果保持一致。
    num_classes: int = 4
    # 标签原始像素值映射时使用的除数，例如 0/80/160/240 -> 0/1/2/3。
    mask_divisor: int = 80
    # 图像归一化均值，通常沿用 ImageNet 预训练配置。
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    # 图像归一化标准差。
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    # 训练阶段随机水平翻转概率；单图 smoke test 时可改成 0。
    train_hflip_prob: float = 0.5
    # DataLoader 工作进程数；Windows、调试或极小数据集场景可降为 0。
    num_workers: int = 8
    # 打开后可配合 non_blocking=True 减少主机到设备的传输等待。
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """管理骨干提取层和分割头规模。"""

    # 与本地骨干权重匹配的 profile，决定某些结构细节的构造方式。
    backbone_profile: str = "SAT493M"
    # 从 ViT 的哪些 block 抽取中间层特征；索引从 0 开始。
    feature_layers: tuple[int, ...] = (11, 17, 23)
    # 多层特征先统一投影到这个通道数，再进行拼接融合。
    fusion_dim: int = 256
    # 解码头逐级上采样时每一层的通道宽度。
    decoder_channels: tuple[int, ...] = (256, 192, 128, 96, 64)
    # GroupNorm 的最大分组数；实际会自动向下寻找可整除的值。
    norm_groups: int = 16
    # 是否使用 backbone 内部归一化后的中间特征。
    use_normed_features: bool = True


@dataclass
class TrainConfig:
    """管理训练过程相关的优化和日志参数。"""

    # 所有随机源共享同一随机种子，便于复现实验。
    seed: int = 3407
    # 训练总轮数；单图 smoke test 可以显著减小。
    epochs: int = 40
    # 每个 batch 的样本数；实际训练时会自动截断到不超过数据集大小。
    batch_size: int = 4
    # 验证时的 batch 大小，通常可以比训练更大一些。
    eval_batch_size: int = 6
    # 梯度累积步数，用于在显存受限时模拟更大的有效 batch。
    grad_accum_steps: int = 2
    # 为 True 时，如果最后一个 batch 不满则丢弃；极小数据集时通常不建议开启。
    drop_last: bool = True
    # 验证集为空时是否回退成“用训练集充当验证集”；只建议 smoke test 使用。
    allow_train_as_val_when_val_empty: bool = False
    # AdamW 初始学习率。
    lr: float = 3e-4
    # 余弦退火结束时衰减到的最小学习率。
    min_lr: float = 1e-5
    # warmup 轮数，可以写成小数，内部会按 step 数换算。
    warmup_epochs: float = 1.0
    # AdamW 权重衰减系数。
    weight_decay: float = 1e-4
    # 梯度裁剪阈值，避免少量 batch 上梯度爆炸。
    grad_clip: float = 1.0
    # 自动混合精度类型；CUDA 支持时推荐 bf16 或 fp16。
    amp_dtype: str = "bf16"
    # 每隔多少个训练 step 打一条日志。
    log_interval: int = 50
    # 每隔多少个 epoch 执行一次验证。
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

    常见用法是：
    1. 在训练或评估脚本中先取一份默认配置。
    2. 按当前实验临时覆盖少量字段。
    3. 把修改后的配置对象继续传给模型、数据集与流程函数。
    """

    return ExperimentConfig()
