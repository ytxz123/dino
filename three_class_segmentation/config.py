import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


DEFAULT_MEAN = tuple(float(value) for value in IMAGENET_DEFAULT_MEAN)
DEFAULT_STD = tuple(float(value) for value in IMAGENET_DEFAULT_STD)


@dataclass
class DatasetConfig:
    train_images: str = "data/train/images"
    train_masks: str = "data/train/masks"
    val_images: str = "data/val/images"
    val_masks: str = "data/val/masks"
    image_size: int = 512
    label_divisor: int = 50
    mean: tuple[float, float, float] = DEFAULT_MEAN
    std: tuple[float, float, float] = DEFAULT_STD


@dataclass
class BackboneConfig:
    weights_path: str = "checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
    out_layers: str = "LAST"
    autocast_dtype: str = "bfloat16"
    check_hash: bool = False


@dataclass
class HeadConfig:
    num_classes: int = 4
    dropout: float = 0.1
    use_batchnorm: bool = False


@dataclass
class OptimizerConfig:
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.01
    grad_clip: float = 1.0


@dataclass
class SchedulerConfig:
    name: str = "WarmupOneCycleLR"
    warmup_iters: int = 500
    pct_start: float = 0.1
    div_factor: float = 25.0
    final_div_factor: float = 1000.0


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 4
    accumulation_steps: int = 4
    num_workers: int = 4
    hflip_prob: float = 0.5
    ce_weight: float = 1.0
    dice_weight: float = 0.0
    class_weight: tuple[float, ...] | None = None
    eval_interval: int = 1
    save_every: int = 1
    log_interval: int = 50
    resume_from: str = ""


@dataclass
class EvalConfig:
    batch_size: int = 4
    num_workers: int = 4
    checkpoint_path: str = ""


@dataclass
class RuntimeConfig:
    output_dir: str = "outputs/three_class_segmentation"
    device: str = "cuda"
    seed: int = 3407
    pin_memory: bool = True


@dataclass
class ThreeClassSegConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


DEFAULT_CONFIG = ThreeClassSegConfig()


def load_config() -> ThreeClassSegConfig:
    return deepcopy(DEFAULT_CONFIG)


def save_config(config: ThreeClassSegConfig, config_path: str) -> None:
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2, ensure_ascii=False)
        file.write("\n")