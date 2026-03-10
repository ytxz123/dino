"""分割子模块的统一导出入口。

这个文件的作用不是实现功能，而是把训练、评估和推理时最常用的
类与函数集中暴露出去。这样外部脚本只需要从 dinov3.segmentation
 导入一次，就能拿到需要的公共接口。
"""

from .datasets import TifInferenceDataset, TifSegmentationDataset, collect_tif_files, segmentation_collate_fn
from .metrics import CombinedSegmentationLoss, compute_metrics, evaluate_model
from .model import (
    ShallowDinoSegmentor,
    build_shallow_dinov3_segmentor,
    load_shallow_dinov3_segmentor_from_checkpoint,
    parse_layer_indices,
)

__all__ = [
    # 数据读取与批处理工具。
    "TifInferenceDataset",
    "TifSegmentationDataset",
    "collect_tif_files",
    "segmentation_collate_fn",
    # 模型构建与恢复工具。
    "ShallowDinoSegmentor",
    "build_shallow_dinov3_segmentor",
    "load_shallow_dinov3_segmentor_from_checkpoint",
    "parse_layer_indices",
    # 训练与评估公共函数。
    "CombinedSegmentationLoss",
    "compute_metrics",
    "evaluate_model",
]