"""分割子模块的统一导出入口。"""

from .datasets import (
    PngInferenceDataset,
    PngSegmentationDataset,
    TifInferenceDataset,
    TifSegmentationDataset,
    collect_png_files,
    collect_tif_files,
    segmentation_collate_fn,
)
from .metrics import CombinedSegmentationLoss, compute_metrics, evaluate_model
from .model import (
    ShallowDinoSegmentor,
    build_shallow_dinov3_segmentor,
    load_shallow_dinov3_segmentor_from_checkpoint,
    parse_layer_indices,
)

__all__ = [
    "PngInferenceDataset",
    "PngSegmentationDataset",
    "TifInferenceDataset",
    "TifSegmentationDataset",
    "collect_png_files",
    "collect_tif_files",
    "segmentation_collate_fn",
    "ShallowDinoSegmentor",
    "build_shallow_dinov3_segmentor",
    "load_shallow_dinov3_segmentor_from_checkpoint",
    "parse_layer_indices",
    "CombinedSegmentationLoss",
    "compute_metrics",
    "evaluate_model",
]