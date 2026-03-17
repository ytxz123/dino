from contextlib import nullcontext
from functools import partial

import torch

from dinov3.eval.segmentation.models import BackboneLayersSet
from dinov3.eval.segmentation.models.heads.linear_head import LinearHead
from dinov3.eval.utils import ModelWithIntermediateLayers
from dinov3.hub.backbones import dinov3_vitl16

from three_class_segmentation.config import ThreeClassSegConfig


BACKBONE_LAYER_MAP = {layer.name: layer for layer in BackboneLayersSet}
AUTOCAST_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _get_backbone_out_indices(model: torch.nn.Module, backbone_out_layers: BackboneLayersSet) -> list[int]:
    n_blocks = getattr(model, "n_blocks", 1)
    if backbone_out_layers == BackboneLayersSet.LAST:
        out_indices = [n_blocks - 1]
    elif backbone_out_layers == BackboneLayersSet.FOUR_LAST:
        out_indices = [index for index in range(n_blocks - 4, n_blocks)]
    else:
        out_indices = [4, 11, 17, 23] if n_blocks == 24 else [i * (n_blocks // 4) - 1 for i in range(1, 5)]
    return out_indices


class FrozenBackboneLinearSegmentor(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        *,
        out_layers: BackboneLayersSet,
        num_classes: int,
        dropout: float,
        use_batchnorm: bool,
        autocast_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        out_indices = _get_backbone_out_indices(backbone, out_layers)
        backbone.requires_grad_(False)
        self.autocast_ctx = self._build_autocast_ctx(autocast_dtype)
        self.backbone = ModelWithIntermediateLayers(
            backbone,
            n=out_indices,
            autocast_ctx=self.autocast_ctx,
            reshape=True,
            return_class_token=False,
        )
        self.head = LinearHead(
            in_channels=[backbone.embed_dim] * len(out_indices),
            n_output_channels=num_classes,
            use_batchnorm=use_batchnorm,
            use_cls_token=False,
            dropout=dropout,
        )

    @staticmethod
    def _build_autocast_ctx(autocast_dtype: torch.dtype):
        if torch.cuda.is_available():
            return partial(torch.autocast, device_type="cuda", enabled=True, dtype=autocast_dtype)
        return partial(nullcontext)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        # 骨干始终只做特征提取，训练时只更新线性分割头。
        with self.autocast_ctx():
            return self.head(features)

    def predict(self, images: torch.Tensor, rescale_to: tuple[int, int]) -> torch.Tensor:
        with torch.inference_mode():
            features = self.backbone(images)
            with self.autocast_ctx():
                return self.head.predict(features, rescale_to=rescale_to)


def build_model(config: ThreeClassSegConfig) -> FrozenBackboneLinearSegmentor:
    backbone = dinov3_vitl16(
        pretrained=True,
        weights=config.backbone.weights_path,
        check_hash=config.backbone.check_hash,
    )
    return FrozenBackboneLinearSegmentor(
        backbone,
        out_layers=BACKBONE_LAYER_MAP[config.backbone.out_layers],
        num_classes=config.head.num_classes,
        dropout=config.head.dropout,
        use_batchnorm=config.head.use_batchnorm,
        autocast_dtype=AUTOCAST_DTYPE_MAP[config.backbone.autocast_dtype],
    )


def load_checkpoint(model: FrozenBackboneLinearSegmentor, checkpoint_path: str, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.head.load_state_dict(checkpoint["head"])
    return checkpoint