from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from dinov3.hub.backbones import Weights, dinov3_vitl16


def _make_group_norm(num_channels: int) -> nn.GroupNorm:
    for groups in (32, 16, 8, 4, 2):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


def _resolve_backbone_weights(weights_spec: str):
    if weights_spec in Weights.__members__:
        return Weights[weights_spec]
    return weights_spec


class ConvGNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            _make_group_norm(out_channels),
            nn.GELU(),
        )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            _make_group_norm(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            _make_group_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = ConvGNAct(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.proj(x)


class LiteSegHead(nn.Module):
    """中等大小的多层特征融合分割头。"""

    def __init__(
        self,
        feature_dim: int,
        num_layers: int,
        projector_dim: int,
        decoder_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [ConvGNAct(feature_dim, projector_dim, kernel_size=1, padding=0) for _ in range(num_layers)]
        )
        self.fuse = ConvGNAct(projector_dim * num_layers, decoder_dim, kernel_size=1, padding=0)
        self.refine_32 = DepthwiseSeparableBlock(decoder_dim, decoder_dim)
        self.up_64 = UpsampleBlock(decoder_dim, decoder_dim // 2)
        self.refine_64 = DepthwiseSeparableBlock(decoder_dim // 2, decoder_dim // 2)
        self.up_128 = UpsampleBlock(decoder_dim // 2, decoder_dim // 4)
        self.refine_128 = DepthwiseSeparableBlock(decoder_dim // 4, decoder_dim // 4)
        self.classifier = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(decoder_dim // 4, num_classes, kernel_size=1),
        )

    def forward(self, features: Iterable[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        x = self.fuse(torch.cat(projected, dim=1))
        x = self.refine_32(x)
        x = self.refine_64(self.up_64(x))
        x = self.refine_128(self.up_128(x))
        x = self.classifier(x)
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)


class FrozenDinoV3Segmenter(nn.Module):
    """冻结 DINOv3 ViT-L/16，仅训练分割头。"""

    def __init__(self, config) -> None:
        super().__init__()
        self.selected_layers = tuple(config.model.selected_layers)

        self.backbone = dinov3_vitl16(
            pretrained=True,
            weights=_resolve_backbone_weights(config.paths.backbone_weights),
            check_hash=config.model.check_hash,
        )
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self.decoder = LiteSegHead(
            feature_dim=config.model.feature_dim,
            num_layers=len(self.selected_layers),
            projector_dim=config.model.projector_dim,
            decoder_dim=config.model.decoder_dim,
            num_classes=config.dataset.num_classes,
            dropout=config.model.dropout,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output_size = images.shape[-2:]
        with torch.no_grad():
            features = self.backbone.get_intermediate_layers(
                images,
                n=self.selected_layers,
                reshape=True,
                return_class_token=False,
                return_extra_tokens=False,
                norm=True,
            )
        return self.decoder(features, output_size=output_size)
