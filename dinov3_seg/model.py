from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from dinov3.checkpointer import init_model_from_checkpoint_for_evals
from dinov3.hub.backbones import Weights, dinov3_vitl16


def make_group_norm(num_channels: int, max_groups: int) -> nn.GroupNorm:
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, norm_groups: int):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            make_group_norm(out_channels, norm_groups),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_groups: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = make_group_norm(out_channels, norm_groups)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class MediumSegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        fusion_dim: int,
        decoder_channels: tuple[int, ...],
        norm_groups: int,
        num_classes: int,
    ):
        super().__init__()
        self.projections = nn.ModuleList(
            [ConvNormAct(in_channels, fusion_dim, kernel_size=1, norm_groups=norm_groups) for _ in range(num_layers)]
        )
        self.fuse = ConvNormAct(fusion_dim * num_layers, decoder_channels[0], kernel_size=3, norm_groups=norm_groups)
        self.up_blocks = nn.ModuleList(
            [
                UpsampleBlock(decoder_channels[i], decoder_channels[i + 1], norm_groups)
                for i in range(len(decoder_channels) - 1)
            ]
        )
        self.classifier = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, features: tuple[torch.Tensor, ...], output_size: tuple[int, int]) -> torch.Tensor:
        # 多层输入先做同维度投影，再在通道维拼接，避免直接拼接 1024 维特征造成解码头过重。
        projected = [proj(feature) for proj, feature in zip(self.projections, features)]
        x = self.fuse(torch.cat(projected, dim=1))
        for block in self.up_blocks:
            x = block(x)
        x = self.classifier(x)
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)


def build_frozen_vitl16(weights_path: str | Path, checkpoint_key: str | None, backbone_profile: str) -> nn.Module:
    profile = Weights[backbone_profile.upper()]
    backbone = dinov3_vitl16(pretrained=False, weights=profile)
    init_model_from_checkpoint_for_evals(backbone, pretrained_weights=weights_path, checkpoint_key=checkpoint_key)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return backbone


class FrozenDinoV3Segmenter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.feature_layers = tuple(sorted(cfg.model.feature_layers))
        self.backbone = build_frozen_vitl16(
            weights_path=cfg.paths.backbone_weights,
            checkpoint_key=cfg.paths.checkpoint_key,
            backbone_profile=cfg.model.backbone_profile,
        )
        self.head = MediumSegmentationHead(
            in_channels=self.backbone.embed_dim,
            num_layers=len(self.feature_layers),
            fusion_dim=cfg.model.fusion_dim,
            decoder_channels=cfg.model.decoder_channels,
            norm_groups=cfg.model.norm_groups,
            num_classes=cfg.data.num_classes,
        )
        self.use_normed_features = cfg.model.use_normed_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # 骨干完全冻结，因此特征提取阶段不保留梯度，显著降低 512x512 训练显存。
        with torch.no_grad():
            features = self.backbone.get_intermediate_layers(
                images,
                n=self.feature_layers,
                reshape=True,
                norm=self.use_normed_features,
            )
        return self.head(features, output_size=images.shape[-2:])
