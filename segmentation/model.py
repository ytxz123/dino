from __future__ import annotations

"""浅层 DINOv3 分割模型定义。

这里实现的是一个面向固定 512x512 PNG 分割任务的轻量结构：
1. 用 DINOv3 backbone 提取浅层 ViT 特征。
2. 用一个小型 CNN 细节分支补偿边缘和纹理信息。
3. 用逐级上采样解码器恢复到像素级输出。

整体目标不是追求最复杂的结构，而是尽量在可读和实用之间取得平衡。
"""

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov3.hub import backbones as dinov3_backbones


def resolve_backbone_weights(weights: str | dinov3_backbones.Weights | None):
    """把字符串形式的权重别名解析成 hub/backbones 中的枚举值。"""
    if weights is None or isinstance(weights, dinov3_backbones.Weights):
        return weights
    return {
        "LVD1689M": dinov3_backbones.Weights.LVD1689M,
        "LVD-1689M": dinov3_backbones.Weights.LVD1689M,
        "SAT493M": dinov3_backbones.Weights.SAT493M,
        "SAT-493M": dinov3_backbones.Weights.SAT493M,
    }.get(str(weights).strip().upper(), weights)


def parse_layer_indices(layer_indices: str | Sequence[int] | None) -> list[int] | None:
    """解析浅层 block 编号配置。"""
    # 支持从命令行传入 "1,2,4,5"，也支持直接传入整数序列。
    if layer_indices is None:
        return None
    if isinstance(layer_indices, str):
        cleaned = [item.strip() for item in layer_indices.split(",") if item.strip()]
        return [int(item) for item in cleaned]
    return [int(item) for item in layer_indices]


def _normalize_detail_dims(detail_dims: str | Sequence[int] | None, default: Sequence[int] = (64, 128, 256)) -> tuple[int, int, int]:
    """把细节分支通道配置统一转换为三元组。"""
    if detail_dims is None:
        return tuple(int(value) for value in default)
    if isinstance(detail_dims, str):
        return tuple(int(item.strip()) for item in detail_dims.split(",") if item.strip())
    return tuple(int(value) for value in detail_dims)


def resolve_shallow_layer_indices(n_blocks: int, requested: Sequence[int] | None = None, num_layers: int = 4) -> list[int]:
    """决定本次训练实际使用哪些浅层 block。"""
    # 如果用户明确指定了层号，直接使用。
    # 否则默认在 backbone 前半段均匀抽取若干层，强调“浅层特征”。
    if requested is not None:
        return sorted(set(int(index) for index in requested))

    shallow_end = max(2, n_blocks // 2)
    shallow_start = 1 if shallow_end > 1 else 0
    positions = torch.linspace(shallow_start, shallow_end - 1, steps=min(num_layers, shallow_end - shallow_start))
    indices = sorted(set(int(round(position.item())) for position in positions))
    if not indices:
        return [0]
    return indices


def _make_norm(num_channels: int) -> nn.GroupNorm:
    """为给定通道数选择一个可整除的 GroupNorm 配置。"""
    # 使用 GroupNorm 而不是 BatchNorm，避免小 batch 训练时统计量不稳定。
    for groups in (32, 16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class ConvNormAct(nn.Sequential):
    # 基础卷积块：Conv -> Norm -> GELU。
    # 这个模块在细节分支和解码分支里重复出现，用来保持实现一致。
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int | None = None):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            _make_norm(out_channels),
            nn.GELU(),
        )


class ResidualConvBlock(nn.Module):
    # 轻量残差块，用于在不大幅增加参数的情况下增强局部建模能力。
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size=3),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            _make_norm(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + x)


class DetailStem(nn.Module):
    """从原始图像中提取高分辨率细节特征。

    DINOv3 的 patch token 更擅长语义表示，但边界细节通常会被 patch 化过程抹平。
    因此这里额外保留一个 CNN 细节分支，用于提供低层纹理和轮廓信息。
    """

    def __init__(self, dims: Sequence[int] = (64, 128, 256)):
        super().__init__()
        dim2, dim4, dim8 = dims
        self.stage2 = nn.Sequential(
            ConvNormAct(3, dim2, stride=2),
            ConvNormAct(dim2, dim2),
        )
        self.stage4 = nn.Sequential(
            ConvNormAct(dim2, dim4, stride=2),
            ConvNormAct(dim4, dim4),
        )
        self.stage8 = nn.Sequential(
            ConvNormAct(dim4, dim8, stride=2),
            ConvNormAct(dim8, dim8),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 分别输出 1/2、1/4、1/8 三个尺度的细节特征。
        feat2 = self.stage2(x)
        feat4 = self.stage4(feat2)
        feat8 = self.stage8(feat4)
        return feat2, feat4, feat8


class ShallowFeatureFusion(nn.Module):
    """融合多个 DINOv3 浅层特征图。

    输入的多层特征都来自同一个 patch 尺度，但语义深度不同。
    先对每一层做通道投影，再拼接融合，得到统一的浅层语义表示。
    """

    def __init__(self, in_channels: int, out_channels: int, num_layers: int):
        super().__init__()
        self.projections = nn.ModuleList([ConvNormAct(in_channels, out_channels, kernel_size=1, padding=0) for _ in range(num_layers)])
        self.fusion = nn.Sequential(
            ConvNormAct(out_channels * num_layers, out_channels),
            ResidualConvBlock(out_channels),
            ResidualConvBlock(out_channels),
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        # 将不同 block 的特征对齐到同一通道维度后再做拼接融合。
        projected = [projection(feature) for projection, feature in zip(self.projections, features)]
        return self.fusion(torch.cat(projected, dim=1))


class UpFusionBlock(nn.Module):
    # 逐级上采样模块。
    # 如果存在 skip feature，则将当前语义特征与高分辨率细节特征融合。
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int = 0):
        super().__init__()
        self.up = ConvNormAct(in_channels, out_channels)
        self.skip_proj = ConvNormAct(skip_channels, out_channels, kernel_size=1, padding=0) if skip_channels > 0 else None
        fusion_in_channels = out_channels * 2 if skip_channels > 0 else out_channels
        self.refine = nn.Sequential(
            ConvNormAct(fusion_in_channels, out_channels),
            ResidualConvBlock(out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        # 这里固定使用双线性插值上采样，兼顾平滑性和实现简洁性。
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up(x)
        if self.skip_proj is not None and skip is not None:
            x = torch.cat([x, self.skip_proj(skip)], dim=1)
        return self.refine(x)


class DINOv3ShallowFeatureExtractor(nn.Module):
    """对 DINOv3 backbone 的一层轻封装。

    主要作用：
    1. 按指定 block 抽取中间层。
    2. 在冻结 backbone 时自动关闭梯度。
    3. 输出 reshape 后的 2D 特征图，方便接分割 decoder。
    """

    def __init__(self, backbone: nn.Module, layer_indices: Sequence[int], freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        self.layer_indices = list(layer_indices)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # reshape=True 会把 patch token 重新整理成 [B, C, H, W] 的二维特征图。
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone.get_intermediate_layers(x, n=self.layer_indices, reshape=True, return_class_token=False)
        else:
            features = self.backbone.get_intermediate_layers(x, n=self.layer_indices, reshape=True, return_class_token=False)
        return list(features)


class ShallowDinoSegmentor(nn.Module):
    """浅层 DINOv3 特征驱动的语义分割网络。

    网络由三部分组成：
    1. DINOv3 浅层特征提取器：负责提供 patch 级语义表示。
    2. 细节分支：负责保留边缘、纹理和高分辨率空间信息。
    3. 解码分支：逐级上采样并融合细节特征，输出像素级类别图。
    """

    def __init__(
        self,
        backbone: nn.Module,
        layer_indices: Sequence[int],
        num_classes: int,
        decoder_dim: int = 256,
        detail_dims: Sequence[int] = (64, 128, 256),
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        input_size: Sequence[int] = (512, 512),
    ):
        super().__init__()
        self.patch_size = int(backbone.patch_size)
        self.input_size = tuple(int(value) for value in input_size)
        if len(self.input_size) != 2:
            raise ValueError(f"Expected 2D input size, got {self.input_size}")
        if any(size % self.patch_size != 0 for size in self.input_size):
            raise ValueError(f"Input size {self.input_size} must be divisible by patch size {self.patch_size}")
        self.layer_indices = list(layer_indices)
        self.freeze_backbone = freeze_backbone
        self.extractor = DINOv3ShallowFeatureExtractor(backbone, layer_indices, freeze_backbone=freeze_backbone)
        self.detail_stem = DetailStem(detail_dims)
        self.shallow_fusion = ShallowFeatureFusion(backbone.embed_dim, decoder_dim, len(self.layer_indices))
        self.decode8 = UpFusionBlock(decoder_dim, detail_dims[2], skip_channels=detail_dims[2])
        self.decode4 = UpFusionBlock(detail_dims[2], detail_dims[1], skip_channels=detail_dims[1])
        self.decode2 = UpFusionBlock(detail_dims[1], detail_dims[0], skip_channels=detail_dims[0])
        self.decode1 = UpFusionBlock(detail_dims[0], detail_dims[0] // 2, skip_channels=0)
        self.head = nn.Sequential(
            nn.Dropout2d(dropout),
            ConvNormAct(detail_dims[0] // 2, detail_dims[0] // 2),
            nn.Conv2d(detail_dims[0] // 2, num_classes, kernel_size=1),
        )

    def train(self, mode: bool = True):
        # 如果 backbone 是冻结的，即使外部调用 model.train()，
        # 也强制让 backbone 维持在 eval 状态，避免内部归一化或 dropout 发生变化。
        super().train(mode)
        if self.freeze_backbone:
            self.extractor.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[-2:]) != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {tuple(x.shape[-2:])}")
        detail2, detail4, detail8 = self.detail_stem(x)
        shallow_features = self.extractor(x)
        fused = self.shallow_fusion(shallow_features)
        decoded = self.decode8(fused, detail8)
        decoded = self.decode4(decoded, detail4)
        decoded = self.decode2(decoded, detail2)
        decoded = self.decode1(decoded)
        return self.head(decoded)


def build_dinov3_backbone(backbone_name: str, backbone_weights: str | None = None, pretrained: bool = True) -> nn.Module:
    """通过 hub/backbones.py 中的构造器实例化 DINOv3 backbone。"""
    builder = getattr(dinov3_backbones, backbone_name)
    resolved_weights = resolve_backbone_weights(backbone_weights)
    builder_kwargs = {"pretrained": pretrained or resolved_weights is not None}
    if resolved_weights is not None:
        builder_kwargs["weights"] = resolved_weights
    return builder(**builder_kwargs)


def build_shallow_dinov3_segmentor(
    backbone_name: str,
    num_classes: int,
    backbone_weights: str | None = None,
    pretrained: bool = True,
    layer_indices: Sequence[int] | None = None,
    num_shallow_layers: int = 4,
    decoder_dim: int = 256,
    detail_dims: Sequence[int] = (64, 128, 256),
    dropout: float = 0.1,
    freeze_backbone: bool = True,
    input_size: Sequence[int] = (512, 512),
) -> ShallowDinoSegmentor:
    """构建完整的浅层 DINOv3 分割模型。"""
    # 先构建 backbone，再根据总 block 数自动解析浅层索引，最后拼装完整分割网络。
    backbone = build_dinov3_backbone(backbone_name, backbone_weights=backbone_weights, pretrained=pretrained)
    selected_indices = resolve_shallow_layer_indices(backbone.n_blocks, requested=layer_indices, num_layers=num_shallow_layers)
    return ShallowDinoSegmentor(
        backbone=backbone,
        layer_indices=selected_indices,
        num_classes=num_classes,
        decoder_dim=decoder_dim,
        detail_dims=detail_dims,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        input_size=input_size,
    )


def load_shallow_dinov3_segmentor_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    backbone_name: str | None = None,
    backbone_weights: str | None = None,
    pretrained: bool | None = None,
    layer_indices: str | Sequence[int] | None = None,
    num_shallow_layers: int | None = None,
    num_classes: int | None = None,
    decoder_dim: int | None = None,
    detail_dims: str | Sequence[int] | None = None,
    dropout: float | None = None,
    freeze_backbone: bool | None = None,
    input_size: Sequence[int] | None = None,
):
    """从 checkpoint 恢复分割模型，并返回模型与原始 checkpoint。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_config = checkpoint.get("model_config", {})
    resolved_layer_indices = (
        parse_layer_indices(layer_indices)
        if layer_indices is not None
        else saved_config.get("resolved_layer_indices", saved_config.get("layer_indices"))
    )
    resolved_model_config = {
        "backbone_name": backbone_name or saved_config["backbone_name"],
        "backbone_weights": backbone_weights or saved_config.get("backbone_weights"),
        "pretrained": pretrained if pretrained is not None else saved_config.get("pretrained", True),
        "layer_indices": resolved_layer_indices,
        "num_shallow_layers": num_shallow_layers or saved_config.get("num_shallow_layers", 4),
        "num_classes": num_classes or saved_config["num_classes"],
        "decoder_dim": decoder_dim or saved_config.get("decoder_dim", 256),
        "detail_dims": _normalize_detail_dims(detail_dims, saved_config.get("detail_dims", (64, 128, 256))),
        "dropout": dropout if dropout is not None else saved_config.get("dropout", 0.1),
        "freeze_backbone": freeze_backbone if freeze_backbone is not None else saved_config.get("freeze_backbone", True),
        "input_size": tuple(input_size) if input_size is not None else tuple(saved_config.get("input_size", (512, 512))),
    }
    model = build_shallow_dinov3_segmentor(**resolved_model_config)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint