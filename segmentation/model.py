import re
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from dinov3.hub import backbones


CHECKPOINTER_DIR = Path(__file__).resolve().parents[1] / "checkpointer"
BACKBONE_NAMES = (
    "dinov3_vits16",
    "dinov3_vits16plus",
    "dinov3_vitb16",
    "dinov3_vitl16",
    "dinov3_vitl16plus",
    "dinov3_vith16plus",
    "dinov3_vit7b16",
)


def resolve_weights_path(weights: str | None) -> str | None:
    if not weights:
        return None
    path = Path(weights).expanduser()
    if not path.is_absolute():
        path = CHECKPOINTER_DIR / path
    return str(path.resolve())


def get_weights_tag(weights: str | None) -> str:
    if not weights:
        return "no_weights"
    resolved = Path(resolve_weights_path(weights) or weights)
    tag = resolved.stem or resolved.name
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", tag)
    return tag


def parse_layers(layers: Iterable[int]) -> list[int]:
    return [int(layer) - 1 for layer in layers]


def build_dinov3_backbone(backbone_name: str, weights: str | None):
    backbone_fn = getattr(backbones, backbone_name)
    kwargs = {"pretrained": True, "check_hash": False}
    resolved_weights = resolve_weights_path(weights)
    if resolved_weights is not None:
        kwargs["weights"] = resolved_weights
    return backbone_fn(**kwargs)


def make_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(32, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            make_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.block(x)


class FrozenDinov3Backbone(nn.Module):
    def __init__(self, backbone_name: str, weights: str | None, layers: Sequence[int]):
        super().__init__()
        self.backbone_name = backbone_name
        self.weights = weights
        self.layers = list(layers)
        self.layer_ids = parse_layers(layers)
        self.backbone = build_dinov3_backbone(backbone_name, weights)
        self.out_channels = self.backbone.embed_dim
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        with torch.no_grad():
            if x.is_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    features = self.backbone.get_intermediate_layers(x, n=self.layer_ids, reshape=True)
            else:
                features = self.backbone.get_intermediate_layers(x, n=self.layer_ids, reshape=True)
        return [feature.float() for feature in features]


class LightweightSegmentationDecoder(nn.Module):
    def __init__(self, in_channels: int, feature_count: int, num_classes: int, decoder_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, decoder_dim, kernel_size=1) for _ in range(feature_count)])
        self.fuse = ConvBlock(decoder_dim * feature_count, decoder_dim)
        self.context = nn.Sequential(ConvBlock(decoder_dim, decoder_dim), nn.Dropout2d(dropout))
        self.up1 = UpBlock(decoder_dim, decoder_dim)
        self.up2 = UpBlock(decoder_dim, decoder_dim // 2)
        self.up3 = UpBlock(decoder_dim // 2, decoder_dim // 4)
        self.up4 = UpBlock(decoder_dim // 4, decoder_dim // 8)
        self.head = nn.Conv2d(decoder_dim // 8, num_classes, kernel_size=1)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            projected.append(projection(feature))

        x = self.fuse(torch.cat(projected, dim=1))
        x = self.context(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.head(x)


class Dinov3SegmentationModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "dinov3_vitb16",
        backbone_weights: str | None = None,
        layers: Sequence[int] = (12,),
        num_classes: int = 4,
        decoder_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = FrozenDinov3Backbone(backbone_name=backbone_name, weights=backbone_weights, layers=layers)
        self.decoder = LightweightSegmentationDecoder(
            in_channels=self.backbone.out_channels,
            feature_count=len(layers),
            num_classes=num_classes,
            decoder_dim=decoder_dim,
            dropout=dropout,
        )
        self.num_classes = num_classes
        self.decoder_dim = decoder_dim

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.train(False)
        self.decoder.train(mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.decoder(features)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def export_config(self) -> dict:
        return {
            "backbone_name": self.backbone.backbone_name,
            "backbone_weights": self.backbone.weights,
            "layers": self.backbone.layers,
            "num_classes": self.num_classes,
            "decoder_dim": self.decoder_dim,
        }


def build_model_from_checkpoint(checkpoint: dict, override_backbone_weights: str | None = None) -> Dinov3SegmentationModel:
    config = checkpoint["config"]
    model = Dinov3SegmentationModel(
        backbone_name=config["backbone_name"],
        backbone_weights=override_backbone_weights or config.get("backbone_weights"),
        layers=config["layers"],
        num_classes=config["num_classes"],
        decoder_dim=config["decoder_dim"],
    )
    model.decoder.load_state_dict(checkpoint["decoder"], strict=True)
    return model