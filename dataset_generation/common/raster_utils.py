from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pyproj import CRS, Transformer
from rasterio import open as rasterio_open
from rasterio.transform import Affine


@dataclass(frozen=True)
class RasterMetadata:
    """GeoTIFF 的基础空间信息。"""

    path: str
    width: int
    height: int
    crs: str
    transform: List[float]

    @property
    def affine(self) -> Affine:
        return Affine(*self.transform)


def read_raster_metadata(path: Path) -> RasterMetadata:
    """读取栅格尺寸、CRS 和仿射变换。"""
    with rasterio_open(path) as dataset:
        return RasterMetadata(
            path=str(path),
            width=int(dataset.width),
            height=int(dataset.height),
            crs=str(dataset.crs) if dataset.crs is not None else "",
            transform=list(dataset.transform)[:6],
        )


def read_rgb_geotiff(path: Path, band_indices: Tuple[int, ...]) -> Tuple[np.ndarray, RasterMetadata]:
    """读取指定波段并转成 HWC 格式的 RGB 图像。"""
    metadata = read_raster_metadata(path)
    with rasterio_open(path) as dataset:
        channels = [dataset.read(int(index)) for index in band_indices]
    image = np.stack(channels, axis=-1)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image, metadata


def read_binary_mask(path: Path, threshold: int) -> np.ndarray:
    """读取单通道 mask，并按阈值转成 0/1 二值图。"""
    with rasterio_open(path) as dataset:
        raw_mask = dataset.read(1)
    return (raw_mask > int(threshold)).astype(np.uint8)


def detect_geojson_crs(geojson_dict: Dict) -> str:
    """从 GeoJSON 中解析 CRS，缺省时回退到 CRS84。"""
    crs_value = geojson_dict.get("crs")
    if not isinstance(crs_value, dict):
        return "OGC:CRS84"
    props = crs_value.get("properties")
    if not isinstance(props, dict):
        return "OGC:CRS84"
    name = str(props.get("name", "")).strip()
    return name or "OGC:CRS84"


def build_transformer(source_crs: str, target_crs: str) -> Transformer:
    """构造世界坐标系转换器。"""
    return Transformer.from_crs(CRS.from_user_input(source_crs), CRS.from_user_input(target_crs), always_xy=True)


def project_coordinates(coordinates, transformer: Transformer) -> np.ndarray:
    """把 GeoJSON 坐标投影到栅格 CRS。"""
    array = np.asarray(coordinates, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    xs, ys = transformer.transform(array[:, 0], array[:, 1])
    return np.stack([xs, ys], axis=1).astype(np.float32)


def world_to_pixel(points_world: np.ndarray, affine: Affine) -> np.ndarray:
    """把世界坐标转换成整图像素坐标。"""
    if points_world.ndim != 2 or points_world.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    inverse_affine = ~affine
    points_pixel: List[List[float]] = []
    for world_x, world_y in points_world.tolist():
        pixel_x, pixel_y = inverse_affine * (float(world_x), float(world_y))
        points_pixel.append([float(pixel_x), float(pixel_y)])
    return np.asarray(points_pixel, dtype=np.float32)


def pixel_to_world(points_xy, affine: Affine) -> List[List[float]]:
    """把像素坐标回投到世界坐标。"""
    output: List[List[float]] = []
    for pixel_x, pixel_y in points_xy:
        world_x, world_y = affine * (float(pixel_x), float(pixel_y))
        output.append([float(world_x), float(world_y)])
    return output