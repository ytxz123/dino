"""RC 数据读取 helper。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from pyproj import CRS, Transformer
from rasterio import open as rasterio_open
from rasterio.transform import Affine

from .common import load_json
from .geometry import dedup_points, ensure_closed_ring


@dataclass(frozen=True)
class RasterMetadata:
    path: str
    width: int
    height: int
    crs: str
    transform: List[float]

    @property
    def affine(self) -> Affine:
        return Affine(*self.transform)


def read_raster_metadata(path: Path) -> RasterMetadata:
    with rasterio_open(path) as dataset:
        return RasterMetadata(
            path=str(path),
            width=int(dataset.width),
            height=int(dataset.height),
            crs=str(dataset.crs) if dataset.crs is not None else "",
            transform=list(dataset.transform)[:6],
        )


def read_rgb_geotiff(path: Path, band_indices: Tuple[int, ...]) -> Tuple[np.ndarray, RasterMetadata]:
    metadata = read_raster_metadata(path)
    with rasterio_open(path) as dataset:
        channels = [dataset.read(int(index)) for index in band_indices]
    image = np.stack(channels, axis=-1)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image, metadata


def read_binary_mask(path: Path, threshold: int) -> np.ndarray:
    with rasterio_open(path) as dataset:
        raw_mask = dataset.read(1)
    return (raw_mask > int(threshold)).astype(np.uint8)


def detect_geojson_crs(geojson_dict: Dict) -> str:
    crs_value = geojson_dict.get("crs")
    if not isinstance(crs_value, dict):
        return "OGC:CRS84"
    properties = crs_value.get("properties")
    if not isinstance(properties, dict):
        return "OGC:CRS84"
    name = str(properties.get("name", "")).strip()
    return name or "OGC:CRS84"


def build_transformer(source_crs: str, target_crs: str) -> Transformer:
    return Transformer.from_crs(CRS.from_user_input(source_crs), CRS.from_user_input(target_crs), always_xy=True)


def project_coordinates(coordinates, transformer: Transformer) -> np.ndarray:
    array = np.asarray(coordinates, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    xs, ys = transformer.transform(array[:, 0], array[:, 1])
    return np.stack([xs, ys], axis=1).astype(np.float32)


def world_to_pixel(points_world: np.ndarray, affine: Affine) -> np.ndarray:
    if points_world.ndim != 2 or points_world.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    inverse_affine = ~affine
    points_pixel: List[List[float]] = []
    for world_x, world_y in points_world.tolist():
        pixel_x, pixel_y = inverse_affine * (float(world_x), float(world_y))
        points_pixel.append([float(pixel_x), float(pixel_y)])
    return np.asarray(points_pixel, dtype=np.float32)


def geojson_lines_to_pixel_features(geojson_dict: Dict, raster_meta: RasterMetadata, category: str) -> List[Dict]:
    source_crs = detect_geojson_crs(geojson_dict)
    transformer = build_transformer(source_crs=source_crs, target_crs=raster_meta.crs)
    output: List[Dict] = []
    for feature in geojson_dict.get("features", []):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry", {})
        if str(geometry.get("type", "")).strip().lower() != "linestring":
            continue
        points_world = project_coordinates(geometry.get("coordinates", []), transformer=transformer)
        points_pixel = dedup_points(world_to_pixel(points_world, affine=raster_meta.affine))
        if points_pixel.ndim != 2 or points_pixel.shape[0] < 2:
            continue
        output.append({"category": str(category), "geometry_type": "line", "points_global": points_pixel.astype(np.float32)})
    return output


def geojson_polygons_to_pixel_features(geojson_dict: Dict, raster_meta: RasterMetadata, category: str) -> List[Dict]:
    source_crs = detect_geojson_crs(geojson_dict)
    transformer = build_transformer(source_crs=source_crs, target_crs=raster_meta.crs)
    output: List[Dict] = []
    for feature in geojson_dict.get("features", []):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry", {})
        if str(geometry.get("type", "")).strip().lower() != "polygon":
            continue
        coordinates = geometry.get("coordinates", [])
        if not isinstance(coordinates, Sequence) or not coordinates:
            continue
        shell = coordinates[0]
        points_world = project_coordinates(shell, transformer=transformer)
        points_pixel = ensure_closed_ring(dedup_points(world_to_pixel(points_world, affine=raster_meta.affine)))
        if points_pixel.ndim != 2 or points_pixel.shape[0] < 4:
            continue
        output.append({"category": str(category), "geometry_type": "polygon", "points_global": points_pixel.astype(np.float32)})
    return output


def load_sample_global_features(
    *,
    lane_path: Path,
    intersection_path: Path,
    raster_meta: RasterMetadata,
    include_lane: bool,
    include_intersection: bool,
) -> List[Dict]:
    output: List[Dict] = []
    if bool(include_lane) and lane_path.is_file():
        output.extend(geojson_lines_to_pixel_features(load_json(lane_path), raster_meta=raster_meta, category="lane_line"))
    if bool(include_intersection) and intersection_path.is_file():
        output.extend(geojson_polygons_to_pixel_features(load_json(intersection_path), raster_meta=raster_meta, category="intersection_polygon"))
    return output


def load_family_raster_and_mask(family: Dict, band_indices: Sequence[int], mask_threshold: int) -> Tuple[np.ndarray, RasterMetadata, np.ndarray | None]:
    image_hwc, raster_meta = read_rgb_geotiff(Path(family["source_image_path"]).resolve(), band_indices=tuple(int(index) for index in band_indices))
    mask_path_text = str(family.get("source_mask_path", "")).strip()
    review_mask = read_binary_mask(Path(mask_path_text), threshold=int(mask_threshold)) if mask_path_text else None
    if review_mask is not None:
        image_hwc = image_hwc.copy()
        image_hwc[review_mask <= 0] = 0
    return image_hwc, raster_meta, review_mask
