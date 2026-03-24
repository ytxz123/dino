from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from dataset_generation.common.geometry_utils import deduplicate_points, ensure_closed_ring
from dataset_generation.common.io_utils import load_json
from dataset_generation.common.raster_utils import (
    RasterMetadata,
    build_transformer,
    detect_geojson_crs,
    project_coordinates,
    world_to_pixel,
)


def load_geojson(path: Path) -> Dict:
    """读取 GeoJSON 文件。"""
    return load_json(path)



def geojson_lines_to_pixel_lines(geojson_dict: Dict, raster_meta: RasterMetadata, category: str) -> List[Dict]:
    """把 GeoJSON 里的 LineString 转成整图像素坐标折线。"""
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
        points_pixel = deduplicate_points(world_to_pixel(points_world, affine=raster_meta.affine))
        if points_pixel.ndim != 2 or points_pixel.shape[0] < 2:
            continue
        output.append(
            {
                "category": str(category),
                "geometry_type": "line",
                "points_global": points_pixel.astype(np.float32),
            }
        )
    return output



def geojson_polygons_to_pixel_rings(geojson_dict: Dict, raster_meta: RasterMetadata, category: str) -> List[Dict]:
    """把 GeoJSON 里的 Polygon 外环转成整图像素坐标闭环。"""
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
        if not isinstance(coordinates, Sequence) or len(coordinates) == 0:
            continue
        shell = coordinates[0]
        points_world = project_coordinates(shell, transformer=transformer)
        points_pixel = ensure_closed_ring(deduplicate_points(world_to_pixel(points_world, affine=raster_meta.affine)))
        if points_pixel.ndim != 2 or points_pixel.shape[0] < 4:
            continue
        output.append(
            {
                "category": str(category),
                "geometry_type": "polygon",
                "points_global": points_pixel.astype(np.float32),
            }
        )
    return output



def load_sample_global_lines(
    sample_dir: Path,
    raster_meta: RasterMetadata,
    lane_relpath: str,
    intersection_relpath: str,
    include_lane: bool,
    include_intersection: bool,
) -> List[Dict]:
    """读取一个样本的 Lane 和 Intersection，并转成统一的像素几何。"""
    output: List[Dict] = []
    if bool(include_lane):
        lane_path = sample_dir / str(lane_relpath)
        if lane_path.is_file():
            output.extend(geojson_lines_to_pixel_lines(load_geojson(lane_path), raster_meta=raster_meta, category="lane_line"))
    if bool(include_intersection):
        intersection_path = sample_dir / str(intersection_relpath)
        if intersection_path.is_file():
            output.extend(
                geojson_polygons_to_pixel_rings(
                    load_geojson(intersection_path),
                    raster_meta=raster_meta,
                    category="intersection_polygon",
                )
            )
    return output
