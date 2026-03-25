"""可视化小工具。"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from PIL import ImageDraw


def draw_endpoint(draw: ImageDraw.ImageDraw, point: Sequence[int], color: Tuple[int, int, int], radius: int = 3) -> None:
    x = int(point[0])
    y = int(point[1])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def draw_polyline(draw: ImageDraw.ImageDraw, points: Iterable[Sequence[int]], color: Tuple[int, int, int], width: int = 3) -> None:
    pts = [tuple(int(value) for value in point[:2]) for point in points]
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=width)
