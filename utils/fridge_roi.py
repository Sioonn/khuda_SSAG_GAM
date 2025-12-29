from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


# Hardcoded fridge ROI polygon (pixel coordinates) for 2560x1440 videos.
# Interpreted as a trapezoid:
#   (0,0) -> (933,0) -> (1503,1440) -> (0,1440)
FRIDGE_ROI_POLYGON_XY: list[tuple[int, int]] = [(0, 0), (933, 0), (1503, 1440), (0, 1440)]


def fridge_roi_polygon_xy() -> list[tuple[int, int]]:
    return list(FRIDGE_ROI_POLYGON_XY)


def polygon_to_mask(
    polygon_xy: Iterable[Tuple[int, int]],
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Rasterize a polygon (x,y points) into a boolean mask of shape (H,W).

    Requires `opencv-python` (cv2).
    """
    import cv2  # type: ignore

    pts = np.array(list(polygon_xy), dtype=np.int32).reshape((-1, 1, 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def inside_ratio(mask: np.ndarray, roi_mask: np.ndarray) -> float:
    """
    area(mask ∩ roi) / area(mask)
    """
    mask_bool = mask.astype(bool)
    area = int(mask_bool.sum())
    if area == 0:
        return 0.0
    inside = int(np.logical_and(mask_bool, roi_mask.astype(bool)).sum())
    return float(inside / max(area, 1))

