from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


FRIDGE_ROI_REFERENCE_SIZE_WH: tuple[int, int] = (2560, 1440)

# Hardcoded fridge ROI polygon (pixel coordinates) for 2560x1440 videos.
# Interpreted as a trapezoid:
#   (0,0) -> (1455,0) -> (2251,924) -> (2367,1440) -> (0,1440)
FRIDGE_ROI_POLYGON_XY: list[tuple[int, int]] = [(0, 0), (1455, 0), (2251, 924), (2367, 1440), (0, 1440)]


def fridge_roi_polygon_xy() -> list[tuple[int, int]]:
    return list(FRIDGE_ROI_POLYGON_XY)


def scaled_fridge_roi_polygon_xy(*, height: int, width: int) -> list[tuple[int, int]]:
    """
    Scale the hardcoded ROI polygon from the reference resolution (2560x1440)
    to the given (width,height). Useful when the input video is resized (e.g. 1280x720).
    """
    ref_w, ref_h = FRIDGE_ROI_REFERENCE_SIZE_WH
    sx = float(width) / float(ref_w)
    sy = float(height) / float(ref_h)

    scaled: list[tuple[int, int]] = []
    for x, y in FRIDGE_ROI_POLYGON_XY:
        xs = int(round(x * sx))
        ys = int(round(y * sy))
        xs = max(0, min(width - 1, xs))
        ys = max(0, min(height - 1, ys))
        scaled.append((xs, ys))
    return scaled


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
