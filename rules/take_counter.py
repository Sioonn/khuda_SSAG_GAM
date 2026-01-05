from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


Point = Tuple[float, float]


@dataclass(frozen=True)
class TakeCounterConfig:
    foods_prompt: str = "objects in refrigerator"

    # Take event when the (ROI-clipped) mask touches ROI boundary for K consecutive frames.
    boundary_thickness_px: int = 4
    boundary_touch_k: int = 1
    min_mask_pixels_in_roi: int = 200


@dataclass(frozen=True)
class TakeEvent:
    frame_index: int
    food_id: int
    inside_ratio: float


@dataclass
class FrameDebug:
    take_count: int = 0
    take_events: list[TakeEvent] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.take_events is None:
            self.take_events = []


def processed_outputs_to_masks_by_id(processed_outputs: Dict[str, Any]) -> Dict[int, np.ndarray]:
    """
    Build {object_id: mask(H,W)} mapping from transformers postprocessed outputs.
    """
    obj_ids = processed_outputs.get("object_ids", None)
    masks = processed_outputs.get("masks", None)
    if obj_ids is None or masks is None:
        return {}

    if hasattr(obj_ids, "detach"):
        obj_ids = obj_ids.detach().cpu().numpy()
    obj_ids = np.asarray(obj_ids).astype(int)

    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)

    masks_by_id: Dict[int, np.ndarray] = {}
    for i in range(int(obj_ids.shape[0])):
        masks_by_id[int(obj_ids[i])] = masks[i]
    return masks_by_id


@dataclass
class FoodState:
    boundary_touch_streak: int = 0
    take_done: bool = False


class BoundaryTouchTakeCounter:
    """
    Online (streaming) take counter using the same boundary-touch rule.
    """

    def __init__(self, *, roi_mask: np.ndarray, config: TakeCounterConfig):
        if roi_mask.dtype != bool:
            roi_mask = roi_mask.astype(bool)
        self.roi_mask = roi_mask
        self.config = config
        self.boundary_band = _roi_boundary_band_mask(
            roi_mask, thickness_px=int(config.boundary_thickness_px)
        )
        self.food_states: Dict[int, FoodState] = {}
        self.take_count: int = 0
        self.taken_object_ids: set[int] = set()

    def update(
        self,
        *,
        frame_index: int,
        foods_by_id: Dict[int, np.ndarray],
    ) -> list[TakeEvent]:
        from utils.fridge_roi import inside_ratio

        events: list[TakeEvent] = []
        for fid, food_mask in foods_by_id.items():
            fid_i = int(fid)
            state = self.food_states.setdefault(fid_i, FoodState())
            if state.take_done:
                continue

            mask_bool = food_mask.astype(bool)
            mask_in_roi = np.logical_and(mask_bool, self.roi_mask)
            area_in_roi = int(mask_in_roi.sum())
            if area_in_roi < int(self.config.min_mask_pixels_in_roi):
                state.boundary_touch_streak = 0
                continue

            touches = bool(np.logical_and(mask_in_roi, self.boundary_band).any())
            if touches:
                state.boundary_touch_streak += 1
            else:
                state.boundary_touch_streak = 0

            if state.boundary_touch_streak >= int(self.config.boundary_touch_k):
                ir = inside_ratio(food_mask, self.roi_mask)
                state.take_done = True
                self.take_count += 1
                self.taken_object_ids.add(fid_i)
                events.append(
                    TakeEvent(
                        frame_index=int(frame_index),
                        food_id=fid_i,
                        inside_ratio=float(ir),
                    )
                )
        return events


def prompt_masks_by_id(
    processed_outputs: Dict[str, Any],
    *,
    prompt: str,
) -> Dict[int, np.ndarray]:
    """
    Returns {object_id: mask} for a given prompt using `prompt_to_obj_ids`.
    Falls back to empty dict if prompt mapping isn't present.
    """
    masks_by_id = processed_outputs_to_masks_by_id(processed_outputs)
    prompt_to_obj_ids = processed_outputs.get("prompt_to_obj_ids", None)
    if prompt_to_obj_ids is None:
        return {}

    def _ids_for_prompt(prompt: str) -> list[int]:
        v = prompt_to_obj_ids.get(prompt, [])
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        if hasattr(v, "tolist"):
            v = v.tolist()
        return [int(x) for x in list(v)]

    ids = _ids_for_prompt(prompt)
    return {i: masks_by_id[i] for i in ids if i in masks_by_id}


def prompt_masks_by_id_or_all(
    processed_outputs: Dict[str, Any],
    *,
    prompt: str,
) -> Dict[int, np.ndarray]:
    """
    If `prompt_to_obj_ids` exists, select masks for that prompt.
    Otherwise, fall back to returning all instance masks.
    """
    if "prompt_to_obj_ids" not in processed_outputs:
        return processed_outputs_to_masks_by_id(processed_outputs)
    masks = prompt_masks_by_id(processed_outputs, prompt=prompt)
    return masks if len(masks) > 0 else processed_outputs_to_masks_by_id(processed_outputs)


def _roi_boundary_band_mask(roi_mask: np.ndarray, *, thickness_px: int) -> np.ndarray:
    """
    Build a band mask around ROI boundary (inside the ROI) for "touch" detection.
    """
    import cv2  # type: ignore

    if thickness_px <= 0:
        raise ValueError("thickness_px must be > 0")

    roi_u8 = (roi_mask.astype(np.uint8) > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(roi_u8, kernel, iterations=1)
    boundary = cv2.subtract(roi_u8, eroded)  # pixels on the ROI edge (inside ROI)
    band = cv2.dilate(boundary, kernel, iterations=int(thickness_px))
    band = np.logical_and(band > 0, roi_u8 > 0)
    return band.astype(bool)


def count_take_events(
    *,
    outputs_per_frame: Dict[int, Dict[str, Any]],
    roi_mask: np.ndarray,
    config: TakeCounterConfig,
) -> Tuple[int, list[TakeEvent], Dict[int, FrameDebug]]:
    """
    Take counting (boundary-touch with safety guards):
    - For each unique food instance (from `config.foods_prompt`),
      clip mask to ROI (mask_in_roi = mask & roi_mask).
    - If mask_in_roi touches the ROI boundary band for `boundary_touch_k` consecutive frames,
      count take_count += 1 (only once per instance).
    Safety guards:
    - Ignore tiny masks: require mask_in_roi area >= `min_mask_pixels_in_roi`.
    """
    from utils.fridge_roi import inside_ratio  # local import to avoid circular deps

    if roi_mask.dtype != bool:
        roi_mask = roi_mask.astype(bool)

    boundary_band = _roi_boundary_band_mask(
        roi_mask, thickness_px=int(config.boundary_thickness_px)
    )

    food_states: Dict[int, FoodState] = {}
    take_events: list[TakeEvent] = []
    debug_by_frame: Dict[int, FrameDebug] = {}
    take_count = 0

    for frame_idx in sorted(outputs_per_frame.keys()):
        processed = outputs_per_frame[frame_idx]
        foods_by_id = prompt_masks_by_id_or_all(processed, prompt=config.foods_prompt)

        events_this_frame: list[TakeEvent] = []

        for fid, food_mask in foods_by_id.items():
            fid_i = int(fid)
            state = food_states.setdefault(fid_i, FoodState())
            if state.take_done:
                continue

            mask_bool = food_mask.astype(bool)
            mask_in_roi = np.logical_and(mask_bool, roi_mask)
            area_in_roi = int(mask_in_roi.sum())
            if area_in_roi < int(config.min_mask_pixels_in_roi):
                state.boundary_touch_streak = 0
                continue

            touches = bool(np.logical_and(mask_in_roi, boundary_band).any())
            if touches:
                state.boundary_touch_streak += 1
            else:
                state.boundary_touch_streak = 0

            if state.boundary_touch_streak >= int(config.boundary_touch_k):
                ir = inside_ratio(food_mask, roi_mask)
                state.take_done = True
                take_count += 1
                ev = TakeEvent(
                    frame_index=int(frame_idx),
                    food_id=fid_i,
                    inside_ratio=float(ir),
                )
                take_events.append(ev)
                events_this_frame.append(ev)

        # Debug snapshot
        debug_by_frame[int(frame_idx)] = FrameDebug(
            take_count=int(take_count),
            take_events=events_this_frame,
        )

    return int(take_count), take_events, debug_by_frame
