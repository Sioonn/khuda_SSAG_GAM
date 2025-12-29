from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


Point = Tuple[float, float]


@dataclass(frozen=True)
class TakeCounterConfig:
    foods_prompt: str = "objects in refrigerator"


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


def count_take_events(
    *,
    outputs_per_frame: Dict[int, Dict[str, Any]],
    roi_mask: np.ndarray,
    config: TakeCounterConfig,
) -> Tuple[int, list[TakeEvent], Dict[int, FrameDebug]]:
    """
    Simplified take counting:
    - For each unique food instance (from `config.foods_prompt`),
      if its segmentation mask has ANY pixel outside the ROI on any frame,
      immediately count take_count += 1 (only once per instance).
    """
    from utils.fridge_roi import inside_ratio  # local import to avoid circular deps

    if roi_mask.dtype != bool:
        roi_mask = roi_mask.astype(bool)

    taken_food_ids: set[int] = set()
    take_events: list[TakeEvent] = []
    debug_by_frame: Dict[int, FrameDebug] = {}
    take_count = 0

    for frame_idx in sorted(outputs_per_frame.keys()):
        processed = outputs_per_frame[frame_idx]
        foods_by_id = prompt_masks_by_id(processed, prompt=config.foods_prompt)

        events_this_frame: list[TakeEvent] = []

        for fid, food_mask in foods_by_id.items():
            fid_i = int(fid)
            if fid_i in taken_food_ids:
                continue

            ir = inside_ratio(food_mask, roi_mask)
            if ir < 1.0:
                taken_food_ids.add(fid_i)
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
