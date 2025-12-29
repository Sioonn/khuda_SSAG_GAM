from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np


def _color_for_id(obj_id: int) -> Tuple[int, int, int]:
    base = (obj_id * 1103515245 + 12345) & 0x7FFFFFFF
    r = 60 + (base % 196)
    g = 60 + ((base // 7) % 196)
    b = 60 + ((base // 49) % 196)
    return (int(b), int(g), int(r))  # BGR


def overlay_take_debug_video(
    *,
    video_frames: list[np.ndarray],
    outputs_per_frame: Dict[int, Dict[str, Any]],
    debug_by_frame: Dict[int, Any],
    out_path: str,
    fps: float,
    roi_polygon_xy: Iterable[Tuple[int, int]] | None = None,
    alpha: float = 0.45,
) -> None:
    """
    Writes a debug MP4 with:
    - instance masks overlay (all objects)
    - ROI polygon outline (optional)
    - HUD: take_count (top-right), frame index
    - highlights foods that have already triggered take (red-ish)
    """
    import cv2  # type: ignore

    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover
        torch = None  # type: ignore

    if len(video_frames) == 0:
        raise ValueError("video_frames is empty")

    first = video_frames[0]
    if not isinstance(first, np.ndarray):
        first = np.asarray(first)
    h, w = first.shape[:2]

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_path}")

    roi_pts = None
    if roi_polygon_xy is not None:
        roi_pts = np.array(list(roi_polygon_xy), dtype=np.int32).reshape((-1, 1, 2))

    taken_obj_ids: set[int] = set()

    for frame_idx in sorted(outputs_per_frame.keys()):
        d = debug_by_frame.get(int(frame_idx))
        if d is not None and hasattr(d, "take_events"):
            for ev in getattr(d, "take_events") or []:
                if hasattr(ev, "food_id"):
                    taken_obj_ids.add(int(getattr(ev, "food_id")))

        frame = video_frames[frame_idx]
        if not isinstance(frame, np.ndarray):
            frame = np.asarray(frame)
        base_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        processed = outputs_per_frame[frame_idx]
        masks = processed.get("masks", None)
        obj_ids = processed.get("object_ids", None)

        if masks is None or obj_ids is None:
            overlay = base_bgr
        else:
            if torch is not None and isinstance(masks, torch.Tensor):
                masks_np = masks.detach().cpu().numpy()
            else:
                masks_np = np.asarray(masks)
            if hasattr(obj_ids, "detach"):
                obj_ids = obj_ids.detach().cpu().numpy()
            obj_ids_np = np.asarray(obj_ids).astype(int)

            overlay_f = base_bgr.astype(np.float32)
            for i in range(int(masks_np.shape[0])):
                m = masks_np[i] > 0.0
                if not m.any():
                    continue
                obj_id = int(obj_ids_np[i])
                if obj_id in taken_obj_ids:
                    color = np.array((0, 0, 255), dtype=np.float32)  # red for taken
                    a = min(0.65, alpha + 0.2)
                else:
                    color = np.array(_color_for_id(obj_id), dtype=np.float32)
                    a = alpha
                overlay_f[m] = a * color + (1.0 - a) * overlay_f[m]
            overlay = overlay_f.astype(np.uint8)

        if roi_pts is not None:
            cv2.polylines(overlay, [roi_pts], isClosed=True, color=(0, 255, 255), thickness=2)

        take_count = 0
        d = debug_by_frame.get(int(frame_idx))
        if d is not None and hasattr(d, "take_count"):
            take_count = int(getattr(d, "take_count") or 0)

        hud = f"take_count={take_count}"
        (tw, th), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        x = max(10, w - tw - 10)
        y = 30
        cv2.rectangle(overlay, (x - 6, y - th - 10), (x + tw + 6, y + 8), (0, 0, 0), -1)
        cv2.putText(
            overlay,
            hud,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"frame={frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(overlay)

    writer.release()
