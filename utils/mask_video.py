from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def _color_for_i(i: int) -> Tuple[int, int, int]:
    base = (i * 1103515245 + 12345) & 0x7FFFFFFF
    r = 60 + (base % 196)
    g = 60 + ((base // 7) % 196)
    b = 60 + ((base // 49) % 196)
    return (int(b), int(g), int(r))  # BGR


def visualize_masks_to_video(
    *,
    video_frames: list[np.ndarray],
    outputs_per_frame: Dict[int, Dict[str, Any]],
    out_path: str,
    fps: float,
    alpha: float = 0.45,
) -> None:
    """
    Render per-frame instance masks (from SAM3 processed outputs) into an overlay video.

    Expected processed_outputs schema per frame:
      - processed_outputs["masks"] : (N, H, W) torch.Tensor or np.ndarray
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

    for frame_idx in sorted(outputs_per_frame.keys()):
        frame = video_frames[frame_idx]
        if not isinstance(frame, np.ndarray):
            frame = np.asarray(frame)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        masks = outputs_per_frame[frame_idx]["masks"]  # (N,H,W)
        if torch is not None and isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        else:
            masks = np.asarray(masks)

        overlay = frame_bgr.astype(np.float32)
        for i in range(int(masks.shape[0])):
            m = masks[i] > 0.0
            if not m.any():
                continue
            color = np.array(_color_for_i(i), dtype=np.float32)
            overlay[m] = alpha * color + (1.0 - alpha) * overlay[m]

        overlay_u8 = overlay.astype(np.uint8)
        cv2.putText(
            overlay_u8,
            f"frame={frame_idx}  num_obj={int(masks.shape[0])}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(overlay_u8)

    writer.release()

