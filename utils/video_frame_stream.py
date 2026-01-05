from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class VideoStreamInfo:
    input_path: str
    original_fps: float
    target_fps: float
    output_size_wh: Tuple[int, int]


def iter_video_frames(
    video_path: str,
    *,
    target_fps: Optional[float] = None,
    resize_width: Optional[int] = None,
    resize_height: Optional[int] = None,
) -> tuple[VideoStreamInfo, Generator[tuple[int, np.ndarray], None, None]]:
    """
    Stream frames from a video using OpenCV, optionally:
    - downsample to `target_fps` by frame skipping (no interpolation)
    - resize each kept frame to (resize_width, resize_height)

    Returns:
      (VideoStreamInfo, generator yielding (out_frame_index, frame_rgb))
    """
    import cv2  # type: ignore

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    original_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if original_fps <= 0:
        cap.release()
        raise RuntimeError(f"Could not read FPS for video: {video_path}")

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if in_w <= 0 or in_h <= 0:
        cap.release()
        raise RuntimeError(f"Could not read frame size for video: {video_path}")

    out_w = int(resize_width) if resize_width is not None else in_w
    out_h = int(resize_height) if resize_height is not None else in_h
    if out_w <= 0 or out_h <= 0:
        cap.release()
        raise ValueError("resize_width/resize_height must be positive when provided")

    if target_fps is None:
        target_fps = original_fps
    target_fps = float(target_fps)
    if target_fps <= 0:
        cap.release()
        raise ValueError("target_fps must be > 0")

    keep_interval = 1.0 if original_fps <= target_fps else (original_fps / target_fps)

    info = VideoStreamInfo(
        input_path=str(video_path),
        original_fps=float(original_fps),
        target_fps=float(target_fps),
        output_size_wh=(out_w, out_h),
    )

    def _gen() -> Generator[tuple[int, np.ndarray], None, None]:
        input_idx = 0
        out_idx = 0
        next_keep = 0.0
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                if input_idx + 1e-6 >= next_keep:
                    if out_w != in_w or out_h != in_h:
                        frame_bgr = cv2.resize(
                            frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA
                        )
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    yield out_idx, frame_rgb
                    out_idx += 1
                    next_keep += keep_interval

                input_idx += 1
        finally:
            cap.release()

    return info, _gen()

