from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class ResampleResult:
    input_path: str
    output_path: str
    original_fps: float
    target_fps: float
    input_frame_count: int
    output_frame_count: int
    approx_duration_sec: float
    output_size_wh: Tuple[int, int]


def get_video_fps(video_path: str) -> float:
    """
    Return the FPS reported by the container/decoder.

    Requires `opencv-python`.
    """
    import cv2  # type: ignore

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if fps <= 0:
        raise RuntimeError(f"Could not read FPS for video: {video_path}")
    return fps


def resample_video_to_fps(
    video_path: str,
    *,
    target_fps: float = 20.0,
    out_path: Optional[str] = None,
    fourcc: str = "mp4v",
    resize_width: Optional[int] = None,
    resize_height: Optional[int] = None,
) -> ResampleResult:
    """
    Downsample a video to `target_fps` by frame skipping (no interpolation).

    Example:
      - input: 400 frames, 10 sec => 40 fps
      - output: keep ~every 2nd frame => 200 frames, saved at 20 fps => 10 sec

    Notes:
    - If `original_fps <= target_fps`, this keeps all frames and just changes container FPS.
      (Duration may change in players if you do this; for your use-case target_fps should be <= original_fps.)
    - Requires `opencv-python`.
    """
    import cv2  # type: ignore

    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    original_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if original_fps <= 0:
        cap.release()
        raise RuntimeError(f"Could not read FPS for video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    input_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Could not read frame size for video: {video_path}")

    if out_path is None:
        p = Path(video_path)
        out_path = str(p.with_name(f"{p.stem}_fps{int(target_fps)}{p.suffix}"))

    out_dir = Path(out_path).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    out_w = int(resize_width) if resize_width is not None else width
    out_h = int(resize_height) if resize_height is not None else height
    if out_w <= 0 or out_h <= 0:
        cap.release()
        raise ValueError("resize_width/resize_height must be positive when provided")

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*fourcc),
        float(target_fps),
        (out_w, out_h),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter for output: {out_path}")

    # Keep frames based on time mapping: output time t=k/target_fps -> input frame ~t*original_fps.
    # Implemented as a sequential "next_keep_index" accumulator to avoid random seeking.
    keep_interval = 1.0 if original_fps <= target_fps else (original_fps / target_fps)
    next_keep = 0.0
    input_idx = 0
    output_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if input_idx + 1e-6 >= next_keep:
            if out_w != width or out_h != height:
                frame_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(frame_bgr)
            output_idx += 1
            next_keep += keep_interval

        input_idx += 1

    cap.release()
    writer.release()

    approx_duration_sec = float(output_idx / target_fps)
    return ResampleResult(
        input_path=str(video_path),
        output_path=str(out_path),
        original_fps=original_fps,
        target_fps=float(target_fps),
        input_frame_count=int(input_frame_count) if input_frame_count > 0 else int(input_idx),
        output_frame_count=int(output_idx),
        approx_duration_sec=approx_duration_sec,
        output_size_wh=(out_w, out_h),
    )
