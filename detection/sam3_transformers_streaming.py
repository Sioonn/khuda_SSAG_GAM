from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from transformers import Sam3VideoModel, Sam3VideoProcessor

from detection.sam3_transformers import normalize_processed_outputs


@dataclass(frozen=True)
class Sam3TransformersStreamingConfig:
    video_path: str
    prompts: list[str]
    model_name: str = "facebook/sam3"
    dtype: torch.dtype = torch.bfloat16
    processing_device: str = "cpu"
    video_storage_device: str = "cpu"
    model_image_size: int | None = None


def init_streaming_session(
    *,
    processor: Sam3VideoProcessor,
    device: torch.device,
    prompts: List[str],
    processing_device: str,
    video_storage_device: str,
    dtype: torch.dtype,
):
    session = processor.init_video_session(
        inference_device=device,
        processing_device=processing_device,
        video_storage_device=video_storage_device,
        dtype=dtype,
    )
    session = processor.add_text_prompt(
        inference_session=session,
        text=prompts if len(prompts) > 1 else prompts[0],
    )
    return session


def iter_video_frames_cv2(video_path: str) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Yields (frame_index, frame_rgb) for all frames in the video.
    Requires `opencv-python`.
    """
    import cv2  # type: ignore

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            yield idx, frame_rgb
            idx += 1
    finally:
        cap.release()

class Sam3TransformersStreamingEngine:
    """
    Loads model+processor once, then creates per-video streaming sessions.
    """

    def __init__(self, *, model_name: str = "facebook/sam3", dtype: torch.dtype = torch.bfloat16, model_image_size: int | None = None):
        self.model_name = model_name
        self.dtype = dtype
        self.model_image_size = model_image_size
        self.device = Accelerator().device

        if model_image_size is not None:
            from transformers import Sam3VideoConfig  # type: ignore

            cfg = Sam3VideoConfig.from_pretrained(model_name)
            cfg.image_size = int(model_image_size)
            self.model = Sam3VideoModel.from_pretrained(model_name, config=cfg).to(
                self.device, dtype=dtype
            )
            self.processor = Sam3VideoProcessor.from_pretrained(
                model_name,
                size={"height": int(model_image_size), "width": int(model_image_size)},
            )
        else:
            self.model = Sam3VideoModel.from_pretrained(model_name).to(self.device, dtype=dtype)
            self.processor = Sam3VideoProcessor.from_pretrained(model_name)

    def create_session(
        self,
        *,
        prompts: List[str],
        processing_device: str,
        video_storage_device: str,
    ):
        return init_streaming_session(
            processor=self.processor,
            device=self.device,
            prompts=prompts,
            processing_device=processing_device,
            video_storage_device=video_storage_device,
            dtype=self.dtype,
        )

    @torch.inference_mode()
    def process_frame(self, *, session, frame_rgb: np.ndarray) -> Dict[str, Any]:
        inputs = self.processor(images=frame_rgb, device=self.device, return_tensors="pt")
        model_outputs = self.model(
            inference_session=session,
            frame=inputs.pixel_values[0],
            reverse=False,
        )
        processed = self.processor.postprocess_outputs(
            session,
            model_outputs,
            original_sizes=inputs.original_sizes,
        )
        return normalize_processed_outputs(processed)


class Sam3TransformersStreamer:
    """
    Backwards-compatible wrapper: keeps a single session and exposes `process_frame(frame_rgb)`.
    """

    def __init__(self, cfg: Sam3TransformersStreamingConfig):
        self.cfg = cfg
        self.engine = Sam3TransformersStreamingEngine(
            model_name=cfg.model_name, dtype=cfg.dtype, model_image_size=cfg.model_image_size
        )
        self.session = self.engine.create_session(
            prompts=list(cfg.prompts),
            processing_device=cfg.processing_device,
            video_storage_device=cfg.video_storage_device,
        )

    def process_frame(self, frame_rgb: np.ndarray) -> Dict[str, Any]:
        return self.engine.process_frame(session=self.session, frame_rgb=frame_rgb)
