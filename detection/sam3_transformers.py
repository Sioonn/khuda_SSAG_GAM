from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from accelerate import Accelerator
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video

from utils.video_processing import resample_video_to_fps


@dataclass(frozen=True)
class Sam3TransformersConfig:
    video_path: str
    prompts: list[str] = None  # type: ignore[assignment]
    target_fps: float = 20.0
    backend: str = "opencv"
    max_frames: int | None = None  # None => all frames
    processing_device: str = "cuda"
    video_storage_device: str = "cuda"
    model_name: str = "facebook/sam3"
    dtype: torch.dtype = torch.bfloat16
    out_dir: str | None = None  # None => <repo_root>/logs/sam3_transformers

    def __post_init__(self):
        if self.prompts is None:
            object.__setattr__(self, "prompts", ["objects in refrigerator"])
        if not isinstance(self.prompts, list) or len(self.prompts) == 0:
            raise ValueError("prompts must be a non-empty list[str]")
        cleaned = [p.strip() for p in self.prompts if p and p.strip()]
        if len(cleaned) == 0:
            raise ValueError("prompts must contain at least one non-empty prompt string")
        object.__setattr__(self, "prompts", cleaned)


def _extract_fps(meta) -> float:
    for attr in ("fps", "video_fps", "frame_rate", "average_fps", "avg_fps"):
        if hasattr(meta, attr):
            v = getattr(meta, attr)
            try:
                return float(v)
            except Exception:
                pass
    try:
        return float(meta)
    except Exception:
        return 0.0


def extract_outputs_per_frame(
    config: Sam3TransformersConfig,
) -> tuple[list[np.ndarray], float, dict[int, dict[str, Any]], Path]:
    """
    Runs SAM3 (Transformers) on the given video and returns per-frame outputs.

    Returns:
      - video_frames: list of RGB frames (numpy arrays)
      - fps: fps value from video metadata (fallbacks to target_fps)
      - outputs_per_frame: {frame_idx: processed_outputs}
      - out_dir: directory where resampled input is written
    """
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(config.out_dir) if config.out_dir is not None else (repo_root / "logs" / "sam3_transformers")
    out_dir.mkdir(parents=True, exist_ok=True)

    resampled_path = str((out_dir / f"{Path(config.video_path).stem}_fps{int(config.target_fps)}.mp4").resolve())
    resample_result = resample_video_to_fps(
        config.video_path,
        target_fps=float(config.target_fps),
        out_path=resampled_path,
    )
    print(
        f"[resample] {resample_result.input_path} ({resample_result.original_fps:.2f}fps, "
        f"{resample_result.input_frame_count} frames) -> {resample_result.output_path} "
        f"({resample_result.target_fps:.2f}fps, {resample_result.output_frame_count} frames)"
    )

    video_frames, video_meta = load_video(resample_result.output_path, backend=config.backend)
    fps = _extract_fps(video_meta) or float(config.target_fps)

    device = Accelerator().device
    model = Sam3VideoModel.from_pretrained(config.model_name).to(device, dtype=config.dtype)
    processor = Sam3VideoProcessor.from_pretrained(config.model_name)

    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device=config.processing_device,
        video_storage_device=config.video_storage_device,
        dtype=config.dtype,
    )
    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=config.prompts if len(config.prompts) > 1 else config.prompts[0],
    )

    outputs_per_frame: Dict[int, dict[str, Any]] = {}
    max_frames = (len(video_frames) - 1) if config.max_frames is None else int(config.max_frames)

    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session, max_frame_num_to_track=max_frames
    ):
        processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
        outputs_per_frame[int(model_outputs.frame_idx)] = processed_outputs

    return video_frames, float(fps), outputs_per_frame, out_dir
