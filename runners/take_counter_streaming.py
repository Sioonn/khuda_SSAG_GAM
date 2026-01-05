from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from detection.sam3_transformers_streaming import (
    Sam3TransformersStreamingEngine,
)
from rules.take_counter import BoundaryTouchTakeCounter, TakeCounterConfig, prompt_masks_by_id_or_all
from utils.fridge_roi import polygon_to_mask, scaled_fridge_roi_polygon_xy
from utils.take_debug_video_streaming import TakeDebugVideoWriter
from utils.video_frame_stream import iter_video_frames


def mask_frame_outside_roi(frame_rgb: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    frame = frame_rgb.copy()
    frame[~roi_mask] = 0
    return frame


@dataclass(frozen=True)
class StreamingRunConfig:
    prompts: list[str]
    foods_prompt: str
    target_fps: float = 15.0
    resize_width: int = 1280
    resize_height: int = 720
    mask_input_outside_roi: bool = True
    processing_device: str = "cpu"
    video_storage_device: str = "cpu"
    model_image_size: int | None = None
    alpha: float = 0.45


@dataclass(frozen=True)
class StreamingVideoResult:
    video_path: str
    take_count: int
    predicted_positive: bool
    summary_path: str
    debug_video_path: Optional[str]


def run_streaming_take_counter_on_video(
    *,
    video_path: str,
    out_dir: Path,
    cfg: StreamingRunConfig,
    engine: Optional[Sam3TransformersStreamingEngine] = None,
) -> StreamingVideoResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem

    stream_info, frames = iter_video_frames(
        video_path,
        target_fps=cfg.target_fps,
        resize_width=cfg.resize_width,
        resize_height=cfg.resize_height,
    )
    w, h = stream_info.output_size_wh

    roi_polygon_xy = scaled_fridge_roi_polygon_xy(height=h, width=w)
    roi_mask = polygon_to_mask(roi_polygon_xy, height=h, width=w)

    if engine is None:
        import torch

        engine = Sam3TransformersStreamingEngine(
            model_name="facebook/sam3",
            dtype=torch.bfloat16,
            model_image_size=cfg.model_image_size,
        )

    # Create a fresh session per video
    session = engine.create_session(
        prompts=list(cfg.prompts),
        processing_device=cfg.processing_device,
        video_storage_device=cfg.video_storage_device,
    )

    take_cfg = TakeCounterConfig(foods_prompt=cfg.foods_prompt)
    counter = BoundaryTouchTakeCounter(roi_mask=roi_mask, config=take_cfg)
    take_events_all: list[dict] = []

    debug_path = str((out_dir / "debug" / f"{stem}_take_debug.mp4").resolve())
    debug_writer = TakeDebugVideoWriter(
        out_path=debug_path,
        fps=float(cfg.target_fps),
        roi_polygon_xy=roi_polygon_xy,
        alpha=cfg.alpha,
    )

    try:
        for frame_idx, frame_rgb in frames:
            frame_for_model = (
                mask_frame_outside_roi(frame_rgb, roi_mask)
                if cfg.mask_input_outside_roi
                else frame_rgb
            )
            processed = engine.process_frame(session=session, frame_rgb=frame_for_model)

            foods_by_id = prompt_masks_by_id_or_all(processed, prompt=cfg.foods_prompt)
            events = counter.update(frame_index=frame_idx, foods_by_id=foods_by_id)
            for ev in events:
                take_events_all.append(ev.__dict__)

            debug_writer.write(
                frame_index=frame_idx,
                frame_rgb=frame_rgb,
                processed_outputs=processed,
                take_count=counter.take_count,
                taken_object_ids=counter.taken_object_ids,
            )
    finally:
        debug_writer.close()
        del session
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    summary_path = out_dir / "summaries" / f"{stem}_take_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video_path": video_path,
                "stream": {
                    "original_fps": stream_info.original_fps,
                    "target_fps": stream_info.target_fps,
                    "resize_wh": [int(w), int(h)],
                },
                "prompts": cfg.prompts,
                "roi_polygon_xy": list(roi_polygon_xy),
                "rule_config": take_cfg.__dict__,
                "take_count": int(counter.take_count),
                "take_events": take_events_all,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    predicted_positive = counter.take_count >= 2
    return StreamingVideoResult(
        video_path=str(video_path),
        take_count=int(counter.take_count),
        predicted_positive=bool(predicted_positive),
        summary_path=str(summary_path),
        debug_video_path=str(debug_path),
    )


def run_streaming_demo() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    # ====== EDIT HERE ======
    test_video = "transform_ex2.MP4"
    video_path = str((repo_root / "data" / test_video).resolve())

    out_dir = (repo_root / "logs" / "take_counter_streaming").resolve()
    cfg = StreamingRunConfig(
        prompts=["objects in refrigerator"],
        foods_prompt="objects in refrigerator",
        target_fps=15.0,
        resize_width=1280,
        resize_height=720,
        mask_input_outside_roi=True,
        processing_device="cpu",
        video_storage_device="cpu",
        model_image_size=None,
        alpha=0.45,
    )

    # Load model once for demo
    import torch

    engine = Sam3TransformersStreamingEngine(model_name="facebook/sam3", dtype=torch.bfloat16)
    result = run_streaming_take_counter_on_video(
        video_path=video_path,
        out_dir=out_dir,
        cfg=cfg,
        engine=engine,
    )
    print("[result] take_count =", result.take_count)
    print("[saved] ", result.summary_path)
    print("[saved] ", result.debug_video_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_streaming_demo())
