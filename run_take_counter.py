from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from detection.sam3_transformers import Sam3TransformersConfig, extract_outputs_per_frame
from rules.take_counter import TakeCounterConfig, count_take_events
from utils.fridge_roi import FRIDGE_ROI_POLYGON_XY, polygon_to_mask
from utils.take_debug_video import overlay_take_debug_video


def main() -> int:
    repo_root = Path(__file__).resolve().parent

    # ====== EDIT HERE ======
    video_path = str((repo_root / "data" / "1203_1_1.MP4").resolve())
    prompts = ["objects in refrigerator"]
    foods_prompt = "objects in refrigerator"

    target_fps = 20.0
    backend = "opencv"
    max_frames = None  # None => all frames

    save_outputs_per_frame_pt = True
    save_debug_video = True
    alpha = 0.45
    # =======================

    det_cfg = Sam3TransformersConfig(
        video_path=video_path,
        prompts=prompts,
        target_fps=target_fps,
        backend=backend,
        max_frames=max_frames,
        processing_device="cuda",
        video_storage_device="cuda",
        out_dir=str((repo_root / "logs" / "take_counter").resolve()),
    )
    video_frames, fps, outputs_per_frame, out_dir = extract_outputs_per_frame(det_cfg)

    if len(video_frames) == 0:
        raise RuntimeError("No video frames loaded.")

    first = video_frames[0]
    if not isinstance(first, np.ndarray):
        first = np.asarray(first)
    h, w = first.shape[:2]

    roi_mask = polygon_to_mask(FRIDGE_ROI_POLYGON_XY, height=h, width=w)
    rule_cfg = TakeCounterConfig(foods_prompt=foods_prompt)
    take_count, take_events, debug_by_frame = count_take_events(
        outputs_per_frame=outputs_per_frame,
        roi_mask=roi_mask,
        config=rule_cfg,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    summary_path = out_dir / f"{stem}_take_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video_path": video_path,
                "fps": float(fps),
                "prompts": prompts,
                "roi_polygon_xy": list(FRIDGE_ROI_POLYGON_XY),
                "rule_config": rule_cfg.__dict__,
                "take_count": int(take_count),
                "take_events": [e.__dict__ for e in take_events],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("[result] take_count =", take_count)
    print("[saved] ", str(summary_path))

    if save_outputs_per_frame_pt:
        import torch

        outputs_path = out_dir / f"{stem}_outputs_per_frame.pt"
        torch.save(
            {
                "video_path": video_path,
                "fps": float(fps),
                "prompts": prompts,
                "outputs_per_frame": outputs_per_frame,
            },
            str(outputs_path),
        )
        print("[saved] ", str(outputs_path))

    if save_debug_video:
        debug_video_path = out_dir / f"{stem}_take_debug.mp4"
        overlay_take_debug_video(
            video_frames=video_frames,
            outputs_per_frame=outputs_per_frame,
            debug_by_frame=debug_by_frame,
            out_path=str(debug_video_path),
            fps=float(fps) if fps else 20.0,
            roi_polygon_xy=FRIDGE_ROI_POLYGON_XY,
            alpha=alpha,
        )
        print("[saved] ", str(debug_video_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
