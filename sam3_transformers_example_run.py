from __future__ import annotations

from pathlib import Path

from utils.mask_video import visualize_masks_to_video
from detection.sam3_transformers import Sam3TransformersConfig, extract_outputs_per_frame


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    # EDIT HERE
    video_path = str((repo_root / "data" / "example2.MP4").resolve())
    prompts = ["objects in refrigerator", "hands"]
    target_fps = 20.0
    backend = "opencv"
    max_frames = None  # None => all frames
    alpha = 0.45
    processing_device = "cuda"
    video_storage_device = "cuda"

    video_frames, fps, outputs_per_frame, out_dir = extract_outputs_per_frame(
        Sam3TransformersConfig(
            video_path=video_path,
            prompts=prompts,
            target_fps=target_fps,
            backend=backend,
            max_frames=max_frames,
            processing_device=processing_device,
            video_storage_device=video_storage_device,
        )
    )

    print(f"Processed {len(outputs_per_frame)} frames")
    frame_0_outputs = outputs_per_frame[min(outputs_per_frame.keys())]
    print(f"Detected {len(frame_0_outputs['object_ids'])} objects")
    print(f"Object IDs: {frame_0_outputs['object_ids'].tolist()}")
    print(f"Scores: {frame_0_outputs['scores'].tolist()}")
    print(f"Boxes shape (XYXY format, absolute coordinates): {frame_0_outputs['boxes'].shape}")
    print(f"Masks shape: {frame_0_outputs['masks'].shape}")
    if "prompt_to_obj_ids" in frame_0_outputs:
        print("prompt_to_obj_ids:")
        for p, obj_ids in frame_0_outputs["prompt_to_obj_ids"].items():
            obj_ids_list = obj_ids.tolist() if hasattr(obj_ids, "tolist") else list(obj_ids)
            print(f"  {p}: {len(obj_ids_list)} objects")

    out_path = str((out_dir / f"{Path(video_path).stem}_masks_overlay.mp4").resolve())
    visualize_masks_to_video(
        video_frames=video_frames,
        outputs_per_frame=outputs_per_frame,
        out_path=out_path,
        fps=fps,
        alpha=alpha,
    )
    print("Saved:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
