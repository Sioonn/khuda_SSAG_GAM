from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from transformers import Sam3VideoModel, Sam3VideoProcessor

from utils.fridge_roi import polygon_to_mask, scaled_fridge_roi_polygon_xy
from utils.video_frame_stream import iter_video_frames


def main() -> int:
    here = Path(__file__).resolve()
    repo_root = here.parent if (here.parent / "data").is_dir() else here.parents[1]
    video_path = str(
        (repo_root / "data" / "예시영상.mp4").resolve()
    )
    prompt = "objects in refrigerator"

    # Keep memory manageable
    target_fps = 15.0
    resize_width, resize_height = 1280, 720

    out_dir = (repo_root / "logs" / "sam3_single").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str((out_dir / "예시영상_sam3_overlay.mp4").resolve())

    stream_info, frames = iter_video_frames(
        video_path,
        target_fps=target_fps,
        resize_width=resize_width,
        resize_height=resize_height,
    )
    w, h = stream_info.output_size_wh

    roi_polygon_xy = scaled_fridge_roi_polygon_xy(height=h, width=w)
    roi_mask = polygon_to_mask(roi_polygon_xy, height=h, width=w)

    device = Accelerator().device
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=dtype)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

    session = processor.init_video_session(
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )
    session = processor.add_text_prompt(inference_session=session, text=prompt)

    import cv2  # type: ignore

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(stream_info.target_fps),
        (int(w), int(h)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_path}")

    def color_for_i(i: int):
        base = (i * 1103515245 + 12345) & 0x7FFFFFFF
        r = 60 + (base % 196)
        g = 60 + ((base // 7) % 196)
        b = 60 + ((base // 49) % 196)
        return (int(b), int(g), int(r))  # BGR

    alpha = 0.45
    roi_pts = np.array(list(roi_polygon_xy), dtype=np.int32).reshape((-1, 1, 2))

    with torch.inference_mode():
        for frame_idx, frame_rgb in frames:
            # Optional: hide outside ROI to reduce false detections
            frame_for_model = frame_rgb.copy()
            frame_for_model[~roi_mask] = 0

            inputs = processor(images=frame_for_model, device=device, return_tensors="pt")
            model_outputs = model(
                inference_session=session,
                frame=inputs.pixel_values[0],
                reverse=False,
            )
            processed = processor.postprocess_outputs(
                session,
                model_outputs,
                original_sizes=inputs.original_sizes,
            )

            masks = processed.get("masks")
            if masks is None:
                masks_np = np.zeros((0, h, w), dtype=np.uint8)
            else:
                masks_np = masks.detach().cpu().numpy() if isinstance(masks, torch.Tensor) else np.asarray(masks)

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            overlay_f = frame_bgr.astype(np.float32)
            for i in range(int(masks_np.shape[0])):
                m = masks_np[i] > 0.0
                if not m.any():
                    continue
                c = np.array(color_for_i(i), dtype=np.float32)
                overlay_f[m] = alpha * c + (1.0 - alpha) * overlay_f[m]

            overlay = overlay_f.astype(np.uint8)
            cv2.polylines(overlay, [roi_pts], isClosed=True, color=(0, 255, 255), thickness=2)
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

            if (frame_idx + 1) % 50 == 0:
                print(f"processed {frame_idx + 1} frames")

    writer.release()
    print("saved:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
