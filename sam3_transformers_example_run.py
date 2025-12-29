from transformers import Sam3VideoModel, Sam3VideoProcessor
from accelerate import Accelerator
import torch

import os
import cv2
import numpy as np


device = Accelerator().device
model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

# Load video frames
from transformers.video_utils import load_video
video_url = "./data/example2.MP4"
video_frames, _ = load_video(video_url, backend="opencv")

# Initialize video inference session
inference_session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
    processing_device="cuda",
    video_storage_device="cuda",
    dtype=torch.bfloat16,
)

# Add text prompt to detect and track objects
text = "objects in refrigerator"
inference_session = processor.add_text_prompt(
    inference_session=inference_session,
    text=text,
)

# Process all frames in the video
outputs_per_frame = {}
for model_outputs in model.propagate_in_video_iterator(
    inference_session=inference_session, max_frame_num_to_track=100
):
    processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
    outputs_per_frame[model_outputs.frame_idx] = processed_outputs

print(f"Processed {len(outputs_per_frame)} frames")

# Access results for a specific frame
frame_0_outputs = outputs_per_frame[0]
print(f"Detected {len(frame_0_outputs['object_ids'])} objects")
print(f"Object IDs: {frame_0_outputs['object_ids'].tolist()}")
print(f"Scores: {frame_0_outputs['scores'].tolist()}")
print(f"Boxes shape (XYXY format, absolute coordinates): {frame_0_outputs['boxes'].shape}")
print(f"Masks shape: {frame_0_outputs['masks'].shape}")

out_path = "./logs/sam3_transformers/example2_masks_overlay.mp4"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

alpha = 0.45
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# fps를 알고 싶으면 위에서 load_video(video_url, backend="opencv")를
# video_frames, fps = load_video(...)로 바꾸고 여기서 fps 사용하세요.
fps = 10.0

first = video_frames[0]
if not isinstance(first, np.ndarray):
    first = np.asarray(first)
h, w = first.shape[:2]

writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

def color_for_i(i: int):
    base = (i * 1103515245 + 12345) & 0x7FFFFFFF
    r = 60 + (base % 196)
    g = 60 + ((base // 7) % 196)
    b = 60 + ((base // 49) % 196)
    return (int(b), int(g), int(r))  # BGR

for frame_idx in sorted(outputs_per_frame.keys()):
    frame = video_frames[frame_idx]
    if not isinstance(frame, np.ndarray):
        frame = np.asarray(frame)

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    masks = outputs_per_frame[frame_idx]["masks"]  # (N,H,W)
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()

    overlay = frame_bgr.astype(np.float32)
    for i in range(masks.shape[0]):
        m = masks[i] > 0.0
        if not m.any():
            continue
        color = np.array(color_for_i(i), dtype=np.float32)
        overlay[m] = alpha * color + (1.0 - alpha) * overlay[m]

    overlay = overlay.astype(np.uint8)
    cv2.putText(
        overlay,
        f"frame={frame_idx}  num_obj={masks.shape[0]}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    writer.write(overlay)

writer.release()
print("Saved:", out_path)
