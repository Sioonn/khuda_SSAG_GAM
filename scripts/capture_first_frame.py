from __future__ import annotations

import argparse
from pathlib import Path


def capture_first_frame(video_path: str, out_path: str) -> str:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read first frame: {video_path}")

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_p), frame_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_path}")
    return str(out_p)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_video = str((repo_root / "data" / "transform_ex.MP4").resolve())
    default_out = str((repo_root / "logs" / "frames" / "transform_ex_frame0.png").resolve())

    parser = argparse.ArgumentParser(description="Capture the first frame of a video and save as an image.")
    parser.add_argument("--video", default=default_video, help="Path to input video.")
    parser.add_argument("--out", default=default_out, help="Path to output image (png/jpg).")
    args = parser.parse_args()

    saved = capture_first_frame(args.video, args.out)
    print(saved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

