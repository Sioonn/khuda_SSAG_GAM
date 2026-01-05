from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import numpy as np


def _color_for_id(obj_id: int) -> Tuple[int, int, int]:
    base = (obj_id * 1103515245 + 12345) & 0x7FFFFFFF
    r = 60 + (base % 196)
    g = 60 + ((base // 7) % 196)
    b = 60 + ((base // 49) % 196)
    return (int(b), int(g), int(r))  # BGR


@dataclass
class TakeDebugVideoWriter:
    out_path: str
    fps: float
    roi_polygon_xy: Optional[Iterable[Tuple[int, int]]] = None
    alpha: float = 0.45

    _writer: Any = None
    _cv2: Any = None
    _size_wh: Optional[Tuple[int, int]] = None
    _roi_pts: Optional[np.ndarray] = None

    def open_if_needed(self, frame_rgb: np.ndarray) -> None:
        if self._writer is not None:
            return
        import cv2  # type: ignore

        self._cv2 = cv2
        h, w = frame_rgb.shape[:2]
        self._size_wh = (w, h)

        out_p = Path(self.out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(out_p), fourcc, float(self.fps), (w, h))
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter: {self.out_path}")

        if self.roi_polygon_xy is not None:
            self._roi_pts = np.array(list(self.roi_polygon_xy), dtype=np.int32).reshape(
                (-1, 1, 2)
            )

    def write(
        self,
        *,
        frame_index: int,
        frame_rgb: np.ndarray,
        processed_outputs: Dict[str, Any],
        take_count: int,
        taken_object_ids: Set[int],
    ) -> None:
        self.open_if_needed(frame_rgb)
        cv2 = self._cv2

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        overlay_f = frame_bgr.astype(np.float32)

        obj_ids = processed_outputs.get("object_ids")
        masks = processed_outputs.get("masks")
        if obj_ids is not None and masks is not None:
            obj_ids_np = np.asarray(obj_ids).astype(int)
            masks_np = np.asarray(masks)
            # masks expected as (N,H,W) uint8(0/1) from normalize_processed_outputs
            for i in range(int(masks_np.shape[0])):
                m = masks_np[i].astype(bool)
                if not m.any():
                    continue
                obj_id = int(obj_ids_np[i])
                if obj_id in taken_object_ids:
                    color = np.array((0, 0, 255), dtype=np.float32)
                    a = min(0.65, self.alpha + 0.2)
                else:
                    color = np.array(_color_for_id(obj_id), dtype=np.float32)
                    a = self.alpha
                overlay_f[m] = a * color + (1.0 - a) * overlay_f[m]

        overlay = overlay_f.astype(np.uint8)
        if self._roi_pts is not None:
            cv2.polylines(overlay, [self._roi_pts], isClosed=True, color=(0, 255, 255), thickness=2)

        w, _ = self._size_wh or (overlay.shape[1], overlay.shape[0])
        hud = f"take_count={int(take_count)}"
        (tw, th), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        x = max(10, w - tw - 10)
        y = 30
        cv2.rectangle(overlay, (x - 6, y - th - 10), (x + tw + 6, y + 8), (0, 0, 0), -1)
        cv2.putText(
            overlay,
            hud,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"frame={int(frame_index)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        self._writer.write(overlay)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

