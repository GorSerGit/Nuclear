"""Сохранение кропов людей."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def padded_box(box: np.ndarray, pad: float, width: int, height: int) -> tuple[int, int, int, int]:
    """Расширяет бокс на pad со всех сторон и обрезает по кадру -> целые x1, y1, x2, y2."""
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = x2 - x1, y2 - y1
    x1, x2 = x1 - pad * w, x2 + pad * w
    y1, y2 = y1 - pad * h, y2 + pad * h
    x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    x2, y2 = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
    return x1, y1, x2, y2


def inner_crop(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Кроп ровно по боксу (для оценки резкости)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = padded_box(box, 0.0, w, h)
    return frame[y1:y2, x1:x2]


def crop_rel_path(cam_id: int, local_id: int, frame_idx: int) -> str:
    """Путь кропа относительно папки данных (всегда через '/')."""
    return f"crops/cam{cam_id}/{local_id}/{frame_idx}.jpg"


def save_crop(frame: np.ndarray, box: np.ndarray, pad: float, jpeg_quality: int, out_path: Path) -> bool:
    """Сохраняет расширенный кроп в JPEG. False, если кроп пустой."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = padded_box(box, pad, w, h)
    if x2 <= x1 or y2 <= y1:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", frame[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    if not ok:
        return False
    out_path.write_bytes(buf.tobytes())   # imwrite не любит не-ASCII пути в Windows
    return True


def read_image(path: Path) -> np.ndarray | None:
    """cv2.imread, работающий с не-ASCII путями."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
