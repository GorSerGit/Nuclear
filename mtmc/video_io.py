"""Чтение кадров потоком: видеофайл (OpenCV) или папка с картинками."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class VideoSource:
    """Источник кадров. Номер кадра — от 0 в исходном видео (для папки — позиция в отсортированном списке)."""

    def __init__(self, path: str | Path, fps: float | None = None, exact_seek: bool = True):
        self.path = Path(path)
        self.exact_seek = exact_seek
        self.is_folder = self.path.is_dir()
        if self.is_folder:
            self.files = sorted(p for p in self.path.iterdir() if p.suffix.lower() in IMAGE_EXT)
            if not self.files:
                raise ValueError(f"в папке {self.path} нет кадров")
            if fps is None:
                raise ValueError("video: для папки с кадрами в паспорте обязателен fps")
            first = cv2.imread(str(self.files[0]))
            if first is None:
                raise ValueError(f"не читается кадр {self.files[0]}")
            self.height, self.width = first.shape[:2]
            self.n_frames = len(self.files)
            self.fps = float(fps)
        else:
            cap = cv2.VideoCapture(str(self.path))
            if not cap.isOpened():
                raise ValueError(f"OpenCV не открывает видео {self.path}")
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
            file_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps is None:
                if not file_fps or file_fps <= 0 or not np.isfinite(file_fps):
                    raise ValueError("fps не задан в паспорте и не читается из файла")
                fps = file_fps
            self.fps = float(fps)

    def frames(self, start: int = 0, end: int | None = None, stride: int = 1) -> Iterator[tuple[int, np.ndarray]]:
        """Кадры с номерами start, start+stride, ... < end. Пропущенные кадры не декодируются полностью."""
        start = max(int(start), 0)
        if self.is_folder:
            stop = self.n_frames if end is None else min(end, self.n_frames)
            for idx in range(start, stop, stride):
                img = cv2.imread(str(self.files[idx]))
                if img is None:
                    log.warning("пропуск нечитаемого кадра %s", self.files[idx])
                    continue
                yield idx, img
            return
        cap = cv2.VideoCapture(str(self.path))
        try:
            idx = 0
            if start > 0:
                if self.exact_seek:
                    while idx < start and cap.grab():
                        idx += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                    idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            while end is None or idx < end:
                need = (idx - start) % stride == 0
                if need:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    yield idx, frame
                else:
                    if not cap.grab():
                        break
                idx += 1
        finally:
            cap.release()

    def count_range(self, start: int, end: int | None, stride: int) -> int | None:
        """Сколько кадров будет выдано (для прогресса)."""
        total = self.n_frames
        stop = total if end is None else (min(end, total) if total else end)
        if stop is None:
            return None
        return max(0, (stop - start + stride - 1) // stride)
