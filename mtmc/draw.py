"""Общие функции рисования: цвета из ID, текст (PIL, кириллица), ffmpeg-писатель."""
from __future__ import annotations

import colorsys
import logging
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

GRAY = (160, 160, 160)

FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf", "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def id_color(i: int) -> tuple[int, int, int]:
    """Детерминированный цвет (BGR) из ID: оттенок (id*0.618)%1, S=V=0.85."""
    h = (int(i) * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.85)
    return int(b * 255), int(g * 255), int(r * 255)


class TextRenderer:
    """Рисует текст через PIL (кириллица, '·'); спрайты кэшируются. Без шрифта — Hershey из OpenCV."""

    def __init__(self, font_path: str | None = None, size: int = 16):
        self.size = size
        self.font = None
        try:
            from PIL import ImageFont
            paths = [font_path] if font_path else []
            paths += FONT_CANDIDATES
            for p in paths:
                if p and Path(p).exists():
                    self.font = ImageFont.truetype(p, size)
                    break
        except ImportError:
            pass
        if self.font is None:
            log.warning("TTF-шрифт не найден, текст будет без кириллицы")
        self._cache: dict = {}

    def sprite(self, text: str, color=(255, 255, 255)) -> np.ndarray:
        """BGRA-спрайт текста (кэш)."""
        key = (text, tuple(color))
        sp = self._cache.get(key)
        if sp is not None:
            return sp
        if self.font is not None:
            from PIL import Image, ImageDraw
            l, t, r, b = self.font.getbbox(text)
            w, h = max(r - min(l, 0), 1), max(b - min(t, 0), 1)
            img = Image.new("RGBA", (w + 2, h + 2), (0, 0, 0, 0))
            ImageDraw.Draw(img).text((1 - min(l, 0), 1 - min(t, 0)), text, font=self.font,
                                     fill=(color[2], color[1], color[0], 255))
            arr = np.array(img)
            sp = arr[:, :, [2, 1, 0, 3]].copy()
        else:
            txt = text.encode("ascii", "replace").decode()
            scale = self.size / 30
            (w, h), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            sp = np.zeros((h + base + 2, w + 2, 4), np.uint8)
            cv2.putText(sp, txt, (1, h + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (*color, 255), 1, cv2.LINE_AA)
        if len(self._cache) > 5000:
            self._cache.clear()
        self._cache[key] = sp
        return sp

    def put(self, img: np.ndarray, text: str, x: int, y: int, color=(255, 255, 255),
            bg: tuple | None = None, pad: int = 2) -> tuple[int, int]:
        """Текст с левым верхним углом (x, y), опционально на плашке. Возвращает (w, h) плашки."""
        sp = self.sprite(text, color)
        h, w = sp.shape[:2]
        W, H = w + 2 * pad, h + 2 * pad
        if bg is not None:
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + W, img.shape[1]), min(y + H, img.shape[0])
            if x2 > x1 and y2 > y1:
                img[y1:y2, x1:x2] = bg
        blend(img, sp, x + pad, y + pad)
        return W, H


def blend(img: np.ndarray, sprite: np.ndarray, x: int, y: int) -> None:
    """Накладывает BGRA-спрайт на BGR-кадр с обрезкой по краям."""
    h, w = sprite.shape[:2]
    H, W = img.shape[:2]
    x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x2 <= x1 or y2 <= y1:
        return
    sp = sprite[y1 - y:y2 - y, x1 - x:x2 - x]
    a = sp[:, :, 3:4].astype(np.float32) / 255.0
    roi = img[y1:y2, x1:x2].astype(np.float32)
    img[y1:y2, x1:x2] = (roi * (1 - a) + sp[:, :, :3].astype(np.float32) * a).astype(np.uint8)


def readable_text_color(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = bg
    return (0, 0, 0) if 0.299 * r + 0.587 * g + 0.114 * b > 150 else (255, 255, 255)


@lru_cache(maxsize=1)
def _which(name: str) -> str | None:
    return shutil.which(name)


def find_ffmpeg(cfg_value: str = "ffmpeg") -> str:
    """Свой путь из конфига важнее всего; иначе свежий ffmpeg из imageio-ffmpeg: системный в PATH бывает старым
    и не понимает нужных опций (-fps_mode появился в ffmpeg 5.1)."""
    if cfg_value != "ffmpeg" and Path(cfg_value).exists():
        return cfg_value
    if cfg_value == "ffmpeg":
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    p = _which(cfg_value)
    if p:
        return p
    raise FileNotFoundError("ffmpeg не найден (установите системный ffmpeg или задайте render.ffmpeg)")


class FFmpegWriter:
    """Пишет BGR-кадры в mp4 (H.264, yuv420p) через pipe."""

    def __init__(self, path: str | Path, width: int, height: int, fps: float, crf: int = 23,
                 preset: str = "veryfast", ffmpeg: str = "ffmpeg", gop: int | None = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.size = (width, height)
        cmd = [find_ffmpeg(ffmpeg), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:.6f}",
               "-i", "-", "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf)]
        if gop:   # частые опорные кадры — быстрая перемотка в браузере
            cmd += ["-g", str(gop), "-keyint_min", str(gop)]
        cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        log.debug("ffmpeg: %s", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.n = 0

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            err = self.proc.stderr.read().decode(errors="replace")
            raise RuntimeError(f"ffmpeg упал: {err}") from None
        self.n += 1

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        rc = self.proc.wait()
        err = self.proc.stderr.read().decode(errors="replace") if self.proc.stderr else ""
        if rc != 0:
            raise RuntimeError(f"ffmpeg завершился с кодом {rc}: {err}")
