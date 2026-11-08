"""Объектив камеры: выпрямление широкоугольного кадра и оценка объектива по прямым линиям сцены.

Гомография «пиксель -> план» верна только для прямолинейного кадра. У камер офиса угол обзора больше 100°, и край
пола в кадре изогнут, поэтому прямая гомография сдвигает людей у края на метры.

Модель для широких объективов — равноудалённая проекция: r = f·θ, где r — расстояние от центра кадра, θ — угол от
оптической оси. Она задаётся в паспорте как K (фокус и центр) и dist = [0]: один коэффициент отличает её от
пяти и более коэффициентов модели OpenCV. Выпрямление считается по формуле, без итераций, поэтому работает до края.
Выпрямленные точки — это пиксели камеры-обскуры с тем же K. H в паспорте относится к ним.

Оценка: в комнате много прямых (стены, двери, плитка, столы), а в кадре они изогнуты. Подбирается такой фокус, при
котором они выпрямляются сильнее всего. Если выигрыш мал, объектив считается обычным и K не пишется.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

MAX_THETA = np.radians(80)      # у края круга изображения направление уже ненадёжно


def is_fisheye(dist: np.ndarray | None) -> bool:
    return dist is not None and np.asarray(dist).size == 1


def undistort_points(pts, K: np.ndarray | None, dist: np.ndarray | None) -> np.ndarray:
    """Пиксели кадра -> пиксели выпрямленного кадра (n, 2)."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if K is None or dist is None:
        return p
    K = np.asarray(K, dtype=np.float64)
    if not is_fisheye(dist):
        return cv2.undistortPoints(p.reshape(-1, 1, 2), K, np.asarray(dist, dtype=np.float64), P=K).reshape(-1, 2)
    f, c = K[0, 0], K[:2, 2]
    v = p - c
    r = np.hypot(*v.T)
    theta = np.minimum(r / f, MAX_THETA)
    k = np.ones(len(p))
    m = r > 1e-9
    k[m] = f * np.tan(theta[m]) / r[m]
    return c + v * k[:, None]


def distort_points(pts, K: np.ndarray | None, dist: np.ndarray | None) -> np.ndarray:
    """Обратно: пиксели выпрямленного кадра -> пиксели кадра. Только для равноудалённой модели."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if K is None or not is_fisheye(dist):
        return p
    K = np.asarray(K, dtype=np.float64)
    f, c = K[0, 0], K[:2, 2]
    v = p - c
    r = np.hypot(*v.T)
    k = np.ones(len(p))
    m = r > 1e-9
    k[m] = f * np.arctan(r[m] / f) / r[m]
    return c + v * k[:, None]


# ---------------------------------------------------------------- оценка по линиям
def edge_pieces(gray: np.ndarray, seg: int = 110, skip_top: int = 40) -> list[np.ndarray]:
    """Гладкие куски контуров длиной seg: почти прямые или плавно изогнутые. Углы и узоры не проходят."""
    e = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    e[:skip_top] = 0                                    # надпись с часами
    contours, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        p = c[:, 0, :].astype(np.float64)
        if len(p) < seg:
            continue
        for i in range(0, len(p) - seg + 1, seg):
            q = p[i:i + seg:4]
            m = q.mean(0)
            axes = np.linalg.svd(q - m, full_matrices=False)[2]
            t, d = (q - m) @ axes[0], (q - m) @ axes[1]
            if t.max() - t.min() < seg * 0.8:           # контур Canny идёт туда и обратно
                continue
            fit = np.polyval(np.polyfit(t, d, 2), t)
            if np.sqrt(np.mean((d - fit) ** 2)) <= 0.8 and np.abs(d).max() <= seg * 0.11:
                out.append(q)
    return out


def _bend(pieces: list[np.ndarray], K: np.ndarray | None, dist: np.ndarray | None, cap: float = 0.03) -> float:
    """Средняя кривизна кусков: толщина их облака точек, отнесённая к длине."""
    total = 0.0
    for q in pieces:
        u = undistort_points(q, K, dist)
        s = np.linalg.svd(u - u.mean(0), compute_uv=False)
        total += min(s[1] / max(s[0], 1e-9), cap)
    return total / max(len(pieces), 1)


@dataclass
class LensFit:
    f: float | None          # фокус в пикселях кадра; None — обычный объектив
    hfov_deg: float
    pieces: int
    bend_raw: float
    bend_fixed: float


def fit_lens(gray: np.ndarray, min_pieces: int = 20, min_gain: float = 0.08) -> LensFit:
    """Фокус равноудалённой модели по одному кадру. Кадр обрабатывается в половинном размере."""
    h, w = gray.shape[:2]
    small = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    pieces = edge_pieces(small)
    raw = _bend(pieces, None, None)
    if len(pieces) < min_pieces:
        return LensFit(None, 0.0, len(pieces), raw, raw)
    W, H = small.shape[1], small.shape[0]
    K = np.array([[1.0, 0, W / 2], [0, 1.0, H / 2], [0, 0, 1.0]])
    zero = np.zeros(1)

    def cost(f: float) -> float:
        K[0, 0] = K[1, 1] = f
        return _bend(pieces, K, zero)

    grid = np.arange(0.35, 1.5, 0.05) * W
    costs = [cost(f) for f in grid]
    i = int(np.argmin(costs))
    fine = grid[i] + np.arange(-0.04, 0.041, 0.01) * W
    fc = [cost(f) for f in fine]
    f, best = float(fine[int(np.argmin(fc))]), float(min(fc))
    if best > raw * (1 - min_gain):
        return LensFit(None, 0.0, len(pieces), raw, best)
    return LensFit(f * 2, float(np.degrees(w / 2 / (f * 2)) * 2), len(pieces), raw, best)


def still_frame(video, fps: float | None = None, n: int = 11) -> np.ndarray | None:
    """Медиана нескольких кадров: люди исчезают, остаются стены. Серый кадр."""
    from .video_io import VideoSource

    vs = VideoSource(video, fps=fps)
    total = vs.n_frames or 0
    if total < n:
        return None
    frames = []
    for k in range(n):
        f0 = int((k + 0.5) * total / n)
        if vs.is_folder:
            img = next(iter(vs.frames(f0, f0 + 1)), (0, None))[1]
        else:
            cap = cv2.VideoCapture(str(vs.path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            ok, img = cap.read()
            cap.release()
            img = img if ok else None
        if img is not None:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return np.median(np.stack(frames), 0).astype(np.uint8) if frames else None


def estimate_lens(video, width: int, height: int, fps: float | None = None) -> tuple[dict, LensFit | None]:
    """Поля паспорта для широкого объектива ({K, dist}) или {}; вторым — результат подбора."""
    gray = still_frame(video, fps)
    if gray is None:
        return {}, None
    fit = fit_lens(gray)
    if fit.f is None:
        return {}, fit
    K = [[fit.f, 0.0, width / 2], [0.0, fit.f, height / 2], [0.0, 0.0, 1.0]]
    return {"K": K, "dist": [0.0]}, fit
