"""Чтение настоящего плана этажа: стены среди мебели и штриховки, камеры, нарисованные поверх цветными секторами.

На архитектурном плане тёмные пиксели — это и стены, и столы, и подписи, и штриховка пола. Простой порог принимает
всё это за стены, и автопривязка считает, что камера никого не видит. Стенами считается только то, что либо толстое,
либо тянется прямой линией дольше мебели.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WALL_THICK_M = 0.1          # толще — стена; тоньше — контур мебели или подпись
WALL_LONG_M = 2.0           # прямая тёмная линия длиннее — перегородка: столы и стулья короче
WALL_MIN_M = 0.4            # сторона самого маленького куска стены (колонна)
DIGIT_MIN = 0.45            # совпадение с образцом цифры: ниже номер считается не прочитанным


def read_plan(path: str | Path) -> np.ndarray:
    """Цветная картинка плана; путь может быть с кириллицей."""
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"не удалось прочитать план {path}: нужен png или jpg")
    return img


def colored(img: np.ndarray) -> np.ndarray:
    """Цветные пометки поверх плана: стенами они не считаются."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] > 80) & (hsv[..., 2] > 80)).astype(np.uint8)
    return cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0


def extract_walls(img: np.ndarray, px_per_m: float, wall_thr: int) -> np.ndarray:
    """bool (H, W): стены плана без мебели, штриховки, подписей и цветных пометок."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = ((gray < wall_thr) & ~colored(img)).astype(np.uint8)
    t = max(3, int(round(WALL_THICK_M * px_per_m)))
    thick = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t)))
    n = max(15, int(round(WALL_LONG_M * px_per_m)))
    lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((1, n), np.uint8)) | \
        cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((n, 1), np.uint8))
    walls = thick | lines
    # обломки меньше колонны — буквы подписей и детали мебели
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(walls, connectivity=8)
    small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < (WALL_MIN_M * px_per_m) ** 2)
    walls[np.isin(lab, small)] = 0
    return cv2.dilate(walls, np.ones((3, 3), np.uint8)) > 0


@dataclass
class CameraMark:
    """Камера, найденная на плане: начало сектора, куда он смотрит, номер из значка (None — не прочитан).

    Камера стоит там, где линии начинаются. Точка, где сошлись бы их продолжения, для этого не годится: у почти
    параллельных линий она уходит на метры за стену."""
    pos: tuple[float, float]
    look_at: tuple[float, float]
    num: int | None


def _red(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (((h <= 14) | (h >= 166)) & (s >= 110) & (v >= 110)).astype(np.uint8)


def _digit_templates() -> dict[int, np.ndarray]:
    from PIL import Image, ImageDraw, ImageFont

    font = None
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            font = ImageFont.truetype(name, 64)
            break
        except OSError:
            continue
    out = {}
    for d in range(1, 10):
        im = Image.new("L", (120, 120), 0)
        ImageDraw.Draw(im).text((30, 15), str(d), fill=255, font=font or ImageFont.load_default())
        out[d] = _normalize(np.asarray(im) > 128)
    return out


def _normalize(mask: np.ndarray) -> np.ndarray:
    """Цифра в рамке 28×28 по высоте, с сохранением пропорций: иначе тонкая единица растягивается в кляксу."""
    ys, xs = np.where(mask)
    out = np.zeros((28, 28), np.uint8)
    if not len(xs):
        return out > 0
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
    w = min(max(int(round(crop.shape[1] * 28 / crop.shape[0])), 1), 28)
    x0 = (28 - w) // 2
    out[:, x0:x0 + w] = cv2.resize(crop, (w, 28), interpolation=cv2.INTER_AREA)
    return out > 127


def read_digit(img: np.ndarray, c: tuple[float, float], r: float) -> int | None:
    """Белая цифра в красном значке. Смотрим только внутрь круга: вокруг него белый фон плана."""
    x, y, q = int(c[0]), int(c[1]), int(r * 0.8)
    x0, y0 = max(x - q, 0), max(y - q, 0)
    crop = img[y0:y + q, x0:x + q]
    if not crop.size:
        return None
    inside = np.zeros(crop.shape[:2], np.uint8)
    cv2.circle(inside, (x - x0, y - y0), int(r * 0.78), 1, -1)
    white = (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) > 170) & (inside > 0)
    if white.sum() < 0.03 * inside.sum():
        return None
    g = _normalize(white)
    score = {d: (t & g).sum() / max((t | g).sum(), 1) for d, t in _digit_templates().items()}
    d = max(score, key=score.get)
    return d if score[d] >= DIGIT_MIN else None


def _segments(mask: np.ndarray, min_len: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Вытянутые компоненты как отрезки (конец, конец)."""
    n, lab = cv2.connectedComponents(mask)
    out = []
    for i in range(1, n):
        ys, xs = np.where(lab == i)
        if len(xs) < 8:
            continue
        pts = np.column_stack((xs, ys)).astype(np.float32)
        c = pts.mean(0)
        u = np.linalg.svd(pts - c, full_matrices=False)[2][0]
        t = (pts - c) @ u
        side = np.abs((pts - c) @ np.array([-u[1], u[0]]))
        if t.max() - t.min() >= min_len and t.max() - t.min() >= 4 * max(np.percentile(side, 95) * 2, 1):
            out.append((c + u * t.min(), c + u * t.max()))
    return out


def _cross(a: tuple, b: tuple) -> np.ndarray | None:
    d1, d2 = a[1] - a[0], b[1] - b[0]
    det = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(det) < 1e-6 * np.linalg.norm(d1) * np.linalg.norm(d2):
        return None
    w = b[0] - a[0]
    return a[0] + d1 * ((w[0] * d2[1] - w[1] * d2[0]) / det)


def find_camera_marks(img: np.ndarray) -> list[CameraMark]:
    """Красные секторы обзора: две линии, сходящиеся к камере, и значок с номером рядом."""
    red = _red(img)
    if red.sum() < 50:
        return []
    filled = np.zeros_like(red)
    cs, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, cs, -1, 1, -1)
    dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    # значки: круги намного толще линий
    badges: list[tuple[tuple[float, float], float]] = []
    if dist.max() >= 7:
        n, lab = cv2.connectedComponents((dist >= 0.55 * dist.max()).astype(np.uint8))
        for i in range(1, n):
            ys, xs = np.where(lab == i)
            k = int(np.argmax(dist[ys, xs]))
            badges.append(((float(xs[k]), float(ys[k])), float(dist[ys[k], xs[k]])))
    no_badge = red.copy()
    for c, r in badges:
        cv2.circle(no_badge, (int(c[0]), int(c[1])), int(r * 1.45), 0, -1)     # вместе с остриём значка
    size = max(img.shape[:2])
    segs = _segments(no_badge, max(20.0, size * 0.02))
    pairs = []
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            x = _cross(segs[i], segs[j])
            if x is None:
                continue
            di = sorted(np.linalg.norm(x - p) for p in segs[i])
            dj = sorted(np.linalg.norm(x - p) for p in segs[j])
            li, lj = di[1] - di[0], dj[1] - dj[0]
            on_seg_i = abs(di[0] + di[1] - np.linalg.norm(segs[i][0] - segs[i][1])) < 1
            on_seg_j = abs(dj[0] + dj[1] - np.linalg.norm(segs[j][0] - segs[j][1])) < 1
            if di[0] + dj[0] > 3 * (li + lj) or (on_seg_i and di[0] > 0.2 * li) or (on_seg_j and dj[0] > 0.2 * lj):
                continue
            pairs.append((di[0] + dj[0], i, j, x))
    used: set[int] = set()
    free = list(range(len(badges)))
    marks = []
    for _, i, j, x in sorted(pairs, key=lambda p: p[0]):
        if i in used or j in used:
            continue
        used |= {i, j}
        far = [max(s, key=lambda p: np.linalg.norm(p - x)) for s in (segs[i], segs[j])]
        look = (far[0] + far[1]) / 2
        near = (min(segs[i], key=lambda p: np.linalg.norm(p - x)) + min(segs[j], key=lambda p: np.linalg.norm(p - x))) / 2
        num = None
        if free:
            k = min(free, key=lambda b: min(np.linalg.norm(np.array(badges[b][0]) - x),
                                            np.linalg.norm(np.array(badges[b][0]) - near)))
            free.remove(k)
            num = read_digit(img, *badges[k])
        marks.append(CameraMark((round(float(near[0]), 1), round(float(near[1]), 1)),
                                (round(float(look[0]), 1), round(float(look[1]), 1)), num))
    return sorted(marks, key=lambda m: (m.num is None, m.num or 0))
