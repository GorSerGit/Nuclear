"""Одежда по зонам: яркость верха и низа фигуры.

OSNet почти не различает людей в тёмном, а светлый верх и тёмный низ видны сразу. Берётся только яркость: неон в офисе
сильно меняет цвет, а яркость заметно меньше. Этого хватает, чтобы запрещать явно разное и находить подмену человека
внутри треклета, но не хватает, чтобы по совпадению цвета считать людей одним: половина офиса в чёрном."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from .config import ReIDCfg
from .crops import padded_box, read_image

ZONES = ((0.18, 0.50), (0.55, 0.90))     # верх и низ, доли высоты фигуры
MIN_PIXELS = 20


def crop_colors(img: np.ndarray, poly: np.ndarray | None = None) -> np.ndarray | None:
    """Яркость верха и низа (0–255). poly — контур фигуры в пикселях кропа; без него средняя полоса кропа."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]
    mask = np.zeros((h, w), np.uint8)
    if poly is not None and len(poly) >= 3:
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
        k = max(3, h // 40) | 1           # край контура задевает фон
        mask = cv2.erode(mask, np.ones((k, k), np.uint8))
    if mask.sum() < MIN_PIXELS:
        mask[:, int(w * 0.3):int(w * 0.7)] = 1
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    y0, y1 = np.percentile(ys, 1), np.percentile(ys, 99)
    out = []
    for a, b in ZONES:
        m = (ys >= y0 + a * (y1 - y0)) & (ys < y0 + b * (y1 - y0))
        if m.sum() < MIN_PIXELS:
            return None
        out.append(float(np.median(gray[ys[m], xs[m]])))
    return np.array(out)


def piece_series(piece: pd.DataFrame, root: Path, sil: dict | None, pad: float,
                 read: Callable[[Path], np.ndarray | None] = read_image) -> tuple[np.ndarray, np.ndarray]:
    """По сохранённым кропам куска: времена (n,) и яркости (n, 2) в порядке времени."""
    times, values = [], []
    c = piece[piece["crop_path"].notna()].sort_values("t")
    for r in c.itertuples():
        img = read(root / r.crop_path)
        if img is None or img.size == 0:
            continue
        poly = (sil or {}).get((int(r.frame_idx), int(r.local_id)))
        if poly is not None:
            x0, y0, _, _ = padded_box(np.array([r.x1, r.y1, r.x2, r.y2]), pad, 10 ** 9, 10 ** 9)
            poly = poly - np.array([x0, y0], np.float32)
        v = crop_colors(img, poly)
        if v is not None:
            times.append(float(r.t))
            values.append(v)
    return np.array(times), np.array(values).reshape(-1, 2)


def clothes_cuts(times: np.ndarray, values: np.ndarray, cfg: ReIDCfg) -> list[float]:
    """Моменты подмены: яркость зоны резко и устойчиво сменилась. Шум отсекается разбросом внутри сторон."""
    side = cfg.clothes_cut_side
    cuts: list[float] = []

    def rec(lo: int, hi: int) -> None:
        if hi - lo < 2 * side:
            return
        best, best_k = -np.inf, None
        for k in range(lo + side, hi - side + 1):
            left, right = values[lo:k], values[k:hi]
            ml, mr = np.median(left, 0), np.median(right, 0)
            dl, dr = np.abs(left - ml), np.abs(right - mr)
            mad = np.maximum(np.median(dl, 0), np.median(dr, 0))
            jump = np.abs(ml - mr)
            score = jump - dl.mean(0) - dr.mean(0)     # медиана терпит чужие кадры: точное место даёт меньший разброс
            for z in range(2):
                if score[z] > best and jump[z] >= cfg.clothes_cut and mad[z] * cfg.clothes_spread < jump[z]:
                    best, best_k = float(score[z]), k
        if best_k is None:
            return
        cuts.append(float((times[best_k - 1] + times[best_k]) / 2))
        rec(lo, best_k)
        rec(best_k, hi)

    rec(0, len(times))
    return sorted(cuts)


def clash(a: np.ndarray | None, b: np.ndarray | None, limit: float) -> bool:
    """Одежда явно разная. Нет данных — не мешаем склейке."""
    return limit > 0 and a is not None and b is not None and float(np.abs(a - b).max()) > limit


def save_colors(root: Path, colors: dict[int, np.ndarray]) -> None:
    (Path(root) / "clothes.json").write_text(json.dumps({str(k): v.tolist() for k, v in colors.items()}),
                                             encoding="utf-8")


def load_colors(root: Path) -> dict[int, np.ndarray]:
    p = Path(root) / "clothes.json"
    if not p.exists():
        return {}
    return {int(k): np.array(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
