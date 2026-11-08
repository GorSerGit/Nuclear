"""Что не надо показывать детектору и что не надо считать человеком.

Пропуск пустых кадров. На записях офиса большая часть кадров пуста, а детектор стоит одинаково на любом кадре.
FrameGate пропускает кадр, только если всё сразу верно:
  - детектор давно (дольше буфера трекера) никого не находил, значит, в трекере не осталось живых треков;
  - проверка детектором была меньше check_s назад;
  - кадр не отличается от последнего пустого кадра.
Поэтому сидящий неподвижно человек не теряется: пока он найден, кадры идут все. Любая ошибка ведёт в сторону
лишней работы, а не потери: смена света только запустит детектор. Если на пропущенных кадрах детектор и правда
никого бы не нашёл, результат совпадает с полной обработкой.

Постоянные изменения, где детектор никого не видит (часы в углу кадра, телевизор), запоминаются и перестают
запускать детектор. Плановая проверка раз в check_s идёт всё равно, поэтому человек там потеряет не больше неё.

Зоны без людей. Телевизор и зеркало показывают настоящих людей, и детектор уверен в них так же, как в живых.
Отсечь их по картинке нельзя, зато они не двигаются: это зоны кадра из паспорта (ignore_zones).
"""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from .config import GateCfg


def zone_mask(polygons: Iterable[np.ndarray], height: int, width: int) -> np.ndarray | None:
    """Маска зон без людей в размере кадра; None, если зон нет."""
    polys = [np.round(np.asarray(p)).astype(np.int32) for p in polygons]
    if not polys:
        return None
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, polys, 1)
    return mask.astype(bool)


def in_zone(mask: np.ndarray | None, box: np.ndarray, share: float) -> bool:
    """Рамка лежит в зоне: низ рамки в зоне и она сама почти вся там.

    Живой человек перед телевизором или зеркалом стоит на полу, низ его рамки вне зоны, и он остаётся.
    Низ берётся чуть выше края рамки: рамка фигуры на экране часто заходит на край зоны."""
    if mask is None:
        return False
    h, w = mask.shape
    x1, y1 = max(int(box[0]), 0), max(int(box[1]), 0)
    x2, y2 = min(int(box[2]), w), min(int(box[3]), h)
    if x2 <= x1 or y2 <= y1:
        return False
    if not mask[max(y2 - 1 - (y2 - y1) // 30, 0), (x1 + x2) // 2]:
        return False
    return float(mask[y1:y2, x1:x2].mean()) >= share


class FrameGate:
    """Решает, нужен ли кадру детектор. want() перед детектором, seen() после него."""

    def __init__(self, cfg: GateCfg, fps: float, track_buffer_s: float, height: int, width: int,
                 zones: np.ndarray | None = None):
        self.cfg = cfg
        self.fps = fps
        self.hold_s = max(cfg.hold_s, track_buffer_s + 1.0)   # за это время трекер забывает всех, кого видел
        self.size = (cfg.width, max(int(round(height * cfg.width / width)), 1))
        self.min_px = max(int(cfg.min_share * self.size[0] * self.size[1]), 1)
        self.quiet = None if zones is None else \
            cv2.resize(zones.astype(np.uint8), self.size, interpolation=cv2.INTER_NEAREST) > 0
        self.ref: np.ndarray | None = None
        self.small: np.ndarray | None = None
        self.changed: np.ndarray | None = None    # что запустило детектор: если людей нет, это шум
        self.hits = np.zeros(self.size[::-1], np.uint16)
        self.noise = np.zeros(self.size[::-1], bool)
        self.last_check = -np.inf
        self.last_person = -np.inf

    def _diff(self) -> np.ndarray:
        d = cv2.absdiff(self.small, self.ref) > self.cfg.diff
        d &= ~self.noise
        if self.quiet is not None:
            d &= ~self.quiet
        return d

    def want(self, frame_idx: int, frame: np.ndarray) -> bool:
        t = frame_idx / self.fps
        # Прореживание до resize: область на полном кадре 2592×1944 дороже самой детекции
        step = max(frame.shape[1] // (self.size[0] * 2), 1)
        gray = cv2.cvtColor(cv2.resize(frame[::step, ::step], self.size, interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)
        self.small = cv2.GaussianBlur(gray, (3, 3), 0)
        self.changed = None
        if self.ref is None or t - self.last_person < self.hold_s:
            return True
        d = self._diff()
        if int(d.sum()) >= self.min_px:
            self.changed = d
            return True
        return t - self.last_check >= self.cfg.check_s

    def seen(self, frame_idx: int, person: bool) -> None:
        t = frame_idx / self.fps
        self.last_check = t
        if person:
            self.last_person = t
            return
        if self.changed is not None:
            self.hits[self.changed] += 1
            noisy = (self.hits >= self.cfg.noise_hits).astype(np.uint8)
            self.noise = cv2.dilate(noisy, np.ones((5, 5), np.uint8)) > 0
        self.ref = self.small
