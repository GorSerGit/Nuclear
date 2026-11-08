"""Геометрия: пиксель -> план, погрешность, многоугольники и зоны. Чистые функции."""
from __future__ import annotations

from typing import Iterable, Mapping

import cv2
import numpy as np

from .lens import undistort_points


def to_floor(H: np.ndarray, pts: Iterable, K: np.ndarray | None = None,
             dist: np.ndarray | None = None) -> np.ndarray:
    """Переводит точки (n, 2) в пикселях в метры на плане. При наличии K/dist сначала убирает дисторсию."""
    p = undistort_points(pts, K, dist).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(p, np.asarray(H, dtype=np.float64))
    return out.reshape(-1, 2)


def floor_sigma(H: np.ndarray, u: float, v: float, px: float = 4.0, base: float = 0.15,
                mult: float = 1.0, K: np.ndarray | None = None, dist: np.ndarray | None = None) -> float:
    """Погрешность положения на плане (м) при сдвиге точки ног на px пикселей."""
    p = to_floor(H, [(u, v), (u + px, v), (u, v + px)], K, dist)
    s = max(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[0]))
    return float(np.hypot(s, base) * mult)


def point_in_polygon(poly: np.ndarray, u: float, v: float) -> bool:
    """True, если точка внутри многоугольника или на его границе."""
    return cv2.pointPolygonTest(np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2),
                                (float(u), float(v)), False) >= 0


def zone_of(zones: Mapping[str, np.ndarray], u: float, v: float) -> str | None:
    """Имя первой зоны (в порядке паспорта), в которую попала точка."""
    for name, poly in zones.items():
        if point_in_polygon(poly, u, v):
            return name
    return None


def camera_from_H(H: np.ndarray, width: int, height: int) -> dict | None:
    """Где стоит камера, по гомографии пола «пиксель -> план» (центр кадра — оптическая ось).
    -> {"pos": (x, y) точка пола под камерой, "height": м, "f": px} или None, если разложить не удалось."""
    try:
        G = np.linalg.inv(np.asarray(H, dtype=np.float64))
        cx, cy = width / 2.0, height / 2.0
        G = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]]) @ G
        h1, h2 = G[:, 0], G[:, 1]
        f2 = -(h1[0] * h2[0] + h1[1] * h2[1]) / (h1[2] * h2[2])
        if not np.isfinite(f2) or f2 <= 0:
            return None
        f = float(np.sqrt(f2))
        A = np.diag([1 / f, 1 / f, 1.0]) @ G
        lam = 2.0 / (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
        ground = to_floor(H, [(cx, height - 1.0)])[0]           # низ кадра — пол перед камерой
        if (A @ np.r_[ground, 1.0])[2] < 0:
            lam = -lam
        r1, r2, t = lam * A[:, 0], lam * A[:, 1], lam * A[:, 2]
        R = np.column_stack([r1, r2, np.cross(r1, r2)])
        C = -R.T @ t
        return {"pos": (float(C[0]), float(C[1])), "height": float(abs(C[2])), "f": f}
    except (np.linalg.LinAlgError, ZeroDivisionError, ValueError):
        return None


def fov_polygon(H: np.ndarray, width: int, height: int, ref_xy: np.ndarray | None = None,
                n_edge: int = 16, K: np.ndarray | None = None, dist: np.ndarray | None = None) -> np.ndarray | None:
    """Участок пола, который видит камера: рамка кадра, спроецированная на план.
    Верх кадра (горизонт и всё, что дальше разумного) отрезается: остаются строки кадра, все точки которых
    лежат на полу перед камерой не дальше 2× размаха точек ref_xy (где камера видела людей)."""
    H = np.asarray(H, dtype=np.float64)

    def proj(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.c_[undistort_points(uv, K, dist), np.ones(len(uv))] @ H.T
        w = p[:, 2]
        if (np.c_[width / 2.0, height - 1.0, 1.0] @ H.T)[0, 2] < 0:   # знак: перед камерой w > 0
            w = -w
        return p[:, :2] / p[:, 2:3], w

    if ref_xy is not None and len(ref_xy) >= 10:
        c = np.median(ref_xy, 0)
        R = 2.0 * float(np.percentile(np.hypot(*(ref_xy - c).T), 99)) + 1.0
    else:
        c = proj(np.array([[width / 2.0, height - 1.0]]))[0][0]
        R = 30.0
    # Граница считается по столбцам кадра, снизу вверх. Обрезка верха целой строкой на широких объективах теряла дальний
    # коридор в середине кадра: углы той же строки уходят далеко, и камера «не видела» половину своих же наблюдений.
    us = np.linspace(0, width - 1, n_edge * 3)
    vs = np.linspace(height - 1, 0, 90)
    keep = []
    for u in us:
        xy, w = proj(np.c_[np.full_like(vs, u), vs])
        ok = (w > 1e-9) & (np.hypot(*(xy - c).T) <= R)
        n = len(vs) if ok.all() else int(np.argmin(ok))          # сколько точек подряд от низа кадра лежат на полу
        if n:
            keep.append((u, vs[n - 1]))
    if len(keep) < 3:
        return None
    border = np.array([*[[u, height - 1.0] for u, _ in keep], *[[u, v] for u, v in reversed(keep)]])
    xy, _ = proj(border)
    return xy.astype(np.float32)


def signed_dist(poly: np.ndarray, x: float, y: float) -> float:
    """Расстояние до границы многоугольника: > 0 внутри, < 0 снаружи (в единицах координат)."""
    return float(cv2.pointPolygonTest(np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2), (float(x), float(y)), True))


def overlap_cams_of(overlaps: Mapping[int, np.ndarray], u: float, v: float) -> str | None:
    """Строка '1,3' с камерами, в чью общую зону попала точка, или None."""
    cams = sorted(int(c) for c, poly in overlaps.items() if point_in_polygon(poly, u, v))
    return ",".join(str(c) for c in cams) if cams else None
