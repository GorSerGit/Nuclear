"""Наблюдение: точка ног, ракурс, окклюзия, качество. Чистые функции без нейросетей и I/O."""
from __future__ import annotations

import cv2
import numpy as np

from .config import QualityCfg

# индексы COCO
NOSE, L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 0, 5, 6, 11, 12, 15, 16


def foot_point(box: np.ndarray, kp: np.ndarray, kp_conf: float = 0.5,
               ratio: float = 1.6) -> tuple[float, float, str]:
    """-> (u, v, source), source in {'ankles', 'body', 'box'}."""
    kp = np.asarray(kp, dtype=float)
    ank = kp[[L_ANK, R_ANK]]
    vis = ank[:, 2] > kp_conf
    if vis.any():
        u, v = ank[vis, :2].mean(0)
        return float(u), float(v), "ankles"
    sh, hp = kp[[L_SH, R_SH]], kp[[L_HIP, R_HIP]]
    if (sh[:, 2] > kp_conf).any() and (hp[:, 2] > kp_conf).any():
        s = sh[sh[:, 2] > kp_conf, :2].mean(0)
        h = hp[hp[:, 2] > kp_conf, :2].mean(0)
        u, v = h + ratio * (h - s)
        return float(u), float(v), "body"
    x1, y1, x2, y2 = [float(x) for x in box]
    return (x1 + x2) / 2, y2, "box"


def view_of(box: np.ndarray, kp: np.ndarray, kp_conf: float = 0.5, side_thr: float = 0.15) -> str:
    """Ракурс: front / back / side / unknown по плечам и носу."""
    kp = np.asarray(kp, dtype=float)
    ls, rs, nose = kp[L_SH], kp[R_SH], kp[NOSE]
    if ls[2] < kp_conf or rs[2] < kp_conf:
        return "unknown"
    dx = (ls[0] - rs[0]) / max(box[2] - box[0], 1)
    if abs(dx) < side_thr:
        return "side"
    if dx > 0:
        return "front" if nose[2] > kp_conf else "side"
    return "back"


def _union_area(rects: list[tuple[float, float, float, float]]) -> float:
    """Площадь объединения прямоугольников (сжатие координат)."""
    if not rects:
        return 0.0
    xs = sorted({r[0] for r in rects} | {r[2] for r in rects})
    ys = sorted({r[1] for r in rects} | {r[3] for r in rects})
    area = 0.0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in rects):
                area += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j])
    return area


def occluded_fractions(boxes: np.ndarray) -> np.ndarray:
    """Для каждого бокса — доля его площади, перекрытая боксами людей с большим y2 (ближе к камере)."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    n = len(boxes)
    occ = np.zeros(n)
    for i in range(n):
        x1, y1, x2, y2 = boxes[i]
        a = max(x2 - x1, 0) * max(y2 - y1, 0)
        if a <= 0:
            continue
        rects = []
        for j in range(n):
            if j == i or boxes[j, 3] <= y2:
                continue
            ix1, iy1 = max(x1, boxes[j, 0]), max(y1, boxes[j, 1])
            ix2, iy2 = min(x2, boxes[j, 2]), min(y2, boxes[j, 3])
            if ix2 > ix1 and iy2 > iy1:
                rects.append((ix1, iy1, ix2, iy2))
        occ[i] = min(_union_area(rects) / a, 1.0)
    return occ


def laplacian_var(gray: np.ndarray) -> float:
    """Дисперсия лапласиана (резкость) серого кропа."""
    if gray is None or gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def quality_factors(box: np.ndarray, kp: np.ndarray, conf: float, occ: float, blur_var: float,
                    frame_w: int, frame_h: int, cfg: QualityCfg, kp_conf: float = 0.5) -> dict[str, float]:
    """Множители качества (раздел 3.6)."""
    x1, y1, x2, y2 = [float(x) for x in box]
    h = y2 - y1
    size = float(np.clip((h - cfg.h_min) / max(cfg.h_good - cfg.h_min, 1e-9), 0, 1))
    e = cfg.edge_px
    near_edge = x1 < e or y1 < e or x2 > frame_w - e or y2 > frame_h - e
    trunc = cfg.trunc_factor if near_edge else 1.0
    occl = 1.0 - float(np.clip(occ / max(cfg.occ_max, 1e-9), 0, 1))
    sharp = 0.5 + 0.5 * float(np.clip(blur_var / max(cfg.blur_ref, 1e-9), 0, 1))
    kp = np.asarray(kp, dtype=float)
    skel = 0.5 + 0.5 * float(np.mean(kp[:, 2] > kp_conf)) if len(kp) else 0.5
    cf = float(np.sqrt(max(conf, 0.0)))
    return {"size": size, "trunc": trunc, "occ": occl, "sharp": sharp, "skel": skel, "conf": cf}


def quality(box: np.ndarray, kp: np.ndarray, conf: float, occ: float, blur_var: float,
            frame_w: int, frame_h: int, cfg: QualityCfg, kp_conf: float = 0.5) -> float:
    """Качество наблюдения q ∈ [0, 1] — произведение множителей."""
    f = quality_factors(box, kp, conf, occ, blur_var, frame_w, frame_h, cfg, kp_conf)
    return float(np.prod(list(f.values())))
