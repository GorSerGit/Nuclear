"""Привязка камеры к плану офиса: гомография «пиксель кадра -> метры плана» по парам точек.

Точки на полу (углы ковра, плитки, ножки столов, дверные проёмы) отмечаются на кадре и на плане.
Нужно ≥ 4 пар, лучше 6–10, разнесённых по всему видимому полу.

Интерактивно (окно OpenCV): щелчок по кадру (слева), затем по тому же месту на плане (справа) — и так далее.
  u — отменить последнюю точку, Enter — сохранить, Esc — выйти без сохранения.
Без окна: --points pts.json с [[u, v, X, Y], ...] (X, Y — пиксели плана).

Метры плана = пиксели плана / px_per_m (ось X вправо, Y вниз) — те же, что использует интерфейс (site.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .lens import undistort_points
from .video_io import VideoSource


def fit_h(pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pairs (n, 4): u, v, X_m, Y_m -> (H, ошибки репроекции в метрах)."""
    if len(pairs) < 4:
        raise ValueError(f"нужно не меньше 4 пар точек, есть {len(pairs)}")
    src = pairs[:, :2].astype(np.float64)
    dst = pairs[:, 2:].astype(np.float64)
    H, _ = cv2.findHomography(src, dst, 0 if len(pairs) < 6 else cv2.RANSAC, 0.3)
    if H is None:
        raise ValueError("гомография не нашлась: точки лежат на одной прямой?")
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    return H / H[2, 2], np.hypot(*(proj - dst).T)


def _click(frame: np.ndarray, plan: np.ndarray) -> list[list[float]]:
    h = max(frame.shape[0], plan.shape[0])
    sf = 900 / h if h > 900 else 1.0
    fr = cv2.resize(frame, None, fx=sf, fy=sf)
    pl = cv2.resize(plan, None, fx=sf, fy=sf)
    W = fr.shape[1]
    pairs: list[list[float]] = []
    pending: list[float] | None = None
    win = "calibrate: kadr (sleva) -> plan (sprava); u - otmena, Enter - sohranit, Esc - vyhod"

    def redraw() -> None:
        canvas = np.zeros((max(fr.shape[0], pl.shape[0]), W + pl.shape[1], 3), np.uint8)
        canvas[:fr.shape[0], :W] = fr
        canvas[:pl.shape[0], W:] = pl
        for i, (u, v, x, y) in enumerate(pairs):
            for (a, b) in ((u * sf, v * sf), (W + x * sf, y * sf)):
                cv2.circle(canvas, (int(a), int(b)), 5, (0, 255, 255), -1)
                cv2.putText(canvas, str(i + 1), (int(a) + 6, int(b) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2)
        if pending:
            cv2.circle(canvas, (int(pending[0] * sf), int(pending[1] * sf)), 6, (0, 0, 255), 2)
        cv2.imshow(win, canvas)

    def on_mouse(ev, x, y, *_):
        nonlocal pending
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        if x < W and pending is None:
            pending = [x / sf, y / sf]
        elif x >= W and pending is not None:
            pairs.append([*pending, (x - W) / sf, y / sf])
            pending = None
        redraw()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            raise SystemExit("калибровка отменена")
        if k in (13, 10):
            break
        if k == ord("u"):
            if pending is not None:
                pending = None
            elif pairs:
                pairs.pop()
            redraw()
    cv2.destroyAllWindows()
    return pairs


def calibrate(passport: str | Path, plan: str | Path | None, px_per_m: float | None,
              points: str | Path | None = None, frame_s: float = 0.0, out: str | Path | None = None) -> dict:
    ppath = Path(passport)
    data = json.loads(ppath.read_text(encoding="utf-8"))
    if not px_per_m:
        raise ValueError("укажите --px-per-m (масштаб плана; для точек уже в метрах — 1)")
    if points:
        pairs = json.loads(Path(points).read_text(encoding="utf-8"))
    else:
        if not plan:
            raise ValueError("нужен --plan (картинка) или --points")
        video = Path(data["video"])
        video = video if video.exists() else ppath.parent / video
        vs = VideoSource(video, fps=data.get("fps"))
        f0 = int(frame_s * vs.fps)
        frame = next(iter(vs.frames(f0, f0 + 1)))[1]
        plan_img = cv2.imdecode(np.fromfile(str(plan), np.uint8), cv2.IMREAD_COLOR)
        pairs = _click(frame, plan_img)
    arr = np.asarray(pairs, dtype=np.float64).reshape(-1, 4)
    arr[:, 2:] /= px_per_m
    if data.get("K") and data.get("dist"):     # H относится к выпрямленному кадру, как у to_floor
        arr[:, :2] = undistort_points(arr[:, :2], np.asarray(data["K"], float), np.asarray(data["dist"], float))
    H, err = fit_h(arr)
    data["H"] = H.tolist()
    data["calib_points"] = np.asarray(pairs).tolist()
    data["calib_px_per_m"] = px_per_m
    dst = Path(out) if out else ppath
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"out": str(dst), "n": len(arr), "err_med_m": float(np.median(err)), "err_max_m": float(err.max())}
