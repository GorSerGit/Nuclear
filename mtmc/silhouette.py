"""Силуэты людей: маска YOLO11-seg на каждом обработанном кадре, сопоставленная с найденными людьми по рамкам.

Работает поверх готовых наблюдений (после process): детекция, трекинг, треклеты и ID не меняются — к каждому
наблюдению (кадр, local_id) добавляется контур фигуры. Результат: observations/camN.sil.parquet
(frame_idx, local_id, poly = [x0, y0, x1, y1, …] в пикселях кадра, iou — совпадение рамок маски и наблюдения).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .config import Config, SilhouetteCfg
from .detect_track import resolve_device, resolve_weights
from .storage import Store
from .video_io import VideoSource

log = logging.getLogger(__name__)

SIL_SCHEMA = pa.schema([("frame_idx", pa.int64()), ("local_id", pa.int64()), ("poly", pa.list_(pa.float32())),
                        ("iou", pa.float32())])


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU рамок (n, 4) × (m, 4)."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def match(obs_boxes: np.ndarray, seg_boxes: np.ndarray, thr: float) -> list[tuple[int, int, float]]:
    """Жадное сопоставление по IoU (лучшие пары первыми), каждая рамка — не больше одного раза."""
    iou = box_iou(obs_boxes, seg_boxes)
    pairs, used_o, used_s = [], set(), set()
    for k in np.argsort(-iou, axis=None):
        i, j = divmod(int(k), iou.shape[1])
        if iou[i, j] < thr:
            break
        if i in used_o or j in used_s:
            continue
        used_o.add(i)
        used_s.add(j)
        pairs.append((i, j, float(iou[i, j])))
    return pairs


def simplify(poly: np.ndarray, box_h: float, frac: float, max_pts: int) -> np.ndarray:
    """Упрощённый контур: допуск frac × высота рамки, не больше max_pts вершин."""
    p = np.asarray(poly, np.float32).reshape(-1, 1, 2)
    if len(p) < 3:
        return p.reshape(-1, 2)
    eps = max(frac * box_h, 0.5)
    out = cv2.approxPolyDP(p, eps, True)
    while len(out) > max_pts:
        eps *= 1.4
        out = cv2.approxPolyDP(p, eps, True)
    return out.reshape(-1, 2)


def fill_between(rows: list[tuple[int, int, list[float], float]], obs: pd.DataFrame,
                 max_gap: int) -> list[tuple[int, int, list[float], float]]:
    """Контуры для кадров без маски: контур ближайшего кадра того же человека, перенесённый за его рамкой.
    Дальше max_gap кадров не переносится: там честнее показать рамку."""
    boxes = {(int(f), int(l)): (x1, y1, x2, y2) for f, l, x1, y1, x2, y2 in
             zip(obs["frame_idx"], obs["local_id"], obs["x1"], obs["y1"], obs["x2"], obs["y2"])}
    have: dict[int, list[tuple[int, list[float], float]]] = {}
    for f, l, poly, iou in rows:
        have.setdefault(l, []).append((f, poly, iou))
    for v in have.values():
        v.sort(key=lambda r: r[0])
    done = {(f, l) for f, l, _, _ in rows}
    out = list(rows)
    for (f, l), box in boxes.items():
        if (f, l) in done or l not in have:
            continue
        fs = [r[0] for r in have[l]]
        i = int(np.searchsorted(fs, f))
        near = min((r for r in have[l][max(i - 1, 0):i + 1]), key=lambda r: abs(r[0] - f))
        if abs(near[0] - f) > max_gap or (near[0], l) not in boxes:
            continue
        sx1, sy1, sx2, sy2 = boxes[(near[0], l)]
        kx, ky = (box[2] - box[0]) / max(sx2 - sx1, 1e-6), (box[3] - box[1]) / max(sy2 - sy1, 1e-6)
        p = np.asarray(near[1], np.float32).reshape(-1, 2)
        p = np.column_stack(((p[:, 0] - sx1) * kx + box[0], (p[:, 1] - sy1) * ky + box[1]))
        out.append((f, l, np.round(p, 1).ravel().tolist(), near[2]))
    return sorted(out, key=lambda r: (r[0], r[1]))


def silhouettes(cam_id: int, data_dir: str | Path, cfg: Config) -> dict:
    """Строит контуры фигур для всех наблюдений камеры; пишет observations/camN.sil.parquet."""
    from ultralytics import YOLO

    sc: SilhouetteCfg = cfg.silhouette
    t0 = time.perf_counter()
    store = Store(data_dir)
    cam = store.get_camera(cam_id)
    obs = store.load_observations(cam_id, columns=["frame_idx", "local_id", "x1", "y1", "x2", "y2"])
    if obs.empty:
        return {"cam_id": cam_id, "observations": 0, "matched": 0}
    by_frame = {int(f): g for f, g in obs.groupby("frame_idx")}
    frames = sorted(by_frame)
    meta = store.load_meta(cam_id)
    stride = int(meta.get("stride") or cfg.video.stride)
    device = resolve_device(sc.device or cfg.detector.device)
    model = YOLO(resolve_weights(sc.weights))
    vs = VideoSource(cam["video_path"], cam["fps"], cfg.video.exact_seek)
    every = max(int(sc.every), 1)
    want = set(frames[::every])
    rows: list[tuple[int, int, list[float], float]] = []
    batch_idx, batch_img = [], []

    def flush() -> None:
        if not batch_img:
            return
        res = model.predict(batch_img, imgsz=sc.imgsz or cfg.detector.imgsz, conf=sc.conf, classes=[0],
                            device=device, retina_masks=True, verbose=False)
        for f, r in zip(batch_idx, res):
            g = by_frame[f]
            if r.masks is None or r.boxes is None or not len(r.boxes):
                continue
            sb = r.boxes.xyxy.float().cpu().numpy()
            polys = r.masks.xy
            ob = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
            for i, j, iou in match(ob, sb, sc.iou_match):
                if j >= len(polys) or len(polys[j]) < 3:
                    continue
                h = ob[i, 3] - ob[i, 1]
                p = simplify(polys[j], h, sc.simplify_frac, sc.max_pts)
                rows.append((f, int(g["local_id"].iloc[i]), np.round(p, 1).ravel().tolist(), iou))
        batch_idx.clear()
        batch_img.clear()

    start, end = frames[0], frames[-1] + 1
    for f, img in tqdm(vs.frames(start, end, stride), total=len(range(start, end, stride)),
                       desc=f"силуэты cam{cam_id}", unit="кадр", leave=False):
        if f not in want:
            continue
        batch_idx.append(f)
        batch_img.append(img)
        if len(batch_img) >= sc.batch:
            flush()
    flush()
    if every > 1:
        rows = fill_between(rows, obs, every * stride * 2)
    df = pd.DataFrame(rows, columns=["frame_idx", "local_id", "poly", "iou"])
    table = pa.Table.from_pandas(df, schema=SIL_SCHEMA, preserve_index=False)
    dst = store.sil_path(cam_id)
    tmp = dst.with_suffix(".tmp")
    pq.write_table(table, tmp)
    tmp.replace(dst)
    out = {"cam_id": cam_id, "observations": int(len(obs)), "matched": int(len(df)),
           "share": round(len(df) / max(len(obs), 1), 3), "wall_s": round(time.perf_counter() - t0, 1)}
    log.info("силуэты cam%s: %s", cam_id, out)
    return out


def load_silhouettes(store: Store, cam_id: int) -> dict[tuple[int, int], np.ndarray]:
    """(frame_idx, local_id) -> контур (n, 2); пусто, если силуэтов нет."""
    p = store.sil_path(cam_id)
    if not p.exists():
        return {}
    df = pq.read_table(p).to_pandas()
    return {(int(f), int(l)): np.asarray(poly, np.float32).reshape(-1, 2)
            for f, l, poly in zip(df["frame_idx"], df["local_id"], df["poly"])}
