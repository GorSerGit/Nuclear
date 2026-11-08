"""Верх и низ тела отдельно, пропорции по позе.

Общий вектор внешности для людей в чёрном почти одинаков, а штаны у них разные. Поэтому кроп делится на верх и низ, и
каждая половина получает свой вектор той же моделью. Пропорция ноги к корпусу берётся из точек позы: рост в кадре зависит
от расстояния, а отношение частей тела почти нет. Камеры смотрят сверху и сжимают ноги, поэтому пропорция только
измеряется и пока не запрещает склейку."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .crops import read_image
from .storage import Store

L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 5, 6, 11, 12, 15, 16
UP, LOW = (0.12, 0.52), (0.50, 1.00)      # доли высоты кропа
MAX_CROPS = 12


def leg_ratio(kp: np.ndarray, conf: float = 0.5) -> float | None:
    """Длина ноги к длине корпуса. kp: (17, 3)."""
    if kp[[L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK], 2].min() < conf:
        return None
    sh, hip, ank = kp[[L_SH, R_SH], :2].mean(0), kp[[L_HIP, R_HIP], :2].mean(0), kp[[L_ANK, R_ANK], :2].mean(0)
    torso = float(np.linalg.norm(hip - sh))
    return float(np.linalg.norm(ank - hip)) / torso if torso > 1.0 else None


def split(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = img.shape[0]
    return img[int(UP[0] * h):int(UP[1] * h)], img[int(LOW[0] * h):int(LOW[1] * h)]


def _clean(vec: dict[int, np.ndarray], cam_of: dict[int, int]) -> dict[int, np.ndarray]:
    """Смещение камеры убирается: неон меняет цвет у всех людей одной камеры одинаково."""
    out = {}
    for cam in set(cam_of.values()):
        ids = [t for t, c in cam_of.items() if c == cam and t in vec]
        if not ids:
            continue
        mean = np.mean([vec[t] for t in ids], 0)
        for t in ids:
            v = vec[t] - 0.5 * mean
            out[t] = v / max(float(np.linalg.norm(v)), 1e-12)
    return out


def part_vectors(store: Store, backend) -> dict[int, dict]:
    """tid -> {"up": вектор верха, "low": вектор низа, "legs": пропорция или None}."""
    import cv2

    ups, lows, legs, cam_of = {}, {}, {}, {}
    for cid in store.load_cameras()["cam_id"]:
        o = store.load_observations(int(cid), columns=["tid", "crop_path", "q", "kp", "is_outlier"])
        o = o[o["tid"].notna() & ~o["is_outlier"]]
        for tid, g in o.groupby("tid"):
            tid = int(tid)
            cam_of[tid] = int(cid)
            r = [leg_ratio(np.asarray(k, float).reshape(17, 3)) for k in g["kp"] if k is not None and len(k) == 51]
            r = [x for x in r if x is not None]
            legs[tid] = float(np.median(r)) if len(r) >= 3 else None
            c = g[g["crop_path"].notna()].sort_values("q", ascending=False).head(MAX_CROPS)
            top, bottom = [], []
            for p in c["crop_path"]:
                img = read_image(Path(store.root) / p)
                if img is None or img.size == 0 or img.shape[0] < 20:
                    continue
                a, b = split(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                top.append(a)
                bottom.append(b)
            if top:
                ups[tid] = backend.embed(top).mean(0).astype(np.float64)
                lows[tid] = backend.embed(bottom).mean(0).astype(np.float64)
    ups, lows = _clean(ups, cam_of), _clean(lows, cam_of)
    return {t: {"up": ups.get(t), "low": lows.get(t), "legs": legs.get(t)} for t in cam_of}


def save(root: str | Path, parts: dict[int, dict]) -> None:
    ids = [t for t, p in parts.items() if p["low"] is not None]
    np.savez_compressed(Path(root) / "parts.npz", tids=np.array(ids, np.int64),
                        up=np.stack([parts[t]["up"] for t in ids]) if ids else np.zeros((0, 1)),
                        low=np.stack([parts[t]["low"] for t in ids]) if ids else np.zeros((0, 1)),
                        legs=np.array([parts[t]["legs"] if parts[t]["legs"] is not None else np.nan for t in ids]))


def load(root: str | Path) -> dict[int, dict]:
    p = Path(root) / "parts.npz"
    if not p.exists():
        return {}
    z = np.load(p)
    return {int(t): {"up": z["up"][i], "low": z["low"][i], "legs": None if np.isnan(z["legs"][i]) else float(z["legs"][i])}
            for i, t in enumerate(z["tids"])}
