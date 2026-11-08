"""Данные для новых функций просмотра: стены плана (3D-вид), цвет одежды треков (поиск по описанию), миниатюры.

Цвет верха и низа берётся по силуэту из сохранённых кропов. Неон в офисе окрашивает всех людей одной камеры, поэтому
у каждой камеры вычитается её средний оттенок: остаётся разница между людьми. Это простая модель цвета, а не
нейросеть «текст — картинка»: она понимает названия цветов и частей одежды, но не фасоны и не мелкие детали."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .clothes import ZONES
from .crops import padded_box, read_image
from .storage import Store

log = logging.getLogger(__name__)
MAX_CROPS = 16
WALL_H_PX = 2.0             # упрощение контура стены, пиксели плана


# ---------------------------------------------------------------- стены
def plan_walls(site_json: str | Path | None) -> list[list[list[float]]]:
    """Контуры стен плана в метрах: многоугольники, которые 3D-вид поднимает вверх."""
    if not site_json:
        return []
    from .config import AutoCalibCfg
    from .planfile import extract_walls, read_plan

    p = Path(site_json)
    d = json.loads(p.read_text(encoding="utf-8"))
    if not d.get("plan_image") or not d.get("px_per_m"):
        return []
    img = Path(d["plan_image"])
    img = img if img.is_absolute() or img.exists() else p.parent / img
    ppm = float(d["px_per_m"])
    try:
        mask = extract_walls(read_plan(img), ppm, AutoCalibCfg().wall_thr).astype(np.uint8)
    except (ValueError, cv2.error, OSError) as e:
        log.warning("стены плана не выделены: %s", e)
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < 0.05 * ppm * ppm:      # мелочь: подписи, значки
            continue
        poly = cv2.approxPolyDP(c, WALL_H_PX, True).reshape(-1, 2)
        if len(poly) >= 3:
            out.append([[round(float(x) / ppm, 2), round(float(y) / ppm, 2)] for x, y in poly])
    return out


# ---------------------------------------------------------------- цвет одежды
def zone_lab(img: np.ndarray, poly: np.ndarray | None) -> np.ndarray | None:
    """Медиана Lab верха и низа фигуры (2, 3) или None."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if poly is not None and len(poly) >= 3:
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
        mask = cv2.erode(mask, np.ones((max(3, h // 40) | 1,) * 2, np.uint8))
    if mask.sum() < 40:
        mask[:, int(w * 0.3):int(w * 0.7)] = 1
    ys, xs = np.nonzero(mask)
    y0, y1 = np.percentile(ys, 1), np.percentile(ys, 99)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = []
    for a, b in ZONES:
        m = (ys >= y0 + a * (y1 - y0)) & (ys < y0 + b * (y1 - y0))
        if m.sum() < 20:
            return None
        v = np.median(lab[ys[m], xs[m]], 0)
        out.append([v[0] * 100 / 255, v[1] - 128, v[2] - 128])      # L 0–100, a и b около нуля для серого
    return np.array(out)


def color_name(lab: np.ndarray) -> str:
    L, a, b = (float(x) for x in lab)
    c = float(np.hypot(a, b))
    hue = (np.degrees(np.arctan2(b, a)) + 360) % 360
    if c < 13 or (L < 22 and c < 18):        # слабый оттенок под офисным светом — это серая гамма
        return "чёрный" if L < 30 else "тёмно-серый" if L < 50 else "серый" if L < 66 else "светло-серый" if L < 78 else "белый"
    if 45 <= hue < 110 and L >= 55 and c < 42:       # тёплый свет офиса делает бежевый «жёлтым»
        return "бежевый"
    if 20 <= hue < 75 and L < 52:
        return "коричневый"
    if hue < 25 or hue >= 345:
        return "красный" if L < 70 else "розовый"
    if hue < 60:
        return "оранжевый"
    if hue < 105:
        return "жёлтый"
    if hue < 170:
        return "зелёный"
    if hue < 280:                            # джинсы под холодным светом уходят к фиолетовому
        return "голубой" if L >= 60 else "синий"
    if hue < 310:
        return "фиолетовый"
    return "розовый"


def track_looks(store: Store, tracks: list[dict], pad: float = 0.05) -> dict[int, dict]:
    """track_id -> {"up": название, "low": название, "up_lab": [...], "low_lab": [...], "thumb": путь кропа}."""
    from .silhouette import load_silhouettes

    want = {tid: t["id"] for t in tracks for tid in t["tids"]}
    samples: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    thumbs: dict[int, tuple[float, str]] = {}
    for cid in store.load_cameras()["cam_id"]:
        cid = int(cid)
        o = store.load_observations(cid, columns=["tid", "frame_idx", "local_id", "x1", "y1", "x2", "y2", "q",
                                                  "crop_path", "is_outlier"])
        o = o[o["tid"].notna() & o["crop_path"].notna() & ~o["is_outlier"]]
        o = o[o["tid"].astype(int).isin(want.keys())]
        if o.empty:
            continue
        sil = load_silhouettes(store, cid)
        o = o.assign(track=o["tid"].astype(int).map(want))
        for tr, g in o.groupby("track"):
            g = g.sort_values("q", ascending=False).head(MAX_CROPS)
            for r in g.itertuples():
                img = read_image(Path(store.root) / r.crop_path)
                if img is None or img.size == 0:
                    continue
                poly = sil.get((int(r.frame_idx), int(r.local_id)))
                if poly is not None:
                    x0, y0, _, _ = padded_box(np.array([r.x1, r.y1, r.x2, r.y2]), pad, 10 ** 9, 10 ** 9)
                    poly = poly - np.array([x0, y0], np.float32)
                v = zone_lab(img, poly)
                if v is not None:
                    samples.setdefault(int(tr), []).append((cid, v, float(r.q)))
                if float(r.q) > thumbs.get(int(tr), (-1.0, ""))[0]:
                    thumbs[int(tr)] = (float(r.q), r.crop_path)
    # оттенок камеры: средний a, b всех фигур этой камеры
    cast: dict[int, np.ndarray] = {}
    for cid in {c for s in samples.values() for c, _, _ in s}:
        ab = np.array([v[:, 1:] for s in samples.values() for c, v, _ in s if c == cid]).reshape(-1, 2)
        cast[cid] = np.median(ab, 0) if len(ab) >= 10 else np.zeros(2)
    out = {}
    for tr, s in samples.items():
        vs = np.array([np.c_[v[:, :1], v[:, 1:] - cast[c]] for c, v, _ in s])
        med = np.median(vs, 0)
        out[tr] = {"up": color_name(med[0]), "low": color_name(med[1]),
                   "up_lab": [round(float(x), 1) for x in med[0]], "low_lab": [round(float(x), 1) for x in med[1]]}
    for tr, (_, path) in thumbs.items():
        out.setdefault(tr, {})["thumb"] = path
    return out


def write_thumbs(store: Store, looks: dict[int, dict], out: Path) -> None:
    """Миниатюра трека: лучший по качеству кроп, высота 160 px."""
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    for tr, lk in looks.items():
        p = lk.pop("thumb", None)
        if not p:
            continue
        img = read_image(Path(store.root) / p)
        if img is None or img.size == 0:
            continue
        h, w = img.shape[:2]
        img = cv2.resize(img, (max(1, int(w * 160 / h)), 160), interpolation=cv2.INTER_AREA)
        cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(out / "thumbs" / f"T{tr}.jpg"))
        lk["thumb"] = f"thumbs/T{tr}.jpg"


def extras(store: Store, tracks: list[dict], site: str | Path | None, out: Path, pad: float = 0.05) -> dict:
    """{"walls": [...], "looks": {track_id: {...}}} для data.js; миниатюры пишутся в out/thumbs."""
    looks = track_looks(store, tracks, pad)
    write_thumbs(store, looks, out)
    return {"walls": plan_walls(site), "looks": {str(k): v for k, v in looks.items()}}


def patch_ui(ui_dir: str | Path, data_dir: str | Path, site: str | Path | None) -> dict:
    """Дописывает новые данные в уже собранный просмотр (без перекодирования видео)."""
    ui_dir = Path(ui_dir)
    text = (ui_dir / "data.js").read_text(encoding="utf-8")
    d = json.loads(text[text.index("{"):text.rstrip().rindex("}") + 1])
    rows = json.loads((Path(data_dir) / "tracks.json").read_text(encoding="utf-8"))["tracks"]
    tracks = [{"id": r["track_id"], "tids": r["tids"]} for r in rows]
    ex = extras(Store(data_dir), tracks, site, ui_dir)
    d["plan"]["walls"] = ex["walls"]
    d["looks"] = ex["looks"]
    (ui_dir / "data.js").write_text("window.MTMC = " + json.dumps(d, ensure_ascii=False, separators=(",", ":")) + ";\n",
                                    encoding="utf-8")
    return {"walls": len(ex["walls"]), "looks": len(ex["looks"])}
