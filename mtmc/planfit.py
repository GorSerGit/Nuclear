"""Уточнение привязки камер к плану по самим людям.

Точки калибровки часто лежат только в ближней части кадра: на офисе у cam5 все точки ниже v = 483, а люди вдали выше
v = 300. За пределами точек гомография экстраполирует и ошибается на 5–13 м, хотя на своих точках точна до сантиметров.
Правило простое: если человека одновременно видит другая камера и он лежит в надёжной зоне той камеры (внутри
многоугольника её точек калибровки), его положение оттуда служит точкой привязки для этой камеры. Пары камер
выбираются по внешности, ложные отсеивает RANSAC. Точки человека не заменяют точки калибровки, а дополняют их, и
новая гомография не принимается, если точки калибровки разошлись."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import Config
from .lens import undistort_points
from .storage import Store

log = logging.getLogger(__name__)


@dataclass
class PlanFitCfg:
    min_sim: float = 0.45          # сходство кусков двух камер, чтобы считать их кандидатом в одного человека
    hull_grow: float = 1.25        # надёжная зона: многоугольник точек калибровки, чуть расширенный от центра
    good_m: float = 1.0            # камера и так согласна с соседями: не трогаем
    grid_s: float = 0.2
    min_points: int = 30           # меньше — камера остаётся как есть
    ransac_m: float = 1.0
    keep_m: float = 1.0            # точки человека дальше этого от модели не участвуют в уточнении
    calib_weight: int = 20         # во столько раз точка калибровки весомее точки человека
    calib_limit_m: float = 0.35    # средняя ошибка на точках калибровки выше этой — модель отвергается
    must_improve: float = 0.6      # медианная ошибка на парах должна упасть хотя бы до этой доли, иначе H остаётся
    rounds: int = 3


def measure_walk(data_dir: str | Path, min_points: int = 25, step: int = 12) -> float | None:
    """Скорость идущих людей на плане, 75-й процентиль: стоящие и ходящие вокруг стола её не занижают."""
    store = Store(data_dir)
    v: list[float] = []
    for cam in store.load_cameras()["cam_id"]:
        o = store.load_observations(int(cam), columns=["t", "tid", "xs", "ys"])
        o = o[o["tid"].notna() & o["xs"].notna()].sort_values("t")
        for _, g in o.groupby("tid"):
            if len(g) >= min_points:
                g = g.iloc[::step]
                v += (np.hypot(g["xs"].diff(), g["ys"].diff()) / g["t"].diff()).dropna().tolist()
    return round(float(np.quantile(v, 0.75)), 1) if len(v) >= 30 else None   # округление: пересчёт не дрожит от запуска к запуску


def apply_h(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    p = np.c_[uv, np.ones(len(uv))] @ H.T
    return p[:, :2] / p[:, 2:3]


def inside(hull: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h = cv2.convexHull(hull.astype(np.float32))
    return np.array([cv2.pointPolygonTest(h, (float(u), float(v)), False) >= 0 for u, v in uv], bool)


def calib_scale(H: np.ndarray, pts: np.ndarray) -> float:
    """Точки калибровки заданы в пикселях плана, H даёт метры: масштаб берём из самой H."""
    m = apply_h(H, pts[:, :2])
    return float(np.median(np.hypot(*m.T) / np.maximum(np.hypot(*pts[:, 2:].T), 1e-9)))


def lens_of(p: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    if p.get("K") and p.get("dist"):
        return np.asarray(p["K"], float), np.asarray(p["dist"], float)
    return None, None


def _table(store: Store, cam: int, H: np.ndarray, lens: tuple, step: float) -> pd.DataFrame:
    """Положения кусков на сетке времени. u, v — выпрямленный кадр: H работает именно в нём."""
    o = store.load_observations(cam, columns=["t", "tid", "u", "v", "is_outlier"])
    o = o[o["tid"].notna() & ~o["is_outlier"] & o["u"].notna()].copy()
    o["tid"] = o["tid"].astype(int)
    o["k"] = np.round(o["t"] / step).astype(np.int64)
    if not o.empty:
        o[["u", "v"]] = undistort_points(o[["u", "v"]].to_numpy(float), *lens)
    g = o.groupby(["tid", "k"], as_index=False)[["u", "v"]].median()
    xy = apply_h(H, g[["u", "v"]].to_numpy(float))
    g["x"], g["y"], g["cam"] = xy[:, 0], xy[:, 1], cam
    return g


def candidates(tabs: dict[int, pd.DataFrame], protos: dict[int, np.ndarray], trusted: dict[int, np.ndarray],
               cfg: PlanFitCfg) -> dict[int, np.ndarray]:
    """Для каждой камеры: (u, v своей камеры, x, y из надёжной зоны другой камеры, номер пары кусков)."""
    out: dict[int, list[np.ndarray]] = {c: [] for c in tabs}
    for c, a in tabs.items():
        for c2, b in tabs.items():
            if c == c2:
                continue
            b = b[inside(trusted[c2], b[["u", "v"]].to_numpy(float))]
            m = a.merge(b, on="k", suffixes=("_a", "_b"))
            if m.empty:
                continue
            pa = np.stack([protos.get(t, np.zeros(1)) if t in protos else np.full_like(next(iter(protos.values())), np.nan)
                           for t in m["tid_a"]])
            pb = np.stack([protos.get(t) if t in protos else np.full_like(next(iter(protos.values())), np.nan)
                           for t in m["tid_b"]])
            sim = np.nansum(pa * pb, axis=1)
            m = m[sim >= cfg.min_sim]
            if not m.empty:
                out[c].append(m[["u_a", "v_a", "x_b", "y_b"]].to_numpy(float))
    return {c: (np.concatenate(v) if v else np.zeros((0, 4))) for c, v in out.items()}


def refit(H: np.ndarray, calib_m: np.ndarray, cand: np.ndarray, cfg: PlanFitCfg) -> tuple[np.ndarray | None, dict]:
    """RANSAC по точкам людей и точкам калибровки, затем уточнение по согласным точкам."""
    if len(cand) < cfg.min_points:
        return None, {"points": len(cand), "why": "мало точек"}
    was = float(np.median(np.hypot(*(apply_h(H, cand[:, :2]) - cand[:, 2:]).T)))
    if was <= cfg.good_m:
        return None, {"points": len(cand), "err_m": round(was, 2), "why": "и так согласна"}
    src = np.vstack([np.repeat(calib_m[:, :2], cfg.calib_weight, axis=0), cand[:, :2]]).astype(np.float64)
    dst = np.vstack([np.repeat(calib_m[:, 2:], cfg.calib_weight, axis=0), cand[:, 2:]]).astype(np.float64)
    new, mask = cv2.findHomography(src, dst, cv2.RANSAC, cfg.ransac_m, maxIters=5000, confidence=0.999)
    if new is None:
        return None, {"points": len(cand), "why": "RANSAC"}
    for _ in range(cfg.rounds):
        e = np.hypot(*(apply_h(new, src) - dst).T)
        keep = e <= cfg.keep_m
        if keep.sum() < 8:
            return None, {"points": len(cand), "why": "мало согласных"}
        new = cv2.findHomography(src[keep], dst[keep], 0)[0]
    calib_err = float(np.mean(np.hypot(*(apply_h(new, calib_m[:, :2]) - calib_m[:, 2:]).T)))
    used = int(np.sum(np.hypot(*(apply_h(new, cand[:, :2]) - cand[:, 2:]).T) <= cfg.keep_m))
    after = float(np.median(np.hypot(*(apply_h(new, cand[:, :2]) - cand[:, 2:]).T)))
    info = {"points": len(cand), "used": used, "calib_err_m": round(calib_err, 3), "err_was_m": round(was, 2),
            "err_now_m": round(after, 2)}
    if calib_err > cfg.calib_limit_m:
        return None, {**info, "why": "разошлись точки калибровки"}
    if used < cfg.min_points:
        return None, {**info, "why": "мало согласных"}
    if after > cfg.must_improve * was:
        return None, {**info, "why": "не стало лучше"}
    return new / new[2, 2], info


def fit_plan(data_dir: str | Path, cfg: Config, passports: dict[int, dict], fit: PlanFitCfg | None = None
             ) -> tuple[dict[int, list], list[dict]]:
    """Новые H камер, которым нашлось уточнение. passports: cam -> паспорт с H и calib_points."""
    from .tracks import tid_protos

    fit = fit or PlanFitCfg()
    store = Store(data_dir)
    trdf = store.load_tracklets()
    tp = {k: v[0] for k, v in tid_protos(store, trdf, cfg.tracks.cam_debias, cfg.tracks.scene_lam).items()}
    if not tp:
        return {}, []
    H = {c: np.asarray(p["H"], float) for c, p in passports.items() if p.get("H") is not None and p.get("calib_points")}
    lens = {c: lens_of(passports[c]) for c in H}
    pts = {c: np.asarray(passports[c]["calib_points"], float) for c in H}
    for c in H:
        pts[c][:, :2] = undistort_points(pts[c][:, :2], *lens[c])
    calib = {c: np.c_[pts[c][:, :2], pts[c][:, 2:] * calib_scale(H[c], pts[c])] for c in H}
    trusted = {c: pts[c][:, :2].mean(0) + (pts[c][:, :2] - pts[c][:, :2].mean(0)) * fit.hull_grow for c in H}
    new: dict[int, list] = {}
    report = []
    for _ in range(2):                    # второй круг: уточнённые камеры помогают соседним
        tabs = {c: _table(store, c, H[c], lens[c], fit.grid_s) for c in H}
        cand = candidates(tabs, tp, trusted, fit)
        for c in H:
            h, info = refit(H[c], calib[c], cand[c], fit)
            report.append({"cam": c, **info})
            if h is not None:
                H[c] = h
                new[c] = h.tolist()
                # зона доверия этой камеры расширяется только на следующий запуск: здесь точки людей ещё проверяются
    return new, report


def save(data_dir: str | Path, new: dict[int, list], report: list[dict]) -> None:
    (Path(data_dir) / "plan_fit.json").write_text(json.dumps({"H": {str(k): v for k, v in new.items()},
                                                              "report": report}, indent=1), encoding="utf-8")


def load(data_dir: str | Path) -> dict[int, list]:
    p = Path(data_dir) / "plan_fit.json"
    if not p.exists():
        return {}
    return {int(k): v for k, v in json.loads(p.read_text(encoding="utf-8"))["H"].items()}

