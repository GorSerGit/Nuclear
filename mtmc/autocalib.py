"""Автоматическая привязка камер к плану офиса по идущим людям. Не получилось — ручная (calibrate).

Шаг 1. Самокалибровка камеры по людям. Модель: обычная камера без дисторсии, центр кадра — оптическая ось,
неизвестны фокус f, наклон вниз, крен и высота камеры h. Человек стоит вертикально, ростом person_height_m:
точка ног (u, v) -> луч на пол -> точка P; голова = P + рост вверх -> проекция должна совпасть с верхом рамки.
Подбор четырёх параметров (least squares, робастная функция потерь) по сотням стоящих людей на разной глубине.
Результат — гомография «пиксель -> пол в метрах» в системе камеры (X вправо, Y вперёд, начало под камерой).
Если в site.json задана высота камеры (height_m), она фиксируется, а подбирается рост.

Шаг 2. Положение на плане. Точка под камерой — из site.json (cameras.N.pos), остаётся поворот φ.
  * линия видимости: камера видит пол, значит отрезок «камера -> точка ног» на плане не пересекает стен
    (тёмные пиксели картинки плана); перебираются все повороты, выбирается тот, где видно больше точек;
  * look_at в site.json (точка плана, куда смотрит камера) — сужает поиск;
  * общие зоны камер: одновременно увиденные люди должны попасть в одни и те же места плана. Совместное
    уточнение поворота, сдвига (≤ pos_prior_m) и масштаба (±10%) всех камер покоординатным спуском.

Шаг 3. Проверка. Статус камеры ok / check / fail по ошибке самокалибровки, доле видимых точек, однозначности
поворота и расхождению с соседями. Для каждой камеры рисуется картинка calib_check/camN.png: сетка 1 м на полу
кадра и точки камеры на плане. fail -> ручная привязка (окно calibrate) или подсказка, как её запустить.
"""
from __future__ import annotations

import json
import logging
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .config import AutoCalibCfg, Config
from .lens import distort_points, undistort_points
from .storage import Store

log = logging.getLogger(__name__)

M_FLIP = np.diag([1.0, -1.0])     # (X вправо, Y вперёд) камеры -> план (x вправо, y вниз): отражение


# ------------------------------------------------------------------ модель камеры
def rotation(pitch: float, roll: float) -> np.ndarray:
    """Мир (X, Y вперёд, Z вверх) -> камера (x вправо, y вниз, z вперёд). pitch > 0 — камера смотрит вниз."""
    R0 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    c, s = math.cos(pitch), math.sin(pitch)
    Rp = np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])
    c, s = math.cos(roll), math.sin(roll)
    Rr = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return Rr @ Rp @ R0


def intrinsics(f: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def ground_to_image(K: np.ndarray, R: np.ndarray, h: float) -> np.ndarray:
    C = np.array([0.0, 0.0, h])
    return K @ np.column_stack([R[:, 0], R[:, 1], -R @ C])


def project(K: np.ndarray, R: np.ndarray, h: float, P: np.ndarray) -> np.ndarray:
    pc = (P - np.array([0.0, 0.0, h])) @ R.T @ K.T
    return pc[:, :2] / pc[:, 2:3]


def back_to_ground(K: np.ndarray, R: np.ndarray, h: float, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T @ R      # = (R^T K^-1 p)^T
    ok = d[:, 2] < -1e-9
    s = np.where(ok, -h / np.where(ok, d[:, 2], -1.0), 0.0)
    P = np.array([0.0, 0.0, h]) + d * s[:, None]
    return P, ok


@dataclass
class Samples:
    foot: np.ndarray      # (n, 2)
    head: np.ndarray      # (n, 2) верх рамки по центру
    hpx: np.ndarray       # рост в кадре, px


def collect_samples(obs: pd.DataFrame, W: int, Hh: int, cfg: AutoCalibCfg, seed: int = 0) -> Samples:
    o = obs
    bw, bh = o["x2"] - o["x1"], o["y2"] - o["y1"]
    m = ((o["foot_src"] == "ankles") & (o["q"] >= cfg.q_min) & (o["occ"] <= cfg.occ_max)
         & (o["y1"] > 2) & (o["y2"] < Hh - 2) & (o["x1"] > 2) & (o["x2"] < W - 2)
         & (bh >= cfg.min_h_px) & (bh / bw.clip(lower=1) >= cfg.aspect_min) & (o["v"] - o["y1"] > cfg.min_h_px * 0.8))
    s = o[m]
    if len(s) > cfg.max_samples:
        s = s.sample(cfg.max_samples, random_state=seed)
    foot = s[["u", "v"]].to_numpy(float)
    head = np.c_[((s["x1"] + s["x2"]) / 2).to_numpy(float), s["y1"].to_numpy(float)]
    return Samples(foot, head, (foot[:, 1] - head[:, 1]).clip(min=1.0))


@dataclass
class SelfCalib:
    ok: bool
    f: float
    pitch: float
    roll: float
    h: float
    person_h: float
    resid: float                  # медиана |ошибки головы| / рост в кадре
    n: int
    at_bound: bool
    H_local: np.ndarray | None     # пиксель -> пол камеры, м
    reasons: list[str] = field(default_factory=list)
    candidates: list = field(default_factory=list)   # [(угол обзора, SelfCalib)] при фиксированных фокусах


def self_calibrate_fov(sm: Samples, W: int, Hh: int, cfg: AutoCalibCfg, height_m: float | None = None) -> SelfCalib:
    """Фокус не подбирается по людям: для каждого угла обзора из сетки — своя калибровка.
    Возвращает калибровку при hfov_prior_deg; остальные — в .candidates (их выбирает совместное уточнение)."""
    cands = []
    for fov in cfg.hfov_grid:
        f = (W / 2) / math.tan(math.radians(fov) / 2)
        cands.append((float(fov), self_calibrate(sm, W, Hh, cfg, intrinsics(f, W / 2, Hh / 2), height_m)))
    prior = min(cands, key=lambda c: abs(c[0] - cfg.hfov_prior_deg))[1]
    prior.candidates = cands
    return prior


def self_calibrate(sm: Samples, W: int, Hh: int, cfg: AutoCalibCfg, K_fixed: np.ndarray | None = None,
                   height_m: float | None = None) -> SelfCalib:
    n = len(sm.foot)
    if n < cfg.min_samples:
        return SelfCalib(False, 0, 0, 0, 0, 0, 1.0, n, False, None,
                         [f"мало стоящих людей в полный рост: {n} < {cfg.min_samples}"])
    cx, cy = (K_fixed[0, 2], K_fixed[1, 2]) if K_fixed is not None else (W / 2, Hh / 2)
    fix_f = K_fixed[0, 0] if K_fixed is not None else None

    def unpack(x):
        f = fix_f if fix_f is not None else math.exp(x[0])
        if height_m is not None:
            return f, x[1], x[2], height_m, x[3]
        return f, x[1], x[2], x[3], cfg.person_height_m

    def resid(x):
        f, pitch, roll, h, ph = unpack(x)
        K, R = intrinsics(f, cx, cy), rotation(pitch, roll)
        P, ok = back_to_ground(K, R, h, sm.foot)
        P[:, 2] = ph
        pr = project(K, R, h, P)
        r = (pr - sm.head) / sm.hpx[:, None]
        r[~ok | ~np.isfinite(r).all(1)] = 1.0
        return np.clip(r, -3, 3).ravel()

    lo = [math.log(0.3 * W), 0.0, math.radians(-30), 1.4 if height_m else 1.0]
    hi = [math.log(4.0 * W), math.radians(85), math.radians(30), 2.05 if height_m else 15.0]
    x4 = 1.7 if height_m else 3.0
    inits = [[math.log(fm * W), math.radians(p), 0.0, x4]
             for p in (5, 15, 25, 40, 55, 70) for fm in (0.5, 0.8, 1.2, 1.8, 2.6)]
    if fix_f is not None:
        inits = [[math.log(fix_f), math.radians(p), 0.0, x4] for p in (5, 15, 25, 40, 55, 70)]
    loss = lambda r: float(np.sum(np.sqrt(1 + (r / 0.05) ** 2) - 1))  # noqa: E731  soft_l1
    inits.sort(key=lambda x: loss(resid(x)))
    best = None
    for x0 in inits[:6]:
        x0 = np.clip(x0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        sol = least_squares(resid, x0, bounds=(lo, hi), loss="soft_l1", f_scale=0.05, max_nfev=400)
        if best is None or sol.cost < best.cost:
            best = sol
    x = best.x
    f, pitch, roll, h, ph = unpack(x)
    r = resid(x).reshape(-1, 2)
    med = float(np.median(np.hypot(r[:, 0], r[:, 1])))
    span = np.array(hi) - np.array(lo)
    free = [1, 2, 3] if fix_f is not None else [0, 1, 2, 3]
    at_bound = bool(any(min(x[i] - lo[i], hi[i] - x[i]) < 0.01 * span[i] for i in free))
    K, R = intrinsics(f, cx, cy), rotation(pitch, roll)
    Hl = np.linalg.inv(ground_to_image(K, R, h))
    # знак: у точек пола перед камерой однородная координата w > 0 (низ кадра — всегда пол перед камерой)
    if Hl[2] @ np.array([cx, Hh - 1.0, 1.0]) < 0:
        Hl = -Hl
    Hl /= abs(Hl[2, 2])
    reasons = []
    if med > cfg.resid_ok:
        reasons.append(f"модель плохо объясняет людей: ошибка {med:.3f} > {cfg.resid_ok}")
    if at_bound:
        reasons.append("параметр камеры упёрся в границу (мало разных расстояний до людей или дисторсия)")
    return SelfCalib(not reasons, f, pitch, roll, h, ph, med, n, at_bound, Hl, reasons)


# ------------------------------------------------------------------ план
@dataclass
class Site:
    ppm: float
    walls: np.ndarray | None        # bool (H, W) — стена
    size_m: tuple[float, float] | None
    image: Path | None
    cams: dict[int, dict]           # cam_id -> {pos: (x, y) м | None, look_at, height_m}


def load_site(path: str | Path | None, wall_thr: int) -> Site | None:
    if not path:
        return None
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    ppm = float(d["px_per_m"])
    walls = size = img_path = None
    if d.get("plan_image"):
        img_path = Path(d["plan_image"])
        img_path = img_path if img_path.is_absolute() or img_path.exists() else p.parent / img_path
        from .planfile import extract_walls, read_plan

        g = read_plan(img_path)
        # настоящий план: мебель, штриховка и пометки — не стены (стены чуть утолщены: луч не проскальзывает)
        walls = extract_walls(g, ppm, wall_thr)
        size = (g.shape[1] / ppm, g.shape[0] / ppm)
    cams = {}
    for k, c in (d.get("cameras") or {}).items():
        cams[int(k)] = {"pos": np.array(c["pos"], float) / ppm if c.get("pos") else None,
                        "look_at": np.array(c["look_at"], float) / ppm if c.get("look_at") else None,
                        "height_m": c.get("height_m")}
    return Site(ppm, walls, size, img_path, cams)


def rot(phi: float) -> np.ndarray:
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s], [s, c]])


def place_points(local: np.ndarray, phi: float, pos: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return local @ (scale * rot(phi) @ M_FLIP).T + pos


def los_score(pts: np.ndarray, pos: np.ndarray, site: Site, near_m: float, steps: int = 32) -> float:
    """Доля точек, к которым от камеры на плане ведёт отрезок без стен."""
    if site.walls is None or not len(pts):
        return float("nan")
    if len(pts) > 500:   # для скорости — равномерная подвыборка
        pts = pts[:: int(math.ceil(len(pts) / 500))]
    return float(np.mean(in_sight(pts, pos, site, near_m, steps)))


def in_sight(pts: np.ndarray, pos: np.ndarray, site: Site, near_m: float, steps: int = 32) -> np.ndarray:
    """Для каждой точки: ведёт ли к ней от камеры отрезок без стен."""
    Hm, Wm = site.walls.shape
    vec = pts - pos
    L = np.hypot(*vec.T)
    # шаг не больше пикселя плана: иначе луч «перепрыгивает» тонкую стену
    steps = int(min(max(steps, np.max(L) * site.ppm * 1.2 + 2), 600))
    s = np.linspace(0, 1, steps)
    smp = pos[None, None, :] + vec[:, None, :] * s[None, :, None]          # (n, steps, 2)
    px = np.round(smp * site.ppm).astype(np.int64)
    inside = (px[..., 0] >= 0) & (px[..., 0] < Wm) & (px[..., 1] >= 0) & (px[..., 1] < Hm)
    wall = np.ones(px.shape[:2], bool)
    wall[inside] = site.walls[px[..., 1][inside], px[..., 0][inside]]
    # камера может быть нарисована прямо на стене: стена, с которой луч начинается (не дальше near_m), не мешает;
    # стена, до которой луч дошёл по свободному месту, — мешает (в том числе стена за спиной камеры)
    lead = np.cumprod(wall, axis=1).astype(bool) & ((s[None, :] * L[:, None]) <= near_m)
    return ~(wall & ~lead).any(1)


def search_phi(local: np.ndarray, pos: np.ndarray, site: Site, cfg: AutoCalibCfg,
               look_at: np.ndarray | None) -> tuple[float, float, float]:
    """-> (φ, доля видимых точек, неоднозначность = лучший далёкий поворот / лучший)."""
    if look_at is not None:
        v = look_at - pos
        # «вперёд» камеры (0, 1) после M_FLIP = (0, -1); φ поворачивает его на направление v
        center = math.atan2(v[1], v[0]) - math.atan2(-1.0, 0.0)
        grid = center + np.radians(np.arange(-30, 30.1, 1.0))
    else:
        grid = np.radians(np.arange(0, 360, 2.0))
    sc = np.array([los_score(place_points(local, p, pos), pos, site, cfg.near_cam_m) for p in grid])
    if np.isnan(sc).all():
        return (float(grid[len(grid) // 2]) if look_at is not None else 0.0), float("nan"), float("nan")
    i = int(np.nanargmax(sc))
    step = float(grid[1] - grid[0])
    fine = grid[i] + np.arange(-step, step + 1e-9, math.radians(0.25))
    fs = np.array([los_score(place_points(local, p, pos), pos, site, cfg.near_cam_m) for p in fine])
    # все повороты, где точки «помещаются» почти одинаково, — плато; берём его середину
    top = np.nanmax(fs)
    good = np.where(fs >= top - 0.005)[0]
    j = int(good[len(good) // 2])
    # плато шире шага сетки — расширяем поиск в обе стороны
    lo_phi, hi_phi = fine[good[0]], fine[good[-1]]
    for sgn in (-1, 1):
        p = lo_phi if sgn < 0 else hi_phi
        for _ in range(120):
            q = p + sgn * math.radians(0.25)
            if los_score(place_points(local, q, pos), pos, site, cfg.near_cam_m) < top - 0.005:
                break
            p = q
        if sgn < 0:
            lo_phi = p
        else:
            hi_phi = p
    best_phi = (lo_phi + hi_phi) / 2 if hi_phi > lo_phi else float(fine[j])
    d = np.abs((grid - grid[i] + np.pi) % (2 * np.pi) - np.pi)
    far = sc[d > math.radians(25)]
    amb = float(np.nanmax(far) / max(sc[i], 1e-9)) if len(far) and look_at is None else 0.0
    return float(best_phi), float(los_score(place_points(local, best_phi, pos), pos, site, cfg.near_cam_m)), amb


# ------------------------------------------------------------------ совместное уточнение
@dataclass
class CamState:
    cid: int
    local: np.ndarray            # точки ног на полу камеры, м (для видимости и соседей)
    bins: np.ndarray             # номер момента времени для каждой точки
    pos0: np.ndarray
    phi: float
    scale: float = 1.0
    shift: np.ndarray = field(default_factory=lambda: np.zeros(2))
    los: float = float("nan")
    amb: float = float("nan")
    need_phi: bool = False       # поворот неизвестен: только по соседям
    cands: list = field(default_factory=list)   # [(угол обзора, точки на полу, SelfCalib)] — варианты фокуса
    ci: int = -1                 # выбранный вариант
    fov_by: str = "типичный угол обзора"

    def plan(self, phi=None, scale=None, shift=None) -> np.ndarray:
        return place_points(self.local, self.phi if phi is None else phi, self.pos0 + (self.shift if shift is None else shift),
                            self.scale if scale is None else scale)


def _tree(states: list[CamState], skip: int) -> tuple[cKDTree | None, int]:
    pts = [np.c_[s.plan(), s.bins * 1000.0] for s in states if s.cid != skip]
    if not pts:
        return None, 0
    P = np.concatenate(pts)
    return cKDTree(P), len(P)


def cross_cost(s: CamState, tree: cKDTree | None, xy: np.ndarray, trunc: float = 1.5) -> tuple[float, float, float]:
    """-> (средняя усечённая дистанция до ближайшего одновременного человека соседей, доля совпавших, медиана совпавших)."""
    if tree is None:
        return 0.0, 0.0, float("nan")
    d, _ = tree.query(np.c_[xy, s.bins * 1000.0], k=1, distance_upper_bound=trunc)
    d = np.minimum(d, trunc)
    m = d < trunc
    return float(d.mean()), float(np.mean(d < 0.5)), float(np.median(d[m])) if m.any() else float("nan")


def refine(states: list[CamState], site: Site | None, cfg: AutoCalibCfg) -> None:
    for rnd in range(cfg.refine_rounds):
        for s in states:
            tree, n = _tree(states, s.cid)

            def J(phi, scale, shift):
                xy = s.plan(phi, scale, shift)
                c, frac, _ = cross_cost(s, tree, xy)
                j = c if frac >= 0.05 else 0.0
                if site is not None and site.walls is not None:
                    j += 2.0 * (1.0 - los_score(xy, s.pos0 + shift, site, cfg.near_cam_m, steps=24))
                j += 0.3 * float(np.hypot(*shift)) / max(cfg.pos_prior_m, 1e-6)
                # Видимость всегда лучше при меньшем масштабе: сжатые точки не задевают стен. Без платы за отход
                # от 1 подбор каждый раз уходит в нижний предел; менять масштаб должны соседние камеры
                j += 0.4 * abs(math.log(scale))
                return j

            if s.need_phi and rnd == 0:
                phis = np.radians(np.arange(0, 360, 3.0))
            else:
                w = 15 if rnd == 0 else 6
                phis = s.phi + np.radians(np.arange(-w, w + 0.1, 1.5 if rnd == 0 else 1.0))

            # Рост людей уже задал метры. Менять масштаб можно только по соседним камерам: одни стены всегда
            # просят его уменьшить (у камер без соседей рост получался 1.36 м вместо 1.70)
            seen = tree is not None and cross_cost(s, tree, s.plan())[1] >= 0.05
            if not seen:
                s.scale = 1.0
            scales = (0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2) if seen else (1.0,)

            def best_pose(phis_):
                b = (J(s.phi, s.scale, s.shift), s.phi, s.scale)
                for phi in phis_:
                    for sc in scales:
                        v = J(phi, sc, s.shift)
                        if v < b[0]:
                            b = (v, phi, sc)
                return b

            # фокус: из вариантов — тот, при котором камера лучше совпадает с соседями (нужна общая зона)
            if rnd >= 1 and len(s.cands) > 1 and tree is not None and cross_cost(s, tree, s.plan())[1] >= 0.05:
                cur = (s.local, s.bins, s.ci, s.phi, s.scale)
                options = []
                for k, (fov, local, cal, bins) in enumerate(s.cands):
                    if fov is None or not cal.ok:     # вариант, который сам плохо объясняет людей, не берём
                        continue
                    s.local, s.bins, s.ci = local, bins, k
                    v, phi, sc = best_pose(s.phi + np.radians(np.arange(-8, 8.1, 2.0)))
                    pen = cfg.f_prior_weight * abs(math.log(math.tan(math.radians(fov) / 2)
                                                            / math.tan(math.radians(cfg.hfov_prior_deg) / 2)))
                    options.append((v + pen, k, phi, sc))
                    s.local, s.bins, s.ci, s.phi, s.scale = cur
                if options:
                    _, k, phi, sc = min(options)
                    if k != s.ci:
                        s.fov_by = "совпадение с соседними камерами"
                    s.local, s.bins, s.ci, s.phi, s.scale = s.cands[k][1], s.cands[k][3], k, phi, sc
            best = best_pose(phis)
            s.phi, s.scale = best[1], best[2]
            r = cfg.pos_prior_m
            grid = np.arange(-r, r + 1e-9, r / 4)
            bj, bs = J(s.phi, s.scale, s.shift), s.shift
            for dx in grid:
                for dy in grid:
                    sh = np.array([dx, dy])
                    v = J(s.phi, s.scale, sh)
                    if v < bj:
                        bj, bs = v, sh
            s.shift = bs


def plan_scale(states: list[CamState], site: Site | None, cams: pd.DataFrame, cfg: AutoCalibCfg) -> float | None:
    """Новый масштаб плана, пикс/м, или None. Нужно минимум две камеры, совпавшие с соседями, и один знак поправки.

    Рост людей задаёт настоящие метры у каждой камеры, а соседние камеры проверяют, что они сошлись на плане.
    Если камеры согласно растягивают или сжимают картину, ошибка не у них, а в масштабе плана. Камеры с ручной H
    посчитаны в старом масштабе, поэтому при них он не меняется."""
    if site is None or site.image is None:
        return None
    for c in cams.itertuples():
        pj = json.loads(c.passport_json)
        method = pj.get("calib_method")
        if method == "manual" or (pj.get("H") is not None and method != "auto"):   # calibrate метку не ставит
            return None
    sure = []
    for s in states:
        tree, _ = _tree(states, s.cid)
        _, frac, med = cross_cost(s, tree, s.plan())
        if frac >= 0.05 and not math.isnan(med) and med <= cfg.overlap_ok_m:
            sure.append(s.scale)
    if len(sure) < 2 or len({np.sign(v - 1.0) for v in sure}) > 1:
        return None
    k = float(np.median(sure))
    return round(site.ppm * k, 2) if abs(k - 1.0) >= 0.03 else None


def write_scale(path: Path, ppm: float) -> None:
    """Пишет масштаб в site.json, первая версия остаётся рядом."""
    keep = path.with_name(path.stem + ".before_scale.json")
    if not keep.exists():
        shutil.copyfile(path, keep)
    d = json.loads(path.read_text(encoding="utf-8"))
    d["px_per_m"] = ppm
    path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ проверка: картинка
def check_image(frame: np.ndarray, H: np.ndarray, plan_xy: np.ndarray, pos: np.ndarray, site: Site | None,
                title: str, K: np.ndarray | None = None, dist: np.ndarray | None = None) -> np.ndarray:
    """Слева — кадр с сеткой 1 м на полу (по найденной H), справа — точки камеры на плане.
    У широкого объектива H относится к выпрямленному кадру, поэтому сетка в кадре изогнута вместе с полом."""
    fr = frame.copy()
    Hi = np.linalg.inv(H)
    lo = np.floor(np.percentile(plan_xy, 2, 0)) - 1
    hi = np.ceil(np.percentile(plan_xy, 98, 0)) + 1
    for k, (a, b) in enumerate([(lo[0], hi[0]), (lo[1], hi[1])]):
        for c in np.arange(a, b + 1):
            t = np.linspace(lo[1 - k], hi[1 - k], 80)
            line = np.c_[np.full_like(t, c), t] if k == 0 else np.c_[t, np.full_like(t, c)]
            ph = np.c_[line, np.ones(len(line))] @ Hi.T
            okp = ph[:, 2] > 1e-9
            uv = distort_points(ph[:, :2] / np.where(okp, ph[:, 2], 1)[:, None], K, dist)
            if site is not None and site.walls is not None:    # сетка только на видимом полу, а не сквозь стены
                okp = okp & in_sight(line, pos, site, 0.6, steps=48)
            for i in range(1, len(uv)):
                if okp[i] and okp[i - 1] and np.all(np.abs(uv[i - 1:i + 1]) < 1e5):
                    cv2.line(fr, tuple(np.int32(uv[i - 1])), tuple(np.int32(uv[i])), (0, 255, 255), 1, cv2.LINE_AA)
    # план
    if site is not None and site.walls is not None:
        pl = np.where(site.walls[..., None], 40, 235).astype(np.uint8).repeat(3, 2)
        ppm = site.ppm
    else:
        span = np.maximum(hi - lo, 1)
        ppm = 400 / span.max()
        pl = np.full((int(span[1] * ppm) + 1, int(span[0] * ppm) + 1, 3), 235, np.uint8)
        plan_xy, pos = plan_xy - lo, pos - lo
    for x, y in (plan_xy * ppm).astype(int)[:: max(1, len(plan_xy) // 1500)]:
        if 0 <= x < pl.shape[1] and 0 <= y < pl.shape[0]:
            cv2.circle(pl, (x, y), 2, (200, 90, 30), -1)
    cv2.circle(pl, tuple(np.int32(pos * ppm)), 7, (0, 0, 220), -1)
    s = fr.shape[0] / pl.shape[0]
    pl = cv2.resize(pl, (max(1, int(pl.shape[1] * s)), fr.shape[0]))
    out = np.concatenate([fr, pl], 1)
    cv2.putText(out, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def _frame(video: str, fps: float | None) -> np.ndarray | None:
    from .video_io import VideoSource

    try:
        vs = VideoSource(video, fps=fps)
        mid = (vs.n_frames or 2) // 2
        return next(iter(vs.frames(mid, mid + 1)))[1]
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ главная функция
def autocalib(data_dir: str | Path, cfg: Config, site_path: str | Path | None = None,
              passports: list[str] | None = None, apply: bool = True, manual: bool = False,
              check_dir: str | Path | None = None, only_missing: bool = False, _scale_pass: int = 0) -> dict:
    ac = cfg.autocalib
    store = Store(data_dir)
    site = load_site(site_path, ac.wall_thr)
    check = Path(check_dir) if check_dir else store.root / "calib_check"
    check.mkdir(parents=True, exist_ok=True)
    files: dict[int, Path] = {}
    for p in passports or []:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        files[int(d["cam_id"])] = Path(p)
    cams = store.load_cameras()
    if cams.empty:
        raise ValueError("в БД нет камер: сначала process (можно без H)")
    res: dict[int, dict] = {}
    states: list[CamState] = []
    lenses: dict[int, tuple] = {}
    for c in cams.itertuples():
        cid = int(c.cam_id)
        pj = json.loads(c.passport_json)
        r: dict = {"cam_id": cid, "status": "fail", "reasons": []}
        res[cid] = r
        if only_missing and pj.get("H") is not None and pj.get("calib_method") != "auto":
            r.update(status="skip", reasons=["H уже задана вручную"])
            continue
        obs = store.load_observations(cid, columns=["t", "x1", "y1", "x2", "y2", "u", "v", "foot_src", "q", "occ"])
        K = np.asarray(pj["K"], float) if pj.get("K") and pj.get("dist") else None
        dist = np.asarray(pj["dist"], float) if K is not None else None
        lenses[cid] = (K, dist)
        sc_site = (site.cams.get(cid) if site else None) or {}
        # Отбор — по кадру: рамки у края отсекаются в его пикселях. Сами точки — без дисторсии, как у to_floor
        sm = collect_samples(obs, int(c.width), int(c.height), ac)
        if K is not None:
            foot, head = undistort_points(sm.foot, K, dist), undistort_points(sm.head, K, dist)
            sm = Samples(foot, head, np.hypot(*(foot - head).T).clip(min=1.0))
            obs[["u", "v"]] = undistort_points(obs[["u", "v"]].to_numpy(float), K, dist)
        if K is not None:
            cal = self_calibrate(sm, int(c.width), int(c.height), ac, K, sc_site.get("height_m"))
        else:
            cal = self_calibrate_fov(sm, int(c.width), int(c.height), ac, sc_site.get("height_m"))
        r["n_samples"] = cal.n
        r["calib_ok"] = cal.ok
        if cal.H_local is None:
            r["reasons"] += cal.reasons
            continue
        sel = obs[obs["q"] >= 0.3]
        if len(sel) > 4000:
            sel = sel.sample(4000, random_state=0)
        uv = sel[["u", "v"]].to_numpy(float)
        bins_all = np.round(sel["t"].to_numpy() / 0.2).astype(np.int64)

        def to_local(Hl):
            ph = np.c_[uv, np.ones(len(uv))] @ Hl.T
            okp = ph[:, 2] > 1e-9
            loc = np.full((len(uv), 2), np.nan)
            loc[okp] = ph[okp, :2] / ph[okp, 2:3]
            return loc

        # у каждого варианта фокуса — свои точки: годятся те, что перед камерой и не дальше 40 м
        cl = []
        for fov, cc in (cal.candidates or [(None, cal)]):
            if cc.H_local is None:
                continue
            loc = to_local(cc.H_local)
            keep = np.isfinite(loc).all(1)
            keep[keep] = np.hypot(*loc[keep].T) < 40
            if keep.sum() >= 20:
                cl.append((fov, loc[keep], cc, bins_all[keep]))
        ci = next((k for k, c_ in enumerate(cl) if c_[2] is cal), -1)
        if ci < 0:
            r["reasons"] += cal.reasons or ["точки ног не проецируются на пол (камера смотрит выше горизонта?)"]
            continue
        pos = sc_site.get("pos")
        if pos is None:
            r["reasons"].append("нет положения камеры на плане (site.json: cameras.<id>.pos)")
            continue
        st = CamState(cid, cl[ci][1], cl[ci][3], np.asarray(pos, float), 0.0, cands=cl, ci=ci)
        if K is not None:
            st.fov_by = "K из паспорта"
        if site is not None and site.walls is not None:
            st.phi, st.los, st.amb = search_phi(st.local, st.pos0, site, ac, sc_site.get("look_at"))
        elif sc_site.get("look_at") is not None:
            st.phi, _, _ = search_phi(st.local, st.pos0, Site(1, None, None, None, {}), ac, sc_site["look_at"])
        else:
            st.need_phi = True
        states.append(st)
    if states:
        refine(states, site, ac)
    fixed = plan_scale(states, site, cams, ac)
    if fixed is not None and _scale_pass < 2:
        old = float(json.loads(Path(site_path).read_text(encoding="utf-8"))["px_per_m"])
        write_scale(Path(site_path), fixed)
        print(f"Масштаб плана уточнён по совпадению камер: {old} -> {fixed} пикс/м", flush=True)
        out = autocalib(data_dir, cfg, site_path, passports, apply, manual, check_dir, only_missing, _scale_pass + 1)
        out["scale_from"] = out.get("scale_from") or old
        return out
    for s in states:
        r = res[s.cid]
        xy = s.plan()
        if site is not None and site.walls is not None:
            s.los = los_score(xy, s.pos0 + s.shift, site, ac.near_cam_m)
        tree, _ = _tree(states, s.cid)
        _, frac, med = cross_cost(s, tree, xy)
        has_ovl = frac >= 0.05
        fov, _, cal, _ = s.cands[s.ci]
        r.update(resid=round(cal.resid, 4), f_px=round(cal.f, 1), hfov_deg=fov, fov_by=s.fov_by,
                 pitch_deg=round(math.degrees(cal.pitch), 1), roll_deg=round(math.degrees(cal.roll), 1),
                 height_m=round(cal.h * s.scale, 2), person_h_m=round(cal.person_h * s.scale, 2))
        r["calib_ok"] = cal.ok
        r["reasons"] += cal.reasons
        S = np.eye(3)
        S[:2, :2] = s.scale * rot(s.phi) @ M_FLIP
        S[:2, 2] = s.pos0 + s.shift
        H = S @ cal.H_local
        H /= abs(H[2, 2])
        r.update(phi_deg=round(math.degrees(s.phi) % 360, 1), scale=s.scale,
                 shift_m=[round(float(v), 2) for v in s.shift], los=None if math.isnan(s.los) else round(s.los, 3),
                 ambiguity=None if math.isnan(s.amb) else round(s.amb, 3),
                 overlap_frac=round(frac, 3), overlap_med_m=None if math.isnan(med) else round(med, 3),
                 H=H.tolist())
        reasons = r["reasons"]
        cross_ok = has_ovl and med <= ac.overlap_ok_m
        if has_ovl and not cross_ok:
            reasons.append(f"расходится с соседними камерами: {med:.2f} м > {ac.overlap_ok_m}")
        # слабые признаки беды: мало совпадений с соседями, но и те далеко; масштаб упёрся в границу
        weak_bad = (not has_ovl and frac >= 0.01 and not math.isnan(med) and med > ac.overlap_ok_m)
        if weak_bad:
            reasons.append(f"с соседями почти не совпадает ({frac:.0%} точек, {med:.2f} м) — проверьте положение камеры")
        if abs(s.scale - 1.0) >= 0.149:
            reasons.append(f"масштаб пришлось поменять на {s.scale - 1:+.0%} — проверьте положение камеры "
                           "и масштаб плана")
            weak_bad = True
        if not r["calib_ok"]:
            r["status"] = "fail"
        elif not math.isnan(s.los):
            amb_bad = not math.isnan(s.amb) and s.amb > ac.ambiguity_max
            if s.los < ac.los_check:
                reasons.append(f"много точек «за стенами»: видно {s.los:.0%}")
                r["status"] = "fail"
            elif s.los >= ac.los_ok and not amb_bad and (cross_ok or not has_ovl):
                r["status"] = "ok"
            elif s.los >= ac.los_ok and amb_bad and cross_ok:
                # по стенам поворот неоднозначен, но соседние камеры его однозначно задают
                r["status"] = "ok"
            else:
                if amb_bad:
                    reasons.append(f"поворот неоднозначен (другой поворот почти так же хорош: {s.amb:.2f})")
                if s.los < ac.los_ok:
                    reasons.append(f"видно {s.los:.0%} точек (< {ac.los_ok:.0%})")
                r["status"] = "check" if (cross_ok or not amb_bad) else "fail"
        elif cross_ok:
            r["status"] = "ok"
        elif not s.need_phi:
            reasons.append("нет плана со стенами и общих зон с соседями: поворот только по look_at — проверьте")
            r["status"] = "check"
        else:
            reasons.append("поворот не определить: нет картинки плана, look_at и общих зон с соседями")
            r["status"] = "fail"
        if weak_bad and r["status"] == "ok":
            r["status"] = "check"
        cam = store.get_camera(s.cid)
        fr = _frame(cam["video_path"], cam["fps"])
        if fr is not None:
            img = check_image(fr, H, xy, s.pos0 + s.shift, site, f"cam{s.cid}: {r['status']}", *lenses[s.cid])
            cv2.imencode(".png", img)[1].tofile(str(check / f"cam{s.cid}.png"))
            r["check_image"] = str(check / f"cam{s.cid}.png")
    for r in res.values():
        r.pop("H_local", None)
    (check / "autocalib.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    applied, manual_done = [], []
    if apply:
        for cid, r in res.items():
            if r["status"] in ("ok", "check"):
                _apply(store, cid, files.get(cid), np.asarray(r["H"]), cfg, {"calib_method": "auto",
                       "calib_auto": {k: r[k] for k in ("status", "resid", "los", "overlap_med_m", "height_m")}})
                applied.append(cid)
        if manual:
            from .calibrate import calibrate

            for cid, r in res.items():
                if r["status"] != "fail":
                    continue
                f = files.get(cid)
                if f is None or site is None or site.image is None:
                    r["reasons"].append("ручная привязка: нужен --passports и plan_image в site.json")
                    continue
                print(f"cam{cid}: автопривязка не удалась ({'; '.join(r['reasons'])}) — ручная привязка")
                calibrate(f, site.image, site.ppm)
                d = json.loads(f.read_text(encoding="utf-8"))
                d["calib_method"] = "manual"
                f.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
                _apply(store, cid, f, np.asarray(d["H"]), cfg, {"calib_method": "manual"})
                r["status"] = "manual"
                manual_done.append(cid)
    plan = {"site": str(site_path) if site_path else None,
            "plan": str(site.image) if site is not None and site.image else None, "ppm": site.ppm if site else None}
    return {"cams": res, "applied": applied, "manual": manual_done, "check_dir": str(check), **plan,
            "files": {cid: str(p) for cid, p in files.items()}}


def _apply(store: Store, cid: int, file: Path | None, H: np.ndarray, cfg: Config, extra: dict) -> None:
    """Пишет H в паспорт (файл — с резервной копией) и пересчитывает треклеты камеры."""
    from .pipeline import retrack

    cam = store.get_camera(cid)
    if file is not None:
        d = json.loads(file.read_text(encoding="utf-8"))
        bak = file.with_name(file.stem + ".before_autocalib.json")
        if not bak.exists():
            shutil.copyfile(file, bak)
    else:
        d = json.loads(cam["passport_json"])
    d["H"] = np.asarray(H).tolist()
    d.update(extra)
    if file is not None and extra.get("calib_method") == "auto":
        file.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    retrack(cid, store.root, cfg, passport_override=d)


def manual_hints(res: dict) -> list[str]:
    """Что делать с камерами, которые не привязались: готовые команды, а не заготовки."""
    fails = sorted(c for c, r in res["cams"].items() if r["status"] == "fail")
    if not fails:
        return []
    out = [f"  Не привязались: {', '.join(f'cam{c}' for c in fails)}."]
    few = [c for c in fails if any("мало стоящих людей" in w for w in res["cams"][c].get("reasons", []))]
    if few:
        out.append(f"  {', '.join(f'cam{c}' for c in few)}: людей в кадре мало. На полной записи их обычно больше, "
                   "поэтому после пробы это не ошибка.")
    out.append("  Вручную: 6–10 пар точек на полу (углы плитки, ножки столов, двери), сначала в кадре, затем на плане. "
               "Enter — сохранить:")
    folder = Path(res["site"]).parent if res.get("site") else None    # рабочие паспорта лежат рядом с site.json
    for c in fails:
        f = res.get("files", {}).get(c)
        passport = folder / Path(f).name if f and folder else f
        if passport and res.get("plan"):
            out.append(f'    python -m mtmc calibrate --passport "{passport}" --plan "{res["plan"]}" '
                       f'--px-per-m {res["ppm"]}')
        else:
            out.append(f"    cam{c}: нужен план (site.json) и паспорт камеры (--passports)")
    return out


def format_autocalib(res: dict) -> str:
    lines = ["Автопривязка камер к плану:"]
    if res.get("scale_from"):
        lines.append(f"  масштаб плана уточнён автоматически: {res['scale_from']} -> {res['ppm']} пикс/м")
    for cid, r in sorted(res["cams"].items()):
        s = (f"  cam{cid}: {r['status'].upper():6s}")
        if "f_px" in r:
            s += (f" высота {r.get('height_m')} м, наклон {r.get('pitch_deg')}°, фокус {r.get('f_px')} px, "
                  f"ошибка {r.get('resid')}, людей {r.get('n_samples')}")
        if r.get("los") is not None:
            s += f"; видно без стен {r['los']:.0%}"
        if r.get("overlap_med_m") is not None and r.get("overlap_frac", 0) >= 0.05:
            s += f"; с соседями {r['overlap_med_m']} м"
        lines.append(s)
        for why in r.get("reasons", []):
            lines.append(f"      - {why}")
    lines += manual_hints(res)
    lines.append(f"  Проверочные картинки: {res['check_dir']} (жёлтая сетка 1 м должна лечь на пол)")
    return "\n".join(lines)
