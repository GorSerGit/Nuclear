"""Сквозные треки (ТЗ хакатона): непрерывные фрагменты наблюдения человека хотя бы одной камерой.

Вход — склейка link (person_id у каждого треклета). Человек — это «скорее всего один и тот же»; трек — это
непрерывность наблюдения. Трек одного person_id рвётся там, где человека не видит НИ ОДНА камера дольше
tracks.end_gap_s: после такого исчезновения фрагменты не объединяются, даже если внешность совпадает
(связь с прежним треком сохраняется только как подсказка prev_track / next_track).

Короткие пропуски (< end_gap_s) — потеря детекции или переход между камерами — трек не рвут.
Шум: трек короче min_track_s или с малым числом наблюдений в счёт не идёт.
"""
from __future__ import annotations

import csv
import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .clothes import clash, load_colors
from .config import Config, TracksCfg
from .storage import Store

log = logging.getLogger(__name__)

TRACKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
  run_id INTEGER, track_id INTEGER, person_id INTEGER, t0 REAL, t1 REAL, cams TEXT,
  n_tracklets INTEGER, n_obs INTEGER, start_reason TEXT, end_reason TEXT, prev_track INTEGER, next_track INTEGER,
  segments_json TEXT, gaps_json TEXT,
  PRIMARY KEY (run_id, track_id));
CREATE TABLE IF NOT EXISTS tracklet_track (
  run_id INTEGER, tid INTEGER, track_id INTEGER,   -- track_id 0: шум, в счёт не идёт
  PRIMARY KEY (run_id, tid));
CREATE TABLE IF NOT EXISTS track_person (
  run_id INTEGER, track_id INTEGER, person_no INTEGER, how TEXT,
  PRIMARY KEY (run_id, track_id));
"""


@dataclass
class Piece:
    """Треклет для построения треков."""
    tid: int
    cam: int
    t0: float
    t1: float
    n_obs: int
    is_weak: bool = False
    exit_zone: bool = False            # треклет закончился в зоне выхода паспорта (zones.<имя>.leads_to)
    q: float | None = None             # средние по наблюдениям: качество, перекрытие, уверенность детектора
    occ: float | None = None
    conf: float | None = None


@dataclass
class Scene:
    """Геометрия для правила «пропуск детекции» и подписи промежутков: зоны обзора камер и положения треклетов."""
    fov: dict[int, np.ndarray]                     # камера -> многоугольник зоны обзора на плане, м
    pos: dict[int, pd.DataFrame]                   # tid -> положение на сетке времени (x, y)

    def end_xy(self, tid: int, last: bool) -> np.ndarray | None:
        p = self.pos.get(tid)
        if p is None or p.empty:
            return None
        r = p.iloc[-1] if last else p.iloc[0]
        return np.array([r["x"], r["y"]], float)

    def depth(self, xy: np.ndarray | None) -> float:
        """Насколько точка внутри ближайшей зоны обзора (м); < 0 — вне всех зон; nan — не известно."""
        from .geometry import signed_dist

        if xy is None or not self.fov:
            return float("nan")
        return max(signed_dist(poly, xy[0], xy[1]) for poly in self.fov.values())

    def gap_kind(self, a_xy: np.ndarray | None, b_xy: np.ndarray | None) -> str:
        """detection — пропал внутри зоны камеры (пропуск детекции); outside — вне обзора всех камер."""
        if a_xy is None or b_xy is None or not self.fov:
            return "unknown"
        return "detection" if self.depth((a_xy + b_xy) / 2) >= 0 else "outside"


@dataclass
class EndTrack:
    track_id: int                      # 1..N по времени начала; 0 — шум
    person_id: int
    t0: float
    t1: float
    tids: list[int]
    cams: list[int]
    segments: list[tuple[int, float, float]]       # (камера, начало, конец) — когда какая камера видела
    gaps: list[tuple[float, float]]                # моменты внутри трека, когда не видела ни одна камера
    n_obs: int
    start_reason: str = ""
    end_reason: str = ""
    prev_track: int | None = None      # предыдущий трек того же человека (по внешности), если был
    next_track: int | None = None
    person_no: int = 0                 # человек P1..: все его треки (ушёл из всех камер и вернулся — тот же P)
    person_how: str = ""               # почему трек отнесён к человеку: первый трек / склейка / внешность 0.94
    pts: list[tuple[float, float, float]] = field(default_factory=list)   # (t, x, y) на плане
    gap_kinds: list[str] = field(default_factory=list)   # к gaps: detection | outside | unknown
    bridged: int = 0                   # сколько пропусков детекции дольше end_gap_s трек пережил
    reject: str = ""                   # почему не в счёт: короткий / фантом / призрак
    q: float | None = None             # средние по наблюдениям трека
    occ: float | None = None
    conf: float | None = None

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


def union_intervals(iv: list[tuple[float, float]], join_gap: float = 0.0) -> list[tuple[float, float]]:
    """Объединение интервалов; соседние с паузой ≤ join_gap сливаются."""
    out: list[list[float]] = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1] + join_gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def can_bridge(last: Piece, nxt: Piece, gap: float, scene: Scene | None, cfg: TracksCfg | None) -> bool:
    """Пропуск детекции, а не уход: человек пропал и появился ВНУТРИ зоны обзора камеры (не ближе fov_margin_m
    к её краю, т.е. не у края кадра), не в зоне выхода, рядом с тем же местом и не позже detect_gap_max_s."""
    if scene is None or cfg is None or gap > cfg.detect_gap_max_s or last.exit_zone:
        return False
    a, b = scene.end_xy(last.tid, last=True), scene.end_xy(nxt.tid, last=False)
    if a is None or b is None:
        return False
    if not (scene.depth(a) >= cfg.fov_margin_m and scene.depth(b) >= cfg.fov_margin_m):
        return False
    return float(np.hypot(*(b - a))) <= 1.0 + cfg.bridge_v_max_mps * gap


def split_person(pieces: list[Piece], end_gap_s: float, scene: Scene | None = None,
                 cfg: TracksCfg | None = None) -> list[list[Piece]]:
    """Режет треклеты одного человека на треки: пауза во всех камерах > end_gap_s — новый трек,
    если только это не пропуск детекции внутри зоны обзора (can_bridge)."""
    out: list[list[Piece]] = []
    cur_end, last = -np.inf, None
    for p in sorted(pieces, key=lambda p: (p.t0, p.t1)):
        gap = p.t0 - cur_end
        if not out or (gap > end_gap_s and not can_bridge(last, p, gap, scene, cfg)):
            out.append([p])
            cur_end, last = p.t1, p
        else:
            out[-1].append(p)
            if p.t1 >= cur_end:
                cur_end, last = p.t1, p
    return out


def bridge_groups(groups: list[tuple[int, list[Piece]]], scene: Scene, cfg: TracksCfg) -> list[tuple[int, list[Piece]]]:
    """Сшивает соседние по времени треки одного человека (склейки), если между ними — пропуск детекции."""
    by: dict[int, list[list[Piece]]] = {}
    for pid, g in groups:
        by.setdefault(pid, []).append(list(g))
    out = []
    for pid, lst in by.items():
        lst.sort(key=lambda g: min(p.t0 for p in g))
        cur = lst[0]
        for g in lst[1:]:
            last = max(cur, key=lambda p: p.t1)
            first = min(g, key=lambda p: p.t0)
            gap = first.t0 - last.t1
            if gap >= -cfg.person_overlap_tol_s and can_bridge(last, first, max(gap, 0.0), scene, cfg):
                cur = cur + g
            else:
                out.append((pid, cur))
                cur = g
        out.append((pid, cur))
    return out


def continue_groups(groups: list[tuple[int, list[Piece]]], scene: Scene, protos: dict | None,
                    cfg: TracksCfg) -> tuple[list[tuple[int, list[Piece]]], int]:
    """Один непрерывный человек, разрезанный на куски: конец трека и начало другого на одной камере совпадают по времени
    и месту. Группы людей здесь не важны: разрез по внешности разводит половинки в разные группы, а ID обязан сохраниться.

    Если продолжений два (рядом шли двое) или у начала два предшественника — неясно, чьё оно, и ничего не соединяется."""
    if cfg.cont_gap_s <= 0:
        return groups, 0
    items = [(pid, list(g)) for pid, g in groups]

    def link_ok(a: list[Piece], b: list[Piece]) -> bool:
        last, first = max(a, key=lambda p: p.t1), min(b, key=lambda p: p.t0)
        gap = first.t0 - last.t1
        if last.cam != first.cam or not 0 <= gap <= cfg.cont_gap_s or last.exit_zone:
            return False
        xa, xb = scene.end_xy(last.tid, True), scene.end_xy(first.tid, False)
        if xa is None or xb is None or float(np.hypot(*(xb - xa))) > cfg.cont_m + 2.0 * gap:
            return False
        pa, pb = (protos or {}).get(last.tid), (protos or {}).get(first.tid)
        return pa is None or pb is None or float(pa[0] @ pb[0]) >= cfg.cont_app

    merged = 0
    changed = True
    while changed:
        changed = False
        for i, (_, a) in enumerate(items):
            nxt = [j for j, (_, b) in enumerate(items) if j != i and link_ok(a, b)]
            if len(nxt) != 1:
                continue
            j = nxt[0]
            if sum(link_ok(x, items[j][1]) for k, (_, x) in enumerate(items) if k != j) != 1:
                continue
            items[i] = (items[i][0], a + items[j][1])
            del items[j]
            merged += 1
            changed = True
            break
    return items, merged


def describe(group: list[Piece], cfg: TracksCfg, scene: Scene | None = None) -> dict:
    """Камеры, интервалы по камерам, невидимые промежутки внутри трека."""
    t0 = min(p.t0 for p in group)
    t1 = max(p.t1 for p in group)
    segs = []
    for cam in sorted({p.cam for p in group}):
        for a, b in union_intervals([(p.t0, p.t1) for p in group if p.cam == cam], cfg.cam_merge_gap_s):
            segs.append((cam, a, b))
    seen = union_intervals([(p.t0, p.t1) for p in group])
    gaps = [(seen[i][1], seen[i + 1][0]) for i in range(len(seen) - 1)]
    kinds = []
    for a, b in gaps:   # где человек был, когда его не видел никто: в зоне камеры или вне всех зон
        before = max((p for p in group if p.t1 <= a + 1e-6), key=lambda p: p.t1, default=None)
        after = min((p for p in group if p.t0 >= b - 1e-6), key=lambda p: p.t0, default=None)
        kinds.append(scene.gap_kind(scene.end_xy(before.tid, True), scene.end_xy(after.tid, False))
                     if scene and before and after else "unknown")
    bridged = sum(1 for a, b in gaps if b - a > cfg.end_gap_s)
    return {"t0": t0, "t1": t1, "segments": segs, "gaps": gaps, "kinds": kinds, "bridged": bridged,
            "cams": sorted({p.cam for p in group}), "n_obs": int(sum(p.n_obs for p in group))}


def hull_of(xy: np.ndarray) -> np.ndarray | None:
    """Наблюдаемый камерой участок пола: выпуклая оболочка точек ног (без 3% дальних выбросов)."""
    import cv2

    xy = xy[np.isfinite(xy).all(1)]
    if len(xy) < 10:
        return None
    d = np.hypot(*(xy - np.median(xy, 0)).T)
    xy = xy[d <= np.percentile(d, 97)]
    return cv2.convexHull(xy.astype(np.float32)).reshape(-1, 2)


def _inside(poly: np.ndarray, xy: np.ndarray) -> np.ndarray:
    import cv2

    return np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in xy], bool)


def _pos_table(tids: list[int], pos: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Положение трека на сетке времени (k -> медиана x, y по всем камерам)."""
    parts = [pos[t] for t in tids if t in pos]
    if not parts:
        return pd.DataFrame(columns=["x", "y"])
    return pd.concat(parts).groupby(level=0).median()


def group_proto(g: list[Piece], protos: dict[int, tuple[np.ndarray, float]]) -> np.ndarray | None:
    acc = None
    for p in g:
        if p.tid in protos:
            v, w = protos[p.tid]
            acc = v * w if acc is None else acc + v * w
    return None if acc is None else acc / max(np.linalg.norm(acc), 1e-12)


def group_quality(g: list[Piece]) -> dict[str, float | None]:
    """Средние q, occ, conf группы треклетов, взвешенные числом наблюдений."""
    out: dict[str, float | None] = {}
    for name in ("q", "occ", "conf"):
        ok = [(getattr(p, name), p.n_obs) for p in g if getattr(p, name) is not None]
        out[name] = float(sum(v * n for v, n in ok) / max(sum(n for _, n in ok), 1)) if ok else None
    return out


def is_ghost(duration: float, q: dict[str, float | None], cfg: TracksCfg) -> bool:
    """Призрак: короткий трек, у которого сразу и плохое качество, и большое перекрытие, и низкая уверенность.
    Каждый признак по отдельности бывает и у человека (мелкий, в толпе), вместе — у ложной детекции рядом с людьми."""
    if duration > cfg.ghost_max_s or None in (q["q"], q["occ"], q["conf"]):
        return False
    return q["q"] <= cfg.ghost_q_max and q["occ"] >= cfg.ghost_occ_min and q["conf"] <= cfg.ghost_conf_max


def refine(raw: list[tuple[int, list[Piece]]], pos: dict[int, pd.DataFrame], hulls: dict[int, np.ndarray],
           cfg: TracksCfg, step: float, protos: dict[int, tuple[np.ndarray, float]] | None = None,
           relaxed: bool = False) -> tuple[list[tuple[int, list[Piece]]], list[tuple[int, list[Piece]]], dict]:
    """Дубли поглощаются соседним треком; фантомы уходят в шум. -> (треки, шум, счётчики).
    relaxed — ослабленное правило дублей: только после сшивки пропусков, иначе оно забирает обрывок человека
    к его параллельной копии раньше, чем обрывки успеют сшиться."""
    cnt = {"absorbed": 0, "phantom": 0}
    protos = protos or {}
    items = [[pid, list(g), _pos_table([p.tid for p in g], pos)] for pid, g in raw]
    span = lambda it: max(p.t1 for p in it[1]) - min(p.t0 for p in it[1])  # noqa: E731
    # 1. дубли: от коротких к длинным
    order = sorted(range(len(items)), key=lambda i: span(items[i]))
    alive = set(range(len(items)))
    for i in order:
        a = items[i]
        if a[2].empty:
            continue
        best = None
        for j in alive:
            if j == i or span(items[j]) < span(a):
                continue
            if cfg.absorb_same_cam_s > 0 and any(
                    p.cam == q.cam and min(p.t1, q.t1) - max(p.t0, q.t0) > cfg.absorb_same_cam_s
                    for p in a[1] for q in items[j][1]):
                continue         # одна камера видела обоих сразу: это двое идут рядом, а не дубль
            b = items[j][2]
            com = a[2].index.intersection(b.index)
            if len(com) < max(3, cfg.absorb_app_overlap * len(a[2])):
                continue
            d = float(np.median(np.hypot(*(a[2].loc[com].to_numpy() - b.loc[com].to_numpy()).T)))
            ok = d <= cfg.absorb_m and len(com) >= cfg.absorb_overlap * len(a[2])
            # по внешности: трек целиком внутри другого по времени, рядом и выглядит так же
            # (допуск после сшивки — end_gap_s: выход за край другого трека короче порога разрыва ничего не значит)
            tol = cfg.end_gap_s if relaxed else 1.0
            inside = (min(p.t0 for p in a[1]) >= min(p.t0 for p in items[j][1]) - tol
                      and max(p.t1 for p in a[1]) <= max(p.t1 for p in items[j][1]) + tol)
            if not ok and inside and d <= cfg.absorb_app_m:
                pa, pb = group_proto(a[1], protos), group_proto(items[j][1], protos)
                need = cfg.absorb_near_sim if relaxed and d <= cfg.absorb_near_m else cfg.absorb_app_sim
                ok = pa is not None and pb is not None and float(pa @ pb) >= need
            if ok and (best is None or d < best[0]):
                best = (d, j)
        if best:
            b = items[best[1]]
            b[1] += a[1]
            b[2] = _pos_table([p.tid for p in b[1]], pos)
            alive.discard(i)
            cnt["absorbed"] += 1
    # 2. фантомы: одна камера, хотя место видят другие
    good, noise = [], []
    for i in sorted(alive):
        pid, g, pt = items[i]
        cams = {p.cam for p in g}
        phantom = False
        if len(cams) == 1 and span(items[i]) <= cfg.phantom_max_s and len(pt):
            xy = pt[["x", "y"]].to_numpy(float)
            others = [c for c, h in hulls.items() if c not in cams and h is not None
                      and _inside(h, xy).mean() >= cfg.phantom_inside]
            phantom = len(others) >= cfg.phantom_min_other_cams
        if phantom:
            cnt["phantom"] += 1
            noise.append((pid, g))
        else:
            good.append((pid, g))
    return good, noise, cnt


def build(persons: dict[int, list[Piece]], cfg: TracksCfg, rec_t0: float = 0.0, rec_t1: float = np.inf,
          edge_s: float = 1.0, pos: dict[int, pd.DataFrame] | None = None,
          hulls: dict[int, np.ndarray] | None = None,
          fov: dict[int, np.ndarray] | None = None,
          protos: dict[int, tuple[np.ndarray, float]] | None = None) -> tuple[list[EndTrack], list[EndTrack], dict]:
    """-> (треки в счёт, шум, счётчики). Треки нумеруются по времени начала.
    pos: tid -> DataFrame(index=k сетки map_step_s, x, y) — для дублей/фантомов (без него правила не применяются).
    fov: камера -> зона обзора на плане — для правила «пропуск детекции» и подписи промежутков."""
    scene = Scene(fov or {}, pos or {}) if pos is not None else None
    groups = [(pid, g) for pid, pieces in persons.items() for g in split_person(pieces, cfg.end_gap_s)]
    cnt = {"absorbed": 0, "phantom": 0, "short": 0, "ghost": 0}
    noise_g: list = []
    if pos is not None:
        groups, noise_g, c = refine(groups, pos, hulls or {}, cfg, cfg.map_step_s, protos)
        cnt.update(c)
    if scene is not None and fov:
        # пропуск детекции сшивается ПОСЛЕ дублей и фантомов: иначе обрывки-дубли склеиваются в лишний трек
        groups = bridge_groups(groups, scene, cfg)
        groups, cnt["continued"] = continue_groups(groups, scene, protos, cfg)
        # после сшивки трек стал длиннее — дубли, которые шли рядом с обеими его частями, теперь поглощаются
        groups, noise2, c = refine(groups, pos, hulls or {}, cfg, cfg.map_step_s, protos, relaxed=True)
        noise_g += noise2
        cnt["absorbed"] += c["absorbed"]
        cnt["phantom"] += c["phantom"]

    def mk(pid: int, g: list[Piece]) -> EndTrack:
        d = describe(g, cfg, scene)
        gq = group_quality(g)
        return EndTrack(0, pid, d["t0"], d["t1"], sorted(p.tid for p in g), d["cams"], d["segments"],
                        d["gaps"], d["n_obs"], gap_kinds=d["kinds"], bridged=d["bridged"],
                        q=gq["q"], occ=gq["occ"], conf=gq["conf"])

    good, noise = [], [mk(pid, g) for pid, g in noise_g]
    for t in noise:
        t.reject = "фантом: видела одна камера, хотя место в обзоре других"
    for pid, g in groups:
        t = mk(pid, g)
        if t.duration < cfg.min_track_s or t.n_obs < cfg.min_track_obs:
            cnt["short"] += 1
            t.reject = "короткий: меньше порога длительности или наблюдений"
        elif is_ghost(t.duration, group_quality(g), cfg):
            cnt["ghost"] += 1
            t.reject = "призрак: плохое качество и большое перекрытие"
        (noise if t.reject else good).append(t)
    cnt["bridged"] = sum(t.bridged for t in good)
    good.sort(key=lambda t: (t.t0, t.t1, t.person_id))
    for i, t in enumerate(good, start=1):
        t.track_id = i
        t.start_reason = "начало записи" if t.t0 - rec_t0 <= edge_s else "появился в обзоре"
        t.end_reason = "конец записи" if rec_t1 - t.t1 <= edge_s else "вышел из обзора всех камер"
    assign_people(good, {}, cfg)
    return good, noise, cnt


def _overlap(a: EndTrack, b: EndTrack) -> float:
    return min(a.t1, b.t1) - max(a.t0, b.t0)


def people_thresholds(good: list[EndTrack], protos: dict[int, np.ndarray], cfg: TracksCfg) -> tuple[float, float]:
    """Пороги сходства под эту съёмку. Заданные в настройках подобраны на EPFL, где один человек даёт 0.95, а через
    широкий объектив настоящие пары дают 0.65–0.75. Мерка из самой сцены: треки, которые одна камера видела
    одновременно, — разные люди; настоящая пара должна быть заметно похожее них. Выше настроек порог не поднимается."""
    neg = []
    for i, a in enumerate(good):
        for b in good[i + 1:]:
            pa, pb = protos.get(a.track_id), protos.get(b.track_id)
            if pa is not None and pb is not None and _overlap(a, b) > 1.0 and set(a.cams) & set(b.cams):
                neg.append(float(pa @ pb))
    if len(neg) < 2:
        return cfg.person_app_min, cfg.person_app_floor
    gap = cfg.person_app_min - cfg.person_app_floor
    app_min = float(min(cfg.person_app_min, np.median(neg) + cfg.person_app_margin))
    return app_min, app_min - gap


def track_clothes(good: list[EndTrack], colors: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Одежда трека: медиана его треклетов."""
    out = {}
    for t in good:
        vals = [colors[tid] for tid in t.tids if tid in colors]
        if vals:
            out[t.track_id] = np.median(vals, 0)
    return out


def seen_twice(a: EndTrack, b: EndTrack, sim: float | None, pos: dict[int, pd.DataFrame] | None, cfg: TracksCfg) -> bool:
    """Одновременные треки могут быть одним человеком: мелкая фигура в дальнем конце коридора видна двум камерам,
    а план там ошибается на метры, и трекам не хватило близости, чтобы соединиться."""
    if cfg.person_twice_sim <= 0 or not pos or sim is None or sim < cfg.person_twice_sim \
            or _overlap(a, b) > cfg.person_twice_s:
        return False
    j = _pos_table(a.tids, pos).join(_pos_table(b.tids, pos), how="inner", lsuffix="_a", rsuffix="_b")
    return len(j) > 0 and float(np.median(np.hypot(j.x_a - j.x_b, j.y_a - j.y_b))) <= cfg.person_twice_m


def track_low(good: list[EndTrack], root: Path) -> dict[int, np.ndarray]:
    """Нижняя половина тела трека: среднее по его кускам."""
    from .bodyparts import load

    parts = load(root)
    out = {}
    for t in good:
        v = [parts[x]["low"] for x in t.tids if x in parts]
        if v:
            m = np.mean(v, 0)
            out[t.track_id] = m / max(float(np.linalg.norm(m)), 1e-12)
    return out


def assign_people(good: list[EndTrack], protos: dict[int, np.ndarray], cfg: TracksCfg,
                  clothes: dict[int, np.ndarray] | None = None, pos: dict[int, pd.DataFrame] | None = None,
                  low: dict[int, np.ndarray] | None = None) -> int:
    """Собирает треки в людей. Треки одного человека не пересекаются по времени, кроме случая seen_twice.
    1) треки одного person_id склейки (рвались паузой во всех камерах) — один человек;
    2) жадно по убыванию сходства внешности: пара ≥ person_app_min, все пары двух групп ≥ person_app_floor,
       ни одна пара треков не идёт одновременно. -> число людей; заполняет person_no, prev/next."""
    idx = {t.track_id: t for t in good}
    parent = {t.track_id: t.track_id for t in good}
    how = {t.track_id: "" for t in good}
    app_min, app_floor = people_thresholds(good, protos, cfg)
    clothes = clothes or {}

    def same_clothes(a: int, b: int) -> bool:
        """Заметная одежда совпала: на тёмной совпадение ничего не значит."""
        ca, cb = clothes.get(a), clothes.get(b)
        return ca is not None and cb is not None and float(np.abs(ca - cb).max()) <= cfg.clothes_near \
            and float(max(ca.max(), cb.max())) >= cfg.clothes_bright

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def members(r: int) -> list[int]:
        return [k for k in parent if root(k) == r]

    def together(a: int, b: int) -> bool:
        pa, pb = protos.get(a), protos.get(b)
        sim = float(pa @ pb) if pa is not None and pb is not None else None
        return _overlap(idx[a], idx[b]) > cfg.person_overlap_tol_s and not seen_twice(idx[a], idx[b], sim, pos, cfg)

    def far_short(a: int, b: int) -> bool:
        """Короткий трек и другой трек далеко по времени: на лестнице под неоном векторы разных людей почти совпадают,
        поэтому за много минут один вид ещё не значит один человек."""
        x, y = idx[a], idx[b]
        if cfg.person_far_sim <= 0 or max(x.t0, y.t0) - min(x.t1, y.t1) <= cfg.person_far_gap_s \
                or min(x.t1 - x.t0, y.t1 - y.t0) > cfg.person_far_short_s:
            return False
        pa, pb = protos.get(a), protos.get(b)
        return pa is None or pb is None or float(pa @ pb) < cfg.person_far_sim

    def can_join(ra: int, rb: int) -> bool:
        A, B = members(ra), members(rb)
        for a in A:
            for b in B:
                if together(a, b) or clash(clothes.get(a), clothes.get(b), cfg.clothes_veto) or far_short(a, b):
                    return False
                la, lb = (low or {}).get(a), (low or {}).get(b)
                if cfg.part_min > 0 and la is not None and lb is not None and float(la @ lb) < cfg.part_min:
                    return False
                pa, pb = protos.get(a), protos.get(b)
                if pa is not None and pb is not None and float(pa @ pb) < app_floor:
                    return False
        return True

    def attach_small() -> None:
        """Короткий трек — обрывок человека, а не новый человек. Он идёт к группе, на которую похож явно больше других.
        Сходство берётся с группой в целом: вектор обрывка шумный, и попарный запрет порвал бы верное соединение."""
        seen = lambda r: sum(idx[m].t1 - idx[m].t0 for m in members(r))  # noqa: E731
        for t in sorted(good, key=lambda x: x.t1 - x.t0):
            a = root(t.track_id)
            pa = protos.get(t.track_id)
            if pa is None or seen(a) > cfg.person_small_s:
                continue
            scored = []
            for r in {root(k) for k in parent} - {a}:
                mem = members(r)
                if seen(r) <= seen(a) or any(together(x, y) or far_short(x, y) for x in members(a) for y in mem):
                    continue
                vs = [protos[m] for m in mem if m in protos]
                if not vs:
                    continue
                g = np.sum(vs, 0)
                scored.append((float(pa @ (g / max(float(np.linalg.norm(g)), 1e-12))), r))
            scored.sort(reverse=True)
            if scored and scored[0][0] >= cfg.person_small_sim \
                    and (len(scored) < 2 or scored[0][0] - scored[1][0] >= cfg.person_small_margin):
                parent[a] = scored[0][1]
                how[t.track_id] = how[t.track_id] or f"обрывок, внешность {scored[0][0]:.2f}"

    by_p: dict[int, list[int]] = {}
    for t in good:
        by_p.setdefault(t.person_id, []).append(t.track_id)
    for lst in by_p.values():
        for a, b in zip(lst, lst[1:]):
            pa, pb = protos.get(a), protos.get(b)
            if pa is not None and pb is not None and float(pa @ pb) < cfg.person_link_min:
                continue         # склейка могла соединить через случайное звено, а треки разделены паузой
            if can_join(root(a), root(b)):
                parent[root(b)] = root(a)
                how[b] = "склейка (время и положение или внешность)"
    pairs = []
    ids = [t.track_id for t in good if t.track_id in protos]
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            s = float(protos[a] @ protos[b])
            need = app_min - (cfg.clothes_bonus if same_clothes(a, b) else 0.0)
            if s >= need and not together(a, b):
                pairs.append((s, a, b))
    for s, a, b in sorted(pairs, reverse=True):
        ra, rb = root(a), root(b)
        if ra != rb and can_join(ra, rb):
            later = b if idx[b].t0 > idx[a].t0 else a
            parent[rb] = ra
            if not how[later]:
                how[later] = f"внешность {s:.2f}"
    if cfg.person_small_s > 0:
        attach_small()
    groups: dict[int, list[EndTrack]] = {}
    for t in good:
        groups.setdefault(root(t.track_id), []).append(t)
    ordered = sorted(groups.values(), key=lambda g: min(t.t0 for t in g))
    for n, g in enumerate(ordered, start=1):
        g.sort(key=lambda t: t.t0)
        for i, t in enumerate(g):
            t.person_no = n
            t.person_how = how[t.track_id] if i else "первое появление"
            t.prev_track = g[i - 1].track_id if i else None
            t.next_track = g[i + 1].track_id if i + 1 < len(g) else None
    return len(ordered)


def tid_protos(store: Store, trdf: pd.DataFrame, cam_debias: float = 0.0,
               scene_lam: float = 0.0) -> dict[int, tuple[np.ndarray, float]]:
    """tid -> (прототип внешности 'all', L2; вес = n_emb)."""
    from .reid import scene_vectors
    from .storage import load_npz

    out = {}
    for r in trdf.itertuples():
        if not isinstance(r.emb_path, str) or not r.n_emb:
            continue
        p = store.root / r.emb_path
        if not p.exists():
            continue
        v = load_npz(p).get("proto_all")
        if v is not None:
            out[int(r.tid)] = (v.astype(np.float64) / max(np.linalg.norm(v), 1e-12), float(r.n_emb))
    if cam_debias <= 0:
        return out
    clean = scene_vectors(store, trdf, cam_debias, scene_lam)
    return {k: (clean[k], w) for k, (_, w) in out.items()}


def track_protos(good: list[EndTrack], tp: dict[int, tuple[np.ndarray, float]]) -> dict[int, np.ndarray]:
    """Прототип внешности трека: среднее прототипов его треклетов с весом n_emb (L2)."""
    out = {}
    for t in good:
        v = group_proto([Piece(tid, 0, 0, 0, 0) for tid in t.tids], tp)
        if v is not None:
            out[t.track_id] = v
    return out


# ------------------------------------------------------------------ ввод-вывод
def _latest_run(store: Store) -> int:
    with sqlite3.connect(store.db_path) as con:
        r = con.execute("SELECT MAX(run_id) FROM tracklet_person").fetchone()[0]
    if r is None:
        raise ValueError("нет склейки людей: сначала python -m mtmc link")
    return int(r)


def _path_points(obs: pd.DataFrame, tids: list[int], step: float) -> list[tuple[float, float, float]]:
    o = obs[obs["tid"].isin(tids) & obs["xs"].notna()]
    if o.empty:
        return []
    k = np.round(o["t"].to_numpy() / step).astype(np.int64)
    g = pd.DataFrame({"k": k, "x": o["xs"].to_numpy(), "y": o["ys"].to_numpy()}).groupby("k").median()
    # скользящая медиана по 5 точкам — убирает одиночные выбросы проекции на краях кадра
    g = g.rolling(5, center=True, min_periods=1).median()
    return [(round(float(i) * step, 3), round(float(r.x), 3), round(float(r.y), 3)) for i, r in g.iterrows()]


def camera_fovs(cams: pd.DataFrame, fl: pd.DataFrame) -> dict[int, np.ndarray]:
    """Зона обзора каждой камеры с H: рамка кадра на плане; без H — оболочка точек, где камера видела людей."""
    from .geometry import fov_polygon

    out = {}
    for c in cams.itertuples():
        cid = int(c.cam_id)
        pp = json.loads(c.passport_json)
        H = pp.get("H")
        ref = fl.loc[fl["cam"] == cid, ["xs", "ys"]].to_numpy(float) if "cam" in fl else None
        K, dist = (np.asarray(pp["K"], float), np.asarray(pp["dist"], float)) if pp.get("K") and pp.get("dist") else (None, None)
        poly = fov_polygon(np.asarray(H), int(c.width), int(c.height), ref, K=K, dist=dist) \
            if H is not None and c.width else None
        if poly is None and ref is not None:
            poly = hull_of(ref)
        if poly is not None:
            out[cid] = poly
    return out


def load_scene(store: Store, cfg: Config, run_id: int) -> dict:
    """Всё, что нужно build(): люди из склейки, положения, зоны обзора, прототипы внешности."""
    pm = store.person_map(run_id)
    trdf = store.load_tracklets()
    cams = store.load_cameras()
    from .passport import frame_time

    spans = [(frame_time(json.loads(c.passport_json).get("clock"), 0, c.fps, c.offset_s),
              frame_time(json.loads(c.passport_json).get("clock"), c.n_frames, c.fps, c.offset_s))
             for c in cams.itertuples()]
    rec_t0, rec_t1 = min(a for a, _ in spans), max(b for _, b in spans)
    # зоны выхода из паспортов: zones.<имя>.leads_to задан — через неё уходят из обзора
    exits: dict[int, set[str]] = {}
    for c in cams.itertuples():
        zones = (json.loads(c.passport_json).get("zones") or {})
        exits[int(c.cam_id)] = {k for k, z in zones.items() if isinstance(z, dict) and z.get("leads_to")}
    persons: dict[int, list[Piece]] = {}
    for r in trdf.itertuples():
        pid = pm.get(int(r.tid))
        if pid is None:
            continue
        ex = isinstance(r.exit_zone, str) and r.exit_zone in exits.get(int(r.cam_id), set())
        persons.setdefault(pid, []).append(Piece(int(r.tid), int(r.cam_id), float(r.t0), float(r.t1),
                                                 int(r.n_obs), bool(r.is_weak), ex))
    obs_l = []
    for c in cams["cam_id"]:
        o = store.load_observations(int(c), columns=["tid", "t", "xs", "ys", "q", "occ", "conf"])
        o["cam"] = int(c)
        obs_l.append(o)
    obs = pd.concat(obs_l)
    obs = obs[obs["tid"].notna()]
    means = obs.groupby("tid")[["q", "occ", "conf"]].mean()
    for pieces in persons.values():
        for p in pieces:
            if p.tid in means.index:
                p.q, p.occ, p.conf = (float(v) for v in means.loc[p.tid])
    fl = obs[obs["xs"].notna()]
    pos, hulls = None, {}
    if len(fl):
        k = np.round(fl["t"].to_numpy() / cfg.tracks.map_step_s).astype(np.int64)
        tab = pd.DataFrame({"tid": fl["tid"].astype(int).to_numpy(), "k": k, "x": fl["xs"].to_numpy(),
                            "y": fl["ys"].to_numpy()}).groupby(["tid", "k"]).median()
        pos = {int(t): g.droplevel(0) for t, g in tab.groupby(level=0)}
        hulls = {int(c): hull_of(g[["xs", "ys"]].to_numpy(float)) for c, g in fl.groupby("cam")}
    fovs = camera_fovs(cams, fl)
    tp = tid_protos(store, trdf, cfg.tracks.cam_debias, cfg.tracks.scene_lam)
    # фантом проверяется по зонам обзора соседних камер (рамка кадра на плане), без H — по облаку их точек
    seen_by = {c: fovs.get(c, h) for c, h in hulls.items()} if cfg.tracks.phantom_by_fov else dict(hulls)
    return {"persons": persons, "pos": pos, "hulls": hulls, "seen_by": seen_by, "fovs": fovs, "protos": tp,
            "rec": (rec_t0, rec_t1), "obs": obs, "cams": cams}


def build_tracks(data_dir: str | Path, cfg: Config, run_id: int | None = None,
                 with_points: bool = True) -> dict:
    """Строит сквозные треки по склейке run_id (по умолчанию последней), пишет в БД и tracks.csv/json."""
    store = Store(data_dir)
    run_id = run_id or _latest_run(store)
    sc = load_scene(store, cfg, run_id)
    persons, hulls, fovs, tp, obs, cams = (sc[k] for k in ("persons", "hulls", "fovs", "protos", "obs", "cams"))
    rec_t0, rec_t1 = sc["rec"]
    good, noise, cnt = build(persons, cfg.tracks, rec_t0, rec_t1, edge_s=max(cfg.tracks.end_gap_s, 1.0),
                             pos=sc["pos"], hulls=sc["seen_by"], fov=fovs, protos=tp)
    n_people = assign_people(good, track_protos(good, tp), cfg.tracks, track_clothes(good, load_colors(store.root)),
                             sc["pos"], track_low(good, store.root))
    if with_points:
        for t in good:
            t.pts = _path_points(obs, t.tids, cfg.tracks.map_step_s)
    _save_db(store, run_id, good, noise)
    res = {"run_id": run_id, "n_tracks": len(good), "n_noise": len(noise), "n_persons": len(persons),
           "n_people": n_people,
           "counters": cnt, "hulls": {c: (h.tolist() if h is not None else []) for c, h in hulls.items()},
           "fovs": {c: p.tolist() for c, p in fovs.items()},
           "record": [rec_t0, rec_t1], "tracks": good, "noise": noise, "end_gap_s": cfg.tracks.end_gap_s,
           "cam_names": {int(c.cam_id): f"cam{c.cam_id}" for c in cams.itertuples()}}
    export(res, store.root, cfg)
    from .report import write_report

    write_report(store, res, cfg)
    return res


def _save_db(store: Store, run_id: int, good: list[EndTrack], noise: list[EndTrack]) -> None:
    with store.connect() as con:
        con.executescript(TRACKS_SCHEMA)
        con.execute("BEGIN IMMEDIATE")
        con.execute("DELETE FROM tracks WHERE run_id=?", (run_id,))
        con.execute("DELETE FROM tracklet_track WHERE run_id=?", (run_id,))
        con.executemany(
            "INSERT INTO tracks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(run_id, t.track_id, t.person_id, t.t0, t.t1, ",".join(map(str, t.cams)), len(t.tids), t.n_obs,
              t.start_reason, t.end_reason, t.prev_track, t.next_track, json.dumps(t.segments),
              json.dumps(t.gaps)) for t in good])
        con.executemany("INSERT INTO tracklet_track VALUES (?,?,?)",
                        [(run_id, tid, t.track_id) for t in good + noise for tid in t.tids])
        con.execute("DELETE FROM track_person WHERE run_id=?", (run_id,))
        con.executemany("INSERT INTO track_person VALUES (?,?,?,?)",
                        [(run_id, t.track_id, t.person_no, t.person_how) for t in good])
        con.execute("COMMIT")


def fmt_t(t: float) -> str:
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m)}:{s:04.1f}"


def track_row(t: EndTrack, names: dict[int, str]) -> dict:
    per_cam = "; ".join(f"{names.get(c, c)} {fmt_t(a)}–{fmt_t(b)}" for c, a, b in t.segments)
    return {"track_id": t.track_id, "start": fmt_t(t.t0), "end": fmt_t(t.t1), "t0_s": round(t.t0, 2),
            "t1_s": round(t.t1, 2), "duration_s": round(t.duration, 1),
            "cameras": ", ".join(str(names.get(c, c)) for c in t.cams), "n_cameras": len(t.cams),
            "per_camera": per_cam, "start_reason": t.start_reason, "end_reason": t.end_reason,
            "unseen_s": round(sum(b - a for a, b in t.gaps), 1),
            "detection_gaps_s": round(sum(b - a for (a, b), k in zip(t.gaps, t.gap_kinds) if k == "detection"), 1),
            "prev_track": t.prev_track or "",
            "next_track": t.next_track or "", "person": f"P{t.person_no}", "person_how": t.person_how,
            "n_tracklets": len(t.tids)}


def people_rows(tracks: list[EndTrack], names: dict[int, str]) -> list[dict]:
    """По человеку: все его треки — моменты, когда его видела хотя бы одна камера."""
    by: dict[int, list[EndTrack]] = {}
    for t in tracks:
        by.setdefault(t.person_no, []).append(t)
    rows = []
    for p, ts in sorted(by.items()):
        ts.sort(key=lambda t: t.t0)
        seen = sum(t.duration - sum(b - a for a, b in t.gaps) for t in ts)
        cams = sorted({c for t in ts for c in t.cams})
        rows.append({
            "person": f"P{p}", "n_tracks": len(ts), "tracks": ", ".join(f"T{t.track_id}" for t in ts),
            "first_seen": fmt_t(ts[0].t0), "last_seen": fmt_t(ts[-1].t1), "visible_s": round(seen, 1),
            "visible_intervals": "; ".join(f"{fmt_t(t.t0)}–{fmt_t(t.t1)} (T{t.track_id})" for t in ts),
            "absent": "; ".join(f"{fmt_t(a.t1)}–{fmt_t(b.t0)}" for a, b in zip(ts, ts[1:])),
            "cameras": ", ".join(str(names.get(c, c)) for c in cams),
            "linked_by": "; ".join(f"T{t.track_id}: {t.person_how}" for t in ts[1:]),
        })
    return rows


def export(res: dict, out_dir: Path, cfg: Config) -> None:
    names = res["cam_names"]
    rows = [track_row(t, names) for t in res["tracks"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tracks.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["track_id"], delimiter=";")
        w.writeheader()
        w.writerows(rows)
    full = {"run_id": res["run_id"], "n_tracks": res["n_tracks"], "n_noise": res["n_noise"],
            "params": asdict(cfg.tracks), "tracks": [{**track_row(t, names), "segments": t.segments,
                                                      "gaps": [[a, b, k] for (a, b), k in zip(t.gaps, t.gap_kinds)],
                                                      "tids": t.tids} for t in res["tracks"]]}
    (out_dir / "tracks.json").write_text(json.dumps(full, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out_dir / "rejected_tracks.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["начало", "конец", "длит_с", "камеры", "наблюдений", "качество", "перекрытие", "уверенность", "причина",
                    "tid"])
        num = lambda v: "" if v is None else round(v, 3)  # noqa: E731
        for t in sorted(res["noise"], key=lambda t: t.t0):
            w.writerow([fmt_t(t.t0), fmt_t(t.t1), round(t.duration, 1), ", ".join(str(names.get(c, c)) for c in t.cams),
                        t.n_obs, num(t.q), num(t.occ), num(t.conf), t.reject, " ".join(map(str, t.tids))])
    prow = people_rows(res["tracks"], names)
    with open(out_dir / "persons.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prow[0]) if prow else ["person"], delimiter=";")
        w.writeheader()
        w.writerows(prow)
    (out_dir / "persons.json").write_text(json.dumps({"n_people": len(prow), "persons": prow}, ensure_ascii=False,
                                                     indent=1), encoding="utf-8")


def format_tracks(res: dict) -> str:
    names = res["cam_names"]
    lines = [f"Сквозных треков: {res['n_tracks']} (людей: {res.get('n_people', '?')}; "
             f"склейка run_id {res['run_id']})",
             f"  не в счёт: {res['n_noise']} (короткие {res['counters'].get('short', 0)}, фантомы "
             f"{res['counters'].get('phantom', 0)}, призраки {res['counters'].get('ghost', 0)}); "
             f"поглощено дублей: {res['counters'].get('absorbed', 0)}; "
             f"пропусков детекции > {res.get('end_gap_s', 2):g} с внутри зоны камеры, трек сохранён: "
             f"{res['counters'].get('bridged', 0)}",
             f"{'ID':>4}  {'начало':>7}  {'конец':>7}  {'длит, с':>7}  камеры"]
    for t in res["tracks"]:
        extra = f"  (P{t.person_no}, раньше: T{t.prev_track})" if t.prev_track else f"  (P{t.person_no})"
        lines.append(f"T{t.track_id:<3}  {fmt_t(t.t0):>7}  {fmt_t(t.t1):>7}  {t.duration:7.1f}  "
                     f"{', '.join(str(names.get(c, c)) for c in t.cams)}{extra}")
    lines.append(f"\nЛюди (все треки человека = когда его видела хотя бы одна камера): {res.get('n_people', '?')}")
    for r in people_rows(res["tracks"], names):
        lines.append(f"{r['person']:<4} в обзоре {r['visible_s']:6.1f} с  треки: {r['visible_intervals']}"
                     + (f"  [{r['linked_by']}]" if r["linked_by"] else ""))
    return "\n".join(lines)
