"""Часть 2: склейка треклетов всех камер в людей (person_id).

Признаки:
  * одновременные треклеты разных камер — расстояние между ними на плане в общие моменты времени
    (главный признак) и сходство прототипов внешности (вето при явной непохожести);
  * последовательные треклеты одной камеры — пауза, расстояние конец→начало, внешность.
Запреты: одновременные треклеты одной камеры; одновременные треклеты разных камер, стоящие далеко друг от друга.
Объединение — жадное: рёбра от самых надёжных, слияние групп только без нарушения запретов.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .clothes import clash, piece_series, save_colors
from .config import Config, LinkCfg
from .reid import scene_vectors
from .storage import Store, load_npz

log = logging.getLogger(__name__)


@dataclass
class Track:
    """Треклет в виде, удобном для склейки."""
    tid: int
    cam: int
    t0: float
    t1: float
    n_obs: int
    keys: np.ndarray                 # (n,) номера ячеек общей сетки времени, по возрастанию
    xy: np.ndarray | None            # (n, 2) положение на плане, м (None — у камеры нет H)
    sigma: np.ndarray | None         # (n,)
    proto: np.ndarray | None = None  # прототип внешности 'all'
    t: np.ndarray | None = None
    clothes: np.ndarray | None = None  # яркость верха и низа
    low: np.ndarray | None = None      # вектор нижней половины тела


@dataclass
class Edge:
    a: int
    b: int
    kind: str          # 'overlap' | 'seq'
    cost: float        # меньше — надёжнее
    conf: float        # 0..1
    info: dict = field(default_factory=dict)


# ------------------------------------------------------------------ чистые функции
def cos_sim(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def make_track(tid: int, cam: int, g: pd.DataFrame, cfg: LinkCfg, proto: np.ndarray | None) -> Track:
    """g — наблюдения одного треклета (t, xs, ys, sigma)."""
    g = g.sort_values("t")
    t = g["t"].to_numpy(float)
    keys = np.round(t / cfg.grid_s).astype(np.int64)
    keys, first = np.unique(keys, return_index=True)
    xy = sig = None
    if "xs" in g and g["xs"].notna().all() and g["sigma"].notna().all():
        xy = g[["xs", "ys"]].to_numpy(float)[first]
        sig = g["sigma"].to_numpy(float)[first]
    return Track(tid, cam, float(t[0]), float(t[-1]), len(g), keys, xy, sig, proto, t[first])


def overlap_stats(a: Track, b: Track) -> dict | None:
    """Расстояния между треклетами в общие моменты времени (оба с планом)."""
    if a.xy is None or b.xy is None:
        return None
    common, ia, ib = np.intersect1d(a.keys, b.keys, assume_unique=True, return_indices=True)
    if len(common) == 0:
        return {"n": 0}
    vec = b.xy[ib] - a.xy[ia]
    d = np.hypot(*vec.T)
    s = np.hypot(a.sigma[ia], b.sigma[ib])
    mid = np.median(vec, 0)
    return {"n": int(len(common)), "median": float(np.median(d)), "sigma": float(np.median(s)), "d": d,
            "offset": mid, "spread": float(np.median(np.hypot(*(vec - mid).T)))}


def camera_shifts(tracks: list[Track], cfg: LinkCfg) -> tuple[dict[int, np.ndarray], list[dict]]:
    """Систематический сдвиг камер на плане по парам треклетов, в которых уверены без координат.

    Ошибка привязки сдвигает всех людей камеры одинаково, поэтому расстояние в паре постоянно. Берутся
    одновременные пары разных камер: внешность почти одна, смещение ровное. На пару камер нужно несколько таких
    пар, иначе один похоже одетый сосед сдвинул бы всю камеру. Сдвиги согласуются по всем парам камер сразу."""
    if cfg.shift_max_m <= 0:
        return {}, []
    votes: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    tr = [t for t in tracks if t.xy is not None and t.proto is not None]
    for i, a in enumerate(tr):
        for b in tr[i + 1:]:
            if a.cam == b.cam or min(a.t1, b.t1) <= max(a.t0, b.t0):
                continue
            sim = cos_sim(a.proto, b.proto)
            if sim is None or sim < cfg.app_strong:
                continue
            st = overlap_stats(a, b)
            if st is None or st["n"] < cfg.shift_min_samples or st["spread"] > cfg.shift_spread_m \
                    or st["median"] > cfg.shift_max_m:
                continue
            key, v = ((a.cam, b.cam), st["offset"]) if a.cam < b.cam else ((b.cam, a.cam), -st["offset"])
            votes[key].append(v)
    pairs = {k: np.median(np.stack(v), 0) for k, v in votes.items() if len(v) >= cfg.shift_min_pairs}
    if not pairs:
        return {}, []
    cams = sorted({c for k in pairs for c in k})
    idx = {c: i for i, c in enumerate(cams)}
    A = np.zeros((len(pairs) + 1, len(cams)))
    B = np.zeros((len(pairs) + 1, 2))
    for row, ((ca, cb), v) in enumerate(pairs.items()):
        A[row, idx[cb]], A[row, idx[ca]] = 1.0, -1.0
        B[row] = v
    A[-1] = 1.0                     # средний сдвиг ноль: положение всей сцены на плане не трогаем
    s = np.linalg.lstsq(A, B, rcond=None)[0]
    shifts = {c: -s[idx[c]] for c in cams}
    if max(float(np.hypot(*v)) for v in shifts.values()) < cfg.shift_min_m:
        return {}, []                # точная привязка: такой сдвиг неотличим от шума, а склейка на нём настроена
    report = [{"cam": c, "dx": round(float(v[0]), 2), "dy": round(float(v[1]), 2)} for c, v in shifts.items()]
    return shifts, report


def apply_shifts(tracks: list[Track], shifts: dict[int, np.ndarray]) -> list[Track]:
    """Копии треклетов со сдвинутыми координатами: исходные наблюдения не меняются."""
    if not shifts:
        return tracks
    return [replace(t, xy=t.xy + shifts[t.cam]) if t.xy is not None and t.cam in shifts else t for t in tracks]


def _edge_pos(tr: Track, end: bool, window_s: float) -> np.ndarray:
    if end:
        m = tr.t >= tr.t[-1] - window_s
    else:
        m = tr.t <= tr.t[0] + window_s
    return tr.xy[m].mean(0)


def legs_differ(a: Track, b: Track, cfg: LinkCfg) -> bool:
    """Низ тела явно разный: у людей в чёрном это заметнее общего вида."""
    if cfg.part_min <= 0 or a.low is None or b.low is None:
        return False
    return float(a.low @ b.low) < cfg.part_min


def _seq_edge(a: Track, b: Track, cfg: LinkCfg) -> Edge | None:
    """Ребро «один закончился — другой начался рядом» (одна камера или передача между камерами)."""
    if legs_differ(a, b, cfg):
        return None
    first, second = (a, b) if a.t1 <= b.t1 else (b, a)
    gap = max(second.t0 - first.t1, 0.0)
    if gap > cfg.seq_max_gap_s or min(a.n_obs, b.n_obs) < cfg.min_obs:
        return None
    sim = cos_sim(a.proto, b.proto)
    if first.xy is not None and second.xy is not None:
        d = float(np.linalg.norm(_edge_pos(second, False, cfg.seq_edge_window_s)
                                 - _edge_pos(first, True, cfg.seq_edge_window_s)))
        cross = a.cam != b.cam
        margin = cfg.seq_cross_margin_m if cross and cfg.seq_cross_margin_m > 0 else cfg.seq_margin_m
        if d > cfg.seq_v_max_mps * gap + margin:
            return None
        # среднее по окну сближает двух людей, разошедшихся навстречу: рядом по времени сверяются сами концы
        if cfg.seq_strict_gap_s > 0 and not cross and gap <= cfg.seq_strict_gap_s and \
                float(np.linalg.norm(second.xy[0] - first.xy[-1])) > cfg.seq_v_max_mps * gap + cfg.seq_margin_m:
            return None
    else:
        if a.cam != b.cam:
            return None          # без плана передачу между камерами не проверить
        d = None
    if sim is not None:
        if sim < cfg.seq_app_min or (gap > cfg.seq_strict_gap_s and sim < cfg.seq_gap_app_min):
            return None
    elif gap > cfg.seq_noapp_gap_s or (cfg.seq_cross_need_app and a.cam != b.cam):
        return None             # обрывок без вектора не должен быть мостом между людьми на разных камерах
    # стоимость: пауза + расстояние + непохожесть; всегда хуже одновременных совпадений
    cost = 1.0 + gap / cfg.seq_max_gap_s + cfg.xy_weight * (d or 0.0) + (1 - sim if sim is not None else 0.5)
    conf = float(np.clip((sim if sim is not None else 0.6) * (1 - gap / cfg.seq_max_gap_s), 0, 1))
    return Edge(first.tid, second.tid, "seq", cost, conf,
                {"gap": round(gap, 2), "d": None if d is None else round(d, 2),
                 "sim": None if sim is None else round(sim, 3), "cross": a.cam != b.cam})


def candidate_edges(tracks: list[Track], cfg: LinkCfg) -> tuple[list[Edge], dict[int, set[int]], dict]:
    """Рёбра-кандидаты и запреты. Возвращает (рёбра, запреты tid->set, счётчики)."""
    cannot: dict[int, set[int]] = defaultdict(set)
    edges: list[Edge] = []
    cnt = defaultdict(int)
    tr = sorted(tracks, key=lambda x: (x.t0, x.tid))
    n = len(tr)
    for i in range(n):
        a = tr[i]
        for j in range(i + 1, n):
            b = tr[j]
            if b.t0 > a.t1 + cfg.seq_max_gap_s:
                break
            if clash(a.clothes, b.clothes, cfg.clothes_veto):
                cannot[a.tid].add(b.tid); cannot[b.tid].add(a.tid)
                cnt["cannot_clothes"] += 1
                continue
            ov = min(a.t1, b.t1) - max(a.t0, b.t0)   # > 0 — пересекаются по времени
            st = overlap_stats(a, b) if ov > 0 else None
            if a.cam == b.cam and ov > cfg.same_cam_overlap_s:
                # одна камера одновременно: разные люди, если только это не дубль в той же точке
                if st is not None and st["n"] >= cfg.min_overlap_samples and st["median"] <= cfg.dup_max_m:
                    cnt["duplicates"] += 1
                else:
                    cannot[a.tid].add(b.tid); cannot[b.tid].add(a.tid)
                    cnt["cannot_same_cam"] += 1
                    continue
            elif st is None or st["n"] < cfg.min_overlap_samples:
                e = _seq_edge(a, b, cfg)
                if e is not None:
                    edges.append(e)
                    cnt["edges_seq_cross" if e.info["cross"] else "edges_seq"] += 1
                continue
            med = st["median"]
            # короткое перекрытие на краях обзоров: это передача между камерами, а план на краю кадра точен хуже
            if a.cam != b.cam and st["n"] < cfg.seq_overlap_samples and med > cfg.d_same_m + cfg.sigma_k * st["sigma"]:
                e = _seq_edge(a, b, cfg)
                if e is not None:
                    edges.append(e)
                    cnt["edges_seq_cross"] += 1
                    continue
            if med > cfg.d_diff_m:
                if st["n"] >= cfg.cannot_far_min_samples:
                    cannot[a.tid].add(b.tid); cannot[b.tid].add(a.tid)
                    cnt["cannot_far"] += 1
                continue
            if min(a.n_obs, b.n_obs) < cfg.min_obs or legs_differ(a, b, cfg):
                continue
            thr = cfg.d_same_m + cfg.sigma_k * st["sigma"]
            near = float(np.mean(st["d"] < cfg.near_m))
            if med > thr or near < cfg.near_frac_min:
                continue
            sim = cos_sim(a.proto, b.proto)
            if sim is not None and sim < cfg.app_reject:
                cnt["vetoed_appearance"] += 1
                continue
            cost = cfg.xy_weight * med / thr - 0.1 * min(st["n"], 50) / 50 \
                - (cfg.overlap_app_w * (sim - cfg.app_reject) if sim is not None else 0)
            conf = float(np.clip((1 - med / thr) * 0.7 + (0.3 * sim if sim is not None else 0.15), 0, 1))
            edges.append(Edge(a.tid, b.tid, "overlap", cost, conf,
                              {"n": st["n"], "median": round(med, 3), "near": round(near, 2),
                               "sim": None if sim is None else round(sim, 3)}))
            cnt["edges_overlap"] += 1
    return drop_skipping(edges, tracks, cfg), cannot, dict(cnt)


def drop_skipping(edges: list[Edge], tracks: list[Track], cfg: LinkCfg) -> list[Edge]:
    """Ребро через паузу не нужно, если за концом куска на той же камере сразу идёт другой: продолжение уже есть, а
    дальний кусок — следующий человек."""
    if not cfg.seq_no_skip:
        return edges
    by = {t.tid: t for t in tracks}

    def has_next(a: Track, skip: int) -> bool:
        for t in tracks:
            gap = t.t0 - a.t1
            if t.tid != skip and t.cam == a.cam and 0 <= gap <= cfg.seq_strict_gap_s and a.xy is not None \
                    and t.xy is not None \
                    and float(np.linalg.norm(t.xy[0] - a.xy[-1])) <= cfg.seq_v_max_mps * gap + cfg.seq_margin_m:
                return True
        return False

    out = []
    for e in edges:
        if e.kind == "seq" and not e.info.get("cross") and e.info.get("gap", 0) > cfg.seq_strict_gap_s \
                and has_next(by[e.a], e.b):
            continue
        out.append(e)
    return out


def cluster(tids: list[int], edges: list[Edge], cannot: dict[int, set[int]]) -> tuple[dict[int, int], dict]:
    """Жадное объединение с запретами. -> (tid -> номер группы, счётчики)."""
    parent = {t: t for t in tids}
    members = {t: {t} for t in tids}
    banned = {t: set(cannot.get(t, ())) for t in tids}   # запреты на уровне группы
    conf_sum = defaultdict(float)
    n_used = defaultdict(int)
    cnt = {"merged": 0, "rejected_conflict": 0}

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in sorted(edges, key=lambda e: (e.kind != "overlap", e.cost, e.a, e.b)):
        if e.a not in parent or e.b not in parent:
            continue
        ra, rb = root(e.a), root(e.b)
        if ra == rb:
            continue
        if banned[ra] & members[rb]:
            cnt["rejected_conflict"] += 1
            continue
        if len(members[ra]) < len(members[rb]):
            ra, rb = rb, ra
        parent[rb] = ra
        members[ra] |= members.pop(rb)
        banned[ra] |= banned.pop(rb)
        conf_sum[ra] += conf_sum.pop(rb, 0.0) + e.conf
        n_used[ra] += n_used.pop(rb, 0) + 1
        cnt["merged"] += 1
    labels = {t: root(t) for t in tids}
    cnt["conf"] = {r: conf_sum[r] / n_used[r] for r in n_used}
    return labels, cnt


def group_tracks(labels: dict[int, int], tracks: list[Track]) -> list[Track]:
    """Группа треклетов -> один «треклет» с общей траекторией (среднее по камерам в каждый момент)."""
    by: dict[int, list[Track]] = defaultdict(list)
    for t in tracks:
        by[labels[t.tid]].append(t)
    out = []
    for k, (root, mem) in enumerate(sorted(by.items())):
        with_xy = [m for m in mem if m.xy is not None]
        keys = np.unique(np.concatenate([m.keys for m in mem]))
        xy = sig = None
        tt = np.concatenate([m.t for m in mem])
        kk = np.concatenate([m.keys for m in mem])
        t_of = dict(zip(kk.tolist(), tt.tolist()))
        if with_xy:
            acc = defaultdict(list)
            for m in with_xy:
                for key, p, s in zip(m.keys.tolist(), m.xy, m.sigma):
                    acc[key].append((p, s))
            keys = np.array(sorted(acc), dtype=np.int64)
            xy = np.array([np.mean([p for p, _ in acc[key]], 0) for key in keys])
            sig = np.array([np.mean([s for _, s in acc[key]]) for key in keys])
        protos = [(m.proto, m.n_obs) for m in mem if m.proto is not None]
        proto = None
        if protos:
            v = sum(p * w for p, w in protos)
            proto = v / max(np.linalg.norm(v), 1e-12)
        t = np.array([t_of[x] for x in keys.tolist()])
        colors = [m.clothes for m in mem if m.clothes is not None]
        out.append(Track(root, -(k + 1), float(t.min()), float(t.max()), sum(m.n_obs for m in mem), keys, xy, sig,
                         proto, t, np.median(colors, 0) if colors else None))
    return out


def group_handoff_edges(groups: list[Track], cfg: LinkCfg) -> list[Edge]:
    """Рёбра передачи между крупными группами: конец одной и начало другой рядом по времени и месту,
    средняя внешность групп почти одинакова (у группы прототип усреднён по многим треклетам и надёжен)."""
    out = []
    big = [g for g in groups if g.n_obs >= cfg.group_handoff_min_obs and g.xy is not None and g.proto is not None]
    for a in big:
        for b in big:
            if a is b or b.t0 < a.t0:
                continue
            ov = a.t1 - b.t0                       # > 0 — короткое пересечение, < 0 — пауза
            if ov > cfg.group_handoff_max_overlap_s or -ov > cfg.seq_max_gap_s or b.t1 <= a.t1:
                continue
            sim = cos_sim(a.proto, b.proto)
            if sim is None or sim < cfg.group_handoff_app or clash(a.clothes, b.clothes, cfg.clothes_veto):
                continue
            gap = max(-ov, 0.0)
            d = float(np.linalg.norm(_edge_pos(b, False, cfg.seq_edge_window_s * 2)
                                     - _edge_pos(a, True, cfg.seq_edge_window_s * 2)))
            if d > cfg.group_handoff_m + cfg.seq_v_max_mps * gap:
                continue
            out.append(Edge(a.tid, b.tid, "overlap", 2.0 + d - sim, float(sim * 0.8),
                            {"handoff": True, "d": round(d, 2), "sim": round(sim, 3), "gap": round(gap, 2)}))
    return out


def group_soft_pairs(groups: list[Track], n_conflicts: dict[tuple[int, int], int],
                     cfg: LinkCfg) -> list[Edge]:
    """Пары групп, которые долго идут рядом и одинаково выглядят: редкие запреты между ними — шум."""
    out = []
    gs = sorted((g for g in groups if g.xy is not None and g.proto is not None), key=lambda g: g.t0)
    for i, a in enumerate(gs):
        for b in gs[i + 1:]:
            if b.t0 > a.t1:
                break
            st = overlap_stats(a, b)
            if st is None or st["n"] < cfg.group_soft_min_samples or st["median"] > cfg.group_soft_m:
                continue
            near = float(np.mean(st["d"] < cfg.group_soft_m))
            sim = cos_sim(a.proto, b.proto)
            if near < cfg.group_soft_near or sim is None or sim < cfg.group_soft_app \
                    or clash(a.clothes, b.clothes, cfg.clothes_veto):
                continue
            nc = n_conflicts.get((min(a.tid, b.tid), max(a.tid, b.tid)), 0)
            if nc > cfg.group_soft_conflict_frac * st["n"]:
                continue
            out.append(Edge(a.tid, b.tid, "overlap", st["median"] - sim, float(sim * near),
                            {"soft": True, "n": st["n"], "median": round(st["median"], 3), "near": round(near, 2),
                             "sim": round(sim, 3), "conflicts": nc}))
    return out


def seam_differs(e: Edge, labels: dict[int, int], tracks: list[Track], cfg: LinkCfg) -> bool:
    """Группа людей в тёмном в среднем похожа на другую такую же, поэтому продолжение сверяется по месту стыка:
    последний кусок одной группы и первый кусок другой."""
    if cfg.group_seam_app_min <= 0 or (e.kind != "seq" and not e.info.get("handoff")):
        return False
    a = [t for t in tracks if labels[t.tid] == e.a and t.proto is not None]
    b = [t for t in tracks if labels[t.tid] == e.b and t.proto is not None]
    if not a or not b:
        return False
    first, second = (a, b) if max(t.t1 for t in a) <= max(t.t1 for t in b) else (b, a)
    end, start = max(first, key=lambda t: t.t1), min(second, key=lambda t: t.t0)
    sim = cos_sim(end.proto, start.proto)
    return (sim is not None and sim < cfg.group_seam_app_min) or legs_differ(end, start, cfg)


def link_groups(labels: dict[int, int], tracks: list[Track], cannot: dict[int, set[int]],
                cfg: LinkCfg) -> tuple[dict[int, int], dict]:
    """Проходы склейки групп по общей траектории; запреты членов переносятся на группы."""
    cnt = defaultdict(int)
    for _ in range(max(cfg.group_passes, 0)):
        g = group_tracks(labels, tracks)
        g_cannot = defaultdict(set)
        n_conf: dict[tuple[int, int], int] = defaultdict(int)
        for t, bans in cannot.items():
            for b in bans:
                ra, rb = labels[t], labels[b]
                if ra != rb:
                    g_cannot[ra].add(rb)
                    if t < b:
                        n_conf[(min(ra, rb), max(ra, rb))] += 1
        soft = group_soft_pairs(g, n_conf, cfg)
        for e in soft:                       # снимаем шумовые запреты между такими группами
            g_cannot[e.a].discard(e.b)
            g_cannot[e.b].discard(e.a)
        cnt["group_soft_edges"] += len(soft)
        edges, c_far, c = candidate_edges(g, cfg)
        edges += soft
        hand = group_handoff_edges(g, cfg)
        edges += hand
        cnt["group_handoff_edges"] += len(hand)
        edges = [e for e in edges if not seam_differs(e, labels, tracks, cfg)]
        for a, bs in c_far.items():
            g_cannot[a] |= bs
        roots = [x.tid for x in g]
        glab, c2 = cluster(roots, edges, g_cannot)
        c2.pop("conf", None)
        cnt["group_merged"] += c2["merged"]
        cnt["group_edges"] += len(edges)
        if c2["merged"] == 0:
            break
        labels = {t: glab[r] for t, r in labels.items()}
    return labels, dict(cnt)


def same_camera_clash(a: list[Track], b: list[Track], cfg: LinkCfg) -> bool:
    """Одна камера видела оба куска одновременно: это точно разные люди, как бы похожи они ни были."""
    return any(x.cam == y.cam and min(x.t1, y.t1) - max(x.t0, y.t0) > cfg.same_cam_overlap_s for x in a for y in b)


def sure_pair(a: Track, b: Track, ma: list[Track], mb: list[Track], sim: float, cfg: LinkCfg) -> bool:
    """Внешность уверенная: координаты дальше проверяются мягко. Вектор короткого куска шумный, поэтому нужна длина,
    а чтобы похожие люди не собрались цепочкой, ни одна пара кусков двух групп не должна быть явно непохожа."""
    if sim < cfg.app_strong or min(a.n_obs, b.n_obs) < cfg.app_sure_min_obs:
        return False
    if any(clash(x.clothes, y.clothes, cfg.clothes_veto) for x in ma for y in mb):
        return False
    sims = (cos_sim(x.proto, y.proto) for x in ma for y in mb)
    return all(s is None or s >= cfg.app_handoff_min for s in sims)


def rescue_groups(labels: dict[int, int], tracks: list[Track], cfg: LinkCfg,
                  thr: float | None = None) -> tuple[dict[int, int], int]:
    """Спасение после точных проходов: координаты разошлись из-за неточной привязки, а человек один.

    Точные проходы настроены на хорошую привязку и не меняются. Здесь рассматривается только то, что осталось
    разорванным: группы разных камер с почти одинаковой внешностью и ровным смещением на плане. Ошибка привязки
    постоянна, а два похоже одетых соседа ходят каждый по-своему. Запрет «далеко» здесь не действует, потому что он
    сам получен из координат; запрет «одна камера видела обоих» действует всегда."""
    if cfg.rescue_max_m <= 0:
        return labels, 0
    merged = 0
    for _ in range(max(cfg.group_passes, 1)):
        groups = [g for g in group_tracks(labels, tracks) if g.xy is not None and g.proto is not None]
        members: dict[int, list[Track]] = defaultdict(list)
        for t in tracks:
            members[labels[t.tid]].append(t)
        options = []
        for i, a in enumerate(groups):
            for b in groups[i + 1:]:
                sim = cos_sim(a.proto, b.proto)
                if sim is None or sim < (cfg.app_strong if thr is None else thr) \
                        or same_camera_clash(members[a.tid], members[b.tid], cfg) \
                        or clash(a.clothes, b.clothes, cfg.clothes_veto):
                    continue
                sure = sure_pair(a, b, members[a.tid], members[b.tid], sim, cfg)
                st = overlap_stats(a, b)
                if st is not None and st["n"] >= cfg.min_overlap_samples:
                    if sure:
                        if st["median"] > cfg.sure_max_m:
                            continue
                    elif st["median"] > cfg.rescue_max_m or st["spread"] > cfg.shift_spread_m:
                        continue
                    d = st["median"]
                else:
                    first, second = (a, b) if a.t1 <= b.t1 else (b, a)
                    gap = max(second.t0 - first.t1, 0.0)
                    if gap > (cfg.handoff_max_s if sure else cfg.seq_max_gap_s):
                        continue
                    d = float(np.linalg.norm(_edge_pos(second, False, cfg.seq_edge_window_s)
                                             - _edge_pos(first, True, cfg.seq_edge_window_s)))
                    spare = cfg.sure_max_m if sure else cfg.seq_margin_m + cfg.app_tol_m
                    if d > cfg.seq_v_max_mps * gap + spare:
                        continue
                options.append((cfg.xy_weight * d - sim, a.tid, b.tid))
        done = 0
        used: set[int] = set()
        for _, ra, rb in sorted(options):
            if ra in used or rb in used:      # группа изменилась: на следующем проходе всё считается заново
                continue
            used |= {ra, rb}
            labels = {t: (ra if r == rb else r) for t, r in labels.items()}
            done += 1
        merged += done
        if not done:
            break
    return labels, merged


def purge_groups(labels: dict[int, int], tracks: list[Track], cfg: LinkCfg) -> tuple[dict[int, int], int]:
    """Разбирает смешанные группы. Люди, идущие вместе, склеиваются по близости координат, поэтому внутри группы куски
    заново раскладываются по внешности, от длинных к коротким. Отделившиеся части потом могут соединиться с другими."""
    if cfg.purge_min <= 0:
        return labels, 0
    members: dict[int, list[Track]] = defaultdict(list)
    for t in tracks:
        members[labels[t.tid]].append(t)
    labels, split = dict(labels), 0
    for mem in members.values():
        seen = sorted((t for t in mem if t.proto is not None), key=lambda t: -t.n_obs)
        parts: list[list[Track]] = []
        for t in seen:
            sims = [cos_sim(sum(x.proto * x.n_obs for x in p), t.proto) for p in parts]
            if sims and max(sims) >= cfg.purge_min:
                parts[int(np.argmax(sims))].append(t)
            else:
                parts.append([t])
        for p in parts[1:]:
            root = p[0].tid
            for t in p:
                labels[t.tid] = root
            split += 1
    return labels, split


def coord_terms(use: list[Track], cfg: LinkCfg) -> tuple[np.ndarray, np.ndarray]:
    """Вклад координат в сходство пары кусков и запрет. Близко вместе — плюс; расходятся — минус; разошлись на всю
    дистанцию coord_far_m — запрет: один человек не бывает в двух дальних местах сразу. Вплотную по времени и месту
    идущие друг за другом куски тоже получают плюс."""
    n = len(use)
    add, ban = np.zeros((n, n)), np.zeros((n, n), bool)
    if cfg.coord_far_m <= 0:
        return add, ban
    for i, a in enumerate(use):
        if a.xy is None:
            continue
        for j in range(i + 1, n):
            b = use[j]
            if b.xy is None:
                continue
            first, second = (a, b) if a.t1 <= b.t1 else (b, a)
            gap = second.t0 - first.t1
            st = overlap_stats(a, b) if a.cam != b.cam and gap < 0 else None
            val = 0.0
            if st is not None and st["n"] >= cfg.min_overlap_samples:
                d = st["median"]
                if d >= cfg.coord_far_m:
                    ban[i, j] = ban[j, i] = True
                    continue
                val = cfg.coord_near if d <= cfg.d_same_m + cfg.sigma_k * st["sigma"] else \
                    -cfg.coord_penalty * d / cfg.coord_far_m
            elif 0 <= gap <= cfg.seq_max_gap_s:
                d = float(np.linalg.norm(_edge_pos(second, False, cfg.seq_edge_window_s)
                                         - _edge_pos(first, True, cfg.seq_edge_window_s)))
                if d <= cfg.seq_v_max_mps * gap + cfg.seq_margin_m:
                    val = cfg.coord_seq
            add[i, j] = add[j, i] = val
    return add, ban


def appearance_clusters(tracks: list[Track], dups: list[tuple[int, int]], cfg: LinkCfg) -> tuple[dict[int, int], int]:
    """Люди по внешности: иерархическая кластеризация с запретом «одна камера видела обоих». Координаты в офисе с
    широкими объективами чаще мешают, чем помогают: на разметке этот способ даёт F 0.78 против 0.54 у склейки по
    траекториям. Куски без вектора остаются сами по себе, их присоединяет attach_rest."""
    labels = {t.tid: t.tid for t in tracks}
    use = [t for t in tracks if t.proto is not None and t.n_obs >= cfg.ahc_min_obs]
    n = len(use)
    if n < 2:
        return labels, 0
    P = np.stack([t.proto for t in use])
    add, far = coord_terms(use, cfg)
    S = P @ P.T + add
    t0, t1 = np.array([t.t0 for t in use]), np.array([t.t1 for t in use])
    cam = np.array([t.cam for t in use])
    shared = np.minimum(t1[:, None], t1[None]) - np.maximum(t0[:, None], t0[None])    # < 0 — пауза между кусками
    ban = far | ((cam[:, None] == cam[None]) & (shared > cfg.same_cam_overlap_s))
    for a, b in cfg.cam_apart:             # камеры в разных концах: человек не бывает на обеих почти одновременно
        pair = ((cam[:, None] == a) & (cam[None] == b)) | ((cam[:, None] == b) & (cam[None] == a))
        ban |= pair & (shared > -cfg.cam_apart_s)
    pos = {t.tid: i for i, t in enumerate(use)}
    for a, b in dups:
        if a in pos and b in pos:
            ban[pos[a], pos[b]] = ban[pos[b], pos[a]] = False
    np.fill_diagonal(ban, True)
    size = np.ones(n)
    obs = np.array([float(t.n_obs) for t in use])
    alive = np.ones(n, bool)
    root = list(range(n))
    merged = 0
    small_pass = False
    while True:
        M = np.where(ban | ~alive[:, None] | ~alive[None], -np.inf, S)
        if small_pass:                     # обрывок идёт к большой группе при более мягком пороге
            M = np.where(np.minimum(obs[:, None], obs[None]) < cfg.ahc_small_obs, M, -np.inf)
        x, y = np.unravel_index(int(np.argmax(M)), M.shape)
        if M[x, y] <= (cfg.ahc_small_thr if small_pass else cfg.ahc_thr):
            if small_pass or cfg.ahc_small_obs <= 0:
                break
            small_pass = True
            continue
        S[x] = S[:, x] = (size[x] * S[x] + size[y] * S[y]) / (size[x] + size[y])
        ban[x] = ban[:, x] = ban[x] | ban[y]
        size[x] += size[y]
        obs[x] += obs[y]
        alive[y] = False
        root = [x if r == y else r for r in root]
        merged += 1
    out = {**labels, **{t.tid: use[root[i]].tid for i, t in enumerate(use)}}
    if cfg.ahc_short_thr > 0:                # короткие куски не строят группы, а только находят свою
        groups: dict[int, list[Track]] = defaultdict(list)
        for i, t in enumerate(use):
            groups[root[i]].append(t)
        for t in sorted((t for t in tracks if t.proto is not None and t.n_obs < cfg.ahc_min_obs), key=lambda t: -t.n_obs):
            best, best_g = cfg.ahc_short_thr, None
            for r, mem in groups.items():
                sim = cos_sim(sum(m.proto for m in mem), t.proto)
                if sim > best and not same_camera_clash([t], mem, cfg):
                    best, best_g = sim, r
            if best_g is not None:
                groups[best_g].append(t)
                out[t.tid] = use[best_g].tid
                merged += 1
    return out, merged


def attach_rest(labels: dict[int, int], tracks: list[Track], edges: list[Edge], cannot: dict[int, set[int]]
                ) -> tuple[dict[int, int], int]:
    """Куски без вектора идут к человеку по времени и месту: продолжение на той же камере или рядом на другой."""
    sizes = Counter(labels.values())
    free = {t.tid for t in tracks if sizes[labels[t.tid]] == 1}
    members: dict[int, set[int]] = defaultdict(set)
    for t, r in labels.items():
        members[r].add(t)
    attached = 0
    for e in sorted(edges, key=lambda e: e.cost):
        for loose, fixed in ((e.a, e.b), (e.b, e.a)):
            if loose in free and fixed not in free and not (cannot.get(loose, set()) & members[labels[fixed]]):
                members[labels[fixed]].add(loose)
                labels[loose] = labels[fixed]
                free.discard(loose)
                attached += 1
                break
    return labels, attached


def duplicate_pairs(obs: pd.DataFrame, cfg: LinkCfg) -> list[tuple[int, int]]:
    """Двойники одной камеры: два треклета на одного человека. Рамка одного почти вся лежит в рамке другого на большей
    части общих кадров. Это видно и без привязки к плану, поэтому смотрим рамки, а не координаты."""
    o = obs[obs["tid"].notna()][["tid", "frame_idx", "x1", "y1", "x2", "y2"]]
    m = o.merge(o, on="frame_idx", suffixes=("_a", "_b"))
    m = m[m["tid_a"] < m["tid_b"]]
    if m.empty:
        return []
    iw = (m[["x2_a", "x2_b"]].min(axis=1) - m[["x1_a", "x1_b"]].max(axis=1)).clip(lower=0)
    ih = (m[["y2_a", "y2_b"]].min(axis=1) - m[["y1_a", "y1_b"]].max(axis=1)).clip(lower=0)
    area_a = (m["x2_a"] - m["x1_a"]) * (m["y2_a"] - m["y1_a"])
    area_b = (m["x2_b"] - m["x1_b"]) * (m["y2_b"] - m["y1_b"])
    m = m.assign(cover=(iw * ih) / pd.concat([area_a, area_b], axis=1).min(axis=1).clip(lower=1.0) >= cfg.dup_cover)
    g = m.groupby(["tid_a", "tid_b"])["cover"].agg(["size", "mean"])
    g = g[(g["size"] >= cfg.dup_min_frames) & (g["mean"] >= cfg.dup_share)]
    return [(int(a), int(b)) for a, b in g.index]


def merge_duplicates(pairs: list[tuple[int, int]], tracks: list[Track], edges: list[Edge], cannot: dict[int, set[int]],
                     cfg: LinkCfg) -> int:
    """Снимает запрет «одна камера видела обоих» с двойников и связывает их первыми. Явно разная внешность — двое."""
    by = {t.tid: t for t in tracks}
    n = 0
    for a, b in pairs:
        if a not in by or b not in by:
            continue
        sim = cos_sim(by[a].proto, by[b].proto)
        if sim is not None and sim < cfg.app_handoff_min:
            continue
        cannot[a].discard(b)
        cannot[b].discard(a)
        edges.append(Edge(a, b, "overlap", -1.0, 1.0, {"duplicate": True, "sim": sim}))
        n += 1
    return n


def scene_app_threshold(tracks: list[Track], cfg: LinkCfg) -> float:
    """Порог сходства для этой съёмки. Сходство зависит от камер и света: на EPFL один человек даёт 0.95, через
    широкий объектив 0.65–0.75. Мерка берётся из самой сцены: кого одна камера видела одновременно, те точно
    разные, и настоящая пара должна быть заметно похожее них."""
    tr = [t for t in tracks if t.proto is not None]
    neg = []
    for i, a in enumerate(tr):
        for b in tr[i + 1:]:
            if a.cam == b.cam and min(a.t1, b.t1) - max(a.t0, b.t0) > 1.0:
                neg.append(cos_sim(a.proto, b.proto))
    if len(neg) < 2:
        return cfg.app_strong
    # Медиана, а не верхний край: на длинной записи всегда есть похоже одетые люди, и край съедает весь диапазон
    return float(np.clip(np.median(neg) + cfg.app_margin, cfg.app_handoff_min, cfg.app_strong))


def appearance_handoffs(labels: dict[int, int], tracks: list[Track], cfg: LinkCfg) -> tuple[dict[int, int], int, float]:
    """Человек перешёл с камеры на камеру, а координаты не помогают: у камеры нет привязки или она неточна.

    Похожесть здесь недостаточна сама по себе: разные люди дают почти те же числа. Поэтому нужны сразу три вещи:
    кусок начался вскоре после конца другого на другой камере, сходство выше порога сцены и оба друг для друга
    лучшие без близкого соперника. Два куска, идущие одновременно, соединяются только при большем сходстве и если
    это почти целиком один и тот же отрезок времени: так выглядят камеры с общей зоной без привязки."""
    thr = scene_app_threshold(tracks, cfg)
    if cfg.handoff_max_s <= 0:
        return labels, 0, thr
    merged = 0
    for _ in range(max(cfg.group_passes, 1)):
        groups = [g for g in group_tracks(labels, tracks) if g.proto is not None]
        members: dict[int, list[Track]] = defaultdict(list)
        for t in tracks:
            members[labels[t.tid]].append(t)
        options: list[tuple[float, int, int]] = []
        for i, a in enumerate(groups):
            for b in groups[i + 1:]:
                first, second = (a, b) if a.t1 <= b.t1 else (b, a)
                sim = cos_sim(a.proto, b.proto)
                if same_camera_clash(members[a.tid], members[b.tid], cfg) \
                        or clash(a.clothes, b.clothes, cfg.clothes_veto):
                    continue
                gap = second.t0 - first.t1
                both_xy = a.xy is not None and b.xy is not None
                if gap < -cfg.handoff_overlap_s:
                    shared = min(a.t1, b.t1) - max(a.t0, b.t0)
                    short = max(min(a.t1 - a.t0, b.t1 - b.t0), 1e-6)
                    if both_xy or shared / short < 0.5 or sim < min(thr + 0.1, cfg.app_strong):
                        continue
                else:
                    if gap > cfg.handoff_max_s or sim < thr:
                        continue
                    last = max(members[first.tid], key=lambda m: m.t1).cam
                    next_cam = min(members[second.tid], key=lambda m: m.t0).cam
                    if last == next_cam:
                        continue
                    if both_xy:
                        d = float(np.linalg.norm(_edge_pos(second, False, cfg.seq_edge_window_s)
                                                 - _edge_pos(first, True, cfg.seq_edge_window_s)))
                        spare = cfg.sure_max_m if sure_pair(a, b, members[a.tid], members[b.tid], sim, cfg) \
                            else cfg.rescue_max_m
                        if d > cfg.seq_v_max_mps * max(gap, 0.0) + spare:
                            continue
                options.append((sim, a.tid, b.tid))
        done = 0
        used: set[int] = set()
        for sim, ra, rb in sorted(options, reverse=True):
            rivals = [s for s, x, y in options if (x, y) != (ra, rb) and {x, y} & {ra, rb}]
            if ra in used or rb in used or (rivals and max(rivals) > sim - cfg.app_ambiguity):
                continue
            used |= {ra, rb}
            labels = {t: (ra if r == rb else r) for t, r in labels.items()}
            done += 1
        merged += done
        if not done:
            break
    return labels, merged, thr


# ------------------------------------------------------------------ ввод-вывод
def _load_tracks(store: Store, cfg: LinkCfg, pad: float = 0.05) -> tuple[list[Track], pd.DataFrame]:
    from .silhouette import load_silhouettes

    cams = store.load_cameras()
    trdf = store.load_tracklets()
    tracks: list[Track] = []
    for cid in cams["cam_id"]:
        obs = store.load_observations(int(cid), columns=["tid", "t", "xs", "ys", "sigma", "is_outlier", "crop_path",
                                                         "frame_idx", "local_id", "x1", "y1", "x2", "y2"])
        obs = obs[obs["tid"].notna() & ~obs["is_outlier"]]
        groups = {int(k): g for k, g in obs.groupby("tid")}
        sil = load_silhouettes(store, int(cid))
        for r in trdf[trdf["cam_id"] == cid].itertuples():
            g = groups.get(int(r.tid))
            if g is None or g.empty:
                continue
            proto = None
            if r.emb_path:
                p = store.root / r.emb_path
                if p.exists():
                    z = load_npz(p)
                    proto = z.get("proto_all")
                    proto = None if proto is None else proto.astype(np.float64)
            track = make_track(int(r.tid), int(cid), g, cfg, proto)
            _, colors = piece_series(g, store.root, sil, pad)
            track.clothes = np.median(colors, 0) if len(colors) >= 2 else None
            tracks.append(track)
    if cfg.cam_debias > 0:
        protos = scene_vectors(store, trdf, cfg.cam_debias, cfg.scene_lam, cfg.occ_limit)
        for t in tracks:
            t.proto = protos.get(t.tid)
    return tracks, trdf


def balance_clothes(tracks: list[Track]) -> None:
    """Выравнивает яркость камер: белая футболка на тёмной камере иначе выглядит как другая одежда."""
    bright = {}
    for cam in {t.cam for t in tracks}:
        vals = np.array([t.clothes for t in tracks if t.cam == cam and t.clothes is not None]).ravel()
        if len(vals) >= 6 and np.percentile(vals, 90) >= 60:
            bright[cam] = float(np.percentile(vals, 90))
    if len(bright) < 2:
        return
    ref = float(np.median(list(bright.values())))
    for t in tracks:
        if t.clothes is not None and t.cam in bright:
            t.clothes = np.clip(t.clothes * np.clip(ref / bright[t.cam], 0.7, 1.6), 0, 255)


def link(data_dir: str | Path, cfg: Config) -> dict:
    """Склеивает треклеты всех камер папки данных. Пишет новый run в runs/persons/tracklet_person."""
    lc = cfg.link
    t_start = time.perf_counter()
    store = Store(data_dir)
    tracks, trdf = _load_tracks(store, lc, cfg.crops.pad)
    cams = store.load_cameras()
    if cams.empty:
        raise ValueError("в БД нет треклетов: сначала process")
    balance_clothes(tracks)
    save_colors(store.root, {t.tid: t.clothes for t in tracks if t.clothes is not None})
    if tracks and lc.part_min > 0:
        from . import bodyparts
        from .reid import make_reid

        import torch

        parts = bodyparts.part_vectors(store, make_reid(cfg.reid, "cuda:0" if torch.cuda.is_available() else "cpu"))
        bodyparts.save(store.root, parts)
        for t in tracks:
            t.low = parts.get(t.tid, {}).get("low")
    shifts, shift_report = camera_shifts(tracks, lc)
    tracks = apply_shifts(tracks, shifts)
    for r in shift_report:
        log.info("cam%s: поправка положения на плане %+.2f, %+.2f м", r["cam"], r["dx"], r["dy"])
    edges, cannot, c1 = candidate_edges(tracks, lc)
    c1["duplicates_merged"] = 0
    dups: list[tuple[int, int]] = []
    for cid in cams["cam_id"]:
        boxes = store.load_observations(int(cid), columns=["tid", "frame_idx", "x1", "y1", "x2", "y2"])
        found = duplicate_pairs(boxes, lc)
        dups += found
        c1["duplicates_merged"] += merge_duplicates(found, tracks, edges, cannot, lc)
    if lc.ahc_thr > 0:
        labels, merged = appearance_clusters(tracks, dups, lc)
        labels, purged = purge_groups(labels, tracks, lc)
        labels, attached = attach_rest(labels, tracks, edges, cannot)
        c2 = {"merged": merged, "attached": attached, "purged": purged}
        conf, first_labels = {}, dict(labels)
    else:
        labels, c2 = cluster([t.tid for t in tracks], edges, cannot)
        conf = c2.pop("conf")
        first_labels = dict(labels)
        labels, c3 = link_groups(labels, tracks, cannot, lc)
        c2.update(c3)
        labels, c2["purged"] = purge_groups(labels, tracks, lc)
        labels, c2["rescued"] = rescue_groups(labels, tracks, lc, scene_app_threshold(tracks, lc))
        labels, c2["handoffs"], c2["app_threshold"] = appearance_handoffs(labels, tracks, lc)

    # номера людей: по времени появления
    by_root: dict[int, list[Track]] = defaultdict(list)
    tmap = {t.tid: t for t in tracks}
    for tid, r in labels.items():
        by_root[r].append(tmap[tid])
    groups = sorted(by_root.items(), key=lambda kv: (min(t.t0 for t in kv[1]), min(t.tid for t in kv[1])))
    tr_idx = trdf.set_index("tid")
    persons, links = [], []
    summary_rows = []
    for pid, (r, mem) in enumerate(groups, start=1):
        sub = tr_idx.loc[[m.tid for m in mem]]
        best = sub.sort_values(["max_q"], ascending=False)
        thumb = next((p for p in best["thumb_path"] if isinstance(p, str)), None)
        t0, t1 = float(sub["t0"].min()), float(sub["t1"].max())
        cs = [conf[first_labels[m.tid]] for m in mem if first_labels[m.tid] in conf]
        persons.append((pid, t0, t1, thumb, float(np.mean(cs)) if cs else None))
        links += [(int(m.tid), pid) for m in mem]
        # время присутствия: объединение интервалов по всем камерам
        iv = sorted((m.t0, m.t1) for m in mem)
        tot, cur0, cur1 = 0.0, iv[0][0], iv[0][1]
        for a, b in iv[1:]:
            if a > cur1:
                tot += cur1 - cur0
                cur0, cur1 = a, b
            else:
                cur1 = max(cur1, b)
        tot += cur1 - cur0
        summary_rows.append({"person_id": pid, "t0": round(t0, 2), "t1": round(t1, 2), "present_s": round(tot, 1),
                             "n_tracklets": len(mem), "cams": ",".join(str(c) for c in sorted({m.cam for m in mem})),
                             "n_obs": int(sub["n_obs"].sum())})
    params = {"link": asdict(lc), "n_tracklets": len(tracks), "counters": {**c1, **c2}}
    with store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        cur = con.execute("INSERT INTO runs (created_at, params_json) VALUES (?, ?)",
                          (time.strftime("%Y-%m-%dT%H:%M:%S"), json.dumps(params, ensure_ascii=False)))
        run_id = cur.lastrowid
        con.executemany("INSERT INTO persons (run_id, person_id, t0, t1, thumb_path, confidence) VALUES (?,?,?,?,?,?)",
                        [(run_id, *p) for p in persons])
        con.executemany("INSERT INTO tracklet_person (run_id, tid, person_id) VALUES (?,?,?)",
                        [(run_id, t, p) for t, p in links])
        con.execute("COMMIT")
    n_major = sum(row["present_s"] >= lc.major_min_s for row in summary_rows)
    edge_log = [asdict(e) for e in edges]
    out = {
        "run_id": run_id, "n_tracklets": len(tracks), "n_persons": len(persons), "n_major": n_major,
        "counters": {**c1, **c2}, "wall_s": round(time.perf_counter() - t_start, 2),
        "persons": summary_rows, "camera_shifts": shift_report,
    }
    (store.root / "runs").mkdir(exist_ok=True)
    (store.root / "runs" / f"link_run{run_id}.json").write_text(
        json.dumps({**out, "edges": edge_log, "params": asdict(lc)}, ensure_ascii=False, indent=1, default=float),
        encoding="utf-8")
    return out


def format_link_summary(s: dict, major_min_s: float) -> str:
    c = s["counters"]
    lines = [
        f"run_id {s['run_id']}: треклетов {s['n_tracklets']} -> людей {s['n_persons']} "
        f"(основных, присутствуют ≥ {major_min_s:.0f} с: {s['n_major']}) за {s['wall_s']} с",
        f"  рёбра: одновременные {c.get('edges_overlap', 0)}, последовательные {c.get('edges_seq', 0)}; "
        f"запреты: одна камера {c.get('cannot_same_cam', 0)}, далеко {c.get('cannot_far', 0)}; "
        f"вето по внешности {c.get('vetoed_appearance', 0)}",
        f"  слияний {c.get('merged', 0)}, отклонено из-за запретов {c.get('rejected_conflict', 0)}, "
        f"запретов по одежде {c.get('cannot_clothes', 0)}, разобрано смешанных {c.get('purged', 0)}, двойников соединено {c.get('duplicates_merged', 0)}, спасено по внешности {c.get('rescued', 0)},переходов между камерами по внешности {c.get('handoffs', 0)} "
        f"(порог сцены {c.get('app_threshold', 0):.2f})",
        *([("  поправка привязки по людям, м: " +
            ", ".join(f"cam{r['cam']} ({r['dx']:+}, {r['dy']:+})" for r in s["camera_shifts"]))]
          if s.get("camera_shifts") else []),
        "  основные люди:",
    ]
    for p in s["persons"]:
        if p["present_s"] >= major_min_s:
            lines.append(f"    P{p['person_id']}: {p['t0']}–{p['t1']} с, присутствует {p['present_s']} с, "
                         f"треклетов {p['n_tracklets']}, камеры {p['cams']}")
    return "\n".join(lines)
