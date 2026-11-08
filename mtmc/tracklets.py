"""Треклеты: нарезка (правила 1–3), отбор кропов, прототипы, номера, сводка. Чистые функции."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .config import ReIDCfg, TrackletCfg

VIEWS = ("front", "back", "side")


# ---------------------------------------------------------------- типы
@dataclass
class EmbPack:
    """Эмбеддинги одного треклета (раздел 4.3). t — только для правила 3, в npz не пишется."""
    embs: np.ndarray                 # (m, d) float32, L2
    q: np.ndarray                    # (m,)
    view: np.ndarray                 # (m,) str
    frame_idx: np.ndarray            # (m,)
    t: np.ndarray                    # (m,)
    model: str = ""
    protos: dict[str, np.ndarray] = field(default_factory=dict)

    def subset(self, mask: np.ndarray) -> "EmbPack":
        return EmbPack(self.embs[mask], self.q[mask], self.view[mask], self.frame_idx[mask], self.t[mask],
                       self.model)

    def __len__(self) -> int:
        return len(self.embs)


@dataclass
class TrackletRow:
    tid: int
    cam_id: int
    local_id: int
    t0: float
    t1: float
    frame0: int
    frame1: int
    n_obs: int
    mean_q: float | None = None
    max_q: float | None = None
    is_weak: int = 0
    start_x: float | None = None
    start_y: float | None = None
    end_x: float | None = None
    end_y: float | None = None
    v_mean: float | None = None
    v0x: float | None = None
    v0y: float | None = None
    v1x: float | None = None
    v1y: float | None = None
    entry_zone: str | None = None
    exit_zone: str | None = None
    overlap_share_json: str | None = None
    start_reason: str | None = None
    end_reason: str | None = None
    n_emb: int = 0
    emb_path: str | None = None
    thumb_path: str | None = None


# ---------------------------------------------------------------- правила 1–2
def _break_reason(a: Any, b: Any, cfg: TrackletCfg, has_floor: bool) -> str | None:
    """Причина разреза между наблюдениями a и b (namedtuple-строки) или None."""
    dt = b.t - a.t
    if dt > cfg.gap_cut_s:
        return "gap"
    dt = max(dt, 0.0)
    if has_floor and np.isfinite(a.x) and np.isfinite(b.x):
        d = float(np.hypot(b.x - a.x, b.y - a.y))
        sa = a.sigma if np.isfinite(a.sigma) else 0.0
        sb = b.sigma if np.isfinite(b.sigma) else 0.0
        if d > cfg.v_max_mps * dt + cfg.jump_sigma_k * float(np.hypot(sa, sb)):
            return "jump"
        return None
    d = float(np.hypot(b.u - a.u, b.v - a.v))
    h = ((a.y2 - a.y1) + (b.y2 - b.y1)) / 2
    if d > (cfg.v_max_heights_ps * dt + cfg.jump_px_margin_h) * h:
        return "jump"
    return None


def cut_tracks(obs: pd.DataFrame, cfg: TrackletCfg, has_floor: bool) -> pd.DataFrame:
    """Правила 1–2. Добавляет колонки piece_id (-1 у выбросов), is_outlier, start_reason/end_reason.

    piece_id — временный номер куска внутри вызова (детерминирован порядком local_id и времени).
    """
    df = obs.copy()
    for c in ("x", "y", "sigma"):
        if c not in df:
            df[c] = np.nan
    df["piece_id"] = -1
    df["is_outlier"] = False
    df["start_reason"] = None
    df["end_reason"] = None
    if df.empty:
        return df
    piece_col = np.full(len(df), -1, dtype=np.int64)
    out_col = np.zeros(len(df), dtype=bool)
    sr_col = np.empty(len(df), dtype=object)
    er_col = np.empty(len(df), dtype=object)
    pos = {idx: k for k, idx in enumerate(df.index)}
    next_piece = 0
    if "occ" not in df:
        df["occ"] = 0.0
    cols = ["t", "x", "y", "sigma", "u", "v", "y1", "y2", "occ"]
    for _lid, g in df.sort_values(["local_id", "t", "frame_idx"]).groupby("local_id", sort=True):
        rows = list(g[cols].itertuples())
        n = len(rows)
        # эпизоды сильного перекрытия с гистерезисом
        occl: dict[Any, bool] = {}
        flag = False
        for r in rows:
            o = r.occ if np.isfinite(r.occ) else 0.0
            if not flag and o >= cfg.occ_cut:
                flag = True
            elif flag and o < cfg.occ_release:
                flag = False
            occl[r.Index] = flag

        def reason_of(a: Any, b: Any) -> str | None:
            rs = _break_reason(a, b, cfg, has_floor)
            if rs is not None or not cfg.occlusion_cut:
                return rs
            if occl[a.Index] != occl[b.Index]:
                return "occlusion"
            if b.t - a.t > cfg.occ_gap_s and (occl[a.Index] or a.occ >= cfg.occ_cut or b.occ >= cfg.occ_cut):
                return "occlusion"
            return None
        pieces: list[tuple[list, str, str]] = []
        cur = [rows[0]]
        start_reason = "track_start"
        prev = rows[0]
        j = 1
        while j < n:
            b = rows[j]
            reason = reason_of(prev, b)
            if reason is None:
                cur.append(b)
                prev = b
                j += 1
                continue
            if reason == "jump" and j + 1 < n:
                c = rows[j + 1]
                if _break_reason(b, c, cfg, has_floor) == "jump" and reason_of(prev, c) is None:
                    out_col[pos[b.Index]] = True
                    j += 1
                    continue
            pieces.append((cur, start_reason, reason))
            cur = [b]
            start_reason = reason
            prev = b
            j += 1
        pieces.append((cur, start_reason, "track_end"))
        for members, sr, er in pieces:
            for r in members:
                k = pos[r.Index]
                piece_col[k] = next_piece
                sr_col[k] = sr
                er_col[k] = er
            next_piece += 1
    df["piece_id"] = piece_col
    df["is_outlier"] = out_col
    df["start_reason"] = sr_col
    df["end_reason"] = er_col
    return df


# ---------------------------------------------------------------- правило 3
def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def scene_cut_threshold(pieces: list[pd.DataFrame], packs: list, cfg: ReIDCfg) -> float:
    """Порог разреза по внешности для этой камеры. На EPFL сходство одного человека с собой высокое, и порог 0.75
    работает, а через широкий объектив тот же человек даёт 0.6–0.75: постоянный порог режет людей на куски (cam2
    офиса: 30 треков трекера стали 89 треклетов). Мерка из самой записи: кого камера видела одновременно, те разные,
    и разрез нужен только там, где человек непохож на себя так же, как на другого."""
    means = []
    for p, pack in zip(pieces, packs):
        if pack is not None and len(pack):
            v = np.asarray(pack.embs, dtype=np.float64).mean(0)
            means.append((float(p["t"].min()), float(p["t"].max()), v / max(np.linalg.norm(v), 1e-12)))
    neg = [float(a[2] @ b[2]) for i, a in enumerate(means) for b in means[i + 1:]
           if min(a[1], b[1]) - max(a[0], b[0]) > 1.0]
    if len(neg) < 2:
        return min(cfg.app_cut_min, cfg.app_cut_thr)
    return float(np.clip(np.median(neg) + cfg.app_cut_margin, cfg.app_cut_min, cfg.app_cut_thr))


def appearance_cuts(embs: np.ndarray, times: np.ndarray, cfg: ReIDCfg) -> list[float]:
    """Правило 3. Возвращает моменты разреза (может быть пусто). Эмбеддинги — в порядке времени."""
    embs = np.asarray(embs, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    order = np.argsort(times, kind="stable")
    embs, times = embs[order], times[order]
    cuts: list[float] = []
    min_side = cfg.app_cut_min_side
    min_total = max(cfg.app_cut_min_total, 2 * min_side)

    def rec(lo: int, hi: int) -> None:
        n = hi - lo
        if n < min_total:
            return
        seg = embs[lo:hi]
        csum = np.cumsum(seg, 0)
        total = csum[-1]
        best_k, best_s = None, np.inf
        for k in range(min_side, n - min_side + 1):
            if times[lo + k - 1] == times[lo + k]:
                continue   # нельзя разрезать между кропами одного момента
            s = _cos(csum[k - 1] / k, (total - csum[k - 1]) / (n - k))
            if s < best_s:
                best_k, best_s = k, s
        if best_k is None or best_s >= cfg.app_cut_thr:
            return
        cuts.append(float((times[lo + best_k - 1] + times[lo + best_k]) / 2))
        rec(lo, lo + best_k)
        rec(lo + best_k, hi)

    rec(0, len(embs))
    return sorted(cuts)


def split_by_times(piece: pd.DataFrame, cuts: list[float]) -> list[pd.DataFrame]:
    """Делит наблюдения куска по моментам разреза (t < cut — влево)."""
    if not cuts:
        return [piece]
    edges = [-np.inf] + sorted(cuts) + [np.inf]
    parts = []
    for a, b in zip(edges[:-1], edges[1:]):
        p = piece[(piece["t"] >= a) & (piece["t"] < b)]
        if len(p):
            parts.append(p)
    return parts


# ---------------------------------------------------------------- кропы и прототипы
def select_crops(piece: pd.DataFrame, time_bins: int, max_crops: int) -> pd.DataFrame:
    """До max_crops//time_bins лучших по q кропов в каждом из time_bins равных отрезков времени."""
    c = piece[piece["crop_path"].notna()]
    if c.empty:
        return c
    per_bin = max(1, max_crops // max(time_bins, 1))
    t0, t1 = float(piece["t"].min()), float(piece["t"].max())
    span = max(t1 - t0, 1e-9)
    b = np.minimum(((c["t"].to_numpy() - t0) / span * time_bins).astype(int), time_bins - 1)
    c = c.assign(_bin=b).sort_values(["_bin", "q", "frame_idx"], ascending=[True, False, True])
    sel = c.groupby("_bin", sort=True).head(per_bin)
    return sel.drop(columns="_bin").sort_values("t")


def compute_prototypes(embs: np.ndarray, q: np.ndarray, views: np.ndarray, proto_min: int = 2) -> dict[str, np.ndarray]:
    """Прототипы по ракурсам (взвешенное по q среднее, L2) + 'all'."""
    out: dict[str, np.ndarray] = {}
    if len(embs) == 0:
        return out
    w = np.maximum(np.asarray(q, dtype=np.float64), 1e-6)

    def proto(mask: np.ndarray) -> np.ndarray:
        v = (embs[mask].astype(np.float64) * w[mask, None]).sum(0)
        return (v / max(np.linalg.norm(v), 1e-12)).astype(np.float32)

    out["all"] = proto(np.ones(len(embs), bool))
    views = np.asarray(views)
    for v in VIEWS:
        m = views == v
        if m.sum() >= proto_min:
            out[v] = proto(m)
    return out


# ---------------------------------------------------------------- номера
def assign_tids(pieces: list[pd.DataFrame], cam_id: int) -> list[int]:
    """tid = cam_id * 1_000_000 + n, n = 1.. по времени начала (затем кадр, local_id)."""
    keys = [(float(p["t"].min()), int(p["frame_idx"].min()), int(p["local_id"].iloc[0]), k)
            for k, p in enumerate(pieces)]
    order = sorted(range(len(pieces)), key=lambda k: keys[k])
    tids = [0] * len(pieces)
    for n, k in enumerate(order, start=1):
        tids[k] = cam_id * 1_000_000 + n
    return tids


# ---------------------------------------------------------------- траектория и сводка
def smooth_and_velocity(piece: pd.DataFrame, cfg: TrackletCfg, has_floor: bool) -> pd.DataFrame:
    """Сглаженные xs, ys и скорости vx, vy (м или px). Возвращает кусок с этими колонками."""
    p = piece.sort_values("t").copy()
    if has_floor and p["x"].notna().all():
        X, Y = p["x"].to_numpy(float), p["y"].to_numpy(float)
    else:
        X, Y = p["u"].to_numpy(float), p["v"].to_numpy(float)
    t = p["t"].to_numpy(float)
    n = len(p)
    w = cfg.smooth_window
    if n >= w and w > cfg.smooth_poly:
        if w % 2 == 0:
            w -= 1
        xs = savgol_filter(X, w, cfg.smooth_poly)
        ys = savgol_filter(Y, w, cfg.smooth_poly)
    else:
        xs, ys = X.copy(), Y.copy()
    if n >= 2:
        vx = np.gradient(xs, t)
        vy = np.gradient(ys, t)
    else:
        vx = np.full(n, np.nan)
        vy = np.full(n, np.nan)
    p["xs"], p["ys"], p["vx"], p["vy"] = xs, ys, vx, vy
    return p


def _f(x: Any) -> float | None:
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def summarize(obs_piece: pd.DataFrame, emb: EmbPack | None, passport: Any, cfg: TrackletCfg,
              tid: int, start_reason: str, end_reason: str, emb_path: str | None = None) -> TrackletRow:
    """Сводка треклета (раздел 3.10). obs_piece должен уже содержать xs, ys, vx, vy."""
    p = obs_piece.sort_values("t")
    first, last = p.iloc[0], p.iloc[-1]
    t = p["t"].to_numpy(float)
    has_floor = getattr(passport, "H", None) is not None
    row = TrackletRow(
        tid=int(tid), cam_id=int(first["cam_id"]), local_id=int(first["local_id"]),
        t0=float(t[0]), t1=float(t[-1]),
        frame0=int(p["frame_idx"].min()), frame1=int(p["frame_idx"].max()),
        n_obs=int(len(p)), mean_q=_f(p["q"].mean()), max_q=_f(p["q"].max()),
        is_weak=int(len(p) < cfg.min_len),
        start_reason=start_reason, end_reason=end_reason,
    )
    if has_floor:
        row.start_x, row.start_y = _f(first["xs"]), _f(first["ys"])
        row.end_x, row.end_y = _f(last["xs"]), _f(last["ys"])
    vx, vy = p["vx"].to_numpy(float), p["vy"].to_numpy(float)
    if np.isfinite(vx).any():
        row.v_mean = _f(np.nanmean(np.hypot(vx, vy)))
        w = cfg.edge_speed_window_s
        m0 = t <= t[0] + w
        m1 = t >= t[-1] - w
        row.v0x, row.v0y = _f(np.nanmean(vx[m0])), _f(np.nanmean(vy[m0]))
        row.v1x, row.v1y = _f(np.nanmean(vx[m1])), _f(np.nanmean(vy[m1]))
    if getattr(passport, "zones", None):
        row.entry_zone = first["zone"] if isinstance(first["zone"], str) else None
        row.exit_zone = last["zone"] if isinstance(last["zone"], str) else None
    overlaps = getattr(passport, "overlaps", None)
    if overlaps:
        oc = p["overlap_cams"].fillna("").astype(str).str.split(",")
        share = {str(c): round(float(oc.apply(lambda s, c=c: str(c) in s).mean()), 4) for c in sorted(overlaps)}
        row.overlap_share_json = json.dumps(share)
    crops = p[p["crop_path"].notna()]
    if len(crops):
        front = crops[crops["view"] == "front"]
        src = front if len(front) else crops
        row.thumb_path = str(src.sort_values(["q", "frame_idx"], ascending=[False, True]).iloc[0]["crop_path"])
    if emb is not None and len(emb):
        row.n_emb = len(emb)
        row.emb_path = emb_path
    return row
