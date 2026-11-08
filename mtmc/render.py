"""Видео с разметкой из сохранённых данных (раздел 5). Нейросети не используются."""
from __future__ import annotations

import json
import logging
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

from .config import Config
from .draw import GRAY, FFmpegWriter, TextRenderer, id_color, readable_text_color
from .storage import Store
from .video_io import VideoSource

log = logging.getLogger(__name__)

COCO_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
              (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]


def fmt_time(t: float) -> str:
    sign = "-" if t < 0 else ""
    t = abs(t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - h * 3600 - m * 60
    return f"{sign}{h:02d}:{m:02d}:{s:05.2f}"


def _infer_stride(frames: np.ndarray) -> int:
    u = np.unique(frames)
    if len(u) < 2:
        return 1
    return int(np.gcd.reduce(np.diff(u))) or 1


class Renderer:
    """Готовит данные камеры и рисует кадры."""

    def __init__(self, store: Store, cam_id: int, cfg: Config, label: str = "tracklet",
                 tids: Iterable[int] | None = None, run_id: int | None = None):
        self.cfg = cfg
        self.rc = cfg.render
        self.cam = store.get_camera(cam_id)
        self.passport = json.loads(self.cam["passport_json"])
        self.meta = store.load_meta(cam_id)
        self.fps = float(self.cam["fps"])
        self.offset = float(self.cam["offset_s"] or 0.0)
        self.label = label
        obs = store.load_observations(cam_id)
        obs = obs[~obs["is_outlier"].fillna(False).astype(bool) & obs["tid"].notna()]
        self.tids_filter = set(int(t) for t in tids) if tids else None
        if self.tids_filter:
            obs = obs[obs["tid"].isin(self.tids_filter)]
        self.obs = obs.sort_values(["tid", "frame_idx"]).reset_index(drop=True)
        self.stride = int(self.meta.get("stride") or _infer_stride(self.obs["frame_idx"].to_numpy()))
        tr = store.load_tracklets(cam_id)
        self.tracklets = tr
        self.weak = set(tr.loc[tr["is_weak"] == 1, "tid"].astype(int))
        self.pmap = store.person_map(run_id) if label == "person" else {}
        self.has_plan = self.passport.get("H") is not None and self.obs["xs"].notna().any()
        self._prepare()

    # ------------------------------------------------------------ подготовка
    def _prepare(self) -> None:
        o = self.obs
        n = len(o)
        self.f = o["frame_idx"].to_numpy(np.int64)
        self.tid = o["tid"].astype("int64").to_numpy()
        self.lid = o["local_id"].to_numpy(np.int64)
        self.box = o[["x1", "y1", "x2", "y2"]].to_numpy(np.float64)
        self.uv = o[["u", "v"]].to_numpy(np.float64)
        self.src = o["foot_src"].to_numpy()
        self.view = o["view"].to_numpy()
        self.q = o["q"].to_numpy(np.float64)
        self.kp = np.stack(o["kp"].to_numpy()).reshape(n, 17, 3) if n else np.zeros((0, 17, 3))
        if self.has_plan:
            xs = o["xs"].to_numpy(np.float64)
            ys = o["ys"].to_numpy(np.float64)
            self.plan = np.stack([xs, ys], 1)
        else:
            self.plan = None
        # сущность (для цвета и следа) и подпись
        self.ent = []
        self.text = []
        self.color = []
        for i in range(n):
            tid, lid = int(self.tid[i]), int(self.lid[i])
            weak = tid in self.weak
            if self.label == "local":
                key, txt, col = ("L", lid), f"#{lid}", id_color(lid)
            elif self.label == "person":
                pid = self.pmap.get(tid)
                if pid is None:
                    key, txt, col = ("T", tid), f"T{tid % 1_000_000}?", GRAY
                else:
                    key, txt, col = ("P", pid), f"P{pid}", id_color(pid)
            else:
                key, txt, col = ("T", tid), f"T{tid % 1_000_000}", id_color(tid)
            if weak and not (self.label == "person" and key[0] == "P"):
                col = GRAY   # слабый треклет серый, если только он не приклеен к человеку
            self.ent.append(key)
            self.text.append(txt)
            self.color.append(col)
        # кадр -> [(i, j, alpha)] : i — наблюдение, j — следующее для интерполяции
        self.by_frame: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
        max_gap = self.stride * max(self.rc.interp_max_gap_steps, 1)
        for i in range(n):
            self.by_frame[int(self.f[i])].append((i, -1, 0.0))
            if i + 1 < n and self.tid[i + 1] == self.tid[i]:
                fa, fb = int(self.f[i]), int(self.f[i + 1])
                if 1 < fb - fa <= max_gap:
                    for fr in range(fa + 1, fb):
                        self.by_frame[fr].append((i, i + 1, (fr - fa) / (fb - fa)))
        # следы по сущностям
        self.trail: dict[tuple, tuple[list[int], np.ndarray, np.ndarray | None]] = {}
        # след ведётся по треклету (а не по человеку): иначе у дублей он прыгает между ними
        self.trail_key = [("T", int(t)) for t in self.tid]
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i, k in enumerate(self.trail_key):
            groups[k].append(i)
        for k, idx in groups.items():
            idx = sorted(idx, key=lambda i: self.f[i])
            self.trail[k] = ([int(self.f[i]) for i in idx], self.uv[idx],
                             self.plan[idx] if self.plan is not None else None)
        # мини-план: масштаб по крайним точкам с отступом
        if self.has_plan:
            p = self.plan[np.isfinite(self.plan).all(1)]
            m = self.rc.minimap_margin_m
            if len(p):
                # по 1–99 процентилям: единичные выбросы далеко за полом не должны сжимать план
                lo, hi = np.percentile(p, 1, axis=0), np.percentile(p, 99, axis=0)
                self.plan_bounds = (lo[0] - m, lo[1] - m, hi[0] + m, hi[1] + m)
            else:
                self.has_plan = False

    def frame_range(self, start_s: float | None, end_s: float | None, clip: bool) -> tuple[int, int | None]:
        if clip and self.tids_filter:
            t = self.tracklets[self.tracklets["tid"].isin(self.tids_filter)]
            if len(t):
                pad = self.rc.clip_pad_s
                a = (t["t0"].min() - pad - self.offset) * self.fps
                b = (t["t1"].max() + pad - self.offset) * self.fps
                return max(0, int(math.floor(a))), int(math.ceil(b)) + 1
        a = int(round(start_s * self.fps)) if start_s is not None else int(self.meta.get("start_frame", 0))
        if end_s is not None:
            b = int(round(end_s * self.fps))
        else:
            b = self.meta.get("end_frame")
        return a, b

    # ------------------------------------------------------------ рисование
    def _state(self, i: int, j: int, a: float):
        if j < 0:
            return self.box[i], self.uv[i], self.kp[i], (self.plan[i] if self.plan is not None else None)
        lerp = lambda x, y: x * (1 - a) + y * a  # noqa: E731
        plan = lerp(self.plan[i], self.plan[j]) if self.plan is not None else None
        kp = lerp(self.kp[i], self.kp[j])
        return lerp(self.box[i], self.box[j]), lerp(self.uv[i], self.uv[j]), kp, plan

    def draw(self, frame: np.ndarray, fidx: int, s: float, tr: TextRenderer, details: bool, skeleton: bool,
             zones_overlay: tuple | None, minimap: bool) -> np.ndarray:
        rc = self.rc
        if zones_overlay is not None:
            ov, mask, labels = zones_overlay
            a = rc.zones_alpha
            frame[mask] = (frame[mask] * (1 - a) + ov[mask] * a).astype(np.uint8)
            for txt, (x, y), col in labels:
                tr.put(frame, txt, x, y, (255, 255, 255), bg=col)
        items = self.by_frame.get(fidx, [])
        trail_frames = int(round(rc.trail_s * self.fps))
        plan_items = []
        kpc = self.cfg.observation.kp_conf
        for i, j, a in items:
            box, uv, kp, plan = self._state(i, j, a)
            col = self.color[i]
            tid = int(self.tid[i])
            weak = tid in self.weak
            thick = rc.weak_thickness if weak else rc.box_thickness
            b = (box * s).round().astype(int)
            # след
            key = self.ent[i]
            fr, uvs, plans = self.trail[self.trail_key[i]]
            lo, hi = bisect_left(fr, fidx - trail_frames), bisect_right(fr, fidx)
            pts = uvs[lo:hi] * s
            pts = np.vstack([pts, uv[None] * s]) if j >= 0 else pts
            if len(pts) >= 2:
                cv2.polylines(frame, [pts.round().astype(np.int32).reshape(-1, 1, 2)], False, col, 2, cv2.LINE_AA)
            if skeleton:
                for e1, e2 in COCO_EDGES:
                    if kp[e1, 2] > kpc and kp[e2, 2] > kpc:
                        cv2.line(frame, tuple((kp[e1, :2] * s).astype(int)), tuple((kp[e2, :2] * s).astype(int)),
                                 col, 1, cv2.LINE_AA)
                for k in range(17):
                    if kp[k, 2] > kpc:
                        cv2.circle(frame, tuple((kp[k, :2] * s).astype(int)), 2, col, -1, cv2.LINE_AA)
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), col, thick, cv2.LINE_AA)
            fu, fv = int(round(uv[0] * s)), int(round(uv[1] * s))
            if self.src[i] == "ankles":
                cv2.circle(frame, (fu, fv), rc.foot_radius, col, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (fu, fv), rc.foot_radius, col, 1, cv2.LINE_AA)
            txt = self.text[i]
            if details:
                txt = f"{txt} · {self.view[i]} · {self.q[i]:.2f}"
            sp = tr.sprite(txt, readable_text_color(col))
            ty = b[1] - sp.shape[0] - 4
            if ty < 0:
                ty = b[1]
            tr.put(frame, txt, b[0], ty, readable_text_color(col), bg=col)
            if minimap and plan is not None and np.isfinite(plan).all():
                plan_items.append((key, plan, col, plans, lo, hi))
        info = (f"{self.cam.get('name') or 'cam' + str(self.cam['cam_id'])}  |  кадр {fidx}  |  "
                f"{fmt_time(fidx / self.fps + self.offset)}  |  людей: {len(items)}")
        tr.put(frame, info, 6, 6, (255, 255, 255), bg=(30, 30, 30), pad=4)
        if minimap and self.has_plan:
            self._minimap(frame, plan_items)
        return frame

    def _minimap(self, frame: np.ndarray, items: list) -> None:
        x0, y0, x1, y1 = self.plan_bounds
        size = min(self.rc.minimap_px, int(0.4 * min(frame.shape[:2])))
        sx = size / max(x1 - x0, 1e-6)
        sy = size / max(y1 - y0, 1e-6)
        k = min(sx, sy)
        pw, ph = max(int((x1 - x0) * k), 20), max(int((y1 - y0) * k), 20)
        H, W = frame.shape[:2]
        pw, ph = min(pw, W - 20), min(ph, H - 20)
        ox, oy = W - pw - 10, H - ph - 10
        roi = frame[oy:oy + ph, ox:ox + pw]
        roi[:] = (roi * 0.35).astype(np.uint8)
        cv2.rectangle(frame, (ox, oy), (ox + pw - 1, oy + ph - 1), (200, 200, 200), 1)

        def to_px(p: np.ndarray) -> np.ndarray:
            return np.stack([ox + (p[..., 0] - x0) * k, oy + (p[..., 1] - y0) * k], -1)

        for key, plan, col, plans, lo, hi in items:
            if plans is not None and hi - lo >= 2:
                tp = plans[lo:hi]
                tp = tp[np.isfinite(tp).all(1)]
                if len(tp) >= 2:
                    cv2.polylines(frame, [to_px(tp).round().astype(np.int32).reshape(-1, 1, 2)], False, col, 1,
                                  cv2.LINE_AA)
            c = to_px(plan).round().astype(int)
            cv2.circle(frame, (int(c[0]), int(c[1])), 4, col, -1, cv2.LINE_AA)

    def zones_overlay(self, W: int, H: int, s: float) -> tuple:
        """Статичный слой зон паспорта: (цветной слой, маска, подписи)."""
        ov = np.zeros((H, W, 3), np.uint8)
        polys = []
        pp = self.passport
        if pp.get("floor_roi"):
            polys.append(("пол", pp["floor_roi"], (80, 200, 80)))
        for name, z in (pp.get("zones") or {}).items():
            to = z.get("leads_to")
            polys.append((f"{name} → {to}" if to else name, z["polygon"], (60, 160, 255)))
        for cam, poly in (pp.get("overlaps") or {}).items():
            polys.append((f"общая с cam{cam}", poly, (255, 120, 200)))
        labels = []
        for name, poly, col in polys:
            pts = (np.asarray(poly, float) * s).round().astype(np.int32).reshape(-1, 1, 2)
            if name != "пол":
                cv2.fillPoly(ov, [pts], col)
            cv2.polylines(ov, [pts], True, col, 3, cv2.LINE_AA)
            c = pts.reshape(-1, 2).mean(0).astype(int)
            labels.append((name, (int(c[0]), int(c[1])), col))
        mask = ov.any(2)
        return ov, mask, labels


def render(cam_id: int, data_dir: str | Path, out: str | Path, cfg: Config, label: str = "tracklet",
           details: bool = False, skeleton: bool = False, zones: bool = False, minimap: bool = False,
           fps_mode: str = "original", scale: float = 1.0, start_s: float | None = None,
           end_s: float | None = None, tids: list[int] | None = None, clip: bool = False,
           run_id: int | None = None) -> dict:
    """Рисует mp4 с разметкой. Возвращает {'frames': n, 'out': path}."""
    store = Store(data_dir)
    r = Renderer(store, cam_id, cfg, label, tids, run_id)
    src = VideoSource(r.cam["video_path"], r.fps, cfg.video.exact_seek)
    a, b = r.frame_range(start_s, end_s, clip)
    if b is None:
        b = src.n_frames
    if fps_mode == "processed":
        st = r.stride
        base = int(r.meta.get("start_frame", 0))
        a = a + ((base - a) % st)
        step, out_fps = st, r.fps / st
    elif fps_mode == "original":
        step, out_fps = 1, r.fps
    else:
        raise ValueError("--fps-mode: original | processed")
    W = int(round(src.width * scale / 2) * 2)
    H = int(round(src.height * scale / 2) * 2)
    s = W / src.width
    # font_size задан для 720p; масштабируем под высоту выходного кадра
    tr = TextRenderer(cfg.render.font, max(11, int(round(cfg.render.font_size * H / 720))))
    zo = r.zones_overlay(W, H, s) if zones else None
    if minimap and not r.has_plan:
        log.warning("--minimap: у камеры нет гомографии/координат плана, мини-план не рисуется")
    writer = FFmpegWriter(out, W, H, out_fps, cfg.render.crf, cfg.render.preset, cfg.render.ffmpeg)
    total = src.count_range(a, b, step)
    n = 0
    try:
        for fidx, frame in tqdm(src.frames(a, b, step), total=total, unit="кадр", desc="render"):
            if (frame.shape[1], frame.shape[0]) != (W, H):
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            frame = r.draw(frame, fidx, s, tr, details, skeleton, zo, minimap)
            writer.write(frame)
            n += 1
    finally:
        writer.close()
    return {"frames": n, "out": str(out), "fps": out_fps, "range": [a, b]}
