"""Все камеры одним видео (сетка 2×N) + панель «вид сверху» с людьми. Нейросети не используются."""
from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .config import Config
from .draw import FFmpegWriter, TextRenderer, readable_text_color
from .render import Renderer, fmt_time
from .storage import Store
from .video_io import VideoSource

log = logging.getLogger(__name__)


class _CamStream:
    """Последовательное чтение кадров камеры по запрошенным номерам (только вперёд)."""

    def __init__(self, r: Renderer, cfg: Config, a: int, b: int):
        self.r = r
        self.src = VideoSource(r.cam["video_path"], r.fps, cfg.video.exact_seek)
        self.a, self.b = a, b
        self.it = self.src.frames(a, b, 1)
        self.cur_idx, self.cur = -1, None

    def get(self, fidx: int) -> np.ndarray | None:
        if fidx < self.a or fidx >= self.b:
            return None
        while self.cur_idx < fidx:
            try:
                self.cur_idx, self.cur = next(self.it)
            except StopIteration:
                self.cur_idx, self.cur = 10 ** 12, None
                return None
        return self.cur if self.cur_idx == fidx else None


def mosaic(data_dir: str | Path, out: str | Path, cfg: Config, cams: list[int] | None = None,
           label: str = "person", run_id: int | None = None, scale: float = 1.0,
           start_s: float | None = None, end_s: float | None = None, plan: bool = True) -> dict:
    store = Store(data_dir)
    all_cams = [int(c) for c in store.load_cameras()["cam_id"]]
    cams = cams or all_cams
    rs = [Renderer(store, c, cfg, label, None, run_id) for c in cams]
    fps = rs[0].fps
    # общий отрезок времени (глобальные часы)
    t_lo = min(r.meta.get("start_frame", 0) / r.fps + r.offset for r in rs)
    t_hi = max((r.meta.get("end_frame") or r.cam["n_frames"]) / r.fps + r.offset for r in rs)
    if start_s is not None:
        t_lo = start_s
    if end_s is not None:
        t_hi = end_s
    n_out = int(math.floor((t_hi - t_lo) * fps))
    streams = []
    for r in rs:
        a = max(0, int(math.floor((t_lo - r.offset) * r.fps)))
        b = int(math.ceil((t_hi - r.offset) * r.fps)) + 1
        streams.append(_CamStream(r, cfg, a, b))
    W0, H0 = streams[0].src.width, streams[0].src.height
    tw, th = int(round(W0 * scale / 2) * 2), int(round(H0 * scale / 2) * 2)
    ncol = 2 if len(rs) > 1 else 1
    nrow = math.ceil(len(rs) / ncol)
    grid_w, grid_h = tw * ncol, th * nrow
    has_plan = plan and any(r.has_plan for r in rs)
    pw = grid_h if has_plan else 0
    W, H = grid_w + pw, grid_h
    W, H = W + W % 2, H + H % 2
    tr = TextRenderer(cfg.render.font, max(11, int(round(cfg.render.font_size * th / 720 * 1.3))))
    tr_plan = TextRenderer(cfg.render.font, max(12, int(round(pw / 40))) if pw else 12)
    # масштаб плана — по всем камерам
    if has_plan:
        b = np.array([r.plan_bounds for r in rs if r.has_plan])
        x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
        pad = 30
        k = min((pw - 2 * pad) / max(x1 - x0, 1e-6), (grid_h - 2 * pad) / max(y1 - y0, 1e-6))
        ox = grid_w + pad + ((pw - 2 * pad) - (x1 - x0) * k) / 2
        oy = pad + ((grid_h - 2 * pad) - (y1 - y0) * k) / 2
    trails: dict[tuple, deque] = defaultdict(deque)
    trail_s = cfg.render.trail_s * 2
    writer = FFmpegWriter(out, W, H, fps, cfg.render.crf, cfg.render.preset, cfg.render.ffmpeg)
    n = 0
    try:
        for k_out in tqdm(range(n_out), unit="кадр", desc="mosaic"):
            t = t_lo + k_out / fps
            canvas = np.zeros((H, W, 3), np.uint8)
            people: dict[tuple, list] = defaultdict(list)
            colors: dict[tuple, tuple] = {}
            texts: dict[tuple, str] = {}
            for ci, (r, st) in enumerate(zip(rs, streams)):
                fidx = int(round((t - r.offset) * r.fps))
                frame = st.get(fidx)
                row, col = divmod(ci, ncol)
                x, y = col * tw, row * th
                if frame is None:
                    tr.put(canvas, f"{r.cam.get('name')}: нет кадра", x + 6, y + 6, (200, 200, 200))
                    continue
                img = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)
                s = tw / frame.shape[1]
                img = r.draw(img, fidx, s, tr, False, False, None, False)
                canvas[y:y + th, x:x + tw] = img
                if has_plan and r.plan is not None:
                    for i, j, a in r.by_frame.get(fidx, []):
                        _, _, _, p = r._state(i, j, a)
                        if p is not None and np.isfinite(p).all():
                            key = r.ent[i]
                            people[key].append(p)
                            colors[key] = r.color[i]
                            texts[key] = r.text[i]
            if has_plan:
                cv2.rectangle(canvas, (grid_w, 0), (W - 1, H - 1), (40, 40, 40), -1)
                cv2.rectangle(canvas, (int(ox), int(oy)), (int(ox + (x1 - x0) * k), int(oy + (y1 - y0) * k)),
                              (90, 90, 90), 1)
                tr_plan.put(canvas, f"вид сверху  |  {fmt_time(t)}  |  людей: {len(people)}", grid_w + 8, 6,
                            (230, 230, 230))
                for key in list(people):
                    p = np.median(people[key], 0)          # медиана по камерам устойчива к выбросу одной камеры
                    if not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1):
                        del people[key]                    # за пределами пола — выброс, не рисуем
                        continue
                    trails[key].append((t, p))
                for key in list(trails):
                    q = trails[key]
                    while q and q[0][0] < t - trail_s:
                        q.popleft()
                    if not q:
                        del trails[key]
                        continue
                    col = colors.get(key, (150, 150, 150))
                    pts = np.array([[ox + (pp[0] - x0) * k, oy + (pp[1] - y0) * k] for _, pp in q])
                    if len(pts) >= 2:
                        cv2.polylines(canvas, [pts.round().astype(np.int32).reshape(-1, 1, 2)], False, col, 2,
                                      cv2.LINE_AA)
                    if key in people:
                        c = pts[-1].round().astype(int)
                        cv2.circle(canvas, (int(c[0]), int(c[1])), 9, col, -1, cv2.LINE_AA)
                        txt = texts.get(key, "")
                        tr_plan.put(canvas, txt, int(c[0]) + 11, int(c[1]) - 10, readable_text_color(col), bg=col)
            writer.write(canvas)
            n += 1
    finally:
        writer.close()
    return {"frames": n, "out": str(out)}
