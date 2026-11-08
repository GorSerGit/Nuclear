"""Отдельное видео на каждого человека: где он ходит по всем камерам. Нейросети не используются.

layout='grid' — все камеры сеткой, рамкой отмечен только этот человек; камеры, где его сейчас не видно, затемнены.
layout='best' — одна камера, на которой человек виден крупнее всего (без мелькания: камера меняется,
только когда текущая его потеряла).
"""
from __future__ import annotations

import logging
import math
import sqlite3
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .config import Config
from .draw import FFmpegWriter, TextRenderer, id_color
from .mosaic import _CamStream
from .render import Renderer, fmt_time
from .storage import Store

log = logging.getLogger(__name__)


def person_table(store: Store, run_id: int | None) -> tuple[int, dict[int, list[int]], dict[int, float]]:
    """-> (run_id, person_id -> [tid], person_id -> секунд присутствия)."""
    with sqlite3.connect(store.db_path) as con:
        if run_id is None:
            run_id = con.execute("SELECT MAX(run_id) FROM tracklet_person").fetchone()[0]
        if run_id is None:
            raise ValueError("нет склейки людей: сначала python -m mtmc link")
        rows = con.execute("SELECT tp.person_id, t.tid, t.t0, t.t1 FROM tracklet_person tp "
                           "JOIN tracklets t USING(tid) WHERE tp.run_id=?", (run_id,)).fetchall()
    tids: dict[int, list[int]] = {}
    iv: dict[int, list[tuple[float, float]]] = {}
    for pid, tid, t0, t1 in rows:
        tids.setdefault(pid, []).append(tid)
        iv.setdefault(pid, []).append((t0, t1))
    present = {}
    for pid, lst in iv.items():
        lst.sort()
        tot, a, b = 0.0, *lst[0]
        for x, y in lst[1:]:
            if x > b:
                tot += b - a
                a, b = x, y
            else:
                b = max(b, y)
        present[pid] = tot + b - a
    return run_id, tids, present


def render_person(store: Store, pid: int, tids: list[int], out: Path, cfg: Config, run_id: int,
                  present_s: float, layout: str = "grid", scale: float = 1.0, pad_s: float = 1.0) -> int:
    cams = [int(c) for c in store.load_cameras()["cam_id"]]
    rs = []
    for c in cams:
        r = Renderer(store, c, cfg, "person", tids, run_id)
        if layout == "best" and r.obs.empty:
            continue
        rs.append(r)
    if not rs:
        return 0
    tr_df = store.load_tracklets()
    tr_df = tr_df[tr_df["tid"].isin(tids)]
    t_lo = float(tr_df["t0"].min()) - pad_s
    t_hi = float(tr_df["t1"].max()) + pad_s
    fps = rs[0].fps
    streams = []
    for r in rs:
        a = max(0, int(math.floor((t_lo - r.offset) * r.fps)))
        b = min(int(math.ceil((t_hi - r.offset) * r.fps)) + 1, r.cam["n_frames"] or 10 ** 9)
        streams.append(_CamStream(r, cfg, a, b))
    W0, H0 = streams[0].src.width, streams[0].src.height
    tw, th = int(round(W0 * scale / 2) * 2), int(round(H0 * scale / 2) * 2)
    if layout == "grid":
        ncol = 2 if len(rs) > 1 else 1
        nrow = math.ceil(len(rs) / ncol)
        gw, gh = tw * ncol, th * nrow
    else:
        gw, gh = tw, th
    bar = max(28, int(round(th * 0.09)))
    W, H = gw + gw % 2, gh + bar + (gh + bar) % 2
    tr = TextRenderer(cfg.render.font, max(11, int(round(cfg.render.font_size * th / 720 * 1.3))))
    tr_bar = TextRenderer(cfg.render.font, max(14, int(bar * 0.6)))
    color = id_color(pid)
    n_out = max(1, int(math.floor((t_hi - t_lo) * fps)))
    writer = FFmpegWriter(out, W, H, fps, cfg.render.crf, cfg.render.preset, cfg.render.ffmpeg)
    cur_cam = None
    n = 0
    try:
        for k in tqdm(range(n_out), unit="кадр", desc=f"P{pid}", leave=False):
            t = t_lo + k / fps
            canvas = np.zeros((H, W, 3), np.uint8)
            visible = []
            tiles = []
            for ci, (r, st) in enumerate(zip(rs, streams)):
                fidx = int(round((t - r.offset) * r.fps))
                frame = st.get(fidx)
                items = r.by_frame.get(fidx, []) if frame is not None else []
                size = max((r.box[i, 3] - r.box[i, 1] for i, _, _ in items), default=0.0)
                if items:
                    name = r.cam.get("name") or f"cam{r.cam['cam_id']}"
                    visible.append(name.split()[-1])          # короткое имя: «c1»
                tiles.append((r, frame, fidx, size))
            if layout == "grid":
                for ci, (r, frame, fidx, size) in enumerate(tiles):
                    row, col = divmod(ci, ncol)
                    x, y = col * tw, bar + row * th
                    if frame is None:
                        continue
                    img = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)
                    img = r.draw(img, fidx, tw / frame.shape[1], tr, False, False, None, False)
                    if size == 0:
                        img = (img * 0.35).astype(np.uint8)           # человека здесь сейчас не видно
                    else:
                        cv2.rectangle(img, (0, 0), (tw - 1, th - 1), color, 3)
                    canvas[y:y + th, x:x + tw] = img
            else:
                seen = [ci for ci, x in enumerate(tiles) if x[3] > 0]
                if cur_cam is None or cur_cam not in seen:
                    if seen:
                        cur_cam = max(seen, key=lambda ci: tiles[ci][3])
                    elif cur_cam is None:
                        cur_cam = 0
                r, frame, fidx, size = tiles[cur_cam]
                if frame is not None:
                    img = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)
                    img = r.draw(img, fidx, tw / frame.shape[1], tr, False, False, None, False)
                    if size == 0:
                        img = (img * 0.35).astype(np.uint8)
                    canvas[bar:bar + th, :tw] = img
            cv2.rectangle(canvas, (0, 0), (W - 1, bar - 1), color, -1)
            where = ", ".join(visible) if visible else "не виден"
            txt = f"P{pid}   {fmt_time(t)}   виден на: {where}"
            tr_bar.put(canvas, txt, 8, max(0, (bar - tr_bar.size) // 2 - 2),
                       (0, 0, 0) if sum(color) > 450 else (255, 255, 255))
            writer.write(canvas)
            n += 1
    finally:
        writer.close()
    return n


def person_videos(data_dir: str | Path, out_dir: str | Path, cfg: Config, persons: list[int] | None = None,
                  min_present_s: float = 20.0, run_id: int | None = None, layout: str = "grid",
                  scale: float = 1.0) -> list[dict]:
    """Рисует по видео на каждого человека (по умолчанию — присутствует не меньше min_present_s)."""
    store = Store(data_dir)
    run_id, tids, present = person_table(store, run_id)
    if persons:
        chosen = [p for p in persons if p in tids]
    else:
        chosen = sorted((p for p in tids if present[p] >= min_present_s), key=lambda p: -present[p])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = []
    for p in tqdm(chosen, desc="люди", unit="чел"):
        out = out_dir / f"person_P{p}.mp4"
        n = render_person(store, p, tids[p], out, cfg, run_id, present[p], layout, scale)
        res.append({"person_id": p, "present_s": round(present[p], 1), "frames": n, "out": str(out)})
    return res
