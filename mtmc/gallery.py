"""Галерея треклетов: строка на треклет, подпись и до 8 кропов по времени (раздел 5.5)."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .crops import read_image
from .draw import GRAY, TextRenderer, id_color
from .render import fmt_time
from .storage import Store
from .tracklets import select_crops

log = logging.getLogger(__name__)


def _write_jpg(path: Path, img: np.ndarray, q: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise RuntimeError(f"не удалось закодировать {path}")
    path.write_bytes(buf.tobytes())


def gallery(cam_id: int, data_dir: str | Path, out: str | Path, cfg: Config) -> list[str]:
    """Собирает сетку кропов; при числе строк > rows_per_file — несколько файлов *_001.jpg, ..."""
    gc = cfg.gallery
    store = Store(data_dir)
    root = store.root
    tr_df = store.load_tracklets(cam_id)
    obs = store.load_observations(cam_id, columns=["tid", "t", "q", "frame_idx", "crop_path", "view"])
    obs = obs[obs["tid"].notna()]
    groups = {int(k): g for k, g in obs.groupby("tid")}
    text = TextRenderer(cfg.render.font, 15)
    ch = gc.crop_h
    cell_w = ch  # ширина ячейки под кроп (люди уже, чем высота)
    W = gc.label_w + gc.crops_per_row * (cell_w + 4) + 4
    rows = []
    for t in tr_df.itertuples():
        row = np.full((ch + 8, W, 3), 24, np.uint8)
        col = GRAY if t.is_weak else id_color(t.tid)
        cv2.rectangle(row, (0, 0), (6, ch + 7), col, -1)
        lines = [
            f"T{t.tid % 1_000_000}  (tid {t.tid}, local #{t.local_id})" + ("  СЛАБЫЙ" if t.is_weak else ""),
            f"{fmt_time(t.t0)} – {fmt_time(t.t1)}  ({t.t1 - t.t0:.1f} с)",
            f"кадры {t.frame0}–{t.frame1}, n={t.n_obs}, q̄={t.mean_q or 0:.2f}",
            f"начало: {t.start_reason}   конец: {t.end_reason}",
            f"зоны: {t.entry_zone or '-'} → {t.exit_zone or '-'}, эмб: {t.n_emb}",
        ]
        y = 6
        for ln in lines:
            _, h = text.put(row, ln, 12, y, (235, 235, 235), pad=1)
            y += h + 3
        g = groups.get(int(t.tid))
        x = gc.label_w
        if g is not None:
            sel = select_crops(g, gc.crops_per_row, gc.crops_per_row)
            for c in sel.itertuples():
                img = read_image(root / c.crop_path)
                if img is None:
                    continue
                h0, w0 = img.shape[:2]
                w1 = max(1, min(cell_w, int(round(w0 * ch / h0))))
                img = cv2.resize(img, (w1, ch), interpolation=cv2.INTER_AREA)
                row[4:4 + ch, x:x + w1] = img
                x += w1 + 4
        rows.append(row)
    out = Path(out)
    if not rows:
        log.warning("у камеры %s нет треклетов", cam_id)
        return []
    chunks = [rows[i:i + gc.rows_per_file] for i in range(0, len(rows), gc.rows_per_file)]
    paths = []
    for k, ch_rows in enumerate(chunks, start=1):
        img = np.vstack(ch_rows)
        p = out if len(chunks) == 1 else out.with_name(f"{out.stem}_{k:03d}{out.suffix or '.jpg'}")
        _write_jpg(p, img, gc.jpeg_quality)
        paths.append(str(p))
    return paths
