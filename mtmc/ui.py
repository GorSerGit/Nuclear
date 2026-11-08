"""Веб-интерфейс: синхронное видео всех камер, силуэты с ID сквозных треков, список треков, шкала времени, план.

build_ui собирает самодостаточную папку:
  index.html, app.js, style.css — из mtmc/ui_static
  videos/camN.mp4             — видео камер в H.264 (браузер не играет avi), опорный кадр каждые ui.gop кадров
  data.js                     — window.MTMC = {камеры, треки, рамки по кадрам, план}
Открывается без сервера (двойной щелчок по index.html) или через python -m mtmc serve --dir <папка>.
Нейросети не используются: всё берётся из БД и parquet.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import Config
from .draw import FFmpegWriter, find_ffmpeg
from .http_video import _range_handler
from .passport import frame_time
from .storage import Store
from .tracks import build_tracks, track_row
from .video_io import VideoSource

log = logging.getLogger(__name__)

STATIC = Path(__file__).with_name("ui_static")


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def _encoders(cfg: Config) -> list[list[str]]:
    """Кодировщики по порядку: видеокарта во много раз быстрее, но есть не у всех."""
    soft = ["-c:v", "libx264", "-preset", cfg.render.preset, "-crf", str(cfg.ui.crf)]
    try:
        import torch

        if cfg.detector.device != "cpu" and torch.cuda.is_available():
            return [["-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(cfg.ui.crf), "-bf", "0"], soft]
    except ImportError:
        pass
    return [soft]


def _direct(src: Path, dst: Path, w: int, h: int, fps: float, end: int | None, cfg: Config) -> bool:
    """Кодирует сам ffmpeg: кадры не ходят через Python, это в разы быстрее.
    Метки времени ставятся заново, чтобы кадр N стоял ровно на N/fps: по этому номеру интерфейс рисует рамки."""
    base = [find_ffmpeg(cfg.render.ffmpeg), "-y", "-loglevel", "error", "-i", os.path.relpath(src), "-an",
            "-vf", f"setpts=N/({fps:.6f}*TB),scale={w}:{h}:flags=area,format=yuv420p",
            "-r", f"{fps:.6f}", "-fps_mode", "cfr", "-g", str(cfg.ui.gop), "-keyint_min", str(cfg.ui.gop),
            "-movflags", "+faststart"]
    if end:
        base += ["-frames:v", str(end)]
    for enc in _encoders(cfg):
        r = subprocess.run([*base, *enc, os.path.relpath(dst)], capture_output=True)
        if r.returncode == 0:
            return True
        log.info("кодировщик %s не сработал: %s", enc[1], r.stderr.decode(errors="replace")[-200:])
    return False


def transcode(src: Path, dst: Path, fps: float, scale: float, cfg: Config,
              end: int | None = None) -> tuple[int, int, int]:
    """Перекодирует видео (или папку кадров) в mp4 для браузера. Повторно не делает, если не изменилось.
    end — конец обработанного отрезка: дальше рамок нет, и кодировать запись незачем."""
    stamp = dst.with_suffix(".json")
    st = src.stat()
    key = {"src": str(src), "mtime": st.st_mtime, "size": st.st_size, "scale": scale, "crf": cfg.ui.crf,
           "gop": cfg.ui.gop, "end": end}
    if dst.exists() and stamp.exists():
        old = json.loads(stamp.read_text(encoding="utf-8"))
        if old.get("key") == key:
            return old["w"], old["h"], old["n"]
    vs = VideoSource(src, fps=fps)
    w, h = _even(vs.width * scale), _even(vs.height * scale)
    t0 = time.perf_counter()
    n = 0
    if not vs.is_folder and _direct(src, dst, w, h, fps, end, cfg):
        cap = cv2.VideoCapture(os.path.relpath(dst))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        log.info("видео %s: %d кадров за %.0f с", dst.name, n, time.perf_counter() - t0)
    if n <= 0:         # ffmpeg мог завершиться успешно и не записать ни кадра: тогда кодируем покадрово

        wr = FFmpegWriter(dst, w, h, fps, crf=cfg.ui.crf, preset=cfg.render.preset, ffmpeg=cfg.render.ffmpeg,
                          gop=cfg.ui.gop)
        n = 0
        for _, fr in tqdm(vs.frames(0, end), total=end or vs.n_frames, desc=f"видео {dst.name}", unit="кадр",
                          leave=False):
            wr.write(cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA) if (fr.shape[1], fr.shape[0]) != (w, h)
                     else fr)
            n += 1
        wr.close()
    stamp.write_text(json.dumps({"key": key, "w": w, "h": h, "n": n}), encoding="utf-8")
    return w, h, n


def video_clock(clock: list | None, n: int, fps: float, offset: float) -> list[list[float]]:
    """[[время в видео, общее время], ...] для интерфейса: камеры с разным ходом часов играют синхронно."""
    frames = [0, *[int(f) for f, _ in clock or [] if 0 < f < n], n]
    return [[round(f / fps, 3), round(frame_time(clock, f, fps, offset), 3)] for f in frames]


def _boxes(obs: pd.DataFrame, tid2track: dict[int, int], s: float, kp_conf: float, skeleton: bool,
           sil: dict | None = None) -> dict:
    """{кадр: [track, x1, y1, x2, y2, kp 17×2 | 0, силуэт [x0, y0, …] | 0] …}; track 0 — шум/без трека.
    Если у наблюдения есть силуэт, поза не передаётся (силуэт важнее, а файл данных меньше)."""
    out: dict[int, list] = {}
    o = obs[obs["tid"].notna()]
    sil = sil or {}
    for r in o.itertuples(index=False):
        tr = tid2track.get(int(r.tid), 0)
        item = [tr, round(r.x1 * s, 1), round(r.y1 * s, 1), round(r.x2 * s, 1), round(r.y2 * s, 1), 0, 0]
        poly = sil.get((int(r.frame_idx), int(r.local_id)))
        if poly is not None and len(poly) >= 3:
            item[6] = np.round(poly * s).astype(int).ravel().tolist()
        elif skeleton and r.kp is not None and len(r.kp) == 51:
            k = np.asarray(r.kp, dtype=float).reshape(17, 3)
            item[5] = np.where(k[:, 2:3] > kp_conf, np.round(k[:, :2] * s), -1).astype(int).ravel().tolist()
        out.setdefault(int(r.frame_idx), []).append(item)
    for f, items in out.items():      # двойники одного трека: на кадре остаётся одна, самая большая рамка
        best: dict[int, list] = {}
        rest = []
        for it in items:
            if not it[0]:
                rest.append(it)
            elif it[0] not in best or (it[3] - it[1]) * (it[4] - it[2]) > \
                    (best[it[0]][3] - best[it[0]][1]) * (best[it[0]][4] - best[it[0]][2]):
                best[it[0]] = it
        out[f] = [*best.values(), *rest]
    return out


def _hull(xy: np.ndarray) -> list[list[float]]:
    """Наблюдаемый камерой участок пола: выпуклая оболочка ног (без 1% дальних выбросов)."""
    xy = xy[np.isfinite(xy).all(1)]
    if len(xy) < 10:
        return []
    med = np.median(xy, 0)
    d = np.hypot(*(xy - med).T)
    xy = xy[d <= np.percentile(d, 99)]
    h = cv2.convexHull(xy.astype(np.float32)).reshape(-1, 2)
    return [[round(float(x), 2), round(float(y), 2)] for x, y in h]


def _load_site(site: str | Path | None, out: Path) -> dict | None:
    """site.json: {"plan_image": "...", "px_per_m": 50, "cameras": {"1": {"pos": [x_px, y_px], "name": ".."}}}.
    Координаты плана — пиксели картинки; метры = пиксели / px_per_m (так же их даёт команда calibrate)."""
    if not site:
        return None
    p = Path(site)
    d = json.loads(p.read_text(encoding="utf-8"))
    ppm = float(d["px_per_m"])
    res: dict = {"px_per_m": ppm, "cameras": {}}
    if d.get("plan_image"):
        img = Path(d["plan_image"])
        img = img if img.is_absolute() or img.exists() else p.parent / img
        dst = out / ("plan" + img.suffix.lower())
        shutil.copyfile(img, dst)
        im = cv2.imdecode(np.fromfile(str(img), np.uint8), cv2.IMREAD_UNCHANGED)
        res.update(image=dst.name, size_m=[im.shape[1] / ppm, im.shape[0] / ppm])
    for k, c in (d.get("cameras") or {}).items():
        pos = c.get("pos")
        res["cameras"][int(k)] = {"pos": [pos[0] / ppm, pos[1] / ppm] if pos else None,
                                  "name": c.get("name"), "dir": c.get("dir")}
    return res


def build_ui(data_dir: str | Path, out: str | Path, cfg: Config, run_id: int | None = None,
             site: str | Path | None = None) -> dict:
    t_start = time.perf_counter()
    out = Path(out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    store = Store(data_dir)
    res = build_tracks(data_dir, cfg, run_id=run_id)
    tid2track = {tid: t.track_id for t in res["tracks"] for tid in t.tids}
    site_d = _load_site(site, out)
    cams_out, boxes = [], {}
    all_xy, starts = [], []
    for c in store.load_cameras().itertuples():
        cid = int(c.cam_id)
        passport = json.loads(c.passport_json)
        src = Path(c.video_path)
        scale = min(1.0, cfg.ui.max_height / float(c.height)) if c.height else 1.0
        meta = store.load_meta(cid)
        starts.append(frame_time(passport.get("clock"), meta.get("start_frame") or 0, float(c.fps),
                                 float(c.offset_s or 0)))
        w, h, n = transcode(src, out / "videos" / f"cam{cid}.mp4", float(c.fps), scale, cfg,
                            meta.get("end_frame"))
        s = w / float(c.width)
        obs = store.load_observations(cid, columns=["frame_idx", "local_id", "tid", "x1", "y1", "x2", "y2", "kp",
                                                    "xs", "ys"])
        from .silhouette import load_silhouettes

        sil = load_silhouettes(store, cid)
        if not sil:
            log.warning("cam%s: силуэтов нет (python -m mtmc silhouettes --data %s) — показываю рамку и позу",
                        cid, data_dir)
        boxes[cid] = _boxes(obs, tid2track, s, cfg.observation.kp_conf, cfg.ui.skeleton, sil)
        xy = obs[["xs", "ys"]].dropna().to_numpy(float)
        all_xy.append(xy)
        sc = dict((site_d or {}).get("cameras", {}).get(cid, {}))
        if not sc.get("pos") and passport.get("H") is not None:
            # положение камеры не задано на плане — берём из её гомографии
            from .geometry import camera_from_H

            cam = camera_from_H(np.asarray(passport["H"]), int(c.width), int(c.height))
            if cam and len(xy) and np.hypot(*(np.array(cam["pos"]) - np.median(xy, 0))) < 30:
                sc["pos"] = [round(cam["pos"][0], 2), round(cam["pos"][1], 2)]
                sc["pos_src"] = "по калибровке"
        cams_out.append({"id": cid, "name": c.name or f"cam{cid}", "short": f"cam{cid}",
                         "video": f"videos/cam{cid}.mp4", "fps": float(c.fps), "offset": float(c.offset_s or 0),
                         "clock": video_clock(passport.get("clock"), n, float(c.fps), float(c.offset_s or 0)),
                         "w": w, "h": h, "n_frames": n, "duration": n / float(c.fps),
                         "stride": int(meta.get("stride") or cfg.video.stride),
                         "hull": [[round(x, 2), round(y, 2)] for x, y in res["hulls"].get(cid, [])],
                         "fov": [[round(x, 2), round(y, 2)] for x, y in res["fovs"].get(cid, [])],
                         "pos": sc.get("pos"), "pos_src": sc.get("pos_src", "план"), "dir": sc.get("dir"),
                         "has_floor": passport.get("H") is not None})
    # границы плана
    if site_d and site_d.get("size_m"):
        bounds = [0.0, 0.0, *site_d["size_m"]]
    else:
        xy = np.concatenate([a for a in all_xy if len(a)]) if any(len(a) for a in all_xy) else np.zeros((1, 2))
        lo, hi = np.percentile(xy, 1, 0) - 1.0, np.percentile(xy, 99, 0) + 1.0
        for cam in cams_out:
            if cam["pos"]:
                lo, hi = np.minimum(lo, np.array(cam["pos"]) - 0.5), np.maximum(hi, np.array(cam["pos"]) + 0.5)
        bounds = [round(float(v), 2) for v in (*lo, *hi)]
    names = res["cam_names"]
    tracks = []
    for t in res["tracks"]:
        row = track_row(t, names)
        tracks.append({"id": t.track_id, "t0": round(t.t0, 3), "t1": round(t.t1, 3), "cams": t.cams,
                       "segs": [[c, round(a, 3), round(b, 3)] for c, a, b in t.segments],
                       "gaps": [[round(a, 3), round(b, 3), k] for (a, b), k in zip(t.gaps, t.gap_kinds)],
                       "person": t.person_no, "person_how": t.person_how, "prev": t.prev_track, "next": t.next_track,
                       "start_reason": t.start_reason, "end_reason": t.end_reason, "n_obs": t.n_obs,
                       "unseen": row["unseen_s"], "pts": [list(p) for p in t.pts]})
    from .ui_extras import extras

    try:
        ex = extras(store, [{"id": t.track_id, "tids": t.tids} for t in res["tracks"]], site, out, cfg.crops.pad)
    except Exception as e:  # noqa: BLE001 — дополнительные данные не должны ломать сборку просмотра
        log.warning("дополнительные данные просмотра не собраны: %s", e)
        ex = {"walls": [], "looks": {}}
    data = {
        "title": f"MTMC — {Path(data_dir).name}", "created": time.strftime("%Y-%m-%d %H:%M"),
        "run_id": res["run_id"], "n_tracks": res["n_tracks"], "n_noise": res["n_noise"],
        "n_people": res["n_people"],
        "params": {"end_gap_s": cfg.tracks.end_gap_s, "min_track_s": cfg.tracks.min_track_s},
        "record": res["record"], "window": [min(starts), max(c["clock"][-1][1] for c in cams_out)],
        "cams": cams_out, "tracks": tracks, "boxes": boxes,
        "plan": {"bounds": bounds, "image": (site_d or {}).get("image"), "map_step_s": cfg.tracks.map_step_s,
                 "walls": ex["walls"]},
        "looks": ex["looks"],
    }
    (out / "data.js").write_text("window.MTMC = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n",
                                 encoding="utf-8")
    for f in STATIC.iterdir():
        if f.is_file():
            shutil.copyfile(f, out / f.name)
    shutil.copyfile(Path(data_dir) / "tracks.csv", out / "tracks.csv")
    for name in ("tracks.json", "persons.csv", "persons.json", "rejected_tracks.csv", "report.md"):
        shutil.copyfile(Path(data_dir) / name, out / name)
    log.info("интерфейс собран за %.1f с", time.perf_counter() - t_start)
    return {"index": str(out / "index.html"), "n_tracks": res["n_tracks"]}


def serve(directory: str | Path, port: int = 8765, open_browser: bool = True) -> None:
    import functools
    import http.server
    import webbrowser

    handler = functools.partial(_range_handler(), directory=str(directory))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/index.html"
        print(f"интерфейс: {url}  (Ctrl+C — остановить)")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
