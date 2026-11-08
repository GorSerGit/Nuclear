"""process(): проход А (кадры) + проход Б (треклеты, Re-ID, сводка) + запись."""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from . import __version__
from .config import Config
from .clothes import clothes_cuts, piece_series
from .crops import crop_rel_path, inner_crop, read_image, save_crop
from .gate import FrameGate, in_zone, zone_mask
from .geometry import floor_sigma, overlap_cams_of, point_in_polygon, to_floor, zone_of
from .observation import foot_point, laplacian_var, occluded_fractions, quality, view_of
from .passport import load_passport
from .storage import CameraRow, Store, obs_to_table
from .tracklets import (EmbPack, appearance_cuts, assign_tids, compute_prototypes, cut_tracks,
                        scene_cut_threshold, select_crops, smooth_and_velocity, split_by_times, summarize)
from .video_io import VideoSource

log = logging.getLogger(__name__)


class Timer:
    """Накопитель времени по этапам."""

    def __init__(self) -> None:
        self.t: dict[str, float] = defaultdict(float)

    def add(self, name: str, dt: float) -> None:
        self.t[name] += dt


def _file_info(path: str | Path | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"path": str(p), "size": p.stat().st_size, "sha256": h.hexdigest()}


def _versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0], "platform": platform.platform(), "mtmc": __version__}
    for mod in ("numpy", "pandas", "scipy", "pyarrow", "pydantic", "cv2", "torch", "ultralytics", "torchreid"):
        try:
            m = __import__(mod)
            out[mod] = str(getattr(m, "__version__", "?"))
        except Exception:
            out[mod] = "n/a"
    try:
        import torch
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return out


# ============================================================ проход А
def pass_a(passport, src: VideoSource, cfg: Config, staging: Path, start_frame: int, end_frame: int | None,
           timer: Timer, counters: Counter) -> tuple[Path, dict]:
    """Детекция, трекинг, наблюдения, кропы. Наблюдения порциями пишутся в staging/obs_raw.parquet."""
    from .detect_track import DetectorTracker

    t_init = time.perf_counter()
    stride = cfg.video.stride
    dtk = DetectorTracker(cfg.detector, cfg.tracker, src.fps, stride, workdir=staging)
    timer.add("model_init", time.perf_counter() - t_init)
    oc = cfg.observation
    H, K, dist = passport.H_np(), passport.K_np(), passport.dist_np()
    roi = passport.floor_roi_np()
    zones = passport.zones_np()
    overlaps = passport.overlaps_np()
    cid = passport.cam_id
    ignored = zone_mask(passport.ignore_np(), src.height, src.width)
    gate = FrameGate(cfg.gate, src.fps, cfg.tracker.track_buffer_s, src.height, src.width, ignored) \
        if cfg.gate.enabled else None
    raw_path = staging / "obs_raw.parquet"
    writer: pq.ParquetWriter | None = None
    buf: list[dict] = []
    crop_counter: dict[int, int] = defaultdict(int)
    local_ids: set[int] = set()
    last_frame = start_frame - 1

    def flush() -> None:
        nonlocal writer, buf
        if not buf:
            return
        table = obs_to_table(pd.DataFrame(buf))
        if writer is None:
            writer = pq.ParquetWriter(raw_path, table.schema)
        writer.write_table(table)
        buf = []

    total = src.count_range(start_frame, end_frame, stride)
    it = src.frames(start_frame, end_frame, stride)
    bar = tqdm(total=total, unit="кадр", desc=f"cam{cid}", dynamic_ncols=True, mininterval=1.0)
    t_loop0 = time.perf_counter()
    try:
        while True:
            t0 = time.perf_counter()
            try:
                frame_idx, frame = next(it)
            except StopIteration:
                break
            t1 = time.perf_counter()
            timer.add("decode", t1 - t0)
            counters["frames"] += 1
            last_frame = frame_idx
            bar.update(1)
            if gate is not None and not gate.want(frame_idx, frame):
                counters["frames_skipped"] += 1
                timer.add("gate", time.perf_counter() - t1)
                continue
            t1b = time.perf_counter()
            timer.add("gate", t1b - t1)
            dets = dtk.step(frame)
            t2 = time.perf_counter()
            timer.add("detect_track", t2 - t1b)
            n_all = len(dets)
            dets = [d for d in dets if not in_zone(ignored, d.box, oc.zone_share)]
            counters["dropped_zone"] += n_all - len(dets)
            if gate is not None:
                gate.seen(frame_idx, bool(dets) or any(not in_zone(ignored, b, oc.zone_share) for b in dtk.candidates))
            if not dets:
                continue
            fh, fw = frame.shape[:2]
            t = passport.frame_time(frame_idx, src.fps)
            boxes = np.stack([d.box for d in dets])
            occ = occluded_fractions(boxes)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t_crop = 0.0
            for i, d in enumerate(dets):
                counters["candidates"] += 1
                u, v, fsrc = foot_point(d.box, d.kp, oc.kp_conf, oc.foot_body_ratio)
                if roi is not None and not point_in_polygon(roi, u, v):
                    counters["dropped_floor_roi"] += 1
                    continue
                x = y = sigma = None
                if H is not None:
                    p = to_floor(H, [(u, v)], K, dist)[0]
                    x, y = float(p[0]), float(p[1])
                    sigma = floor_sigma(H, u, v, oc.sigma_px, oc.sigma_base_m,
                                        float(oc.sigma_mult.get(fsrc, 1.0)), K, dist)
                view = view_of(d.box, d.kp, oc.kp_conf, oc.view_side_thr)
                blur = laplacian_var(inner_crop(gray, d.box))
                q = quality(d.box, d.kp, d.conf, float(occ[i]), blur, fw, fh, oc.quality, oc.kp_conf)
                crop_path = None
                if q >= cfg.crops.min_q:
                    k = crop_counter[d.local_id]
                    crop_counter[d.local_id] += 1
                    if k % max(cfg.crops.every, 1) == 0:
                        tc = time.perf_counter()
                        rel = crop_rel_path(cid, d.local_id, frame_idx)
                        if save_crop(frame, d.box, cfg.crops.pad, cfg.crops.jpeg_quality, staging / rel):
                            crop_path = rel
                            counters["crops"] += 1
                        t_crop += time.perf_counter() - tc
                local_ids.add(d.local_id)
                buf.append({
                    "cam_id": cid, "frame_idx": frame_idx, "t": t, "local_id": d.local_id, "tid": None,
                    "x1": d.box[0], "y1": d.box[1], "x2": d.box[2], "y2": d.box[3], "conf": d.conf,
                    "kp": d.kp.reshape(-1).tolist(), "u": u, "v": v, "foot_src": fsrc,
                    "x": x, "y": y, "sigma": sigma, "view": view, "q": q, "occ": float(occ[i]),
                    "zone": zone_of(zones, u, v) if zones else None,
                    "overlap_cams": overlap_cams_of(overlaps, u, v) if overlaps else None,
                    "crop_path": crop_path, "is_outlier": False,
                })
                counters["observations"] += 1
            t3 = time.perf_counter()
            timer.add("crops", t_crop)
            timer.add("observation", t3 - t2 - t_crop)
            if len(buf) >= cfg.io.flush_rows:
                flush()
            bar.set_postfix(obs=counters["observations"], tracks=len(local_ids), refresh=False)
    finally:
        bar.close()
        flush()
        if writer is not None:
            writer.close()
    timer.add("pass_a_total", time.perf_counter() - t_loop0)
    counters["local_tracks"] = len(local_ids)
    info = {"detector_weights": dtk.weights_path, "tracker": dtk.tracker_dict, "device": dtk.device,
            "last_frame": last_frame}
    return raw_path, info


# ============================================================ проход Б
def _embed_pieces(pieces: list[pd.DataFrame], backend, cfg: Config, staging: Path,
                  timer: Timer) -> list[EmbPack | None]:
    """Отбор кропов и эмбеддинги батчами для всех кусков."""
    if backend is None:
        return [None] * len(pieces)
    t0 = time.perf_counter()
    jobs: list[tuple[int, pd.DataFrame]] = []
    for k, p in enumerate(pieces):
        sel = select_crops(p, cfg.reid.time_bins, cfg.reid.max_crops)
        if len(sel):
            jobs.append((k, sel))
    flat: list[tuple[int, Any]] = [(k, r) for k, sel in jobs for r in sel.itertuples()]
    feats: dict[int, list[tuple[Any, np.ndarray]]] = defaultdict(list)
    chunk = max(cfg.reid.batch, 1) * 4
    for i in tqdm(range(0, len(flat), chunk), desc="re-id", unit="батч", leave=False):
        part = flat[i:i + chunk]
        imgs, keep = [], []
        for k, r in part:
            img = read_image(staging / r.crop_path)
            if img is None or img.size == 0:
                log.warning("не читается кроп %s", r.crop_path)
                continue
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            keep.append((k, r))
        if not imgs:
            continue
        e = backend.embed(imgs)
        for (k, r), vec in zip(keep, e):
            feats[k].append((r, vec))
    out: list[EmbPack | None] = [None] * len(pieces)
    for k, lst in feats.items():
        lst.sort(key=lambda rv: (rv[0].t, rv[0].frame_idx))
        out[k] = EmbPack(
            embs=np.stack([v for _, v in lst]).astype(np.float32),
            q=np.array([r.q for r, _ in lst], dtype=np.float32),
            view=np.array([r.view for r, _ in lst], dtype=str),
            frame_idx=np.array([r.frame_idx for r, _ in lst], dtype=np.int64),
            t=np.array([r.t for r, _ in lst], dtype=np.float64),
            model=backend.name,
        )
    timer.add("reid", time.perf_counter() - t0)
    return out


def pass_b(obs: pd.DataFrame, passport, cfg: Config, staging: Path, backend, timer: Timer,
           counters: Counter, sil: dict | None = None) -> tuple[pd.DataFrame, list, dict[int, EmbPack]]:
    """Нарезка на треклеты (правила 1–3 и подмена одежды), сводки, итоговые наблюдения."""
    cid = passport.cam_id
    has_floor = passport.H is not None
    t0 = time.perf_counter()
    cut = cut_tracks(obs, cfg.tracklets, has_floor)
    counters["outliers"] = int(cut["is_outlier"].sum()) if len(cut) else 0
    kept = cut[cut["piece_id"] >= 0]
    raw_pieces = [g.sort_values("t") for _, g in kept.groupby("piece_id", sort=True)]
    for p in raw_pieces:
        sr = p["start_reason"].iloc[0]
        if sr in ("gap", "jump", "occlusion"):
            counters[f"cuts_{sr}"] += 1
    timer.add("cut_rules_1_2", time.perf_counter() - t0)

    packs = _embed_pieces(raw_pieces, backend, cfg, staging, timer)

    t1 = time.perf_counter()
    cut_cfg = replace(cfg.reid, app_cut_thr=scene_cut_threshold(raw_pieces, packs, cfg.reid))
    log.info("cam%s: порог разреза по внешности %.2f", cid, cut_cfg.app_cut_thr)
    final: list[tuple[pd.DataFrame, EmbPack | None, str, str]] = []
    for p, pack in zip(raw_pieces, packs):
        sr, er = p["start_reason"].iloc[0], p["end_reason"].iloc[0]
        cuts = appearance_cuts(pack.embs, pack.t, cut_cfg) if pack is not None and len(pack) else []
        if "crop_path" in p:
            found = clothes_cuts(*piece_series(p, staging, sil, cfg.crops.pad), cfg.reid)
            counters["cuts_clothes"] += len(found)
            cuts = sorted({*cuts, *found})
        if not cuts:
            final.append((p, pack, sr, er))
            continue
        parts = split_by_times(p, cuts)
        edges = [-np.inf] + cuts + [np.inf]
        counters["cuts_appearance"] += len(parts) - 1
        for j, part in enumerate(parts):
            a, b = float(part["t"].min()), float(part["t"].max())
            sub = None
            if pack is not None:
                m = (pack.t >= a) & (pack.t <= b)
                sub = pack.subset(m) if m.any() else None
            final.append((part, sub, sr if j == 0 else "appearance", er if j == len(parts) - 1 else "appearance"))
    tids = assign_tids([f[0] for f in final], cid)
    rows, embs = [], {}
    upd = []
    for (part, pack, sr, er), tid in zip(final, tids):
        sm = smooth_and_velocity(part, cfg.tracklets, has_floor)
        emb_path = None
        if pack is not None and len(pack):
            pack.protos = compute_prototypes(pack.embs, pack.q, pack.view, cfg.reid.proto_min)
            emb_path = f"embeddings/cam{cid}/{tid}.npz"
            embs[tid] = pack
        row = summarize(sm, pack, passport, cfg.tracklets, tid, sr, er, emb_path)
        rows.append(row)
        upd.append(pd.DataFrame({"tid": tid, "xs": sm["xs"], "ys": sm["ys"], "vx": sm["vx"], "vy": sm["vy"]},
                                index=sm.index))
    rows.sort(key=lambda r: r.tid)
    out = cut.drop(columns=["piece_id", "start_reason", "end_reason"])
    for c in ("tid", "xs", "ys", "vx", "vy"):
        out[c] = np.nan
    if upd:
        u = pd.concat(upd)
        out.loc[u.index, ["tid", "xs", "ys", "vx", "vy"]] = u[["tid", "xs", "ys", "vx", "vy"]].to_numpy()
    out["tid"] = out["tid"].astype("Int64")
    counters["tracklets"] = len(rows)
    counters["tracklets_weak"] = sum(r.is_weak for r in rows)
    counters["tracklets_with_emb"] = len(embs)
    timer.add("summarize", time.perf_counter() - t1)
    return out.sort_values(["frame_idx", "local_id"], kind="stable"), rows, embs


# ============================================================ process
def seed_everything(seed: int = 0) -> None:
    """Повторяемость: фиксированные генераторы случайных чисел и детерминированные свёртки cuDNN."""
    import os
    import random

    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def process(camera: str | Path, data_dir: str | Path, cfg: Config, start_s: float | None = None,
            end_s: float | None = None) -> dict:
    """Полная обработка одной камеры. Возвращает сводку запуска (она же пишется в runs/*.json)."""
    from .reid import make_reid

    seed_everything(0)
    t_all = time.perf_counter()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    passport = load_passport(camera)
    cid = passport.cam_id
    src = VideoSource(passport.video_path, passport.fps, cfg.video.exact_seek)
    fps, stride = src.fps, max(int(cfg.video.stride), 1)
    start_frame = int(round(start_s * fps)) if start_s else 0
    end_frame = int(round(end_s * fps)) if end_s is not None else None
    store = Store(data_dir)
    staging = store.new_staging(cid)
    timer, counters = Timer(), Counter()
    log.info("камера %s: %s, %dx%d, fps=%.3f, кадров=%s, stride=%d", cid, src.path, src.width, src.height,
             fps, src.n_frames, stride)
    try:
        raw_path, a_info = pass_a(passport, src, cfg, staging, start_frame, end_frame, timer, counters)
        obs = pd.read_parquet(raw_path) if raw_path.exists() else pd.DataFrame()
        if obs.empty:
            obs = obs_to_table(pd.DataFrame()).to_pandas()
        backend = None
        reid_weights = None
        if len(obs):
            tr = time.perf_counter()
            backend = make_reid(cfg.reid, a_info["device"])
            reid_weights = cfg.reid.weights if backend is not None else None
            timer.add("model_init", time.perf_counter() - tr)
        obs_final, rows, embs = pass_b(obs, passport, cfg, staging, backend, timer, counters)

        t_w = time.perf_counter()
        cam_row = CameraRow(
            cam_id=cid, name=passport.name, video_path=str(passport.video_path), fps=fps,
            offset_s=passport.offset_s, width=src.width, height=src.height, n_frames=src.n_frames,
            passport_json=json.dumps(passport.raw, ensure_ascii=False),
            processed_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        last = a_info["last_frame"] + 1
        if end_frame is not None:
            last = min(end_frame, src.n_frames) if src.n_frames else end_frame
        elif src.n_frames:
            last = src.n_frames
        meta = {"cam_id": cid, "stride": stride, "start_frame": start_frame,
                "end_frame": int(last), "fps": fps, "offset_s": passport.offset_s,
                "frames_processed": counters["frames"], "processed_at": cam_row.processed_at}
        store.replace_camera(cam_row, obs_final, rows, embs, staging=staging, meta=meta)
        timer.add("write", time.perf_counter() - t_w)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    sil = None
    if cfg.silhouette.enabled and len(obs):
        from .silhouette import silhouettes

        ts = time.perf_counter()
        sil = silhouettes(cid, data_dir, cfg)
        timer.add("silhouettes", time.perf_counter() - ts)
        # Яркость одежды точна только по силуэту, а он считается по записанным наблюдениям. Без этого прохода
        # средняя полоса кропа задевает фон: на офисе 35 ложных разрезов вместо 11, и люди рассыпаются на части.
        tr = time.perf_counter()
        retrack(cid, data_dir, cfg)
        timer.add("retrack_with_silhouettes", time.perf_counter() - tr)
    wall = time.perf_counter() - t_all
    tm = dict(timer.t)
    speeds = {"overall_fps": counters["frames"] / wall if wall else None}
    for k in ("decode", "detect_track", "observation", "crops"):
        if tm.get(k):
            speeds[f"{k}_fps"] = counters["frames"] / tm[k]
    if tm.get("pass_a_total"):
        speeds["pass_a_fps"] = counters["frames"] / tm["pass_a_total"]
    video_seconds = counters["frames"] * stride / fps
    speeds["realtime_factor"] = video_seconds / wall if wall else None
    summary = {
        "cam_id": cid, "camera": str(camera), "video": str(passport.video_path),
        "started_at": started, "wall_s": round(wall, 2),
        "range_frames": [start_frame, meta["end_frame"]], "fps": fps, "stride": stride,
        "counters": {
            "frames_processed": counters["frames"],
            "frames_skipped": counters["frames_skipped"],
            "observations": counters["observations"],
            "dropped_floor_roi": counters["dropped_floor_roi"],
            "dropped_zone": counters["dropped_zone"],
            "local_tracks": counters["local_tracks"],
            "tracklets": counters["tracklets"],
            "tracklets_weak": counters["tracklets_weak"],
            "tracklets_with_emb": counters["tracklets_with_emb"],
            "cuts_gap": counters["cuts_gap"],
            "cuts_jump": counters["cuts_jump"],
            "cuts_occlusion": counters["cuts_occlusion"],
            "cuts_appearance": counters["cuts_appearance"],
            "outliers": counters["outliers"],
            "crops": counters["crops"],
        },
        "timings_s": {k: round(v, 3) for k, v in tm.items()},
        "speed": {k: (round(v, 2) if v is not None else None) for k, v in speeds.items()},
        "device": a_info["device"],
        "versions": _versions(),
        "weights": {"detector": _file_info(a_info["detector_weights"]), "reid": _file_info(reid_weights)},
        "tracker_yaml": a_info["tracker"],
        "silhouettes": sil,
        "reid_model": backend.name if backend is not None else None,
        "config": cfg.to_dict(),
    }
    run_file = store.root / "runs" / f"cam{cid}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    run_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["run_file"] = str(run_file)
    return summary


def reproject(obs: pd.DataFrame, passport, oc) -> pd.DataFrame:
    """Пересчитывает положение на плане (x, y, sigma) из точки ног (u, v) по гомографии паспорта."""
    obs = obs.copy()
    H, K, dist = passport.H_np(), passport.K_np(), passport.dist_np()
    if H is None or obs.empty:
        obs[["x", "y", "sigma"]] = np.nan
        return obs
    uv = obs[["u", "v"]].to_numpy(np.float64)
    xy = to_floor(H, uv, K, dist)
    obs["x"], obs["y"] = xy[:, 0], xy[:, 1]
    obs["sigma"] = [floor_sigma(H, u, v, oc.sigma_px, oc.sigma_base_m, float(oc.sigma_mult.get(s, 1.0)), K, dist)
                    for u, v, s in zip(uv[:, 0], uv[:, 1], obs["foot_src"])]
    return obs


def retrack(cam_id: int, data_dir: str | Path, cfg: Config, passport_override: dict | None = None) -> dict:
    """Только проход Б: заново режет треки камеры из сохранённых наблюдений и кропов (без детекции).

    Нужен, чтобы быстро пробовать пороги нарезки. Паспорт берётся из БД (как был при process).
    passport_override — новый паспорт (например, с H от autocalib): положения на плане пересчитываются.
    """
    from .passport import parse_passport
    from .reid import make_reid
    from .silhouette import load_silhouettes

    t_all = time.perf_counter()
    store = Store(data_dir)
    cam = store.get_camera(cam_id)
    raw = passport_override if passport_override is not None else json.loads(cam["passport_json"])
    # видео ищем по абсолютному пути из БД: относительный путь паспорта зависит от папки запуска
    passport = parse_passport({**raw, "video": cam["video_path"]}, base_dir=".")
    passport._raw = raw
    if passport_override is not None:
        cam = {**cam, "passport_json": json.dumps(raw, ensure_ascii=False)}
    meta = store.load_meta(cam_id)
    obs = store.load_observations(cam_id)
    obs = obs.drop(columns=["tid", "xs", "ys", "vx", "vy"]).assign(is_outlier=False)
    if passport_override is not None:
        obs = reproject(obs, passport, cfg.observation)
    staging = store.new_staging(cam_id)
    timer, counters = Timer(), Counter()
    try:
        # кропы копируются в staging: replace_camera подменяет папку целиком
        shutil.copytree(store.crops_dir(cam_id), staging / "crops" / f"cam{cam_id}")
        backend = make_reid(cfg.reid, "cuda:0" if _cuda() else "cpu") if len(obs) else None
        obs_final, rows, embs = pass_b(obs, passport, cfg, staging, backend, timer, counters,
                                       load_silhouettes(store, cam_id))
        cam_row = CameraRow(**{k: cam[k] for k in ("cam_id", "name", "video_path", "fps", "offset_s", "width",
                                                   "height", "n_frames", "passport_json")},
                            processed_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        meta["retracked_at"] = cam_row.processed_at
        store.replace_camera(cam_row, obs_final, rows, embs, staging=staging, meta=meta)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"cam_id": cam_id, "wall_s": round(time.perf_counter() - t_all, 2), **dict(counters)}


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def format_summary(s: dict) -> str:
    c = s["counters"]
    sp = s["speed"]
    lines = [
        f"Камера {s['cam_id']}: обработано кадров {c['frames_processed']} за {s['wall_s']} с "
        f"({sp.get('overall_fps')} кадр/с, x{sp.get('realtime_factor')} от реального времени, {s['device']})",
        f"  пустых кадров без детектора: {c.get('frames_skipped', 0)}",
        f"  наблюдений: {c['observations']}  (отброшено по floor_roi: {c['dropped_floor_roi']}, "
        f"на экранах и в зеркалах: {c.get('dropped_zone', 0)})",
        f"  локальных треков: {c['local_tracks']}",
        f"  треклетов: {c['tracklets']}  (слабых: {c['tracklets_weak']}, с эмбеддингами: {c['tracklets_with_emb']})",
        f"  разрезов: пауза {c['cuts_gap']}, скачок {c['cuts_jump']}, перекрытие {c.get('cuts_occlusion', 0)}, "
        f"внешность {c['cuts_appearance']}; "
        f"выбросов: {c['outliers']}",
        f"  кропов: {c['crops']}",
        "  время по этапам, с: " + ", ".join(f"{k}={v}" for k, v in s["timings_s"].items()),
        f"  сводка запуска: {s.get('run_file')}",
    ]
    return "\n".join(lines)
