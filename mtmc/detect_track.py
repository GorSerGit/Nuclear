"""Обёртка над ultralytics YOLO11-pose + BoT-SORT."""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .config import DetectorCfg, TrackerCfg

log = logging.getLogger(__name__)


@dataclass
class Detection:
    local_id: int
    box: np.ndarray        # (4,) x1, y1, x2, y2
    conf: float
    kp: np.ndarray         # (17, 3) x, y, c


def resolve_device(device: str) -> str:
    """cuda недоступна -> cpu с предупреждением."""
    if str(device).startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                log.warning("CUDA недоступна, работаем на CPU (медленно)")
                return "cpu"
        except ImportError:
            return "cpu"
    return device


def tracker_yaml_dict(tcfg: TrackerCfg, fps: float, stride: int) -> dict:
    """Словарь для yaml трекера; track_buffer пересчитан из секунд в обработанные кадры."""
    d = {
        "tracker_type": tcfg.tracker_type,
        "track_high_thresh": tcfg.track_high_thresh,
        "track_low_thresh": tcfg.track_low_thresh,
        "new_track_thresh": tcfg.new_track_thresh,
        "track_buffer": max(1, int(round(tcfg.track_buffer_s * fps / stride))),
        "match_thresh": tcfg.match_thresh,
        "fuse_score": tcfg.fuse_score,
        "gmc_method": tcfg.gmc_method,
        "proximity_thresh": tcfg.proximity_thresh,
        "appearance_thresh": tcfg.appearance_thresh,
        "with_reid": tcfg.with_reid,
        "model": tcfg.model,
    }
    return d


def resolve_weights(weights: str) -> str:
    """Если файла нет — отдаём ultralytics голое имя, он скачает веса сам."""
    p = Path(weights)
    if p.exists():
        return str(p)
    log.warning("веса %s не найдены, ultralytics попробует скачать %s", weights, p.name)
    return p.name


class DetectorTracker:
    """Детектор со скелетом + трекер. step() вызывается на каждом обрабатываемом кадре, состояние сохраняется."""

    def __init__(self, cfg: DetectorCfg, tracker_cfg: TrackerCfg, fps: float, stride: int,
                 workdir: str | Path | None = None):
        from ultralytics import YOLO

        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.half = bool(cfg.half) and self.device != "cpu" and self.device != "mps"
        self.weights_path = resolve_weights(cfg.weights)
        self.model = YOLO(self.weights_path)
        self.tracker_dict = tracker_yaml_dict(tracker_cfg, fps, stride)
        wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="mtmc_trk_"))
        wd.mkdir(parents=True, exist_ok=True)
        self.tracker_yaml = wd / "tracker.yaml"
        with open(self.tracker_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.tracker_dict, f, allow_unicode=True)
        log.info("трекер: %s", self.tracker_dict)
        # ultralytics >= 8.4: half заменён на quantize; поддерживаем обе версии
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT
            new_api = "quantize" in DEFAULT_CFG_DICT
        except Exception:
            new_api = False
        self.precision = {"quantize": 16 if self.half else None} if new_api else {"half": self.half}
        self.new_track_thresh = tracker_cfg.track_high_thresh if tracker_cfg.new_track_thresh is None \
            else tracker_cfg.new_track_thresh
        self.candidates = np.zeros((0, 4))

    def step(self, frame_bgr: np.ndarray) -> list[Detection]:
        res = self.model.track(
            frame_bgr, persist=True, tracker=str(self.tracker_yaml), classes=[0],
            conf=self.cfg.conf, imgsz=self.cfg.imgsz, device=self.device,
            verbose=False, **self.precision,
        )[0]
        boxes = res.boxes
        self.candidates = np.zeros((0, 4))
        if boxes is None or len(boxes) == 0:
            return []
        if boxes.id is None:
            # Новый трек подтверждается на следующем кадре: человек уже есть, а id ещё нет. Слабее порога новых
            # треков трек не начнётся, и такие рамки (мебель, край шторы) пропуску кадров не мешают.
            keep = boxes.conf.float().cpu().numpy() >= self.new_track_thresh
            self.candidates = boxes.xyxy.float().cpu().numpy()[keep]
            return []
        ids = boxes.id.int().cpu().numpy()
        xyxy = boxes.xyxy.float().cpu().numpy()
        conf = boxes.conf.float().cpu().numpy()
        if res.keypoints is not None and res.keypoints.data is not None and len(res.keypoints.data):
            kps = res.keypoints.data.float().cpu().numpy()
            if kps.shape[-1] == 2:   # нет уверенностей
                kps = np.concatenate([kps, np.ones(kps.shape[:-1] + (1,))], -1)
        else:
            kps = np.zeros((len(ids), 17, 3), dtype=np.float32)
        return [Detection(int(ids[i]), xyxy[i].astype(np.float64), float(conf[i]), kps[i].astype(np.float64))
                for i in range(len(ids))]
