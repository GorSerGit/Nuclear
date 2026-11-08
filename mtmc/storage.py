"""SQLite + parquet + npz: схема, запись, чтение, атомарная замена данных камеры."""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .tracklets import EmbPack, TrackletRow

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
  cam_id        INTEGER PRIMARY KEY,
  name          TEXT,
  video_path    TEXT NOT NULL,
  fps           REAL NOT NULL,
  offset_s      REAL NOT NULL DEFAULT 0,
  width         INTEGER, height INTEGER,
  n_frames      INTEGER,
  passport_json TEXT NOT NULL,
  processed_at  TEXT
);
CREATE TABLE IF NOT EXISTS tracklets (
  tid           INTEGER PRIMARY KEY,
  cam_id        INTEGER NOT NULL REFERENCES cameras,
  local_id      INTEGER NOT NULL,
  t0 REAL NOT NULL, t1 REAL NOT NULL,
  frame0 INTEGER NOT NULL, frame1 INTEGER NOT NULL,
  n_obs         INTEGER NOT NULL,
  mean_q REAL, max_q REAL,
  is_weak       INTEGER NOT NULL DEFAULT 0,
  start_x REAL, start_y REAL, end_x REAL, end_y REAL,
  v_mean REAL, v0x REAL, v0y REAL, v1x REAL, v1y REAL,
  entry_zone TEXT, exit_zone TEXT,
  overlap_share_json TEXT,
  start_reason TEXT, end_reason TEXT,
  n_emb         INTEGER NOT NULL DEFAULT 0,
  emb_path      TEXT,
  thumb_path    TEXT
);
CREATE INDEX IF NOT EXISTS ix_tr_cam_time ON tracklets (cam_id, t0);
CREATE TABLE IF NOT EXISTS runs (run_id INTEGER PRIMARY KEY, created_at TEXT, params_json TEXT);
CREATE TABLE IF NOT EXISTS persons (
  run_id INTEGER, person_id INTEGER, t0 REAL, t1 REAL, thumb_path TEXT, confidence REAL,
  PRIMARY KEY (run_id, person_id));
CREATE TABLE IF NOT EXISTS tracklet_person (
  run_id INTEGER, tid INTEGER REFERENCES tracklets, person_id INTEGER,
  PRIMARY KEY (run_id, tid));
"""

OBS_COLUMNS = ["cam_id", "frame_idx", "t", "local_id", "tid", "x1", "y1", "x2", "y2", "conf", "kp",
               "u", "v", "foot_src", "x", "y", "sigma", "xs", "ys", "vx", "vy", "view", "q", "occ",
               "zone", "overlap_cams", "crop_path", "is_outlier"]

OBS_SCHEMA = pa.schema([
    ("cam_id", pa.int32()), ("frame_idx", pa.int64()), ("t", pa.float64()), ("local_id", pa.int64()),
    ("tid", pa.int64()),
    ("x1", pa.float32()), ("y1", pa.float32()), ("x2", pa.float32()), ("y2", pa.float32()),
    ("conf", pa.float32()), ("kp", pa.list_(pa.float32())),
    ("u", pa.float32()), ("v", pa.float32()), ("foot_src", pa.string()),
    ("x", pa.float64()), ("y", pa.float64()), ("sigma", pa.float64()),
    ("xs", pa.float64()), ("ys", pa.float64()), ("vx", pa.float64()), ("vy", pa.float64()),
    ("view", pa.string()), ("q", pa.float32()), ("occ", pa.float32()),
    ("zone", pa.string()), ("overlap_cams", pa.string()), ("crop_path", pa.string()),
    ("is_outlier", pa.bool_()),
])

TRACKLET_FIELDS = [f.name for f in dataclasses.fields(TrackletRow)]


@dataclass
class CameraRow:
    cam_id: int
    name: str | None
    video_path: str
    fps: float
    offset_s: float
    width: int | None
    height: int | None
    n_frames: int | None
    passport_json: str
    processed_at: str | None


def obs_to_table(df: pd.DataFrame) -> pa.Table:
    """DataFrame наблюдений -> таблица со строгой схемой (NaN -> NULL)."""
    d = df.copy()
    for c in OBS_COLUMNS:
        if c not in d:
            d[c] = None
    d = d[OBS_COLUMNS]
    arrays = []
    for f in OBS_SCHEMA:
        col = d[f.name]
        if f.name == "kp":
            arrays.append(pa.array([None if v is None else [float(x) for x in v] for v in col], type=f.type))
            continue
        if pa.types.is_string(f.type):
            vals = [v if isinstance(v, str) else None for v in col]
            arrays.append(pa.array(vals, type=f.type))
            continue
        if pa.types.is_boolean(f.type):
            arrays.append(pa.array(col.fillna(False).astype(bool).to_numpy(), type=f.type))
            continue
        num = pd.to_numeric(col, errors="coerce")
        mask = num.isna().to_numpy()
        if pa.types.is_integer(f.type):
            vals = num.fillna(0).astype("int64").to_numpy()
        else:
            vals = num.astype("float64").to_numpy()
        arrays.append(pa.array(vals, type=f.type, mask=mask))
    return pa.Table.from_arrays(arrays, schema=OBS_SCHEMA)


def save_npz(path: Path, pack: EmbPack) -> None:
    """Пишет эмбеддинги треклета (раздел 4.3)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "embs": pack.embs.astype(np.float16),
        "q": np.asarray(pack.q, dtype=np.float32),
        "view": np.asarray(pack.view, dtype=str),
        "frame_idx": np.asarray(pack.frame_idx, dtype=np.int64),
        "model": np.array(pack.model),
    }
    for k, v in pack.protos.items():
        data[f"proto_{k}"] = np.asarray(v, dtype=np.float32)
    with open(path, "wb") as f:
        np.savez(f, **data)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _rmtree(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p, ignore_errors=False)


class Store:
    """Папка данных: mtmc.sqlite, observations/, embeddings/, crops/, runs/."""

    def __init__(self, data_dir: str | Path):
        self.root = Path(data_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in ("observations", "embeddings", "crops", "runs", ".staging"):
            (self.root / sub).mkdir(exist_ok=True)
        self.db_path = self.root / "mtmc.sqlite"
        with self.connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=60, isolation_level=None)
        con.execute("PRAGMA busy_timeout=60000")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    # --- пути
    def obs_path(self, cam_id: int) -> Path:
        return self.root / "observations" / f"cam{cam_id}.parquet"

    def sil_path(self, cam_id: int) -> Path:
        """Силуэты (контуры фигур) наблюдений камеры — отдельно от наблюдений, ключ (frame_idx, local_id)."""
        return self.root / "observations" / f"cam{cam_id}.sil.parquet"

    def meta_path(self, cam_id: int) -> Path:
        return self.root / "observations" / f"cam{cam_id}.meta.json"

    def emb_dir(self, cam_id: int) -> Path:
        return self.root / "embeddings" / f"cam{cam_id}"

    def crops_dir(self, cam_id: int) -> Path:
        return self.root / "crops" / f"cam{cam_id}"

    def new_staging(self, cam_id: int) -> Path:
        p = self.root / ".staging" / f"cam{cam_id}_{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
        p.mkdir(parents=True, exist_ok=False)
        return p

    # --- запись
    def replace_camera(self, cam: CameraRow, obs: pd.DataFrame, tracklets: list[TrackletRow],
                       embs: dict[int, EmbPack], staging: Path | None = None,
                       meta: dict | None = None) -> None:
        """Заменяет данные камеры целиком. Сначала всё пишется в staging, затем подмена;
        при любой ошибке старые данные остаются на месте."""
        cid = cam.cam_id
        st = staging or self.new_staging(cid)
        st_obs = st / "observations.parquet"
        st_meta = st / "meta.json"
        st_emb = st / "embeddings"
        st_crops = st / "crops"
        st_crops.mkdir(exist_ok=True)
        st_emb.mkdir(exist_ok=True)
        try:
            # 1. файлы — в staging
            if "frame_idx" in obs:
                obs = obs.sort_values(["frame_idx", "local_id"], kind="stable")
            pq.write_table(obs_to_table(obs), st_obs)
            st_meta.write_text(json.dumps(meta or {}, ensure_ascii=False, indent=1), encoding="utf-8")
            for tid, pack in embs.items():
                save_npz(st_emb / f"{tid}.npz", pack)
            # 2. БД — в транзакции, пока без commit
            con = self.connect()
            try:
                con.execute("BEGIN IMMEDIATE")
                con.execute("DELETE FROM tracklet_person WHERE tid IN (SELECT tid FROM tracklets WHERE cam_id=?)",
                            (cid,))
                con.execute("DELETE FROM tracklets WHERE cam_id=?", (cid,))
                con.execute(
                    "INSERT OR REPLACE INTO cameras (cam_id, name, video_path, fps, offset_s, width, height, "
                    "n_frames, passport_json, processed_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (cid, cam.name, cam.video_path, cam.fps, cam.offset_s, cam.width, cam.height,
                     cam.n_frames, cam.passport_json, cam.processed_at))
                ph = ",".join("?" * len(TRACKLET_FIELDS))
                con.executemany(
                    f"INSERT INTO tracklets ({','.join(TRACKLET_FIELDS)}) VALUES ({ph})",
                    [tuple(getattr(r, f) for f in TRACKLET_FIELDS) for r in tracklets])
                # 3. подмена файлов
                self._swap_files(cid, st_obs, st_meta, st_emb, st_crops)
                try:
                    con.execute("COMMIT")
                except Exception:
                    self._restore(cid)
                    raise
            except Exception:
                try:
                    con.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            finally:
                con.close()
            self._drop_backups(cid)
        finally:
            # staging больше не нужен ни при успехе, ни при ошибке (старые данные не тронуты)
            shutil.rmtree(st, ignore_errors=True)

    def _targets(self, cid: int) -> list[Path]:
        return [self.obs_path(cid), self.meta_path(cid), self.emb_dir(cid), self.crops_dir(cid)]

    @staticmethod
    def _bak(p: Path) -> Path:
        return p.with_name(p.name + ".bak")

    def _swap_files(self, cid: int, st_obs: Path, st_meta: Path, st_emb: Path, st_crops: Path) -> None:
        targets = self._targets(cid)
        sources = [st_obs, st_meta, st_emb, st_crops / f"cam{cid}"]
        if not sources[3].exists():
            sources[3].mkdir(parents=True)
        for t in targets:   # старые -> .bak
            b = self._bak(t)
            if b.exists():
                _rmtree(b) if b.is_dir() else b.unlink()
            if t.exists():
                os.replace(t, b)
        try:
            for s, t in zip(sources, targets):
                t.parent.mkdir(parents=True, exist_ok=True)
                os.replace(s, t)
        except Exception:
            self._restore(cid)
            raise

    def _restore(self, cid: int) -> None:
        for t in self._targets(cid):
            b = self._bak(t)
            if b.exists():
                if t.exists():
                    _rmtree(t) if t.is_dir() else t.unlink()
                os.replace(b, t)

    def _drop_backups(self, cid: int) -> None:
        for t in self._targets(cid):
            b = self._bak(t)
            if b.exists():
                shutil.rmtree(b, ignore_errors=True) if b.is_dir() else b.unlink()

    # --- чтение
    def load_observations(self, cam_id: int, columns: list[str] | None = None) -> pd.DataFrame:
        p = self.obs_path(cam_id)
        if not p.exists():
            raise FileNotFoundError(f"нет наблюдений камеры {cam_id}: {p}")
        return pd.read_parquet(p, columns=columns)

    def load_meta(self, cam_id: int) -> dict:
        p = self.meta_path(cam_id)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def load_tracklets(self, cam_id: int | None = None) -> pd.DataFrame:
        with self.connect() as con:
            if cam_id is None:
                return pd.read_sql_query("SELECT * FROM tracklets ORDER BY cam_id, t0, tid", con)
            return pd.read_sql_query("SELECT * FROM tracklets WHERE cam_id=? ORDER BY t0, tid", con,
                                     params=(cam_id,))

    @staticmethod
    def _video_now(path: str, passport_json: str) -> str:
        """Абсолютный путь из БД; если папку проекта перенесли и его нет — путь из паспорта."""
        if Path(path).exists():
            return path
        p = json.loads(passport_json or "{}").get("video")
        return str(Path(p).resolve()) if p and Path(p).exists() else path

    def load_cameras(self) -> pd.DataFrame:
        with self.connect() as con:
            df = pd.read_sql_query("SELECT * FROM cameras ORDER BY cam_id", con)
        df["video_path"] = [self._video_now(v, p) for v, p in zip(df["video_path"], df["passport_json"])]
        return df

    def get_camera(self, cam_id: int) -> dict:
        with self.connect() as con:
            con.row_factory = sqlite3.Row
            r = con.execute("SELECT * FROM cameras WHERE cam_id=?", (cam_id,)).fetchone()
        if r is None:
            raise KeyError(f"камера {cam_id} не обработана (нет в БД)")
        d = dict(r)
        d["video_path"] = self._video_now(d["video_path"], d["passport_json"])
        return d

    def person_map(self, run_id: int | None = None) -> dict[int, int]:
        """tid -> person_id для run_id (по умолчанию последний)."""
        with self.connect() as con:
            if run_id is None:
                r = con.execute("SELECT MAX(run_id) FROM tracklet_person").fetchone()
                run_id = r[0] if r else None
                if run_id is None:
                    return {}
            rows = con.execute("SELECT tid, person_id FROM tracklet_person WHERE run_id=?", (run_id,)).fetchall()
        return {int(t): int(p) for t, p in rows}
