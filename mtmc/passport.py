"""Паспорт камеры: pydantic-модель и валидация (раздел 2.2 ТЗ)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr, ValidationError, field_validator


class PassportError(ValueError):
    """Ошибка паспорта с именем поля."""


def frame_time(clock: list | None, frame_idx: float, fps: float, offset_s: float = 0.0) -> float:
    """Время кадра по карте часов; за её концами время идёт с обычной скоростью."""
    if not clock:
        return float(frame_idx) / fps + offset_s
    a = np.asarray(clock, dtype=np.float64)
    if frame_idx <= a[0, 0]:
        return float(a[0, 1] + (frame_idx - a[0, 0]) / fps) + offset_s
    if frame_idx >= a[-1, 0]:
        return float(a[-1, 1] + (frame_idx - a[-1, 0]) / fps) + offset_s
    return float(np.interp(frame_idx, a[:, 0], a[:, 1])) + offset_s


def _check_polygon(poly: Any, name: str = "polygon") -> list[list[float]]:
    if not isinstance(poly, (list, tuple)) or len(poly) < 3:
        raise ValueError("многоугольник должен содержать не менее 3 точек")
    out = []
    for p in poly:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError("каждая точка многоугольника — пара [x, y]")
        out.append([float(p[0]), float(p[1])])
    return out


def _check_matrix(m: Any, rows: int, cols: int) -> list[list[float]]:
    arr = np.asarray(m, dtype=float)
    if arr.shape != (rows, cols):
        raise ValueError(f"должна быть матрица {rows}×{cols}, получено {list(arr.shape)}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("матрица содержит не-числа")
    return arr.tolist()


class Zone(BaseModel):
    model_config = ConfigDict(extra="ignore")
    polygon: list[list[float]]
    leads_to: str | None = None

    @field_validator("polygon", mode="before")
    @classmethod
    def _poly(cls, v: Any) -> list[list[float]]:
        return _check_polygon(v)


class Passport(BaseModel):
    """Паспорт одной камеры. Обязательны только cam_id и video."""

    model_config = ConfigDict(extra="ignore")

    cam_id: int
    video: str
    name: str | None = None
    fps: float | None = None
    offset_s: float = 0.0
    H: list[list[float]] | None = None
    K: list[list[float]] | None = None
    dist: list[float] | None = None
    floor_roi: list[list[float]] | None = None
    zones: dict[str, Zone] | None = None
    overlaps: dict[int, list[list[float]]] | None = None
    ignore_zones: dict[str, list[list[float]]] | None = None    # экраны и зеркала: люди там не настоящие
    clock: list[list[float]] | None = None    # [[кадр, общее время, с], ...]: файл записан с неровной частотой

    _video_path: Path = PrivateAttr()
    _raw: dict = PrivateAttr(default_factory=dict)

    @field_validator("cam_id")
    @classmethod
    def _cam(cls, v: int) -> int:
        if v < 0 or v > 9000:
            raise ValueError("cam_id должен быть в диапазоне 0..9000")
        return v

    @field_validator("fps")
    @classmethod
    def _fps(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("fps должен быть > 0")
        return v

    @field_validator("H", mode="before")
    @classmethod
    def _h(cls, v: Any) -> Any:
        if v is None:
            return None
        m = _check_matrix(v, 3, 3)
        if abs(np.linalg.det(np.asarray(m))) < 1e-15:
            raise ValueError("гомография вырождена (det = 0)")
        return m

    @field_validator("K", mode="before")
    @classmethod
    def _k(cls, v: Any) -> Any:
        return None if v is None else _check_matrix(v, 3, 3)

    @field_validator("dist", mode="before")
    @classmethod
    def _dist(cls, v: Any) -> Any:
        if v is None:
            return None
        arr = np.asarray(v, dtype=float).ravel()
        if arr.size not in (1, 4, 5, 8, 12, 14):
            raise ValueError("dist: ожидается 4, 5, 8, 12 или 14 коэффициентов (формат OpenCV) "
                             "или [0] для широкого объектива (mtmc/lens.py)")
        return arr.tolist()

    @field_validator("floor_roi", mode="before")
    @classmethod
    def _roi(cls, v: Any) -> Any:
        return None if v is None else _check_polygon(v)

    @field_validator("ignore_zones", mode="before")
    @classmethod
    def _ignore(cls, v: Any) -> Any:
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("ignore_zones — словарь 'название' -> многоугольник")
        return {str(k): _check_polygon(p) for k, p in v.items()}

    @field_validator("overlaps", mode="before")
    @classmethod
    def _ovl(cls, v: Any) -> Any:
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("overlaps — словарь 'номер камеры' -> многоугольник")
        out = {}
        for k, poly in v.items():
            try:
                cam = int(k)
            except (TypeError, ValueError):
                raise ValueError(f"ключ '{k}' должен быть номером камеры") from None
            try:
                out[cam] = _check_polygon(poly)
            except ValueError as e:
                raise ValueError(f"камера {k}: {e}") from None
        return out

    # --- удобные представления ---
    @property
    def video_path(self) -> Path:
        return self._video_path

    @property
    def raw(self) -> dict:
        """Паспорт как был прочитан из файла."""
        return self._raw

    @property
    def has_floor(self) -> bool:
        return self.H is not None

    def H_np(self) -> np.ndarray | None:
        return None if self.H is None else np.asarray(self.H, dtype=np.float64)

    def K_np(self) -> np.ndarray | None:
        if self.K is None or self.dist is None:
            return None
        return np.asarray(self.K, dtype=np.float64)

    def dist_np(self) -> np.ndarray | None:
        if self.K is None or self.dist is None:
            return None
        return np.asarray(self.dist, dtype=np.float64)

    def floor_roi_np(self) -> np.ndarray | None:
        return None if self.floor_roi is None else np.asarray(self.floor_roi, dtype=np.float32)

    def zones_np(self) -> dict[str, np.ndarray]:
        if not self.zones:
            return {}
        return {k: np.asarray(z.polygon, dtype=np.float32) for k, z in self.zones.items()}

    def frame_time(self, frame_idx: float, fps: float) -> float:
        """Общее время кадра, с. Есть карта часов — по ней, иначе номер кадра / fps + сдвиг."""
        return frame_time(self.clock, frame_idx, fps, self.offset_s)

    def ignore_np(self) -> list[np.ndarray]:
        return [np.asarray(p, dtype=np.float32) for p in (self.ignore_zones or {}).values()]

    def overlaps_np(self) -> dict[int, np.ndarray]:
        if not self.overlaps:
            return {}
        return {k: np.asarray(p, dtype=np.float32) for k, p in self.overlaps.items()}


def _format_errors(e: ValidationError) -> str:
    lines = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err.get("loc", ())) or "(паспорт)"
        msg = err.get("msg", "")
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
        lines.append(f"поле '{loc}': {msg}")
    return "; ".join(lines)


def parse_passport(data: dict, base_dir: str | Path = ".") -> Passport:
    """Валидирует словарь паспорта. Ошибки — PassportError с именем поля."""
    if not isinstance(data, dict):
        raise PassportError("паспорт должен быть JSON-объектом")
    try:
        p = Passport.model_validate(data)
    except ValidationError as e:
        raise PassportError(_format_errors(e)) from None
    candidates = [Path(p.video)]
    if not Path(p.video).is_absolute():
        candidates.append(Path(base_dir) / p.video)
    for c in candidates:
        if c.exists():
            p._video_path = c.resolve()
            break
    else:
        raise PassportError(f"поле 'video': файл или папка не найдены: {p.video}")
    p._raw = data
    return p


def load_passport(path: str | Path) -> Passport:
    """Читает и проверяет паспорт; относительный video ищется от CWD, затем от папки паспорта."""
    path = Path(path)
    if not path.exists():
        raise PassportError(f"файл паспорта не найден: {path}")
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise PassportError(f"паспорт {path}: неверный JSON: {e}") from None
    return parse_passport(data, base_dir=path.parent)
