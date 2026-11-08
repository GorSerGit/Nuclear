"""Re-ID: интерфейс ReIDBackend и реализация на torchreid (OSNet)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import ReIDCfg

log = logging.getLogger(__name__)


class ReIDBackend(Protocol):
    name: str
    dim: int

    def embed(self, crops_rgb: list[np.ndarray]) -> np.ndarray:
        """(n, dim), строки L2-нормированы."""
        ...


def l2norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


class TorchReIDBackend:
    """OSNet через torchreid.utils.FeatureExtractor; flip-TTA: сумма прямого и отражённого, L2."""

    def __init__(self, cfg: ReIDCfg, device: str):
        try:
            from torchreid.utils import FeatureExtractor
        except ImportError:
            try:
                from torchreid.reid.utils import FeatureExtractor  # новые версии пакета
            except ImportError as e:
                raise ImportError("torchreid не установлен: pip install torchreid "
                                  "или из github.com/KaiyangZhou/deep-person-reid; "
                                  "либо reid.backend: none в конфиге") from e
        w = Path(cfg.weights)
        if not w.exists():
            raise FileNotFoundError(f"reid.weights: нет файла {w} (scripts/download_weights)")
        self.cfg = cfg
        self.extractor = FeatureExtractor(model_name=cfg.model_name, model_path=str(w), device=device,
                                          verbose=False)
        self.name = f"torchreid/{cfg.model_name}/{w.name}"
        self.dim = int(self.embed([np.zeros((256, 128, 3), np.uint8)]).shape[1])

    def _raw(self, imgs: list[np.ndarray]) -> np.ndarray:
        out = []
        for i in range(0, len(imgs), self.cfg.batch):
            f = self.extractor(imgs[i:i + self.cfg.batch])
            out.append(f.detach().float().cpu().numpy())
        return np.concatenate(out, 0)

    def embed(self, crops_rgb: list[np.ndarray]) -> np.ndarray:
        if not crops_rgb:
            return np.zeros((0, getattr(self, "dim", 512)), np.float32)
        f = self._raw(crops_rgb)
        if self.cfg.flip_tta:
            f = f + self._raw([np.ascontiguousarray(c[:, ::-1]) for c in crops_rgb])
        return l2norm(f).astype(np.float32)


class PersonViTBackend:
    """PersonViT ViT-S (MSMT17): прямая загрузка весов в timm, чужой код модели не выполняется. Вектор берётся после
    BatchNorm: на записи офиса он различает людей в тёмном намного лучше OSNet (AUC 0.94 против 0.79)."""

    def __init__(self, cfg: ReIDCfg, device: str):
        import timm
        import torch
        from safetensors.torch import load_file

        w = Path(cfg.weights)
        if not w.exists():
            raise FileNotFoundError(f"reid.weights: нет файла {w} (scripts/download_weights)")
        state = load_file(str(w))
        self.torch, self.cfg, self.device = torch, cfg, device
        self.model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0, img_size=(256, 128))
        self.model.load_state_dict({k[5:]: v for k, v in state.items() if k.startswith("base.") and ".fc." not in k
                                    and not k.startswith("base.fc")})
        self.neck = torch.nn.BatchNorm1d(384)
        self.neck.load_state_dict({k[11:]: v for k, v in state.items() if k.startswith("bottleneck.")})
        self.model.eval().to(device)
        self.neck.eval().to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
        self.name = f"personvit/{w.name}"
        self.dim = 384

    def embed(self, crops_rgb: list[np.ndarray]) -> np.ndarray:
        import cv2

        if not crops_rgb:
            return np.zeros((0, self.dim), np.float32)
        torch, out = self.torch, []
        for i in range(0, len(crops_rgb), self.cfg.batch):
            part = [cv2.resize(c, (128, 256), interpolation=cv2.INTER_CUBIC) for c in crops_rgb[i:i + self.cfg.batch]]
            b = torch.from_numpy(np.stack(part)).permute(0, 3, 1, 2).float().to(self.device) / 255
            b = (b - self.mean) / self.std
            with torch.no_grad():
                v = self.model(b)
                if self.cfg.flip_tta:
                    v = (v + self.model(b.flip(3))) / 2
                out.append(self.neck(v).float().cpu().numpy())
        return l2norm(np.concatenate(out, 0)).astype(np.float32)


def camera_debias(vectors: dict[int, np.ndarray], cam_of: dict[int, int], share: float) -> dict[int, np.ndarray]:
    """Убирает смещение камеры: свет и ракурс добавляют к векторам всех людей камеры общую часть."""
    if share <= 0:
        return vectors
    by_cam: dict[int, list[np.ndarray]] = {}
    for k, v in vectors.items():
        by_cam.setdefault(cam_of[k], []).append(v)
    mean = {c: np.mean(v, 0) for c, v in by_cam.items()}
    return {k: l2norm(v - share * mean[cam_of[k]]) for k, v in vectors.items()}


def clean_proto(z: dict, occ: dict[int, float] | None, limit: float) -> np.ndarray:
    """Прототип без вырезок, где человека заслоняет сосед: иначе в вектор попадает и второй человек."""
    e, q = z["embs"].astype(np.float64), z["q"].astype(np.float64)
    keep = np.array([(occ or {}).get(int(f), 0.0) <= limit for f in z["frame_idx"]])
    if limit <= 0 or keep.sum() < 2 or keep.all():
        return l2norm(z["proto_all"].astype(np.float64))
    return l2norm((e[keep] * np.maximum(q[keep], 1e-6)[:, None]).sum(0))


def scene_vectors(store, trdf, share: float, lam: float, occ_limit: float = 0.0) -> dict[int, np.ndarray]:
    """Прототипы треклетов под эту сцену. Сначала вычитается смещение камеры, затем подавляется разброс внутри
    треклетов: всё, что меняется у одного человека (поза, ракурс, свет), для сравнения людей только шум. Разметка не
    нужна: треклет сам служит классом. На вырезках офиса AUC 0.966 -> 0.991."""
    from .storage import load_npz

    protos, cam_of, parts, occ = {}, {}, [], {}
    if occ_limit > 0:
        for cid in trdf["cam_id"].unique():
            o = store.load_observations(int(cid), columns=["tid", "frame_idx", "occ"]).dropna(subset=["tid"])
            for tid, g in o.groupby("tid"):
                occ[int(tid)] = dict(zip(g["frame_idx"].astype(int), g["occ"].astype(float)))
    for r in trdf.itertuples():
        p = store.root / r.emb_path if isinstance(r.emb_path, str) and r.n_emb else None
        if p is None or not p.exists():
            continue
        z = load_npz(p)
        if z.get("proto_all") is None:
            continue
        protos[int(r.tid)] = clean_proto(z, occ.get(int(r.tid)), occ_limit)
        cam_of[int(r.tid)] = int(r.cam_id)
        e = z["embs"].astype(np.float64)
        if len(e) >= 2:
            parts.append(e - e.mean(0))
    out = camera_debias(protos, cam_of, share)
    if lam <= 0 or not parts:
        return out
    rest = np.concatenate(parts)
    w, v = np.linalg.eigh(rest.T @ rest / len(rest) + lam * np.eye(rest.shape[1]))
    white = (v * w ** -0.5) @ v.T
    return {k: l2norm(white @ x) for k, x in out.items()}


def make_reid(cfg: ReIDCfg, device: str = "cpu") -> ReIDBackend | None:
    """Создаёт бэкенд по cfg.backend. 'none' -> None (эмбеддинги не считаются, правило 3 не работает)."""
    backend = cfg.backend.lower()
    if backend in ("none", "off", ""):
        return None
    if backend == "torchreid":
        return TorchReIDBackend(cfg, cfg.device or device)
    if backend == "personvit":
        return PersonViTBackend(cfg, cfg.device or device)
    raise ValueError(f"reid.backend: неизвестный бэкенд '{cfg.backend}'")
