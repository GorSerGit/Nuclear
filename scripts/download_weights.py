"""Скачивает веса в weights/: YOLO11m-pose и YOLO11m-seg (ultralytics), OSNet x1.0 MSMT17 (torchreid model zoo).

python scripts/download_weights.py [--out weights]
"""
from __future__ import annotations

import argparse
import socket
import sys
import urllib.request
from pathlib import Path

YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}"
# torchreid MODEL_ZOO: osnet_x1_0, MSMT17 (combineall=True, лучше переносится на другие сцены)
OSNET_GDRIVE_ID = "1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x"
# PersonViT ViT-S, MSMT17 (Apache 2.0, arXiv:2408.05398): неофициальная конвертация весов в safetensors
PERSONVIT_URL = "https://huggingface.co/maennyn/personvit-reid-msmt17-vit-s/resolve/main/model.safetensors"


def fetch(url: str, dst: Path) -> None:
    print(f"-> {dst}  ({url})")
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        if tmp.stat().st_size <= 0:
            raise ValueError(f"получен пустой файл: {dst.name}")
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="weights")
    ap.add_argument("--yolo", default="yolo11m-pose.pt")
    ap.add_argument("--seg", default="yolo11m-seg.pt", help="модель силуэтов (маски людей)")
    a = ap.parse_args()
    socket.setdefaulttimeout(60)      # urlretrieve сам не ограничен: при обрыве связи ждал бы вечно
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in (a.yolo, a.seg):
        y = out / name
        if not y.is_file() or y.stat().st_size <= 0:
            fetch(YOLO_URL.format(name=name), y)
        else:
            print(f"есть {y}")
    p = out / "personvit_msmt17_s.safetensors"
    if not p.is_file() or p.stat().st_size <= 0:
        fetch(PERSONVIT_URL, p)
    else:
        print(f"есть {p}")
    o = out / "osnet_x1_0_msmt17.pth"
    if not o.is_file() or o.stat().st_size <= 0:
        try:
            import gdown
        except ImportError:
            print("pip install gdown  (нужен для Google Drive)", file=sys.stderr)
            return 1
        print(f"-> {o}  (Google Drive {OSNET_GDRIVE_ID})")
        tmp = o.with_suffix(o.suffix + ".part")
        try:
            downloaded = gdown.download(id=OSNET_GDRIVE_ID, output=str(tmp), quiet=False)
            if not downloaded or not tmp.is_file() or tmp.stat().st_size <= 0:
                raise ValueError(f"не удалось скачать {o.name}")
            tmp.replace(o)
        finally:
            tmp.unlink(missing_ok=True)
    else:
        print(f"есть {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
