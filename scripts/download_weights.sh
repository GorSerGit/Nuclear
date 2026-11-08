#!/usr/bin/env bash
# Скачивает веса моделей в weights/ (YOLO11m-pose и OSNet x1.0 MSMT17).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p weights
if [ ! -f weights/yolo11m-pose.pt ]; then
  curl -L -o weights/yolo11m-pose.pt \
    https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt
fi
if [ ! -f weights/osnet_x1_0_msmt17.pth ]; then
  python -m pip show gdown >/dev/null 2>&1 || python -m pip install gdown
  python -m gdown 1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x -O weights/osnet_x1_0_msmt17.pth
fi
ls -la weights
