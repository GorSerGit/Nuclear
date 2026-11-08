"""Разметка плана в браузере: масштаб, положение камер, направление. Сохраняет configs/office/site.json.

    python scripts/site_tool.py [--configs configs/office] [--videos data/office] [--port 8790] [--no-browser]

План — картинка (png, jpg) в папке configs/office. PDF сначала сохраните как картинку.
"""
from __future__ import annotations

import argparse
import base64
import http.server
import json
import os
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

PAGE = (ROOT / "scripts" / "site_tool.html")
IMAGES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
PNG_HEADER = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"


def first_frame(video: Path, at_s: float = 2.0) -> str | None:
    """Кадр камеры в base64: чтобы понять, какая камера где на плане."""
    cap = cv2.VideoCapture(os.path.relpath(video, ROOT))
    cap.set(cv2.CAP_PROP_POS_MSEC, at_s * 1000)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    k = 240 / max(img.shape[1], 1)
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, None, fx=k, fy=k), [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode() if ok else None


def state(cfg_dir: Path, videos_dir: Path) -> dict:
    from office import ensure_passports, find_videos

    videos = find_videos(videos_dir) if videos_dir.exists() else []
    files = ensure_passports(videos, cfg_dir) if videos else []
    by_name = {v.name: v for v in videos}
    site_path = cfg_dir / "site.json"
    site = json.loads(site_path.read_text(encoding="utf-8")) if site_path.exists() else {}
    cams = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        v = by_name.get(Path(d["video"]).name)
        cams.append({"id": int(d["cam_id"]), "name": d.get("name") or f.stem, "frame": first_frame(v) if v else None,
                     "zones": d.get("ignore_zones") or {}})
    plans = sorted(p.name for p in cfg_dir.iterdir() if p.suffix.lower() in IMAGES) if cfg_dir.exists() else []
    return {"cams": cams, "plans": plans, "site": site, "marks": {p: plan_marks(cfg_dir / p) for p in plans}}


def passport_of(cfg_dir: Path, cam_id: int) -> tuple[Path, dict]:
    from office import passport_files

    for f in passport_files(cfg_dir):
        d = json.loads(f.read_text(encoding="utf-8"))
        if int(d["cam_id"]) == cam_id:
            return f, d
    raise ValueError(f"нет паспорта камеры {cam_id}")


def full_frame(cfg_dir: Path, videos_dir: Path, cam_id: int, at_s: float = 2.0) -> bytes:
    """Кадр в исходном размере: координаты зон — пиксели кадра."""
    _, d = passport_of(cfg_dir, cam_id)
    cap = cv2.VideoCapture(os.path.relpath(videos_dir / Path(d["video"]).name, ROOT))
    cap.set(cv2.CAP_PROP_POS_MSEC, at_s * 1000)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"не читается видео камеры {cam_id}")
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def save_zones(cfg_dir: Path, cam_id: int, zones: dict) -> Path:
    """Экраны и зеркала камеры. Пустой словарь убирает поле."""
    f, d = passport_of(cfg_dir, cam_id)
    if any(len(p) < 3 for p in zones.values()):
        raise ValueError("в зоне должно быть не меньше 3 точек")
    d.pop("ignore_zones", None)
    if zones:
        d["ignore_zones"] = zones
    f.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    return f


def plan_marks(path: Path, *, uploaded: bool = False) -> list[dict]:
    """Камеры, нарисованные на плане красными секторами. Ошибка чтения — не повод останавливать разметку."""
    from mtmc.planfile import find_camera_marks, read_plan

    if uploaded:
        with path.open("rb") as file:
            header = file.read(24)
        if len(header) != 24 or not header.startswith(PNG_HEADER):
            raise ValueError("Выберите файл в формате PNG.")
        width = int.from_bytes(header[16:20], "big")
        height = int.from_bytes(header[20:24], "big")
        if not 0 < width * height <= 32_000_000:
            raise ValueError("Размер изображения должен быть не больше 32 мегапикселей.")
    try:
        image = read_plan(path)
    except (ValueError, cv2.error) as e:
        if uploaded:
            raise ValueError("PNG повреждён или не читается.") from e
        return []
    try:
        return [{"pos": list(m.pos), "look_at": list(m.look_at), "num": m.num} for m in find_camera_marks(image)]
    except (ValueError, cv2.error):
        return []


def upload_plan(cfg_dir: Path, name: str, length: int, source) -> dict:
    from mtmc.http_video import upload_media

    if length > 32 * 1024 * 1024:
        raise ValueError("PNG должен занимать не больше 32 МБ.")
    marks = upload_media(cfg_dir, name, length, source, (".png",), "плана",
                         validate=lambda path: plan_marks(path, uploaded=True))
    return {"name": name, "marks": marks}


def save_site(cfg_dir: Path, d: dict, require_all: bool = True) -> Path:
    """Проверяет то, без чего дальше не пойдёт, и пишет site.json."""
    image = d.get("plan_image")
    if not isinstance(image, str) or Path(image).name != image or not (cfg_dir / image).is_file():
        raise ValueError("не выбрана картинка плана")
    if not d.get("px_per_m") or float(d["px_per_m"]) <= 0:
        raise ValueError("не задан масштаб")
    from office import passport_files

    cams = d.get("cameras") or {}
    need = [str(json.loads(f.read_text(encoding="utf-8"))["cam_id"])
            for f in passport_files(cfg_dir)] if require_all else []
    need = need or list(cams)
    miss = [c for c in need if "pos" not in cams.get(c, {})]
    if miss or not cams:
        raise ValueError("не отмечено положение камер: " + (", ".join(f"cam{c}" for c in miss) or "всех"))
    path = cfg_dir / "site.json"
    path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def plan_response(cfg_dir: Path, name: str) -> tuple[bytes, str] | None:
    path = cfg_dir / Path(unquote(name)).name
    suffix = path.suffix.lower()
    if suffix in IMAGES and path.is_file() and path.resolve().parent == cfg_dir.resolve():
        return path.read_bytes(), "image/" + suffix.lstrip(".").replace("jpg", "jpeg")
    return None


def get_response(cfg_dir: Path, videos_dir: Path, path: str) -> tuple[bytes, str] | None:
    if path in ("/", "/index.html"):
        return PAGE.read_bytes(), "text/html; charset=utf-8"
    if path == "/state":
        return json.dumps(state(cfg_dir, videos_dir), ensure_ascii=False).encode(), "application/json"
    if path.startswith("/plan/"):
        return plan_response(cfg_dir, path[6:])
    if path.startswith("/frame/"):
        return full_frame(cfg_dir, videos_dir, int(path[7:])), "image/jpeg"
    return None


def post_response(cfg_dir: Path, path: str, body: dict, require_all: bool = True) -> dict | None:
    if path == "/zones":
        saved = save_zones(cfg_dir, int(body["cam"]), body["zones"])
    elif path == "/save":
        saved = save_site(cfg_dir, body, require_all)
    else:
        return None
    return {"path": str(saved)}


def make_handler(cfg_dir: Path, videos_dir: Path):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                response = get_response(cfg_dir, videos_dir, urlsplit(self.path).path)
                if response is None:
                    self.send_error(404)
                else:
                    self.send(200, *response)
            except ValueError as e:
                self.send(404, str(e).encode(), "text/plain; charset=utf-8")

        def do_POST(self):
            try:
                url = urlsplit(self.path)
                n = int(self.headers.get("Content-Length") or 0)
                if url.path == "/upload":
                    host = self.headers.get("Host")
                    port = self.server.server_port
                    origin = self.headers.get("Origin")
                    if (host not in (f"127.0.0.1:{port}", f"localhost:{port}")
                            or (origin and origin != f"http://{host}")
                            or self.headers.get("Sec-Fetch-Site") == "cross-site"):
                        self.send_error(403)
                        return
                    if self.headers.get_content_type() != "application/octet-stream":
                        self.send(415, json.dumps({"error": "Ожидается PNG-файл."}).encode(), "application/json")
                        return
                    self.connection.settimeout(60)
                    name = parse_qs(url.query).get("name", [""])[0]
                    result = upload_plan(cfg_dir, name, n, self.rfile)
                    self.send(201, json.dumps(result, ensure_ascii=False).encode(), "application/json")
                    return
                body = json.loads(self.rfile.read(n))
                response = post_response(cfg_dir, url.path, body)
                if response is None:
                    self.send_error(404)
                else:
                    self.send(200, json.dumps(response, ensure_ascii=False).encode(), "application/json")
            except FileExistsError as e:
                self.send(409, json.dumps({"error": str(e)}, ensure_ascii=False).encode(), "application/json")
            except (ValueError, KeyError, TypeError, OSError) as e:
                self.send(400, json.dumps({"error": str(e)}, ensure_ascii=False).encode(), "application/json")

    return H


def main() -> int:
    ap = argparse.ArgumentParser(description="Разметка плана: масштаб и камеры -> site.json")
    ap.add_argument("--configs", default="configs/office")
    ap.add_argument("--videos", default="data/office")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    cfg_dir, videos_dir = ROOT / a.configs, ROOT / a.videos
    cfg_dir.mkdir(parents=True, exist_ok=True)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(cfg_dir, videos_dir))
    url = f"http://127.0.0.1:{a.port}/"
    print(f"Разметка плана: {url}  (закройте окно, когда сохраните)", flush=True)
    if not a.no_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
