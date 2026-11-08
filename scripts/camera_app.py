"""Приложение для настройки камер: номер камеры, затем точки на полу и привязка к плану.

    python scripts/camera_app.py [--configs configs/office] [--videos data/office] [--port 8791] [--no-browser]

Порядок работы: у каждой записи выбирается номер камеры, задаётся план и масштаб, затем для камеры щёлкаются пары точек
на полу: в кадре и на плане. Сетка 1 м сразу показывает, легла ли привязка на пол. Сохраняется в configs/office/camN.json.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mtmc.calibrate import fit_h  # noqa: E402
from mtmc.lens import distort_points, undistort_points  # noqa: E402
from office import ensure_lens, find_videos, passport_files, probe  # noqa: E402
from site_tool import IMAGES, plan_response  # noqa: E402

PAGE = ROOT / "scripts" / "camera_app.html"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def passport_of_video(cfg_dir: Path, video: str) -> tuple[Path | None, dict | None]:
    for f in passport_files(cfg_dir):
        d = read_json(f)
        if Path(d["video"]).name == video:
            return f, d
    return None, None


def passport_of_cam(cfg_dir: Path, cam_id: int) -> tuple[Path, dict]:
    for f in passport_files(cfg_dir):
        d = read_json(f)
        if int(d["cam_id"]) == cam_id:
            return f, d
    raise ValueError(f"камере {cam_id} ещё не назначена запись")


def site_of(cfg_dir: Path) -> dict:
    p = cfg_dir / "site.json"
    return read_json(p) if p.exists() else {"cameras": {}}


def state(cfg_dir: Path, videos_dir: Path) -> dict:
    rows = []
    for v in find_videos(videos_dir) if videos_dir.exists() else []:
        _, d = passport_of_video(cfg_dir, v.name)
        d = d or {}
        rows.append({"video": v.name, "cam_id": d.get("cam_id"), "calibrated": d.get("H") is not None,
                     "method": d.get("calib_method"), "lens": bool(d.get("K")), "pairs": d.get("calib_points") or [],
                     "zones": len(d.get("ignore_zones") or {})})
    site = site_of(cfg_dir)
    plans = sorted(p.name for p in cfg_dir.iterdir() if p.suffix.lower() in IMAGES) if cfg_dir.exists() else []
    return {"videos": rows, "plans": plans, "plan_image": site.get("plan_image"), "px_per_m": site.get("px_per_m")}


def video_frame(videos_dir: Path, video: str, at_s: float) -> bytes:
    cap = cv2.VideoCapture(os.path.relpath(videos_dir / Path(video).name, ROOT))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(at_s, 0.0) * 1000)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"не читается запись {video} на {at_s:g} с")
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def assign(cfg_dir: Path, videos_dir: Path, video: str, cam_id: int) -> dict:
    """Назначает записи номер камеры. Смена номера переносит паспорт и положение камеры на плане."""
    if not 1 <= cam_id <= 9000:
        raise ValueError("номер камеры должен быть от 1 до 9000")
    path = videos_dir / Path(video).name
    if not path.exists():
        raise ValueError(f"нет записи {video}")
    f, d = passport_of_video(cfg_dir, path.name)
    for other in passport_files(cfg_dir):
        o = read_json(other)
        if int(o["cam_id"]) == cam_id and Path(o["video"]).name != path.name:
            raise ValueError(f"номер {cam_id} уже у записи {Path(o['video']).name}")
    target = cfg_dir / f"cam{cam_id}.json"
    if d is None:
        fps, _ = probe(path)
        d = {"cam_id": cam_id, "name": path.stem, "video": os.path.relpath(path, ROOT).replace("\\", "/"),
             "fps": round(fps, 3), "offset_s": 0.0}
        write_json(target, d)
        ensure_lens([target], {path.name: path})
        return read_json(target)
    old = int(d["cam_id"])
    if old != cam_id:
        d["cam_id"] = cam_id
        write_json(target, d)
        if f != target:
            f.unlink()
        site = site_of(cfg_dir)
        cams = site.get("cameras") or {}
        if str(old) in cams:
            cams[str(cam_id)] = cams.pop(str(old))
            write_json(cfg_dir / "site.json", site)
    return d


def set_plan(cfg_dir: Path, plan_image: str, px_per_m: float) -> dict:
    if not (cfg_dir / Path(plan_image).name).exists():
        raise ValueError("нет такой картинки плана")
    if float(px_per_m) <= 0:
        raise ValueError("масштаб должен быть больше нуля")
    site = site_of(cfg_dir)
    site["plan_image"], site["px_per_m"] = Path(plan_image).name, float(px_per_m)
    write_json(cfg_dir / "site.json", site)
    return {**site, "rescaled": rescale_cameras(cfg_dir, float(px_per_m))}


def rescale_cameras(cfg_dir: Path, ppm: float) -> int:
    """H переводит кадр в метры плана: с новым масштабом метры меняются, а точки в пикселях плана остаются верными."""
    n = 0
    for f in passport_files(cfg_dir):
        d = read_json(f)
        old = d.get("calib_px_per_m")
        if d.get("H") is None or not old or float(old) == ppm:
            continue
        k = float(old) / ppm
        d["H"] = (np.diag([k, k, 1.0]) @ np.asarray(d["H"], float)).tolist()
        d["calib_px_per_m"] = ppm
        write_json(f, d)
        n += 1
    return n


def line_px(cfg_dir: Path, cam_id: int, pairs: list, points: list) -> dict:
    """Длина линии, отмеченной на полу в кадре, в пикселях плана."""
    _, d = passport_of_cam(cfg_dir, cam_id)
    H, _, _ = solve(d, pairs, 1.0)
    K, dist = lens_of(d)
    uv = undistort_points(np.asarray(points, float).reshape(-1, 2), K, dist).reshape(-1, 1, 2)
    a, b = cv2.perspectiveTransform(uv, H).reshape(-1, 2)
    return {"px": round(float(np.hypot(*(a - b))), 2)}


def lens_of(d: dict) -> tuple:
    return (np.asarray(d["K"], float), np.asarray(d["dist"], float)) if d.get("K") and d.get("dist") else (None, None)


def solve(d: dict, pairs: list, ppm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """pairs [[u, v, X, Y]] в пикселях кадра и плана -> (H, ошибки по парам в метрах, точки на плане в метрах)."""
    arr = np.asarray(pairs, dtype=np.float64).reshape(-1, 4)
    if len(arr) < 4:
        raise ValueError(f"нужно не меньше 4 пар точек, есть {len(arr)}")
    arr[:, 2:] /= ppm
    K, dist = lens_of(d)
    arr[:, :2] = undistort_points(arr[:, :2], K, dist)
    H, err = fit_h(arr)
    return H, err, arr[:, 2:]


def grid_lines(H: np.ndarray, floor: np.ndarray, d: dict, margin_m: float = 1.0, n: int = 40) -> list[list[list[float]]]:
    """Сетка 1 м вокруг отмеченных точек в пикселях кадра: широкий объектив изгибает её вместе с полом."""
    Hi = np.linalg.inv(H)
    K, dist = lens_of(d)
    lo, hi = np.floor(floor.min(0) - margin_m), np.ceil(floor.max(0) + margin_m)
    sign = np.sign((Hi @ np.r_[floor.mean(0), 1.0])[2]) or 1.0
    out = []
    for axis in (0, 1):
        for c in np.arange(lo[axis], hi[axis] + 1):
            t = np.linspace(lo[1 - axis], hi[1 - axis], n)
            line = np.c_[np.full(n, c), t] if axis == 0 else np.c_[t, np.full(n, c)]
            ph = np.c_[line, np.ones(n)] @ Hi.T
            good = ph[:, 2] * sign > 1e-9
            uv = distort_points(ph[:, :2] / np.where(good, ph[:, 2], 1.0)[:, None], K, dist)
            piece: list[list[float]] = []
            for ok, p in zip(good, uv):
                if ok:
                    piece.append([round(float(p[0]), 1), round(float(p[1]), 1)])
                elif piece:
                    out.append(piece)
                    piece = []
            if len(piece) > 1:
                out.append(piece)
    return out


def preview(cfg_dir: Path, cam_id: int, pairs: list) -> dict:
    _, d = passport_of_cam(cfg_dir, cam_id)
    ppm = float(site_of(cfg_dir).get("px_per_m") or 0)
    if ppm <= 0:
        raise ValueError("сначала задайте масштаб плана")
    H, err, floor = solve(d, pairs, ppm)
    return {"errors_m": [round(float(e), 3) for e in err], "median_m": round(float(np.median(err)), 3),
            "max_m": round(float(err.max()), 3), "grid": grid_lines(H, floor, d)}


def save(cfg_dir: Path, cam_id: int, pairs: list) -> dict:
    f, d = passport_of_cam(cfg_dir, cam_id)
    ppm = float(site_of(cfg_dir).get("px_per_m") or 0)
    if ppm <= 0:
        raise ValueError("сначала задайте масштаб плана")
    H, err, _ = solve(d, pairs, ppm)
    d.update({"H": H.tolist(), "calib_points": np.asarray(pairs, float).tolist(), "calib_px_per_m": ppm,
              "calib_method": "manual"})
    d.pop("calib_auto", None)
    write_json(f, d)
    return {"path": str(f), "median_m": round(float(np.median(err)), 3), "max_m": round(float(err.max()), 3)}


def get_response(cfg_dir: Path, videos_dir: Path, path: str, query: dict) -> tuple[bytes, str] | None:
    if path in ("/", "/index.html"):
        return PAGE.read_bytes(), "text/html; charset=utf-8"
    if path == "/state":
        return json.dumps(state(cfg_dir, videos_dir), ensure_ascii=False).encode(), "application/json"
    if path == "/frame":
        frame = video_frame(videos_dir, query["video"][0], float(query.get("t", ["2"])[0]))
        return frame, "image/jpeg"
    if path.startswith("/plan/"):
        return plan_response(cfg_dir, path[6:])
    return None


def post_response(cfg_dir: Path, videos_dir: Path, path: str, body: dict) -> dict | None:
    if path == "/assign":
        return assign(cfg_dir, videos_dir, body["video"], int(body["cam_id"]))
    if path == "/plan":
        return set_plan(cfg_dir, body["plan_image"], float(body["px_per_m"]))
    if path == "/line":
        return line_px(cfg_dir, int(body["cam_id"]), body["pairs"], body["points"])
    if path == "/preview":
        return preview(cfg_dir, int(body["cam_id"]), body["pairs"])
    if path == "/save":
        return save(cfg_dir, int(body["cam_id"]), body["pairs"])
    return None


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

        def reply(self, data, code: int = 200) -> None:
            self.send(code, json.dumps(data, ensure_ascii=False).encode(), "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            try:
                response = get_response(cfg_dir, videos_dir, url.path, parse_qs(url.query))
                if response is None:
                    self.send_error(404)
                else:
                    self.send(200, *response)
            except (ValueError, KeyError) as e:
                self.reply({"error": str(e)}, 400)

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                response = post_response(cfg_dir, videos_dir, urlparse(self.path).path, json.loads(self.rfile.read(n)))
                if response is None:
                    self.send_error(404)
                else:
                    self.reply(response)
            except (ValueError, KeyError, TypeError) as e:
                self.reply({"error": str(e)}, 400)

    return H


def main() -> int:
    ap = argparse.ArgumentParser(description="Настройка камер: номер, точки, привязка к плану")
    ap.add_argument("--configs", default="configs/office")
    ap.add_argument("--videos", default="data/office")
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    cfg_dir, videos_dir = ROOT / a.configs, ROOT / a.videos
    cfg_dir.mkdir(parents=True, exist_ok=True)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(cfg_dir, videos_dir))
    url = f"http://127.0.0.1:{a.port}/"
    print(f"Настройка камер: {url}  (закройте окно, когда закончите)", flush=True)
    if not a.no_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
