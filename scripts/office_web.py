"""Локальный интерфейс разметки, запуска и просмотра офисных записей."""
from __future__ import annotations

import argparse
import base64
import hashlib
import http.server
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import webbrowser
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import office
from office_dependencies import MODES, Dependencies, package_versions
from office_platform import open_yaml
from mtmc.http_video import _range_handler, upload_media

ROOT = office.ROOT
STATIC = ROOT / "scripts/office_web_static"
HISTORY = ROOT / "results_office_generations"
DEFAULTS = {"videos": "data/office", "configs": "configs/office", "data": "results_office", "ui": "ui_office"}
LEGACY = {
    "legacy-full": ("Вся запись", ROOT / "ui_office", ROOT / "results_office"),
    "legacy-trial": ("Отрезок записи", ROOT / "ui_office_trial", ROOT / "results_office_trial"),
}
OUTPUTS = (HISTORY, *(folder for _, ui, data in LEGACY.values() for folder in (ui, data)))
VIEWER = ROOT / "mtmc/ui_static"
VIEWER_ASSETS = ("index.html", "app.js", "style.css", "features.js")
PLAN_IMAGES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class BusyError(ValueError):
    pass


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def source_dir(options: dict, name: str) -> Path:
    value = options.get(name) or DEFAULTS[name]
    if not isinstance(value, str):
        raise ValueError("Путь к папке должен быть строкой.")
    path = (ROOT / value.strip()).resolve()
    if (not path.is_relative_to(ROOT) or any(path.is_relative_to(folder) for folder in OUTPUTS)
            or not path.is_dir()):
        raise ValueError(f"Выберите существующую папку внутри проекта: {value}")
    return path


def source_video(folder: Path, name: str) -> Path:
    path = (folder / name).resolve()
    if Path(name).name != name or path.parent != folder or path.suffix.lower() not in office.VIDEO_EXT or not path.is_file():
        raise FileNotFoundError("Запись не найдена.")
    return path


def config_paths(configs: Path) -> tuple[Path, Path]:
    paths = tuple(p.resolve() for p in office.config_sources(configs))
    if any(not p.is_relative_to(ROOT) or p.suffix.lower() not in (".yaml", ".yml") for p in paths):
        raise ValueError("Конфиг должен находиться внутри проекта.")
    return paths


def output_name(value: str, default: str) -> str:
    name = value or default
    is_reserved = getattr(os.path, "isreserved", lambda p: Path(p).is_reserved())
    if (not isinstance(name, str) or not re.fullmatch(r"[\w-]{1,60}", name)
            or name.casefold() == "inputs" or is_reserved(name)):
        raise ValueError("Имя папки результата может содержать только буквы, цифры, дефис и подчёркивание.")
    return name


def generation_options(body: dict) -> tuple[dict, list[tuple[Path, Path, dict]]]:
    videos, configs = (source_dir(body, k) for k in ("videos", "configs"))
    entries = office.camera_inputs(office.find_videos(videos), configs)
    by_cam = {int(d["cam_id"]): (v, p, d) for v, p, d in entries}
    if len(by_cam) != len(entries):
        raise ValueError("В паспортах повторяются номера камер.")
    cameras = body.get("cameras", list(by_cam))
    if not isinstance(cameras, list) or not cameras or any(type(c) is not int or c not in by_cam for c in cameras):
        raise ValueError("Выберите хотя бы одну камеру из списка записей.")
    cameras = sorted(set(cameras))
    chosen = [by_cam[c] for c in cameras]
    for v, _, _ in chosen:
        source_video(videos, v.name)

    start, end = body.get("start"), body.get("end")
    for value in (start, end):
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError("Границы отрезка должны быть неотрицательными числами.")
    if end is not None and end <= (start or 0):
        raise ValueError("Конец отрезка должен быть позже начала.")
    quality = body.get("quality", "auto")
    if quality not in ("auto", "fast", "high"):
        raise ValueError("Неизвестный режим качества.")
    device = body.get("device", "auto")
    if device not in MODES:
        raise ValueError("Неизвестное устройство.")
    silhouettes, fresh = body.get("silhouettes", True), body.get("fresh", False)
    if type(silhouettes) is not bool or type(fresh) is not bool:
        raise ValueError("Параметры силуэтов и пересчёта должны быть переключателями.")
    data, ui = (output_name(body.get(k), DEFAULTS[k]) for k in ("data", "ui"))
    if data.casefold() == ui.casefold():
        raise ValueError("Укажите разные имена папок для данных и просмотра.")
    return {
        "videos": videos.relative_to(ROOT).as_posix(), "configs": configs.relative_to(ROOT).as_posix(),
        "cameras": cameras, "start": start, "end": end, "quality": quality, "device": device,
        "silhouettes": silhouettes, "fresh": fresh, "data": data, "ui": ui,
    }, chosen


def snapshot_inputs(folder: Path, configs: Path, entries: list[tuple[Path, Path, dict]],
                    yaml_files: tuple[bytes, bytes | None]) -> None:
    folder.mkdir()
    base, extra = yaml_files
    (folder / "config.yaml").write_bytes(base)
    # Пустой снимок не даёт конвейеру подхватить будущий файл переопределений.
    (folder / "overrides.yaml").write_bytes(extra if extra is not None else b"{}\n")
    for video, _, passport in entries:
        # Папка видео из формы имеет приоритет перед старым путём в паспорте.
        write_json(folder / f"cam{int(passport['cam_id'])}.json",
                   {**passport, "video": video.relative_to(ROOT).as_posix()})
    site = configs / "site.json"
    if site.is_file():
        plan = read_json(site)
        image = plan.get("plan_image")
        if not isinstance(image, str) or not image:
            raise ValueError("В site.json не выбрана картинка плана.")
        source = (configs / image).resolve()
        if not source.is_file():
            source = (ROOT / image).resolve()
        if not source.is_file() or not source.is_relative_to(ROOT) or source.suffix.lower() not in PLAN_IMAGES:
            raise ValueError("Картинка плана из site.json не найдена в проекте.")
        shutil.copyfile(source, folder / source.name)
        write_json(folder / "site.json", {**plan, "plan_image": source.name})


def cache_key(inputs: Path, options: dict, videos: list[Path]) -> str:
    """Не переносим кэш при смене кода, входных данных или настроек."""
    key = hashlib.sha256()
    settings = {k: options[k] for k in ("videos", "cameras", "start", "end", "quality", "silhouettes",
                                      "actual_device", "gpu_name")}
    key.update(json.dumps(settings, sort_keys=True).encode())
    key.update(json.dumps(package_versions(), sort_keys=True).encode())
    key.update(sys.version.encode())
    sources = [ROOT / "scripts/office.py", *(ROOT / "mtmc").glob("*.py")]
    for folder, paths in ((inputs, inputs.iterdir()), (ROOT, sources)):
        for path in sorted(paths):
            key.update(path.relative_to(folder).as_posix().encode())
            key.update(hashlib.sha256(path.read_bytes()).digest())
    for path in sorted(videos + list((ROOT / "weights").glob("*"))):
        stat = path.stat()
        key.update(f"{path.relative_to(ROOT)}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return key.hexdigest()


def viewer_exists(folder: Path) -> bool:
    return folder.resolve() == folder and all((folder / f).is_file() for f in ("index.html", "data.js"))


class Generations:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.worker: threading.Thread | None = None
        self.active: int | None = None

    def folders(self) -> list[Path]:
        if HISTORY.resolve() != HISTORY:
            raise ValueError("Папка генераций должна находиться внутри проекта.")
        if not HISTORY.exists():
            return []
        return sorted((p for p in HISTORY.iterdir() if re.fullmatch(r"[1-9][0-9]*", p.name) and p.is_dir()
                       and p.resolve() == p), key=lambda p: int(p.name), reverse=True)

    def saved(self) -> list[dict]:
        runs = []
        for path in self.folders():
            manifest = path / "generation.json"
            if manifest.resolve() != manifest:
                continue
            try:
                run = read_json(manifest)
            except FileNotFoundError:
                continue
            if str(run["number"]) == path.name:
                runs.append(run)
        return runs

    def output(self, run: dict, key: str) -> Path:
        options = run["options"]
        trial = options["start"] is not None or options["end"] is not None
        return HISTORY / str(run["number"]) / (output_name(options[key], DEFAULTS[key]) + ("_trial" if trial else ""))

    def list(self) -> list[dict]:
        items = []
        for run in self.saved():
            number, status = run["number"], run["status"]
            if status == "running" and number != self.active:
                status = "interrupted"
            viewer = (f"/view/{number}/workspace/index.html"
                      if status == "ready" and viewer_exists(self.output(run, "ui")) else None)
            items.append({**run, "id": str(number), "status": status, "viewer": viewer})
        legacy = []
        for key, (label, ui, _) in LEGACY.items():
            if viewer_exists(ui):
                mtime = (ui / "index.html").stat().st_mtime
                legacy.append({"id": key, "label": label, "status": "ready", "viewer": f"/view/{key}/workspace/index.html",
                               "created": datetime.fromtimestamp(mtime, timezone.utc).isoformat()})
        return items + sorted(legacy, key=lambda item: item["created"], reverse=True)

    def start(self, body: dict, dependencies: Dependencies) -> int:
        if self.active is not None:
            raise BusyError("Дождитесь завершения текущей генерации.")
        options, chosen = generation_options(body)
        options["actual_device"] = dependencies.device(options["device"])
        options["gpu_name"] = dependencies.state.get("gpu_name") if options["actual_device"] == "gpu" else None
        configs = ROOT / options["configs"]
        base, extra = config_paths(configs)
        yaml_files = base.read_bytes(), extra.read_bytes() if extra.is_file() else None
        office.merge_config(*yaml_files)
        existing = self.folders()
        HISTORY.mkdir(exist_ok=True)
        sequence = HISTORY / "sequence.json"
        last = read_json(sequence)["last"] if sequence.exists() else 0
        number = max(last, int(existing[0].name) if existing else 0) + 1
        path = HISTORY / str(number)
        path.mkdir()
        write_json(sequence, {"last": number})
        inputs = path / "inputs"
        run = {"number": number, "created": datetime.now(timezone.utc).isoformat(), "status": "running",
               "options": options}
        try:
            snapshot_inputs(inputs, configs, chosen, yaml_files)
            key = cache_key(inputs, options, [v for v, _, _ in chosen])
        except (ValueError, OSError) as e:
            run.update(status="failed", error=str(e))
            write_json(path / "generation.json", run)
            raise
        run["cache_key"] = key
        previous = next((item for item in self.saved() if item["status"] == "ready" and item.get("cache_key") == key
                         and (self.output(item, "data") / "office_state.json").is_file()
                         and viewer_exists(self.output(item, "ui"))), None) if not options["fresh"] else None
        write_json(path / "generation.json", run)
        self.active = number
        self.worker = threading.Thread(target=self.run, args=(run, path, previous))
        self.worker.start()
        return number

    def run(self, run: dict, path: Path, previous: dict | None) -> None:
        options = run["options"]
        cmd = [sys.executable, str(ROOT / "scripts/office.py"), "--no-serve",
               "--videos", options["videos"], "--configs", (path / "inputs").relative_to(ROOT).as_posix(),
               "--data", (path / options["data"]).relative_to(ROOT).as_posix(),
               "--ui", (path / options["ui"]).relative_to(ROOT).as_posix(),
               "--quality", options["quality"], "--device", options["actual_device"],
               "--cameras", *map(str, options["cameras"])]
        for key in ("start", "end"):
            if options[key] is not None:
                cmd += [f"--{key}", str(options[key])]
        if not options["silhouettes"]:
            cmd.append("--no-silhouettes")
        if options["fresh"]:
            cmd.append("--fresh")
        run["status"] = "failed"
        try:
            with (path / "job.log").open("w", encoding="utf-8", buffering=1) as log:
                try:
                    device_label = options["gpu_name"] or options["actual_device"].upper()
                    log.write(f"Устройство: выбрано {options['device']}, используется {device_label}.\n")
                    if previous:
                        log.write(f"Копирую совместимые данные из генерации №{previous['number']}.\n")
                        for key in ("data", "ui"):
                            shutil.copytree(self.output(previous, key), self.output(run, key))
                    result = subprocess.run(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                            stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUTF8": "1",
                                                                            "PYTHONUNBUFFERED": "1"})
                    if result.returncode == 0 and viewer_exists(self.output(run, "ui")):
                        run["status"] = "ready"
                    else:
                        run["error"] = f"Обработка завершилась без готового результата. Код выхода {result.returncode}."
                except OSError as e:
                    run["error"] = str(e)
                    log.write(f"СТОП: {e}\n")
        except OSError as e:
            run["error"] = str(e)
        finally:
            with self.lock:
                try:
                    write_json(path / "generation.json", run)
                finally:
                    self.active = None

    def clear(self) -> None:
        if self.active is not None:
            raise BusyError("Дождитесь завершения обработки перед очисткой.")
        folders = self.folders()
        for _, ui, data in LEGACY.values():
            if (ui / "videos").is_dir() or viewer_exists(ui):
                folders.append(ui)
            if (data / "run_config.yaml").is_file() or (data / "mtmc.sqlite").is_file():
                folders.append(data)
        for path in folders:
            if path.resolve() != path or path.parent not in (HISTORY, ROOT):
                raise ValueError(f"Небезопасный путь для очистки: {path}")
        for path in folders:
            shutil.rmtree(path)

    def viewer(self, key: str) -> Path | None:
        if key in LEGACY:
            folder = LEGACY[key][1]
        elif re.fullmatch(r"[1-9][0-9]*", key):
            path = HISTORY / key / "generation.json"
            if path.resolve() != path or not path.is_file():
                return None
            run = read_json(path)
            if str(run["number"]) != key or run["status"] != "ready":
                return None
            folder = self.output(run, "ui")
        else:
            return None
        return folder if viewer_exists(folder) else None


def make_handler(generations: Generations, dependencies: Dependencies):
    range_handler = _range_handler()
    auto_lock = threading.Lock()

    class Handler(range_handler):
        extensions_map = {**range_handler.extensions_map, ".mkv": "video/x-matroska", ".ts": "video/mp2t"}

        def handle_one_request(self) -> None:
            try:
                super().handle_one_request()
            except ConnectionError:
                self.close_connection = True

        def end_headers(self):
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            super().end_headers()

        def allowed(self) -> bool:
            host = self.headers.get("Host")
            port = self.server.server_port
            origin = self.headers.get("Origin")
            if (host in (f"127.0.0.1:{port}", f"localhost:{port}") and (not origin or origin == f"http://{host}")
                    and self.headers.get("Sec-Fetch-Site") != "cross-site"):
                return True
            self.send_error(403)
            return False

        def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def reply(self, body, status: int = 200) -> None:
            self.send_bytes(json.dumps(body, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

        def translate_path(self, _path):
            return str(self.file_path)

        def serve_file(self, path: Path | None, head: bool) -> None:
            if path is None or not path.is_file():
                self.send_error(404)
                return
            self.file_path = path
            if head:
                super().do_HEAD()
            else:
                super().do_GET()

        def do_GET(self):
            self.get()

        def do_HEAD(self):
            self.get(head=True)

        def get(self, head: bool = False) -> None:
            if not self.allowed():
                return
            url = urlsplit(self.path)
            path, _, suffix = url.path[1:].partition("/")
            query = parse_qs(url.query)
            options = {k: v[0] for k, v in query.items()}
            try:
                if url.path in ("/", "/index.html", "/office.css", "/office.js"):
                    self.serve_file(STATIC / (path or "index.html"), head)
                elif path == "view":
                    key, _, relative = suffix.partition("/")
                    folder = generations.viewer(key)
                    if folder and relative.startswith("workspace/"):
                        relative = relative.removeprefix("workspace/")
                        if relative in VIEWER_ASSETS:
                            self.serve_file(VIEWER / relative, head)
                            return
                    file = (folder / unquote(relative)).resolve() if folder else None
                    self.serve_file(file if file and file.is_relative_to(folder) else None, head)
                elif path == "source":
                    self.serve_file(source_video(source_dir(options, "videos"), unquote(suffix)), head)
                elif head:
                    self.send_error(404)
                elif path == "poster":
                    if not dependencies.state["base_ready"]:
                        self.send_error(404)
                        return
                    import site_tool

                    image = site_tool.first_frame(source_video(source_dir(options, "videos"), unquote(suffix)), at_s=0)
                    if image is None:
                        self.send_error(404)
                    else:
                        self.send_bytes(base64.b64decode(image), "image/jpeg")
                elif url.path == "/api/videos":
                    videos, configs = (source_dir(options, k) for k in ("videos", "configs"))
                    with generations.lock:
                        entries = office.camera_inputs(office.find_videos(videos), configs)
                    self.reply([{"camera": int(d["cam_id"]), "name": v.name, "size": v.stat().st_size}
                                for v, _, d in entries])
                elif url.path == "/api/generations":
                    with generations.lock:
                        runs = generations.list()
                    self.reply(runs)
                elif url.path == "/api/dependencies":
                    self.reply(dependencies.status())
                elif match := re.fullmatch(r"api/generations/([1-9][0-9]*)/log", url.path[1:]):
                    log = HISTORY / match[1] / "job.log"
                    if log.resolve() != log or not (log.parent / "generation.json").is_file():
                        self.send_error(404)
                        return
                    tail = b""
                    if log.is_file():
                        with log.open("rb") as f:
                            f.seek(max(log.stat().st_size - 16_384, 0))
                            tail = f.read(16_384)
                    log_text = tail.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
                    self.reply({"text": re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", log_text)})
                elif path in ("plan", "calibration"):
                    if not dependencies.state["base_ready"]:
                        self.reply({"error": "Установите базовые библиотеки во вкладке «Зависимости»."}, 503)
                        return
                    import camera_app
                    import camera_auto
                    import site_tool

                    configs, videos = (source_dir(options, k) for k in ("configs", "videos"))
                    with generations.lock:
                        if path == "plan":
                            response = site_tool.get_response(configs, videos, "/" + suffix)
                        elif suffix in ("", "index.html"):
                            response = camera_auto.page(), "text/html; charset=utf-8"
                        else:
                            response = camera_app.get_response(configs, videos, "/" + suffix, query)
                    if response is None:
                        self.send_error(404)
                    else:
                        self.send_bytes(*response)
                else:
                    self.send_error(404)
            except FileNotFoundError:
                self.send_error(404)
            except ConnectionError:
                raise
            except (ValueError, KeyError, TypeError, OSError) as e:
                self.reply({"error": str(e)}, 400)

        def do_POST(self):
            if not self.allowed():
                return
            url = urlsplit(self.path)
            path, _, suffix = url.path[1:].partition("/")
            try:
                length = int(self.headers.get("Content-Length", "0"))
                options = {k: v[0] for k, v in parse_qs(url.query).items()}
                if url.path == "/api/videos/upload":
                    if self.headers.get_content_type() != "application/octet-stream":
                        self.reply({"error": "Ожидается видеофайл."}, 415)
                        return
                    self.connection.settimeout(60)
                    upload_media(source_dir(options, "videos"), options["name"], length, self.rfile,
                                 office.VIDEO_EXT, "видео")
                    self.reply({"ok": True}, 201)
                    return
                if url.path == "/plan/upload":
                    if not dependencies.state["base_ready"]:
                        self.reply({"error": "Сначала установите базовые библиотеки."}, 503)
                        return
                    if self.headers.get_content_type() != "application/octet-stream":
                        self.reply({"error": "Ожидается PNG-файл."}, 415)
                        return
                    import site_tool

                    self.connection.settimeout(60)
                    result = site_tool.upload_plan(source_dir(options, "configs"), options.get("name", ""),
                                                   length, self.rfile)
                    self.reply(result, 201)
                    return
                if self.headers.get_content_type() != "application/json":
                    self.reply({"error": "Ожидается запрос JSON."}, 415)
                    return
                if not 0 < length <= 1_000_000:
                    raise ValueError("Неверный размер запроса.")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Ожидается объект JSON.")
                if path in ("plan", "calibration") and not dependencies.state["base_ready"]:
                    self.reply({"error": "Сначала установите базовые библиотеки."}, 503)
                    return
                if url.path == "/calibration/auto":
                    import camera_auto

                    configs, videos = (source_dir(options, k) for k in ("configs", "videos"))
                    # Первый расчёт может занять несколько минут. Не блокируем остальные вкладки.
                    if not auto_lock.acquire(blocking=False):
                        raise BusyError("Авторазметка уже выполняется. Дождитесь её завершения.")
                    try:
                        result = camera_auto.auto_mark(configs, videos, int(body["cam_id"]), body["pairs"])
                    finally:
                        auto_lock.release()
                    self.reply(result)
                    return
                with generations.lock:
                    if url.path == "/api/generations":
                        self.reply({"number": generations.start(body, dependencies)}, 202)
                    elif url.path == "/api/config/open":
                        paths = dict(zip(("base", "office"), config_paths(source_dir(body, "configs"))))
                        file = paths.get(body.get("kind"))
                        if file is None or not file.is_file():
                            raise ValueError("Выбранный файл YAML не найден.")
                        open_yaml(file)
                        self.reply({"file": file.name})
                    elif url.path == "/api/dependencies/install":
                        if generations.active is not None:
                            raise BusyError("Дождитесь окончания обработки.")
                        dependencies.start(body.get("device", "auto"))
                        self.reply({"ok": True}, 202)
                    elif url.path == "/api/dependencies/driver":
                        if type(body.get("install")) is not bool:
                            raise ValueError("Нужно подтвердить или отклонить установку.")
                        dependencies.confirm(body["install"])
                        self.reply({"ok": True}, 202)
                    elif url.path == "/api/generations/clear":
                        if body.get("confirm") is not True:
                            raise ValueError("Подтвердите очистку результатов.")
                        generations.clear()
                        self.reply({"ok": True})
                    elif path in ("plan", "calibration"):
                        import camera_app
                        import site_tool

                        configs, videos = (source_dir(options, k) for k in ("configs", "videos"))
                        if path == "plan":
                            response = site_tool.post_response(configs, "/" + suffix, body, require_all=False)
                        else:
                            response = camera_app.post_response(configs, videos, "/" + suffix, body)
                        if response is None:
                            self.send_error(404)
                        else:
                            self.reply(response)
                    else:
                        self.send_error(404)
            except (BusyError, FileExistsError) as e:
                self.reply({"error": str(e)}, 409)
            except ConnectionError:
                raise
            except (ValueError, KeyError, TypeError, OSError) as e:
                self.reply({"error": str(e)}, 400)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--dependencies", action="store_true", help="открыть вкладку зависимостей")
    args = parser.parse_args()
    os.chdir(ROOT)
    generations = Generations()
    lock_path = HISTORY / "server.sqlite"
    if lock_path.resolve() != lock_path:
        parser.exit(1, "Папка генераций должна находиться внутри проекта.\n")
    HISTORY.mkdir(exist_ok=True)
    try:
        with closing(sqlite3.connect(lock_path, timeout=0)) as lock:
            # Один сервер может распоряжаться историей, даже если запустить второй на другом порту.
            lock.execute("BEGIN EXCLUSIVE")
            default_videos = ROOT / DEFAULTS["videos"]
            if default_videos.resolve() == default_videos:
                default_videos.mkdir(parents=True, exist_ok=True)
            dependencies = Dependencies()
            with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(generations, dependencies)) as server:
                url = f"http://127.0.0.1:{server.server_port}/"
                print(f"MTMC — {url}", flush=True)
                if not args.no_browser:
                    webbrowser.open(url + ("#dependencies" if args.dependencies else ""))
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    if generations.worker and generations.worker.is_alive():
                        print("Ожидаю завершения текущей обработки.", flush=True)
                        generations.worker.join()
                    if dependencies.worker and dependencies.worker.is_alive():
                        print("Ожидаю завершения задачи установки.", flush=True)
                        dependencies.worker.join()
    except sqlite3.OperationalError as e:
        parser.exit(1, f"Не удалось открыть историю. Проверьте, не запущен ли уже сайт. {e}\n")
    except OSError as e:
        parser.exit(1, f"Не удалось запустить сайт на порту {args.port}. {e}\n")


if __name__ == "__main__":
    main()
