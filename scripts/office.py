"""Запуск на новых видео одной командой: положили записи в data/office — получили интерфейс.

    python scripts/office.py              вся запись
    python scripts/office.py --end 60     проба: первые 60 с, в отдельные папки *_trial

Что делает сам:
- заводит паспорта камер в configs/office (свои правки в существующих паспортах не трогает);
- подбирает настройки под машину (без CUDA: CPU, imgsz 640) и шаг link.grid_s под fps записи;
- обрабатывает камеры по одной и запоминает готовые: после обрыва повторный запуск продолжит с того же места;
- если есть configs/office/site.json — привязывает камеры к плану;
- если H появилась после обработки — пересчитывает план без повторной детекции;
- склеивает, строит треки, собирает интерфейс и открывает его.

Без плана (H) камеры не связываются между собой: каждая камера получается сама по себе.
Об этом программа предупреждает. Детекция при этом не пропадает: добавьте план и запустите ещё раз.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import is_dataclass
from pathlib import Path
from typing import get_args, get_type_hints

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".ts", ".mts", ".wmv"}
PY = sys.executable
# Детекцию повторяем, только если изменились эти разделы настроек. tracklets пересчитываются из наблюдений за секунды,
# link и tracks — при каждом запуске: смена их порогов не должна стоить часов работы.
DETECT_PARTS = ("video", "gate", "detector", "tracker", "observation", "crops", "reid", "silhouette")
SPEED_CPU_640 = 10      # замер: Ryzen 7 4700U, stride 2, 25 кадр/с, с силуэтами — обработка в 8–12 раз медленнее записи


def say(msg: str = "") -> None:
    print(msg, flush=True)


def stop(msg: str) -> None:
    say()
    say(f"СТОП: {msg}")
    sys.exit(1)


def keep_awake() -> None:
    """Не даём ноутбуку заснуть, пока программа работает (Windows)."""
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def mtmc(*args: str) -> None:
    """Шаг конвейера отдельным процессом: память модели освобождается, индикаторы работают."""
    r = subprocess.run([PY, "-m", "mtmc", *args], cwd=ROOT)
    if r.returncode != 0:
        stop(f"шаг «{args[0]}» завершился с ошибкой. Запустите ещё раз: готовые камеры не пересчитываются.")


def find_videos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXT)


def probe(path: Path) -> tuple[float, float]:
    """fps и длительность, с."""
    import cv2

    cap = cv2.VideoCapture(os.path.relpath(path, ROOT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok = cap.isOpened()
    cap.release()
    if not ok or fps <= 0:
        stop(f"не удалось прочитать видео {path.name} (или в нём нет fps). Проверьте файл.")
    return fps, n / fps


def passport_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.json")
                  if not p.name.startswith("site") and not p.name.endswith(".before_autocalib.json"))


def number_from_names(videos: list[Path]) -> dict[str, int]:
    """Номера камер из имён файлов (cam3.mp4 -> 3): так они совпадают с планом организаторов.
    Только если у всех файлов номера есть и они разные; иначе нумерация по порядку."""
    nums = {}
    for v in videos:
        m = re.search(r"(\d{1,4})(?!.*\d)", v.stem)
        if not m:
            return {}
        nums[v.name] = int(m.group(1))
    ok = len(set(nums.values())) == len(nums) and all(0 < n <= 9000 for n in nums.values())
    return nums if ok else {}


def camera_inputs(videos: list[Path], folder: Path) -> list[tuple[Path, Path, dict]]:
    """Видео, путь паспорта и его данные. Для новых видео подбирает номера без записи на диск."""
    known: dict[str, tuple[Path, dict]] = {}
    used: set[int] = set()
    for p in passport_files(folder):
        d = json.loads(p.read_text(encoding="utf-8"))
        known[Path(d["video"]).name] = (p, d)
        used.add(int(d["cam_id"]))
    by_name = number_from_names(videos)
    entries = []
    for v in videos:
        if v.name in known:
            p, d = known[v.name]
            entries.append((v, p, d))
            continue
        cam = by_name.get(v.name)
        if cam is None or cam in used:
            cam = next(i for i in range(1, 9000) if i not in used)
        used.add(cam)
        rel = os.path.relpath(v, ROOT).replace("\\", "/")
        d = {"cam_id": cam, "name": v.stem, "video": rel, "offset_s": 0.0}
        entries.append((v, folder / f"cam{cam}.json", d))
    return sorted(entries, key=lambda entry: entry[1])


def ensure_passports(videos: list[Path], folder: Path, cameras: list[int] | None = None) -> list[Path]:
    """Новому видео — новый паспорт; существующие паспорта не меняются."""
    entries = camera_inputs(videos, folder)
    if cameras is not None:
        missing = set(cameras) - {int(d["cam_id"]) for _, _, d in entries}
        if missing:
            stop("нет видео для камер " + ", ".join(map(str, sorted(missing))))
        entries = [(v, p, d) for v, p, d in entries if int(d["cam_id"]) in cameras]
    folder.mkdir(parents=True, exist_ok=True)
    for v, p, d in entries:
        if not p.exists():
            fps, _ = probe(v)
            d["fps"] = round(fps, 3)
            p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
            say(f"  новый паспорт: cam{d['cam_id']} — {v.name}")
    return [p for _, p, _ in entries]


def ensure_lens(files: list[Path], by_name: dict[str, Path]) -> None:
    """Широкоугольные объективы: оценка по прямым линиям сцены, один раз для каждой камеры (mtmc/lens.py).

    Камера без K, но с H не трогается: её H посчитана по кривому кадру. Если у камеры линий мало, а соседние с
    таким же кадром широкоугольные, берётся их фокус: в одном помещении камеры обычно одинаковые."""
    import cv2

    from mtmc.lens import estimate_lens

    todo: list[tuple[Path, dict, int, int]] = []
    foci = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("K"):
            foci.append(float(d["K"][0][0]))
        if d.get("K") or d.get("H") is not None or d.get("lens_checked"):
            continue
        cap = cv2.VideoCapture(os.path.relpath(by_name[Path(d["video"]).name], ROOT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        lens, fit = estimate_lens(os.path.relpath(by_name[Path(d["video"]).name], ROOT), w, h, d.get("fps"))
        d["lens_checked"] = True
        d.update(lens)
        if lens:
            foci.append(lens["K"][0][0])
            say(f"  cam{d['cam_id']}: широкий объектив, угол обзора {fit.hfov_deg:.0f}°, кадр будет выпрямляться")
        else:
            todo.append((f, d, w, h))
        f.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    if len(foci) >= 2:
        focus = sorted(foci)[len(foci) // 2]
        for f, d, w, h in todo:
            d.update({"K": [[focus, 0.0, w / 2], [0.0, focus, h / 2], [0.0, 0.0, 1.0]], "dist": [0.0],
                      "lens_source": "как у соседних камер"})
            f.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
            say(f"  cam{d['cam_id']}: линий мало, объектив взят как у соседних камер")


def trial_passports(files: list[Path], data: Path) -> list[Path]:
    """Проба работает на копиях паспортов: H, найденная по минуте видео, не должна попасть в рабочие паспорта."""
    folder = data / "passports"
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        t = folder / f.name
        if d.get("H") is None and t.exists():
            old = json.loads(t.read_text(encoding="utf-8"))
            if old.get("K") == d.get("K"):        # H считается для конкретного объектива
                d.update({k: v for k, v in old.items() if k == "H" or k.startswith("calib_")})
        t.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        out.append(t)
    return out


def config_sources(folder: Path) -> tuple[Path, Path]:
    base, extra = folder / "config.yaml", folder / "overrides.yaml"
    return (base if base.is_file() else ROOT / "configs/config.yaml",
            extra if extra.is_file() else ROOT / "configs/office/overrides.yaml")


def merge_config(base: bytes, extra: bytes | None) -> dict:
    import yaml
    from mtmc.config import config_from_dict

    def validate(section, prefix: str = "") -> None:
        for key, expected in get_type_hints(type(section)).items():
            value = getattr(section, key)
            path = prefix + key
            if is_dataclass(value):
                validate(value, path + ".")
                continue
            allowed = get_args(expected) or (expected,)
            if not any(type(value) is kind or (kind is float and type(value) is int) for kind in allowed):
                raise ValueError(f"неверный тип значения «{path}»")
            if type(value) is float and not math.isfinite(value):
                raise ValueError(f"значение «{path}» должно быть конечным")
            if path in {"video.stride", "detector.imgsz", "silhouette.imgsz", "silhouette.every",
                        "silhouette.batch", "reid.batch"} and value is not None and value <= 0:
                raise ValueError(f"значение «{path}» должно быть больше нуля")

    def read(raw: bytes, label: str) -> dict:
        try:
            value = yaml.safe_load(raw)
            if value is None:
                return {}
            if not isinstance(value, dict):
                raise ValueError("ожидается словарь YAML")
            validate(config_from_dict(value))
            return value
        except (yaml.YAMLError, ValueError) as e:
            raise ValueError(f"{label}: {e}") from e

    data = read(base, "config.yaml")
    if extra is not None:
        for part, values in read(extra, "overrides.yaml").items():
            data.setdefault(part, {}).update(values)
    return config_from_dict(data).to_dict()


def make_config(data: Path, fps: float, quality: str, silhouettes: bool, cfg_dir: Path = ROOT / "configs/office",
                device: str = "auto") -> Path:
    """Конфиг этого запуска: configs/config.yaml + поправки под машину и fps записи."""
    import torch
    import yaml

    from mtmc.config import WALK_MPS, rescale_metres

    base, extra = config_sources(cfg_dir)
    d = merge_config(base.read_bytes(), extra.read_bytes() if extra.is_file() else None)
    walk = data / "walk.json"
    if walk.exists():
        rescale_metres(d, json.loads(walk.read_text(encoding="utf-8"))["mps"] / WALK_MPS)
    gpu = device != "cpu" and torch.cuda.is_available()
    selected = "cuda:0" if gpu else "cpu"
    for section in ("detector", "reid", "silhouette"):
        d[section]["device"] = selected
    if not gpu:
        d["detector"]["device"] = "cpu"
        d["detector"]["half"] = False
    if quality == "fast" or (quality == "auto" and not gpu):
        d["detector"]["imgsz"] = 640
        d["silhouette"]["imgsz"] = 640
    if not gpu:
        d["silhouette"]["every"] = 2         # маски — почти половина времени на процессоре; между ними контур сдвигается
    d["silhouette"]["enabled"] = silhouettes
    d["link"]["grid_s"] = round(d["video"]["stride"] / fps, 5)      # шаг общей сетки времени зависит от fps
    data.mkdir(parents=True, exist_ok=True)
    out = data / "run_config.yaml"
    out.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
    say(f"  устройство: {'видеокарта' if gpu else 'процессор'}, imgsz {d['detector']['imgsz']}, "
        f"stride {d['video']['stride']}, силуэты: {'да' if silhouettes else 'нет'}")
    return out


def check_site(site: Path, cams: list[int]) -> bool:
    """Ошибки плана ловим до многочасовой обработки."""
    if not site.exists():
        return False
    d = json.loads(site.read_text(encoding="utf-8"))
    img = d.get("plan_image")
    if not img or not (site.parent / img).exists() and not Path(img).exists():
        stop(f"в {site.name} указана картинка плана «{img}», но такого файла нет рядом с ним.")
    if not d.get("px_per_m"):
        stop(f"в {site.name} не задан масштаб px_per_m (пикселей плана на метр).")
    miss = [c for c in cams if "pos" not in (d.get("cameras") or {}).get(str(c), {})]
    if miss:
        stop(f"в {site.name} нет положения (pos) камер: {', '.join(map(str, miss))}. "
             f"Номера камер — в configs/office/cam*.json.")
    return True


def fingerprint(video: Path, passport: dict, cfg: Path, start: float | None, end: float | None) -> str:
    """Отпечаток обработки: видео, срез, настройки. H и calib_* в него не входят: их дописывает привязка,
    а план пересчитывается без детекции."""
    keep = {k: passport.get(k) for k in ("cam_id", "fps", "offset_s", "K", "dist", "floor_roi", "zones", "overlaps",
                                         "ignore_zones", "clock")}
    st = video.stat()
    return hashlib.sha256(json.dumps([video.name, st.st_size, st.st_mtime_ns, keep, config_part(cfg, DETECT_PARTS),
                                      start, end], ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def config_part(cfg: Path, parts: tuple[str, ...]) -> dict:
    """Часть настроек: от неё зависит, какой шаг надо повторять."""
    import yaml

    d = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    return {k: d.get(k) for k in parts}


def fit_plan_step(files: list[Path], data: Path, cfg_path: Path) -> list[Path]:
    """Уточнение привязки по людям (mtmc/planfit.py). Рабочие паспорта не меняются: уточнённая H относится к этой
    папке результатов. -> паспорта для склейки и интерфейса."""
    from mtmc.config import load_config
    from mtmc.planfit import fit_plan, save

    passports = {int(d["cam_id"]): d for d in (json.loads(f.read_text(encoding="utf-8")) for f in files)}
    passports = {c: d for c, d in passports.items() if d.get("H") is not None}
    new, report = fit_plan(data, load_config(cfg_path), passports)
    save(data, new, report)
    for r in report[-len(passports):] if passports else []:
        say(f"  уточнение привязки: {r}")
    if not new:
        return files
    folder = data / "passports_fit"
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        if int(d["cam_id"]) in new:
            d["H"] = new[int(d["cam_id"])]
            say(f"  cam{d['cam_id']}: привязка к плану уточнена по людям, которых видят соседние камеры")
        t = folder / f.name
        t.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        out.append(t)
    return out


def sync_h(files: list[Path], data: Path, cfg_path: Path) -> None:
    """H в паспорте появилась или изменилась после обработки — пересчитать план и треклеты без детекции."""
    from mtmc.config import load_config
    from mtmc.pipeline import retrack
    from mtmc.storage import Store

    store = Store(data)
    cams = store.load_cameras().set_index("cam_id")
    cfg = load_config(cfg_path)
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        cid = int(d["cam_id"])
        if cid not in cams.index or d.get("H") is None:
            continue
        if json.loads(cams.loc[cid, "passport_json"]).get("H") != d["H"]:
            say(f"  cam{cid}: новая привязка к плану, пересчёт без детекции")
            retrack(cid, data, cfg, passport_override=d)


def part_range(args: list[str]) -> tuple[float | None, float]:
    """part_office.bat: [секунды, с какой секунды] -> (--start, --end). Без секунд спрашивает."""
    if not args:
        try:
            args = input("Сколько секунд видео обработать: ").split()
        except EOFError:
            args = []
    try:
        seconds = float(args[0])
        begin = float(args[1]) if len(args) > 1 else 0.0
    except (IndexError, ValueError):
        seconds, begin = 0.0, 0.0
    if seconds <= 0 or begin < 0:
        stop("нужно число секунд больше нуля. Пример: part_office.bat 300 или part_office.bat 300 600")
    say(f"Обрабатываю отрезок записи: с {begin:g} по {begin + seconds:g} секунду.")
    return (begin or None), begin + seconds


def main() -> int:
    ap = argparse.ArgumentParser(description="Новые видео -> интерфейс, одной командой")
    ap.add_argument("--videos", default="data/office", help="папка с записями")
    ap.add_argument("--configs", default="configs/office", help="паспорта и site.json")
    ap.add_argument("--cameras", type=int, nargs="+", help="номера обрабатываемых камер; по умолчанию все")
    ap.add_argument("--data", default="results_office")
    ap.add_argument("--ui", default="ui_office")
    ap.add_argument("--start", type=float, default=None, help="с какой секунды видео")
    ap.add_argument("--end", type=float, default=None, help="до какой секунды видео (проба)")
    ap.add_argument("--quality", choices=["auto", "fast", "high"], default="auto",
                    help="auto: на CPU imgsz 640, на видеокарте 1280; high: 1280 всегда (на CPU в 3 раза дольше)")
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="устройство обработки")
    ap.add_argument("--no-silhouettes", action="store_true", help="без силуэтов (быстрее, в интерфейсе рамка и поза)")
    ap.add_argument("--fresh", action="store_true", help="пересчитать все камеры заново")
    ap.add_argument("--no-serve", action="store_true", help="не открывать интерфейс в конце")
    ap.add_argument("--part", action="store_true", help="часть записи: office.py --part СЕКУНДЫ [С_КАКОЙ_СЕКУНДЫ]")
    ap.add_argument("part_args", nargs="*", help="для --part: сколько секунд и с какой секунды")
    a = ap.parse_args()
    if a.part:
        a.start, a.end = part_range(a.part_args)
    elif a.part_args:
        ap.error(f"лишние параметры: {' '.join(a.part_args)}")
    if any(t is not None and (not math.isfinite(t) or t < 0) for t in (a.start, a.end)):
        ap.error("границы отрезка должны быть неотрицательными числами")
    if a.end is not None and a.end <= (a.start or 0):
        ap.error("конец отрезка должен быть позже начала")

    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    if a.device == "gpu" and not torch.cuda.is_available():
        stop("выбрано GPU, но PyTorch не видит CUDA. Проверьте вкладку «Зависимости».")
    actual_device = "gpu" if a.device != "cpu" and torch.cuda.is_available() else "cpu"
    say(f"Устройство: выбрано {a.device}, используется {actual_device.upper()}.")
    os.chdir(ROOT)
    keep_awake()
    videos_dir, cfg_dir = ROOT / a.videos, ROOT / a.configs
    trial = a.start is not None or a.end is not None
    data = ROOT / (a.data + "_trial" if trial else a.data)
    ui = ROOT / (a.ui + "_trial" if trial else a.ui)
    videos_dir.mkdir(parents=True, exist_ok=True)

    say("== Подготовка")
    videos = find_videos(videos_dir)
    if not videos:
        stop(f"в папке {videos_dir} нет видео. Положите туда записи всех камер и запустите снова.")
    files = ensure_passports(videos, cfg_dir, a.cameras)
    say(f"  видео: {len(files)}")
    ensure_lens(files, {v.name: v for v in videos})
    if trial:
        data.mkdir(parents=True, exist_ok=True)
        files = trial_passports(files, data)
    passports ={f: json.loads(f.read_text(encoding="utf-8")) for f in files}
    by_name = {v.name: v for v in videos}
    if a.cameras is not None and (data / "mtmc.sqlite").is_file():
        from mtmc.storage import Store

        if set(Store(data).load_cameras()["cam_id"]) - set(a.cameras):
            stop("в папке результатов есть другие камеры. Задайте отдельные папки --data и --ui.")

    infos = [probe(by_name[Path(d["video"]).name]) for d in passports.values()]
    rates = sorted({round(i[0], 2) for i in infos})
    if len(rates) > 1:
        say(f"  ВНИМАНИЕ: у камер разный fps {rates}; порог времени возьмём по наибольшему")
    if len(files) < 2:
        say("  ВНИМАНИЕ: камера одна — сквозные треки сводятся к трекам этой камеры")
    if a.start is not None and any(duration > 0 and a.start >= duration for _, duration in infos):
        stop("начало отрезка за пределами одной из выбранных записей.")
    if a.end is not None and any(round((a.start or 0) * fps) >= round(a.end * fps) for fps, _ in infos):
        stop("отрезок слишком короткий для частоты кадров одной из выбранных записей.")
    max_fps = max(i[0] for i in infos)
    cfg = make_config(data, max_fps, a.quality, not a.no_silhouettes, cfg_dir, actual_device)
    has_site = check_site(cfg_dir / "site.json", [int(d["cam_id"]) for d in passports.values()])

    no_h = [int(d["cam_id"]) for d in passports.values() if d.get("H") is None]
    if no_h and not has_site:
        say()
        say("  ВНИМАНИЕ: плана нет (configs/office/site.json), камеры не привязаны.")
        say("  Без него камеры НЕ свяжутся между собой: один человек получит свой трек на каждой камере.")
        say("  Обработка не пропадёт: добавьте план и запустите снова — детекция не повторится.")

    total = sum(min(i[1], a.end or i[1]) - (a.start or 0) for i in infos)
    if actual_device == "cpu":
        say(f"  оценка времени на этом ноутбуке: около {total * SPEED_CPU_640 / 3600:.1f} ч "
            f"(видео всего {total / 60:.0f} мин). Держите его на зарядке.")

    say()
    say("== Обработка камер")
    state_path = data / "office_state.json"
    state = {} if a.fresh or not state_path.exists() else json.loads(state_path.read_text(encoding="utf-8"))
    for f, d in passports.items():
        cid, v = str(d["cam_id"]), by_name[Path(d["video"]).name]
        fp = fingerprint(v, d, cfg, a.start, a.end)
        if state.get(cid) == fp:
            say(f"  cam{cid}: уже обработана, пропускаю")
            continue
        say(f"  cam{cid}: {v.name}")
        cmd = ["process", "--camera", str(f.relative_to(ROOT)), "--data", str(data.relative_to(ROOT)),
               "--config", str(cfg.relative_to(ROOT)), "--log-level", "WARNING"]
        if a.start is not None:
            cmd += ["--start", str(a.start)]
        if a.end is not None:
            cmd += ["--end", str(a.end)]
        mtmc(*cmd)
        state[cid] = fp
        state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")

    pp = [str(f.relative_to(ROOT)) for f in files]
    data_rel, cfg_rel, ui_rel = str(data.relative_to(ROOT)), str(cfg.relative_to(ROOT)), str(ui.relative_to(ROOT))
    cut_hash = lambda: hashlib.sha256(json.dumps(config_part(cfg, ("tracklets",)),  # noqa: E731
                                                 sort_keys=True).encode()).hexdigest()
    was = cut_hash()
    from mtmc.planfit import measure_walk

    mps = measure_walk(data)
    if mps:                 # меры расстояний берутся из шага людей на этом плане: нарезку это тоже меняет
        (data / "walk.json").write_text(json.dumps({"mps": mps}), encoding="utf-8")
        make_config(data, max_fps, a.quality, not a.no_silhouettes, cfg_dir, actual_device)
        say(f"  шаг людей на плане: {mps:.2f} м/с")
    cut = cut_hash()
    if state.get("_tracklets", was) != cut:
        say()
        say("== Пороги нарезки изменились: пересчёт треклетов без детекции")
        mtmc("retrack", "--data", data_rel, "--config", cfg_rel)
    state["_tracklets"] = cut
    state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
    # Камеры из старой копии результатов нарезаны ещё без силуэтов: детекция годится, нарезку надо повторить.
    uncut = [m for m in sorted((data / "observations").glob("*.meta.json"))
             if m.with_name(m.name.replace(".meta.json", ".sil.parquet")).exists()
             and "retracked_at" not in json.loads(m.read_text(encoding="utf-8"))]
    if uncut:
        say()
        say("== Камеры нарезаны без силуэтов: пересчёт треклетов без детекции")
        mtmc("retrack", "--data", data_rel, "--config", cfg_rel)
    site_rel = str((cfg_dir / "site.json").relative_to(ROOT)) if has_site else None
    if has_site and no_h:
        say()
        say("== Привязка камер к плану")
        # Без --manual: окно ручной привязки остановило бы всё до щелчков человека. Камеры, которые
        # не привязались, перечисляются в конце с готовой командой.
        mtmc("autocalib", "--data", data_rel, "--site", site_rel, "--passports", *pp,
             "--only-missing", "--config", cfg_rel)
    fitted = fit_plan_step(files, data, cfg)
    sync_h(fitted, data, cfg)
    pp = [str(f.relative_to(ROOT)) for f in fitted]

    say()
    say("== Склейка, треки, интерфейс")
    run = ["run", "--skip-process", "--passports", *pp, "--data", data_rel, "--ui", ui_rel, "--config", cfg_rel]
    if site_rel:
        run += ["--site", site_rel]
    mtmc(*run)

    n = 0
    tracks_csv = data / "tracks.csv"
    if tracks_csv.exists():
        n = max(len(tracks_csv.read_text(encoding="utf-8-sig").splitlines()) - 1, 0)
    left = [f for f in files if json.loads(f.read_text(encoding="utf-8")).get("H") is None]
    say()
    say(f"Готово. Сквозных треков: {n}. Таблица: {tracks_csv}")
    say(f"Отчёт с условиями съёмки: {data / 'report.md'}. Отброшенное с причинами: {data / 'rejected_tracks.csv'}")
    if left:
        names = ", ".join(str(json.loads(f.read_text(encoding="utf-8"))["cam_id"]) for f in left)
        say(f"ВНИМАНИЕ: камеры без привязки к плану ({names}) не связаны с другими: число треков ненадёжно.")
        if has_site:
            d = json.loads((cfg_dir / "site.json").read_text(encoding="utf-8"))
            plan = cfg_dir / d["plan_image"]
            say("Привяжите их вручную (6–10 пар точек на полу, Enter — сохранить), затем запустите снова:")
            for f in left:    # рабочий паспорт, а не копия пробы: иначе ручная привязка пропадёт
                say(f'  .venv\\Scripts\\python.exe -m mtmc calibrate --passport "{(cfg_dir / f.name).relative_to(ROOT)}" '
                    f'--plan "{plan.relative_to(ROOT)}" --px-per-m {d["px_per_m"]}')
    if a.no_serve:
        say(f"Интерфейс: python -m mtmc serve --dir {ui_rel}")
        return 0
    say("Интерфейс открывается в браузере. Закройте это окно, когда закончите смотреть.")
    subprocess.run([PY, "-m", "mtmc", "serve", "--dir", ui_rel], cwd=ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
