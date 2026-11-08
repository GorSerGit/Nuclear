#@title MTMC: отрезок видео на GPU Colab
# Среда выполнения -> Сменить среду выполнения -> GPU.
# Перед запуском на Google Диске в папке MTMC должны лежать:
#   Nuclear_colab.zip              код, настройки камер, план и веса моделей
#   videos/camera_new_1.mkv … 5    записи камер
# Запускается одной ячейкой. После обрыва запустите её снова: готовые камеры не считаются заново.
SECONDS = 150          #@param {type:"integer"}
START = 150            #@param {type:"integer"}
QUALITY = "high"       #@param ["high", "auto", "fast"]
SILHOUETTES = True     #@param {type:"boolean"}
DRIVE_FOLDER = "MTMC"  #@param {type:"string"}
# SECONDS — сколько секунд обработать, 0 — всю запись. START — с какой секунды.
# QUALITY: high — imgsz 1280, как при подборе порогов, самый точный и самый долгий вариант.
#          fast — imgsz 640, примерно в 3 раза быстрее, дальние люди теряются чаще.

import io, os, shutil, subprocess, sys, tarfile, threading, time, zipfile
from pathlib import Path

from google.colab import drive
drive.mount("/content/drive", force_remount=True)
DRIVE = Path("/content/drive/MyDrive") / DRIVE_FOLDER
WORK = Path("/content/Nuclear")
BUNDLE = DRIVE / "Nuclear_colab.zip"
SUFFIX = "_trial" if SECONDS > 0 else ""       # отрезок считается в свои папки, полный запуск они не трогают
BACKUP = DRIVE / f"results_office{SUFFIX}_backup.tar"
RESULTS, UI = WORK / f"results_office{SUFFIX}", WORK / f"ui_office{SUFFIX}"

# --- проверки до долгих шагов
videos = sorted(p for p in (DRIVE / "videos").glob("*") if p.is_file()) if (DRIVE / "videos").exists() else []
if not BUNDLE.exists():
    raise SystemExit(f"Нет {BUNDLE}: загрузите Nuclear_colab.zip в папку {DRIVE_FOLDER} на Диске")
if not videos:
    raise SystemExit(f"Нет записей: положите их в {DRIVE / 'videos'}")
print("Записи:", ", ".join(f"{p.name} ({p.stat().st_size / 2**20:.0f} МБ)" for p in videos))
r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
if r.returncode != 0:
    raise SystemExit("Видеокарты нет: Среда выполнения -> Сменить среду выполнения -> GPU, затем запустите снова")
print("Видеокарта:", r.stdout.strip())

# --- код и библиотеки
stamp_file, bundle_id = WORK / ".bundle", f"{BUNDLE.stat().st_size}-{BUNDLE.stat().st_mtime}"
if not stamp_file.exists() or stamp_file.read_text() != bundle_id:       # новый архив с Диска заменяет код и настройки
    print("Распаковываю проект...", flush=True)
    with zipfile.ZipFile(BUNDLE) as z:
        z.extractall(WORK)
    stamp_file.write_text(bundle_id)
pip = [sys.executable, "-m", "pip", "install", "-q"]
subprocess.run([*pip, "ultralytics==8.4.202", "lap==0.5.14", "tensorboard", "gdown", "imageio-ffmpeg", "timm",
                "safetensors"], check=True)
if subprocess.run([*pip, "torchreid==0.2.5"]).returncode != 0:
    subprocess.run([*pip, "git+https://github.com/KaiyangZhou/deep-person-reid"], check=True)
import matplotlib, torch
fonts = Path("/usr/share/fonts/truetype/dejavu")          # шрифты подписей: в Colab их нет
fonts.mkdir(parents=True, exist_ok=True)
for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
    shutil.copy2(Path(matplotlib.get_data_path()) / "fonts/ttf" / name, fonts / name)
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("torch не видит видеокарту: перезапустите среду выполнения и запустите ячейку снова")
import yaml
cfg_path = WORK / "configs/config.yaml"
c = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
vram = torch.cuda.get_device_properties(0).total_memory / 2**30
c["silhouette"]["batch"] = 16 if vram >= 14 else 4     # больший пакет не влезает в память и резко замедляет работу
cfg_path.write_text(yaml.safe_dump(c, allow_unicode=True, sort_keys=False), encoding="utf-8")
print(f"Видеопамять {vram:.0f} ГБ, пакет масок {c['silhouette']['batch']}")

# --- видео: с Диска читается медленно, поэтому копия на локальный диск. Время файла сохраняется: по нему узнаются готовые камеры
(WORK / "data/office").mkdir(parents=True, exist_ok=True)
for p in videos:
    dst = WORK / "data/office" / p.name
    if not dst.exists() or dst.stat().st_size != p.stat().st_size:
        print("Копирую", p.name, flush=True)
        shutil.copy2(p, dst)

# --- продолжение после обрыва
if BACKUP.exists() and not RESULTS.exists():
    print("Восстанавливаю готовые камеры с Диска...", flush=True)
    with tarfile.open(BACKUP) as t:
        t.extractall(WORK, filter="fully_trusted")


def save_backup() -> None:
    """Один файл на Диске: тысячи мелких кропов копировались бы туда очень долго."""
    tmp = BACKUP.with_suffix(".tmp")
    with tarfile.open(tmp, "w") as t:
        t.add(RESULTS, arcname=RESULTS.name)
    tmp.replace(BACKUP)


stop_watch = threading.Event()


def watch() -> None:
    """После каждой готовой камеры — копия на Диск. Файл состояния обновляется только когда камера записана целиком."""
    state, seen = RESULTS / "office_state.json", None
    while not stop_watch.wait(30):
        if state.exists() and state.stat().st_mtime != seen:
            seen = state.stat().st_mtime
            try:
                save_backup()
                print(f"[{time.strftime('%H:%M')}] готовые камеры сохранены на Диск", flush=True)
            except OSError as e:
                print("Не удалось сохранить на Диск:", e, flush=True)


# --- запуск
cmd = [sys.executable, "scripts/office.py", "--no-serve", "--quality", QUALITY]
if SECONDS > 0:
    cmd += ["--part", str(SECONDS), str(START)]
if not SILHOUETTES:
    cmd.append("--no-silhouettes")
threading.Thread(target=watch, daemon=True).start()
started = time.time()
proc = subprocess.Popen(cmd, cwd=WORK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                        env=dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1"))
text = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace", newline="")    # newline="": видно \r индикатора
line, shown = "", 0.0
for ch in iter(lambda: text.read(1), ""):
    if ch == "\n":
        print(line, flush=True)
        line = ""
    elif ch == "\r":                  # индикатор: не чаще раза в минуту, иначе вывод ячейки распухает
        if time.time() - shown > 60 and line.strip():
            print(line[-120:], flush=True)
            shown = time.time()
        line = ""
    else:
        line += ch
code = proc.wait()
stop_watch.set()
print(f"\nВремя: {(time.time() - started) / 3600:.1f} ч, код завершения {code}")
if code != 0:
    raise SystemExit("Запуск завершился с ошибкой: текст выше. Запустите ячейку снова: готовые камеры сохранены")

# --- результат на Диск
save_backup()
stamp = time.strftime("%Y%m%d_%H%M")
archive = shutil.make_archive(str(DRIVE / f"{UI.name}_{stamp}"), "zip", UI)
for name in ("tracks.csv", "persons.csv", "report.md", "rejected_tracks.csv"):
    if (RESULTS / name).exists():
        shutil.copy2(RESULTS / name, DRIVE / name)
print("На Диске:", Path(archive).name, f"({Path(archive).stat().st_size / 2**20:.0f} МБ), tracks.csv, persons.csv, report.md")
print("Распакуйте архив на компьютере и откройте index.html")

import pandas as pd
from IPython.display import Markdown, display
display(pd.read_csv(RESULTS / "tracks.csv", sep=";", encoding="utf-8-sig"))
display(Markdown((RESULTS / "report.md").read_text(encoding="utf-8")))
