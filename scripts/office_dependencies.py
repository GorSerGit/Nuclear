"""Проверка и фоновая установка зависимостей локального сайта."""
from __future__ import annotations

import importlib.metadata
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
from contextlib import closing
from pathlib import Path

from office import ROOT
from office_platform import (LINUX_DRIVERS, NVIDIA_DRIVERS, linux_nvidia_present, nvidia_info,
                             open_nvidia_drivers, windows_driver)

HISTORY = ROOT / "results_office_generations"
STATE = HISTORY / "dependencies.json"
LOG = HISTORY / "dependencies.log"
MODES = ("auto", "cpu", "gpu")
TORCH = "2.7.1"
VISION = "0.22.1"
PYPI = "https://pypi.org/simple"
BASE = {"numpy", "opencv-python", "scipy", "pyyaml"}
AFTER_TORCH = {"ultralytics", "timm", "torchreid", "tensorboard", "lap", "safetensors"}
WEIGHTS = ("yolo11m-pose.pt", "yolo11m-seg.pt", "osnet_x1_0_msmt17.pth", "personvit_msmt17_s.safetensors")
BOOTSTRAP_PROBE = "import cv2, numpy, scipy.optimize, yaml"
RUNTIME_PROBE = """
import json
import imageio_ffmpeg, pandas, pyarrow, pydantic, safetensors, timm, torch, torchreid, ultralytics
from torchvision.ops import nms

def check(device):
    boxes = torch.tensor([[0., 0., 2., 2.]], device=device)
    scores = torch.tensor([0.9], device=device)
    if nms(boxes, scores, 0.5).cpu().tolist() != [0]:
        raise RuntimeError("torchvision NMS")

check("cpu")
imageio_ffmpeg.get_ffmpeg_exe()
result = {"cpu": True, "gpu": False, "runtime_cuda": torch.version.cuda, "gpu_name": None}
try:
    if torch.cuda.is_available():
        check("cuda:0")
        result.update(gpu=True, gpu_name=torch.cuda.get_device_name(0))
except Exception as error:
    result["cuda_error"] = str(error)
print(json.dumps(result))
"""


def requirements() -> dict[str, str]:
    specs = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        spec = line.partition("#")[0].strip()
        if spec:
            name = spec.split("==")[0].lower()
            if name not in {"torch", "torchvision", "pytest"}:
                specs[name] = spec
    return specs


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def package_versions() -> dict[str, str | None]:
    return {name: version(name) for name in (*requirements(), "torch", "torchvision")}


def base_matches() -> bool:
    specs = requirements()
    return all(specs[name].lower() == f"{name}=={version(name)}" for name in BASE)


def weight_status() -> dict[str, bool]:
    out = {}
    for name in WEIGHTS:
        try:
            out[name] = (ROOT / "weights" / name).stat().st_size > 0
        except OSError:
            out[name] = False
    return out


def choose_cuda(info: dict) -> str | None:
    try:
        limit = tuple(map(int, info["cuda"].split(".")))
        arch = tuple(map(int, (info.get("arch") or "0.0").split(".")))
    except (AttributeError, KeyError, ValueError):
        return None
    for capability, tag in (((12, 8), "cu128"), ((12, 6), "cu126"), ((11, 8), "cu118")):
        if limit >= capability and (arch < (12, 0) or tag == "cu128"):
            return tag
    return None


def probe(code: str, timeout: int = 90) -> str:
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            env={**os.environ, "PYTHONUTF8": "1"}, timeout=timeout)
    if result.returncode:
        raise ValueError((result.stderr or result.stdout or f"Код выхода {result.returncode}").strip()[-1600:])
    return result.stdout


def stream(args: list[str], emit) -> None:
    with subprocess.Popen([sys.executable, *args], cwd=ROOT, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace", env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}) as child:
        for line in child.stdout:
            emit(line.rstrip())
        if child.wait() != 0:
            raise ValueError(f"Команда завершилась с кодом {child.returncode}. Подробности в журнале.")


def pip_install(specs, emit, index: str = PYPI, constraints: Path | None = None) -> None:
    args = ["-m", "pip", "--isolated", "install", "--no-input", "--index-url", index, "--progress-bar", "on"]
    if constraints is not None:
        args.extend(("-c", str(constraints)))
    stream([*args, *specs], emit)


def saved_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(data: dict) -> None:
    HISTORY.mkdir(exist_ok=True)
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temp.replace(STATE)


class Dependencies:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.worker: threading.Thread | None = None
        self.offer: dict | None = None
        saved = saved_state()
        self.mode = saved.get("mode", "auto")
        self.state = {**saved, "base_ready": False, "cpu": False, "gpu": False}
        self.state.setdefault("stage", "Проверка окружения")
        self.state.setdefault("phase", "idle")
        if self.state["phase"] in ("working", "driver"):
            self.state.update(phase="idle", stage="Предыдущая установка прервана. Можно повторить.")
        self.launch(self.initial_check)

    def change(self, **fields) -> None:
        with self.lock:
            self.state.update(fields)

    def emit(self, line: str) -> None:
        with LOG.open("a", encoding="utf-8") as log:
            log.write(line + "\n")

    def stage(self, message: str) -> None:
        self.change(stage=message)
        self.emit("== " + message)

    def launch(self, task) -> None:
        self.worker = threading.Thread(target=task)
        self.worker.start()

    def busy(self) -> bool:
        return self.state["phase"] in ("working", "driver") or bool(self.worker and self.worker.is_alive())

    def inspect(self) -> None:
        info = nvidia_info()
        self.change(nvidia=info, cpu=False, gpu=False, runtime_cuda=None, gpu_name=None,
                    cuda_error=None, error=None)
        if not base_matches():
            self.change(base_ready=False, error="Установите версии базовых библиотек из requirements.txt.")
            return
        try:
            probe(BOOTSTRAP_PROBE)
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            self.change(base_ready=False, error=str(e))
            return
        self.change(base_ready=True)
        try:
            if not all(package_versions().values()) or not all(weight_status().values()):
                self.change(error="Не все пакеты обработки или веса установлены.")
                return
            health = json.loads(probe(RUNTIME_PROBE).strip().splitlines()[-1])
            self.change(**health)
        except (OSError, ValueError, IndexError, subprocess.SubprocessError) as e:
            self.change(error=str(e))

    def initial_check(self) -> None:
        self.inspect()
        if self.state["phase"] == "reboot":
            try:
                boot = windows_driver("Boot").get("boot")
            except (OSError, ValueError, subprocess.SubprocessError):
                boot = None
            if boot and boot != self.state.get("boot"):
                if choose_cuda(self.state["nvidia"]):
                    self.change(phase="working", stage="После перезагрузки драйвер доступен.")
                    self.install()
                else:
                    self.change(needs_driver=True)
                    self.finish("error", "После перезагрузки совместимая CUDA не найдена.")
        elif self.state["stage"] == "Проверка окружения":
            self.change(stage="Проверка завершена." if self.state["cpu"] else "Требуется установка.")

    def status(self) -> dict:
        with self.lock:
            state = dict(self.state)
        try:
            with LOG.open("rb") as f:
                f.seek(max(LOG.stat().st_size - 24_576, 0))
                log = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", f.read().decode("utf-8", errors="replace"))
        except OSError:
            log = ""
        return {**state, "busy": self.busy(), "python": sys.version.split()[0], "packages": package_versions(),
                "weights": weight_status(), "log": log, "offer": self.offer if state["phase"] == "driver" else None,
                "driver_help": NVIDIA_DRIVERS if os.name == "nt" else LINUX_DRIVERS}

    def device(self, requested: str) -> str:
        if self.busy():
            raise ValueError("Дождитесь завершения проверки или установки зависимостей.")
        if requested == "gpu" and not self.state["gpu"]:
            raise ValueError("Выбран GPU, но CUDA не прошла проверку. Откройте вкладку «Зависимости».")
        if requested != "cpu" and self.state["gpu"]:
            return "gpu"
        if self.state["cpu"]:
            return "cpu"
        raise ValueError("Обработка недоступна. Установите зависимости во вкладке «Зависимости».")

    def start(self, mode: str) -> None:
        if self.busy():
            raise ValueError("Проверка или установка зависимостей уже выполняется.")
        if self.state["phase"] == "reboot":
            raise ValueError("Сначала перезагрузите компьютер для завершения установки драйвера.")
        if mode not in MODES:
            raise ValueError("Неизвестное устройство.")
        self.mode, self.offer = mode, None
        LOG.write_text("", encoding="utf-8")
        save_state({"phase": "working"})
        self.change(phase="working", stage="Подготовка", error=None, cpu=False, gpu=False, needs_driver=False)
        self.launch(lambda: self.install())

    def confirm(self, install_driver: bool) -> None:
        if self.state["phase"] != "driver" or not self.offer or (self.worker and self.worker.is_alive()):
            raise ValueError("Нет готового предложения Windows Update.")
        self.change(phase="working")
        self.launch(lambda: self.install(install_driver))

    def finish(self, phase: str, message: str, **details) -> None:
        self.change(phase=phase, stage=message, **details)
        self.emit("== " + message)
        save_state({"phase": phase, "stage": message, "mode": self.mode, **details}
                   if phase in ("reboot", "restart") else {})

    def install(self, install_driver: bool | None = None) -> None:
        try:
            specs = requirements()
            tag = None
            if install_driver is None:
                self.stage("Библиотеки разметки")
                if not base_matches() and any(name in sys.modules for name in ("cv2", "numpy", "scipy", "yaml")):
                    self.finish("restart", "Перезапустите сайт через !start.bat или start.sh для обновления разметки.")
                    return
                if not base_matches() or not self.state["base_ready"]:
                    self.change(base_ready=False)
                    pip_install([specs[name] for name in sorted(BASE)], self.emit)
                probe(BOOTSTRAP_PROBE)
                self.change(base_ready=True)
                self.stage("Проверка драйвера NVIDIA" if self.mode != "cpu" else "Подготовка сборки для CPU")
                info = nvidia_info()
                self.change(nvidia=info)
                tag = choose_cuda(info) if self.mode != "cpu" else None
                if self.mode != "cpu" and not tag:
                    try:
                        self.offer = windows_driver("Find", progress=self.emit)
                    except (ValueError, OSError) as e:
                        self.emit(str(e))
                        self.offer = None
                    if self.offer and self.offer.get("id"):
                        self.change(phase="driver", stage="Windows Update нашёл подходящий драйвер.")
                        return
                    if self.offer and self.offer.get("warning"):
                        self.emit(self.offer["warning"])
                    hardware = bool(info["driver"] or (self.offer or {}).get("hardware") or linux_nvidia_present())
                    self.change(needs_driver=hardware or self.mode == "gpu")
                    if hardware:
                        open_nvidia_drivers()
            elif install_driver:
                info = nvidia_info()
                if not choose_cuda(info):
                    self.stage("Установка драйвера NVIDIA через Windows Update")
                    result = windows_driver("Elevate", self.offer, self.emit)
                    if result.get("reboot"):
                        self.finish("reboot", "Перезагрузите компьютер и снова откройте сайт для проверки CUDA.",
                                    boot=result["boot"])
                        return
                    info = nvidia_info()
                self.change(nvidia=info)
                tag = choose_cuda(info)
                self.change(needs_driver=not bool(tag))
            elif self.mode == "gpu":
                raise ValueError("Установка драйвера отклонена. Для режима GPU нужна рабочая CUDA.")
            else:
                self.change(needs_driver=True)

            if self.mode == "gpu" and not tag:
                raise ValueError("Совместимая CUDA недоступна. Воспользуйтесь официальной инструкцией по драйверу.")
            self.stage("Библиотеки обработки")
            common = [spec for name, spec in specs.items() if name not in BASE | AFTER_TORCH]
            pip_install(common, self.emit)
            self.install_torch(tag or "cpu", specs)
            self.stage("Веса моделей")
            stream([str(ROOT / "scripts/download_weights.py")], self.emit)
            self.stage("Проверка установленных пакетов и устройства")
            stream(["-m", "pip", "check"], self.emit)
            self.inspect()
            if not self.state["cpu"]:
                raise ValueError(self.state["error"] or "Библиотеки обработки не прошли проверку.")
            if tag and not self.state["gpu"]:
                self.emit(self.state.get("cuda_error") or "PyTorch не смог выполнить операцию CUDA.")
                if self.mode == "gpu":
                    raise ValueError("CUDA не прошла проверку PyTorch и torchvision.")
                self.stage("CUDA недоступна, устанавливаю сборку для CPU")
                self.install_torch("cpu", specs)
                self.inspect()
                if not self.state["cpu"]:
                    raise ValueError(self.state["error"] or "Сборка CPU не прошла проверку.")
            self.finish("done", "Готово. GPU доступен." if self.state["gpu"] else "Готово. Доступна обработка на CPU.")
        except (ValueError, OSError, subprocess.SubprocessError) as e:
            self.change(error=str(e))
            self.finish("error", f"Установка остановлена. {e}")

    def install_torch(self, tag: str, specs: dict) -> None:
        pair = [f"torch=={TORCH}+{tag}", f"torchvision=={VISION}+{tag}"]
        self.stage(f"PyTorch и torchvision ({tag})")
        pip_install(pair, self.emit, f"https://download.pytorch.org/whl/{tag}")
        self.stage("Библиотеки моделей")
        constraints = HISTORY / "pip-constraints.txt"
        constraints.write_text("\n".join([*pair, *(specs[name] for name in sorted(BASE))]) + "\n", encoding="utf-8")
        try:
            pip_install([specs[name] for name in sorted(AFTER_TORCH)], self.emit, constraints=constraints)
        finally:
            constraints.unlink(missing_ok=True)


def prepare() -> None:
    if saved_state().get("phase") != "restart":
        return
    with closing(sqlite3.connect(HISTORY / "server.sqlite", timeout=0)) as lock:
        lock.execute("BEGIN EXCLUSIVE")
        if saved_state().get("phase") != "restart":
            return
        try:
            specs = requirements()
            pip_install([specs[name] for name in sorted(BASE)], print)
            probe(BOOTSTRAP_PROBE)
            save_state({"phase": "idle", "stage": "Библиотеки разметки готовы. Продолжите установку на сайте."})
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            save_state({"phase": "error", "stage": str(e)})


if __name__ == "__main__" and sys.argv[1:] == ["--prepare"]:
    prepare()
