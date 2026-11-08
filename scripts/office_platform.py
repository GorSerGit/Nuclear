"""Действия на компьютере с локальным сервером."""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path

from office import ROOT

NVIDIA_DRIVERS = "https://www.nvidia.com/en-us/drivers/"
LINUX_DRIVERS = "https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/latest/choose-an-installation-method.html"

def linux_nvidia_present() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    for device in Path("/sys/bus/pci/devices").glob("*"):
        try:
            if ((device / "vendor").read_text().strip() == "0x10de"
                    and (device / "class").read_text().strip().startswith("0x03")):
                return True
        except OSError:
            continue
    return False

def open_yaml(path: Path) -> None:
    if os.name == "nt":
        try:
            os.startfile(path)
        except OSError:
            subprocess.Popen(["notepad.exe", str(path)])
    elif os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        subprocess.Popen(["xdg-open", str(path)])
    else:
        raise ValueError("На компьютере с сервером нет графического сеанса для открытия YAML.")


def open_nvidia_drivers() -> None:
    if os.name == "nt":
        webbrowser.open(NVIDIA_DRIVERS)


def nvidia_info() -> dict:
    result = {"name": None, "driver": None, "cuda": None, "arch": None}
    exe = "nvidia-smi"
    if os.name == "nt":
        system = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/nvidia-smi.exe"
        if system.is_file():
            exe = str(system)
    try:
        def query(*args: str) -> str:
            return subprocess.check_output([exe, *args], stderr=subprocess.DEVNULL,
                                           text=True, errors="replace", timeout=10)

        summary = query()
        match = re.search(r"CUDA(?: UMD)? Version:\s*(\d+\.\d+)", summary)
        if match:
            result["cuda"] = match[1]
        try:
            name, driver, arch = next(csv.reader(query("--query-gpu=name,driver_version,compute_cap",
                                                       "--format=csv,noheader").splitlines(), skipinitialspace=True))
            result.update(name=name, driver=driver, arch=arch)
        except (ValueError, StopIteration, subprocess.SubprocessError):
            driver = re.search(r"(?:KMD|Driver) Version:\s*([\d.]+)", summary)
            result["driver"] = driver[1] if driver else None
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def windows_driver(action: str, offer: dict | None = None, progress=None) -> dict:
    """Windows Update возвращает предложение, а установка идёт через отдельный процесс с UAC."""
    if os.name != "nt":
        return {}
    with tempfile.TemporaryDirectory(prefix="mtmc-driver-") as temp:
        report = Path(temp) / "report.json"
        command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                   "-File", str(ROOT / "scripts/nvidia_update.ps1"),
                   "-Action", action, "-Report", str(report)]
        if offer is not None:
            command += ["-UpdateId", offer["id"], "-Revision", str(offer["revision"])]
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace") as process:
            for line in process.stdout:
                if progress:
                    progress(line.rstrip())
            code = process.wait()
        if not report.is_file():
            raise ValueError(f"Windows Update не вернул результат (код {code}).")
        data = json.loads(report.read_text(encoding="utf-8-sig"))
        if code or data.get("error"):
            raise ValueError(data.get("error") or f"Windows Update завершился с кодом {code}.")
        return data
