"""Архив для Colab: код, настройки офиса, план и веса. Без результатов, видео и окружения.

python scripts/make_colab_zip.py     пишет Nuclear_colab.zip в корень проекта
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAKE = ["mtmc", "scripts", "configs", "weights", "tests", "requirements.txt", "SOLUTION.md", "DECISIONS.md", "README.md",
        "НОВЫЕ_ВИДЕО.md"]
SKIP = {"__pycache__", ".pytest_cache"}


def main() -> None:
    out = ROOT / "Nuclear_colab.zip"
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for item in TAKE:
            p = ROOT / item
            for f in ([p] if p.is_file() else sorted(p.rglob("*"))):
                if f.is_file() and not set(f.relative_to(ROOT).parts) & SKIP and f.suffix != ".bak":
                    z.write(f, f.relative_to(ROOT).as_posix())
                    n += 1
    print(f"{out}: файлов {n}, {out.stat().st_size / 2**20:.0f} МБ")


if __name__ == "__main__":
    main()
