"""HTTP-перемотка и загрузка локальных медиафайлов без зависимостей обработки."""

import os
import re
import shutil
import tempfile
from pathlib import Path


def upload_media(folder: Path, name: str, length: int, source, extensions, kind: str, validate=None):
    """Проверяет временный файл перед публикацией и сохраняет занятые имена."""
    is_reserved = getattr(os.path, "isreserved", lambda p: Path(p).is_reserved())
    if (not name or name != name.strip() or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            or is_reserved(name) or Path(name).suffix.lower() not in extensions):
        raise ValueError(f"Недопустимое имя или формат {kind}.")
    dest = folder / name
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(f"Файл «{name}» уже существует.")
    if length <= 0:
        raise ValueError("Пустой файл или неизвестный размер.")
    if length > shutil.disk_usage(folder).free:
        raise ValueError("Недостаточно места для файла.")
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=folder, prefix=".mtmc-upload-", suffix=".part", delete=False) as f:
            temp = Path(f.name)
            remaining = length
            while remaining:
                chunk = source.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise ValueError("Загрузка оборвалась. Файл не сохранён.")
                f.write(chunk)
                remaining -= len(chunk)
        result = validate(temp) if validate else None
        if os.name == "nt":
            temp.rename(dest)
        else:
            os.link(temp, dest)
        return result
    except FileExistsError as e:
        raise FileExistsError(f"Файл «{name}» уже существует.") from e
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def _range_handler():
    """SimpleHTTPRequestHandler с поддержкой Range: без неё браузер не может перематывать mp4."""
    import http.server

    class H(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):   # без шума в консоли
            pass

        def send_head(self):
            rng = self.headers.get("Range")
            path = self.translate_path(self.path)
            if not rng or os.path.isdir(path) or not os.path.exists(path):
                return super().send_head()
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            size = os.path.getsize(path)
            a = int(m.group(1)) if m and m.group(1) else 0
            b = int(m.group(2)) if m and m.group(2) else size - 1
            b = min(b, size - 1)
            if a > b:
                self.send_error(416)
                return None
            f = open(path, "rb")
            f.seek(a)
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
            self.send_header("Content-Length", str(b - a + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self._remaining = b - a + 1
            return f

        def copyfile(self, src, dst):
            n = getattr(self, "_remaining", None)
            self._remaining = None
            try:
                if n is None:
                    return super().copyfile(src, dst)
                while n > 0:
                    buf = src.read(min(1 << 16, n))
                    if not buf:
                        break
                    dst.write(buf)
                    n -= len(buf)
            except ConnectionError:      # браузер сам обрывает загрузку при перемотке — это не ошибка
                self.close_connection = True

        def end_headers(self):
            if not self.headers.get("Range"):
                self.send_header("Accept-Ranges", "bytes")
            super().end_headers()

    return H


