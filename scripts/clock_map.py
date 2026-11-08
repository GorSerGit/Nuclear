"""Общее время камер по часам, впечатанным в кадр.

python scripts/clock_map.py read <камера> <папка>     читает секунды по всей записи, пишет <папка>/clockN.json
python scripts/clock_map.py write <папка>             записывает сдвиг старта в offset_s файлов configs/office/camN.json

Камеры начали запись не одновременно. Сравнивать часы надо по номеру кадра, как работает обработка: перемотка по времени
файла даёт ложные расхождения до 12 с. Читаются только секунды: они на тёмном фоне, а минуты восстанавливаются из смены секунд. Образцы цифр
собираются из той же записи: секунды идут подряд, поэтому порядок появления единиц задаёт цифры.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
X0 = {1: 0, 2: 0, 3: 0, 4: 1950, 5: 0}
Y = {1: (10, 62), 2: (10, 62), 3: (10, 62), 4: (22, 82), 5: (10, 62)}
TENS, UNITS = (547, 581), (580, 614)        # положение цифр секунд от начала строки
FIRST = {1: 39, 2: 40, 3: 40, 4: 43, 5: 42}  # секунды на первом кадре, прочитаны глазами
BASE = 17 * 3600 + 38 * 60                  # 17:38:00: от этого момента считается общее время
THR = 170


def cell(frame: np.ndarray, c: int, box: tuple[int, int]) -> np.ndarray:
    y0, y1 = Y[c]
    x0 = X0[c]
    g = cv2.cvtColor(frame[y0:y1, x0 + box[0]:x0 + box[1]], cv2.COLOR_BGR2GRAY)
    return (g > THR).astype(np.uint8)


def score(a: np.ndarray, b: np.ndarray, shift: int = 4) -> float:
    best = 0.0
    for dx in range(-shift, shift + 1):
        s = np.roll(b, dx, 1)
        u = (a | s).sum()
        best = max(best, float((a & s).sum() / max(u, 1)))
    return best


def learn(c: int, cap: cv2.VideoCapture, fps: float) -> dict[int, np.ndarray]:
    """Образцы десяти цифр из первых 40 с."""
    reps: dict[int, np.ndarray] = {}
    digit = FIRST[c] % 10
    last_change = 0
    prev = None
    step = max(int(fps // 5), 1)
    for i in range(int(fps * 40)):
        if not cap.grab():
            break
        if i % step:
            continue
        ok, fr = cap.retrieve()
        if not ok:
            break
        u = cell(fr, c, UNITS)
        if prev is None:
            reps[digit] = u
        elif score(u, prev) < 0.75:
            digit = (digit + max(int(round((i - last_change) / fps)), 1)) % 10
            last_change = i
            reps.setdefault(digit, u)
        prev = u
    return reps


def read(c: int, out: Path) -> None:
    path = f"data/office/camera_new_{c}.mkv"
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    reps = learn(c, cap, fps)
    cap.release()
    print(f"cam{c}: образцов цифр {sorted(reps)}", flush=True)
    cap = cv2.VideoCapture(path)
    step = max(int(fps // 5), 1)
    rows = []
    i = 0
    while cap.grab():
        if i % step == 0:
            ok, fr = cap.retrieve()
            if not ok:
                break
            pair = []
            for box in (TENS, UNITS):
                m = cell(fr, c, box)
                s = {d: score(m, r) for d, r in reps.items()}
                d = max(s, key=s.get)
                pair.append((d, s[d]))
            rows.append([i, pair[0][0] * 10 + pair[1][0], round(min(pair[0][1], pair[1][1]), 3)])
        i += 1
    cap.release()
    (out / f"clock{c}.json").write_text(json.dumps({"frames": i, "fps": fps, "rows": rows}), encoding="utf-8")
    print(f"cam{c}: прочитано {len(rows)} меток, кадров {i}", flush=True)


def ticks(c: int, folder: Path) -> tuple[np.ndarray, float]:
    """Смены секунд: [[кадр, время по часам, с], ...]. Смена засчитывается, если держится на следующей метке."""
    j = json.loads((folder / f"clock{c}.json").read_text(encoding="utf-8"))
    r = np.array(j["rows"], float)
    t, prev = float(38 * 60 + FIRST[c]), FIRST[c]
    out = [[0.0, t]]
    for i in range(1, len(r) - 1):
        fr, ss, sc = r[i]
        if sc < 0.6 or r[i + 1, 1] != ss or ss == prev:
            continue
        d = (ss - prev) % 60
        if d <= 25:
            t += d
            prev = ss
            out.append([fr, t])
    return np.array(out), float(j["fps"])


def write(folder: Path) -> None:
    """Сдвиг старта камер. Часы идут ровно с номером кадра, поэтому карта не нужна: хватает постоянного сдвига."""
    shift = {}
    for c in FIRST:
        a, fps = ticks(c, folder)
        rest = a[:, 1] - a[:, 0] / fps
        shift[c] = float(np.median(rest))
        print(f"cam{c}: разброс {np.ptp(rest):.2f} с за запись")
    base = min(shift.values())
    for c, v in shift.items():
        f = ROOT / "configs/office" / f"cam{c}.json"
        d = json.loads(f.read_text(encoding="utf-8"))
        d["offset_s"] = round(v - base, 2)
        f.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"cam{c}: сдвиг {d['offset_s']} с")


if __name__ == "__main__":
    if sys.argv[1] == "read":
        read(int(sys.argv[2]), Path(sys.argv[3]))
    elif sys.argv[1] == "write":
        write(Path(sys.argv[2]))
