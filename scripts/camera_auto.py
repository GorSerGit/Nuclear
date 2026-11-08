"""Авторазметка точек для приложения настройки камер: 2 ручные пары -> 10 разнесённых пар «кадр <-> план».

    python scripts/camera_auto.py [--configs configs/office] [--videos data/office] [--port 8791] [--no-browser]

Это надстройка: scripts/camera_app.py и camera_app.html не меняются. Здесь запускается то же приложение, на странице
появляется кнопка «Авторазметка». Паспорта меняются только кнопкой «Сохранить привязку», как и раньше.

Как считается:
1. Объектив и наклон. На неподвижном кадре ищутся отрезки линий. Стены, двери и плитка идут по трём взаимно
   перпендикулярным направлениям, поэтому поворот камеры находится прямо по кадру. Один объектив подбирается сразу по
   всем камерам: r -> угол θ = x(1 + k x²), x = r / f. Результат хранится в site_auto_lens.json рядом с паспортами.
2. Положение. При известном повороте точка пола F = C + h·a, где a считается из пикселя. Неизвестны только C и h,
   поэтому хватает двух пар, а 4–5 пар дают заметно точнее. Подбор устойчив к выбросам: не сходящиеся пары
   показываются отдельно. Из четырёх возможных поворотов на 90° остаётся тот, что сходится с парами.
3. Новые пары. На плане берутся точки пола, видимые из камеры, и проецируются в кадр.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import threading
import webbrowser
import http.server
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares, minimize  # noqa: E402

import camera_app as app  # noqa: E402

CACHE = "site_auto_lens.json"    # office.passport_files считает паспортом любой *.json, кроме site*
PX_SCALE = 40.0         # пиксели кадра: ошибки крупнее этой считаются выбросами мягко
PX_INLIER = 90.0        # пара согласна с решением, если расхождение в кадре меньше
MIN_BASE_M = 1.5        # две пары ближе этого расстояния положение не определяют
H_RANGE = (1.2, 6.0)    # допустимая высота камеры, м
MAX_THETA = 1.1         # дальше от оси модель объектива не проверялась, рад
N_PAIRS = 10
STEP_M = 0.4            # шаг точек-кандидатов на плане
MAX_RANGE_M = 14.0


# ---------------------------------------------------------------- модель камеры
def rays(p: np.ndarray, lens: dict) -> np.ndarray:
    """Пиксели кадра (n, 2) -> направления в системе камеры (n, 3), z = 1."""
    v = np.asarray(p, float).reshape(-1, 2) - [lens["cx"], lens["cy"]]
    r = np.hypot(*v.T) + 1e-9
    x = r / lens["f"]
    th = np.minimum(x * (1 + lens["k"] * x * x), 1.45)
    return np.c_[v / r[:, None] * np.tan(th)[:, None], np.ones(len(v))]


def project(P: np.ndarray, C: np.ndarray, h: float, R: np.ndarray, lens: dict) -> tuple[np.ndarray, np.ndarray]:
    """Точки пола в метрах плана -> (пиксели кадра, угол от оптической оси)."""
    c3 = np.c_[np.asarray(P, float).reshape(-1, 2) - C, np.full(len(P), h)] @ R.T
    a = np.hypot(c3[:, 0], c3[:, 1])
    th = np.arctan2(a, c3[:, 2])
    k = lens["k"]
    x = th.copy()
    for _ in range(8):
        x = x - (x * (1 + k * x * x) - th) / (1 + 3 * k * x * x)
    r = x * lens["f"]
    s = r / (a + 1e-9)
    return np.c_[lens["cx"] + c3[:, 0] * s, lens["cy"] + c3[:, 1] * s], th


def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """План (x вправо, y вниз, z в пол) -> камера. yaw — курс на плане, pitch — наклон вниз от горизонта."""
    f = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
    r = np.cross([0, 0, 1.0], f)
    r /= np.linalg.norm(r)
    c, s = np.cos(roll), np.sin(roll)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ np.stack([r, np.cross(f, r), f])


# ---------------------------------------------------------------- объектив и поворот по кадру
def still_frame(video: Path, n: int = 9) -> np.ndarray:
    """Медиана кадров в половинном размере: люди исчезают, стены остаются."""
    cap = cv2.VideoCapture(os.path.relpath(video, ROOT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * total / n))
        ok, img = cap.read()
        if ok:
            frames.append(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        raise ValueError(f"не читается запись {video.name}")
    return np.median(np.stack(frames), 0).astype(np.uint8)


def segments(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Отрезки линий в пикселях полного кадра и их веса по длине."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    if found is None:
        return np.zeros((0, 4)), np.zeros(0)
    L = found.reshape(-1, 4) * 2.0
    H, W = gray.shape[0] * 2, gray.shape[1] * 2
    length = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])
    far = np.maximum(np.hypot(L[:, 0] - W / 2, L[:, 1] - H / 2), np.hypot(L[:, 2] - W / 2, L[:, 3] - H / 2))
    top = H * 0.036                                       # надпись с часами
    keep = (length >= 44) & (far < W * 0.56) & (L[:, 1] > top) & (L[:, 3] > top)
    return L[keep], np.minimum(length[keep], 300.0)


def normals(L: np.ndarray, lens: dict) -> np.ndarray:
    n = np.cross(rays(L[:, :2], lens), rays(L[:, 2:], lens))
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def agreement(R: np.ndarray, N: np.ndarray, w: np.ndarray, sigma: float) -> float:
    """Доля длины отрезков, идущих по осям помещения."""
    return float((w * np.exp(-((N @ R) ** 2).min(1) / (2 * sigma ** 2))).sum() / max(w.sum(), 1e-9))


def coarse_rotation(N: np.ndarray, w: np.ndarray) -> np.ndarray:
    grid = itertools.product(np.radians(np.arange(0, 90, 3)), np.radians(np.arange(15, 78, 3)),
                             np.radians(np.arange(-12, 13, 3)))
    return np.array(max(grid, key=lambda v: agreement(rotation(*v), N, w, 0.02)))


def refine_rotation(x0: np.ndarray, N: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    o = minimize(lambda v: -agreement(rotation(*v), N, w, 0.008), x0, method="Nelder-Mead",
                 options={"xatol": 1e-3, "fatol": 2e-4, "maxiter": 200})
    return o.x, float(-o.fun)


def fit_scene(data: list[tuple[np.ndarray, np.ndarray]], W: int, H: int) -> tuple[dict, list[np.ndarray], list[float]]:
    """Один объектив на все камеры и поворот каждой: (объектив, [yaw, pitch, roll], согласие линий)."""
    base = {"f": 0.6 * W, "k": 0.1, "cx": W / 2, "cy": H / 2}
    starts = [coarse_rotation(normals(L, base), w) for L, w in data]
    best = None
    for f in (0.54, 0.57, 0.60, 0.63, 0.66):
        for k in (0.04, 0.10, 0.16, 0.22):
            lens = {**base, "f": f * W, "k": k}
            fits = [refine_rotation(x, normals(L, lens), w) for x, (L, w) in zip(starts, data)]
            total = float(np.mean([s for _, s in fits]))
            if best is None or total > best[0]:
                best = (total, lens, [x for x, _ in fits], [s for _, s in fits])
    return best[1], best[2], best[3]


def scene(cfg_dir: Path, videos_dir: Path) -> dict:
    """Объектив и повороты камер, один раз на набор записей. Первый расчёт занимает минуты."""
    videos = sorted(app.find_videos(videos_dir), key=lambda p: p.name)
    if not videos:
        raise ValueError("нет записей")
    key = [[v.name, v.stat().st_size] for v in videos]
    f = cfg_dir / CACHE
    if f.exists():
        d = app.read_json(f)
        if d.get("key") == key:
            return d
    frames = [still_frame(v) for v in videos]
    H, W = frames[0].shape[0] * 2, frames[0].shape[1] * 2
    lens, rot, score = fit_scene([segments(i) for i in frames], W, H)
    d = {"key": key, "lens": lens, "size": [W, H],
         "cameras": {v.name: {"yaw_pitch_roll": [float(a) for a in x], "agreement": round(s, 3)}
                     for v, x, s in zip(videos, rot, score)}}
    app.write_json(f, d)
    return d


# ---------------------------------------------------------------- положение по парам
def floor_slopes(uv: np.ndarray, R: np.ndarray, lens: dict) -> tuple[np.ndarray, np.ndarray]:
    """F = C + h·a: коэффициенты a и признак, что луч идёт в пол."""
    d = rays(uv, lens) @ R
    down = d[:, 2] > 0.02
    return d[:, :2] / np.where(down, d[:, 2], 1.0)[:, None], down


def solve_position(a: np.ndarray, F: np.ndarray, idx: list[int]) -> tuple[np.ndarray, float | None]:
    """Наименьшие квадраты для C и h по выбранным парам. Дальние точки неустойчивы и весят меньше."""
    A, b = [], []
    for i in idx:
        w = 1.0 / (1.0 + float(a[i] @ a[i]))
        A += [[w, 0, w * a[i, 0]], [0, w, w * a[i, 1]]]
        b += [w * F[i, 0], w * F[i, 1]]
    x = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    return x[:2], float(x[2]) if H_RANGE[0] <= x[2] <= H_RANGE[1] else None


def locate(pairs: np.ndarray, ppm: float, x: np.ndarray, lens: dict, look_at: np.ndarray | None = None) -> dict:
    """Поза камеры по парам [[u, v, X, Y]] (пиксели плана): 4 поворота на 90°, устойчивый подбор в пикселях кадра."""
    F = pairs[:, 2:] / ppm
    uv = pairs[:, :2]
    n = len(F)
    if n < 2:
        raise ValueError("нужно 2 пары точек: щёлкните 2 точки на полу в кадре и на плане")
    if max(np.hypot(*(F[i] - F[j])) for i, j in itertools.combinations(range(n), 2)) < MIN_BASE_M:
        raise ValueError(f"точки слишком близко: разнесите их хотя бы на {MIN_BASE_M:g} м")
    best = None
    for q in range(4):
        yaw = x[0] + q * np.pi / 2
        R = rotation(yaw, x[1], x[2])
        a, down = floor_slopes(uv, R, lens)
        seeds = []
        for i, j in itertools.combinations(range(n), 2):
            if down[i] and down[j] and np.hypot(*(F[i] - F[j])) >= MIN_BASE_M:
                C, h = solve_position(a, F, [i, j])
                if h is not None:
                    seeds.append([*C, h])
        for s in seeds[:6]:
            o = least_squares(lambda v: (project(F, v[:2], v[2], R, lens)[0] - uv).ravel(), s, loss="soft_l1",
                              f_scale=PX_SCALE, bounds=([-np.inf, -np.inf, H_RANGE[0]], [np.inf, np.inf, H_RANGE[1]]))
            if best is None or o.cost < best["cost"]:
                px = np.hypot(*(project(F, o.x[:2], o.x[2], R, lens)[0] - uv).T)
                best = {"cost": float(o.cost), "C": o.x[:2], "h": float(o.x[2]), "yaw": yaw, "R": R, "px": px,
                        "inliers": [m for m in range(n) if px[m] <= PX_INLIER]}
    if best is None:
        raise ValueError("пары не сходятся ни с одним положением камеры: проверьте, что точки лежат на полу")
    best["err"] = best["px"]
    if n == 2 and look_at is not None:                    # двух пар мало для уверенного выбора из четырёх курсов
        d = np.asarray(look_at) - best["C"]
        off = abs((best["yaw"] - np.arctan2(d[1], d[0]) + np.pi) % (2 * np.pi) - np.pi)
        best["look_warning"] = bool(off > np.radians(100))
    return best

# ---------------------------------------------------------------- новые пары
def visible_floor(walls: np.ndarray, ppm: float, C: np.ndarray) -> np.ndarray:
    """Узлы сетки на плане (метры), к которым от камеры идёт отрезок без стен."""
    Hp, Wp = walls.shape
    ax = np.arange(-MAX_RANGE_M, MAX_RANGE_M + STEP_M, STEP_M)
    P = C + np.stack(np.meshgrid(ax, ax), -1).reshape(-1, 2)
    P = P[np.hypot(*(P - C).T) <= MAX_RANGE_M]
    px = np.rint(P * ppm).astype(int)
    P = P[(px[:, 0] >= 0) & (px[:, 1] >= 0) & (px[:, 0] < Wp) & (px[:, 1] < Hp)]
    t = np.linspace(0.0, 1.0, 96)[None, :, None]
    path = np.rint((C[None, None, :] + (P[:, None, :] - C[None, None, :]) * t) * ppm).astype(int)
    path[..., 0] = path[..., 0].clip(0, Wp - 1)
    path[..., 1] = path[..., 1].clip(0, Hp - 1)
    blocked = walls[path[..., 1], path[..., 0]]
    blocked &= np.logical_or.accumulate(~blocked, axis=1)  # камера висит на стене: стена в начале пути не мешает
    return P[~blocked.any(1)]


def spread(P: np.ndarray, n: int) -> list[int]:
    """Выбор n разнесённых точек: каждый раз самая дальняя от уже выбранных."""
    chosen = [int(np.argmin(np.hypot(*(P - P.mean(0)).T)))]
    d = np.hypot(*(P - P[chosen[0]]).T)
    while len(chosen) < min(n, len(P)):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.hypot(*(P - P[i]).T))
    return chosen


def make_pairs(pose: dict, walls: np.ndarray | None, ppm: float, lens: dict, W: int, H: int) -> list[list[float]]:
    C, h, R = pose["C"], pose["h"], pose["R"]
    if walls is not None:
        P = visible_floor(walls, ppm, C)
    else:
        ax = np.arange(-MAX_RANGE_M, MAX_RANGE_M + STEP_M, STEP_M)
        P = C + np.stack(np.meshgrid(ax, ax), -1).reshape(-1, 2)
    if not len(P):
        raise ValueError("на плане не нашлось пола вокруг камеры: проверьте пары и план")
    uv, th = project(P, C, h, R, lens)
    ok = (th < MAX_THETA) & (uv[:, 0] > W * 0.03) & (uv[:, 0] < W * 0.97) & (uv[:, 1] > H * 0.05) & (uv[:, 1] < H * 0.97)
    P, uv = P[ok], uv[ok]
    if len(P) < 4:
        raise ValueError("в кадр попало слишком мало пола: проверьте пары")
    return [[round(float(uv[i, 0]), 1), round(float(uv[i, 1]), 1),
             round(float(P[i, 0] * ppm), 1), round(float(P[i, 1] * ppm), 1)] for i in spread(P, N_PAIRS)]


def plan_walls(cfg_dir: Path) -> np.ndarray | None:
    from mtmc.config import AutoCalibCfg
    from mtmc.planfile import extract_walls, read_plan

    site = app.site_of(cfg_dir)
    if not site.get("plan_image"):
        return None
    image = Path(site["plan_image"])
    if not image.is_absolute() and not image.exists():
        image = cfg_dir / image
    return extract_walls(read_plan(image), float(site["px_per_m"]), AutoCalibCfg().wall_thr)


def auto_mark(cfg_dir: Path, videos_dir: Path, cam_id: int, given: list) -> dict:
    """Новые пары по ручным парам. Ничего не записывает в паспорт."""
    _, d = app.passport_of_cam(cfg_dir, cam_id)
    site = app.site_of(cfg_dir)
    ppm = float(site.get("px_per_m") or 0)
    if ppm <= 0:
        raise ValueError("сначала задайте масштаб плана")
    pairs = np.asarray(given, float).reshape(-1, 4)
    if len(pairs) < 2 or not np.isfinite(pairs).all():
        raise ValueError("нужно не меньше двух пар точек с корректными координатами")
    walls = plan_walls(cfg_dir)
    sc = scene(cfg_dir, videos_dir)
    W, H = sc["size"]
    cam = sc["cameras"].get(Path(d["video"]).name)
    if cam is None:
        raise ValueError("для этой записи нет расчёта: обновите страницу")
    look = (site.get("cameras") or {}).get(str(cam_id), {}).get("look_at")
    pose = locate(pairs, ppm, np.array(cam["yaw_pitch_roll"]), sc["lens"], np.array(look) / ppm if look else None)
    new = make_pairs(pose, walls, ppm, sc["lens"], W, H)
    outliers = [i for i in range(len(pairs)) if i not in pose["inliers"]]
    weak = cam["agreement"] < 0.25
    return {"pairs": new, "outliers": outliers, "used": len(pose["inliers"]), "height_m": round(pose["h"], 2),
            "camera_px": [round(float(v * ppm)) for v in pose["C"]], "weak_lines": weak,
            "check_heading": bool(pose.get("look_warning")),
            "fit_px": round(float(np.median(pose["px"])), 1)}


# ---------------------------------------------------------------- приложение
BUTTON_JS = r"""
(()=>{const b=document.createElement('button');b.id='auto';b.textContent='Авторазметка';
b.title='По 2 и более парам точек строит 10 разнесённых пар. В паспорт записывает только «Сохранить привязку».';
$('clear').after(b);
b.onclick=async()=>{if(cur===null)return hint('Сначала выберите камеру слева.');
  if(pairs.length<2)return hint('Отметьте 2 пары точек на полу подальше друг от друга, затем нажмите «Авторазметка».');
  const camera=cur,anchors=JSON.stringify(pairs);
  b.disabled=true;hint('Считаю. Первый запуск изучает записи всех камер и занимает несколько минут…');
  try{const j=await api('./auto',{cam_id:camera.cam_id,pairs});
    if(cur!==camera)return;
    if(JSON.stringify(pairs)!==anchors)return hint('Точки изменились во время расчёта. Запустите авторазметку снова.');
    pairs=j.pairs;pending=null;await refresh();
    const n=j.outliers.length?` Не учтены пары: ${j.outliers.map(i=>i+1).join(', ')}, они не сходятся с остальными.`:'';
    const w=(j.weak_lines?' В кадре мало прямых линий, проверьте особенно внимательно.':'')+(j.check_heading?' Направление камеры расходится с планом, проверьте сетку.':'');
    hint(`Готово: высота камеры ${j.height_m} м, по ${j.used} парам.${n}${w} Если сетка легла на пол, нажмите «Сохранить привязку». Если нет, снова нажмите «Калибровать», вернутся сохранённые точки.`);
  }catch(e){if(cur===camera)hint(e.message);}finally{b.disabled=false;}};})();
"""


def page() -> bytes:
    return app.PAGE.read_text(encoding="utf-8").replace("</body>", f"<script>{BUTTON_JS}</script></body>").encode()


def make_handler(cfg_dir: Path, videos_dir: Path):
    base = app.make_handler(cfg_dir, videos_dir)

    class H(base):
        def do_GET(self):
            if self.path.split("?")[0] in ("/", "/index.html"):
                self.send(200, page(), "text/html; charset=utf-8")
            else:
                super().do_GET()

        def do_POST(self):
            if self.path != "/auto":
                return super().do_POST()
            try:
                b = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                self.reply(auto_mark(cfg_dir, videos_dir, int(b["cam_id"]), b["pairs"]))
            except (ValueError, KeyError, TypeError) as e:
                self.reply({"error": str(e)}, 400)

    return H


def main() -> int:
    ap = argparse.ArgumentParser(description="Настройка камер с авторазметкой точек")
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
    print(f"Настройка камер с авторазметкой: {url}  (закройте окно, когда закончите)", flush=True)
    if not a.no_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
