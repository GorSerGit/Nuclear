"""Отчёт по запуску: результат вместе с условиями, в которых он получен.

Число треков без условий съёмки мало что значит: по этому же отчёту видно, насколько людям хватало размера в кадре,
как часто они перекрывались, какие камеры привязаны к плану и что отброшено. Пишется в <data>/report.md.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .config import Config
from .storage import Store


def camera_conditions(store: Store, cfg: Config) -> list[dict]:
    """Условия съёмки по камерам: то, что определяет, чему в результате можно верить."""
    out = []
    qc = cfg.observation.quality
    for c in store.load_cameras().itertuples():
        cid = int(c.cam_id)
        o = store.load_observations(cid, columns=["y1", "y2", "occ", "conf", "xs"])
        if o.empty:
            out.append({"cam": cid, "name": c.name, "n": 0, "warn": ["людей не найдено"]})
            continue
        h = o["y2"] - o["y1"]
        sil_path = store.sil_path(cid)
        sil = pq.ParquetFile(sil_path).metadata.num_rows if sil_path.exists() else 0
        r = {"cam": cid, "name": c.name, "size": f"{c.width}×{c.height}", "fps": round(float(c.fps), 2),
             "n": len(o), "h": float(h.median()), "small": float((h < qc.h_min).mean()),
             "occ": float((o["occ"] >= qc.occ_max).mean()), "conf": float(o["conf"].median()),
             "sil": sil / len(o), "floor": bool(o["xs"].notna().any())}
        w = []
        if r["small"] > 0.3:
            w.append(f"люди мелкие: {r['small']:.0%} наблюдений ниже {qc.h_min} px, детекция и внешность ненадёжны")
        if r["occ"] > 0.3:
            w.append(f"частые перекрытия: {r['occ']:.0%} наблюдений, возможна смена ID внутри камеры")
        if not r["floor"]:
            w.append("нет привязки к плану: камера не связана с другими")
        if r["sil"] < 0.5:
            w.append(f"силуэт только у {r['sil']:.0%} наблюдений, остальные показаны рамкой")
        r["warn"] = w
        out.append(r)
    return out


def write_report(store: Store, res: dict, cfg: Config) -> Path:
    from .tracks import fmt_t, track_row

    names = res["cam_names"]
    conds = camera_conditions(store, cfg)
    meta = {int(c): store.load_meta(int(c)) for c in store.load_cameras()["cam_id"]}
    strides = sorted({m.get("stride") for m in meta.values() if m.get("stride")})
    a, b = res["record"]
    tc, lc = cfg.tracks, cfg.link
    lines = ["# Результат запуска", "",
             f"**Сквозных треков: {res['n_tracks']}.** Людей: {res.get('n_people', '?')}. "
             f"Запись {fmt_t(0)}–{fmt_t(b - a)}, камер: {len(conds)}.", "",
             "## Треки", "",
             "| ID | Начало | Конец | Длит., с | Камеры | Человек | Как закончился |",
             "|---|---|---|---|---|---|---|"]
    for t in res["tracks"]:
        r = track_row(t, names)
        lines.append(f"| T{t.track_id} | {r['start']} | {r['end']} | {r['duration_s']} | {r['cameras']} | "
                     f"{r['person']} | {r['end_reason']} |")
    lines += ["", "Интервалы по камерам: `tracks.csv`, столбец `per_camera`.", "",
              "## Условия съёмки", "",
              "Результат относится к этим условиям. На других ракурсах и размерах людей он может быть другим.", "",
              "| Камера | Кадр | Наблюдений | Рост в кадре, px | Мелкие | С перекрытием | Уверенность | С силуэтом | План |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in conds:
        if not r["n"]:
            lines.append(f"| cam{r['cam']} {r['name'] or ''} | | 0 | | | | | | |")
            continue
        lines.append(f"| cam{r['cam']} {r['name'] or ''} | {r['size']} | {r['n']} | {r['h']:.0f} | {r['small']:.0%} | "
                     f"{r['occ']:.0%} | {r['conf']:.2f} | {r['sil']:.0%} | {'да' if r['floor'] else '**нет**'} |")
    warn = [f"- cam{r['cam']}: {w}" for r in conds for w in r["warn"]]
    lines += ["", "**Предупреждения:**" if warn else "Предупреждений по условиям нет.", *warn, ""]
    why = Counter(t.reject.split(":")[0] for t in res["noise"])
    lines += ["## Что не вошло в счёт", "",
              f"Отброшено {len(res['noise'])}: " + (", ".join(f"{k} {v}" for k, v in why.most_common()) or "ничего")
              + ". Каждый случай с причиной — в `rejected_tracks.csv`. Настоящего человека фильтр тоже может "
                "отбросить, поэтому длинные строки этого файла стоит просмотреть.",
              f"Поглощено дублей: {res['counters'].get('absorbed', 0)}. Трек сохранён через пропуск детекции: "
              f"{res['counters'].get('bridged', 0)} раз.", "",
              "## Настройки этого запуска", "",
              f"- обрабатывался каждый {', '.join(map(str, strides)) or '?'}-й кадр, `imgsz` {cfg.detector.imgsz};",
              f"- трек завершается, если человека не видит ни одна камера дольше {tc.end_gap_s:g} с;",
              f"- короткий пропуск внутри зоны обзора сохраняет трек: до {tc.detect_gap_max_s:g} с и рядом с тем же местом;",
              f"- один человек на разных камерах: не дальше {lc.d_same_m:g} м на плане плюс погрешность;",
              f"- не считается шумом трек от {tc.min_track_s:g} с и от {tc.min_track_obs} наблюдений.", ""]
    path = store.root / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
