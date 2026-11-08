"""Диагностика прогона офиса: качество по ручной разметке, точность плана, масштаб.
python scripts/office_diag.py results_office_trial
Разметка сделана на отрезке 150–300 с: на другом отрезке оценка не имеет смысла."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_office_labels import evaluate, load_labels  # noqa: E402


def load_obs(data: str, cams) -> pd.DataFrame:
    cols = ["t", "tid", "xs", "ys", "x1", "y1", "x2", "y2"]
    parts = [pd.read_parquet(f"{data}/observations/cam{c}.parquet", columns=cols).assign(cam=c) for c in cams]
    o = pd.concat(parts)
    o = o[o["tid"].notna() & o["xs"].notna()].copy()
    o["tid"] = o["tid"].astype(int)
    o["h"] = o["y2"] - o["y1"]
    return o


def speeds(o: pd.DataFrame) -> None:
    """Идущий человек — около 1.2–1.5 м/с. Иначе масштаб плана неверен, а с ним все пороги в метрах."""
    print("\nСкорость людей, м/с (норма: 75% около 1.2–1.5):")
    for c, oc in o.groupby("cam"):
        v = []
        for _, g in oc.sort_values("t").groupby("tid"):
            if len(g) >= 25:
                g = g.iloc[::12]
                v += (np.hypot(g["xs"].diff(), g["ys"].diff()) / g["t"].diff()).dropna().tolist()
        print(f"  cam{c}: медиана {np.median(v):.2f}, 75% {np.quantile(v, .75):.2f}, 90% {np.quantile(v, .9):.2f}")


def pairs(o: pd.DataFrame, labels: dict[int, str]) -> None:
    """Один человек на двух камерах в одни моменты: расстояние на плане и есть ошибка привязки."""
    o = o.assign(who=o["tid"].map(labels), k=(o["t"] / .2).round().astype(int))
    o = o[o["who"].notna()]
    g = o.groupby(["who", "k", "cam"], as_index=False).agg(x=("xs", "median"), y=("ys", "median"), h=("h", "median"))
    m = g.merge(g, on=["who", "k"], suffixes=("_a", "_b"))
    m = m[m["cam_a"] < m["cam_b"]]
    m["dx"], m["dy"] = m["x_b"] - m["x_a"], m["y_b"] - m["y_a"]
    m["d"] = np.hypot(m["dx"], m["dy"])
    m["hmin"] = m[["h_a", "h_b"]].min(axis=1)
    print("\nОшибка плана по парам камер, м (один человек, по разметке):")
    t = m.groupby(["cam_a", "cam_b"]).agg(n=("d", "size"), med=("d", "median"), p90=("d", lambda s: s.quantile(.9)),
                                          dx=("dx", "median"), dy=("dy", "median"))
    print(t.round(2).to_string())
    print("\nОшибка плана по росту меньшей фигуры в кадре, px:")
    b = pd.cut(m["hmin"], [0, 150, 250, 400, 3000])
    print(m.groupby(b, observed=True)["d"].describe(percentiles=[.5, .9])[["count", "50%", "90%"]].round(2).to_string())


def parts_report(data: str, labels: dict[int, str]) -> None:
    """Насколько половины тела и пропорции разделяют людей: один человек и разные, по разметке."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mtmc.bodyparts import load

    p = {t: v for t, v in load(data).items() if t in labels}
    if len(p) < 2:
        print("\nПоловины тела: нет данных")
        return
    rows = []
    ids = sorted(p)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            la, lb = p[a]["legs"], p[b]["legs"]
            rows.append((labels[a] == labels[b], float(p[a]["up"] @ p[b]["up"]), float(p[a]["low"] @ p[b]["low"]),
                         abs(la - lb) if la is not None and lb is not None else np.nan))
    r = pd.DataFrame(rows, columns=["same", "up", "low", "legs"])
    print("\nПоловины тела, сходство кусков (10%, 50%, 90%), разные и один человек:")
    print(r.groupby("same")[["up", "low", "legs"]].quantile([.1, .5, .9]).round(2).to_string())


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    data = sys.argv[1] if len(sys.argv) > 1 else "results_office_trial"
    r = evaluate(data)
    table = r.pop("table")
    print("Качество по разметке:", r)
    print(table.to_string())
    fit = Path(data, "plan_fit.json")
    if fit.exists():
        print("\nУточнение привязки:")
        for x in json.loads(fit.read_text(encoding="utf-8"))["report"]:
            print(" ", x)
    cams = sorted(int(p.name[3]) for p in Path(data, "observations").glob("cam?.parquet"))
    o = load_obs(data, cams)
    speeds(o)
    labels = load_labels(data)
    pairs(o, labels)
    parts_report(data, labels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
