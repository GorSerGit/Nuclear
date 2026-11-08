"""Оценка склейки людей офиса по ручной разметке кусков (configs/labels_office_150s.json).
python scripts/eval_office_labels.py results_office_trial
Чистота: доля наблюдений номера, принадлежащая его главному человеку. Полнота: доля наблюдений человека у его главного
номера. Размечены куски от 30 наблюдений там, где человека видно уверенно; остальные в оценку не входят."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_labels(results: str | Path) -> dict[int, str]:
    """Разметка записана через камеру, номер трекера и кадры: номера кусков меняются при пересчёте. -> tid -> человек."""
    d = json.loads((ROOT / "configs/labels_office_150s.json").read_text(encoding="utf-8"))
    db = sqlite3.connect(Path(results) / "mtmc.sqlite")
    t = pd.read_sql("select tid, cam_id, local_id, frame0, frame1 from tracklets", db)
    out: dict[int, str] = {}
    for who, items in d.items():
        for it in items:
            s = t[(t.cam_id == it["cam"]) & (t.local_id == it["local_id"])]
            ov = s[["frame1"]].min(axis=1).clip(upper=it["f1"]) - s["frame0"].clip(lower=it["f0"])
            for tid in s.tid[ov >= 0.5 * (s.frame1 - s.frame0)]:
                out[int(tid)] = who
    return out


def evaluate(results: str | Path) -> dict:
    db = sqlite3.connect(Path(results) / "mtmc.sqlite")
    run = db.execute("select max(run_id) from tracks").fetchone()[0]
    q = """select x.tid, p.person_no, t.n_obs from tracklet_track x join track_person p
           on p.run_id = x.run_id and p.track_id = x.track_id join tracklets t on t.tid = x.tid where x.run_id = ?"""
    df = pd.read_sql(q, db, params=(run,))
    labels = load_labels(results)
    df["who"] = df["tid"].map(labels)
    lab = df[df["who"].notna()]
    ct = lab.pivot_table(index="person_no", columns="who", values="n_obs", aggfunc="sum", fill_value=0)
    pur = float(ct.max(axis=1).sum() / ct.values.sum())
    rec = float(ct.max(axis=0).sum() / ct.values.sum())
    n_real = lab["who"].nunique()
    return {"purity": round(pur, 3), "recall": round(rec, 3), "f": round(2 * pur * rec / (pur + rec), 3),
            "people": int(df["person_no"].nunique()), "people_labeled": int(lab["person_no"].nunique()),
            "real": int(n_real), "table": ct}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    r = evaluate(sys.argv[1] if len(sys.argv) > 1 else "results_office_trial")
    t = r.pop("table")
    print(r)
    if "-v" in sys.argv:
        print(t.to_string())

