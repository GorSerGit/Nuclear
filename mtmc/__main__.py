"""CLI: python -m mtmc process | render | gallery | stats"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_config


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")


def _default_config(path: str | None) -> str | None:
    if path:
        return path
    p = Path("configs/config.yaml")
    return str(p) if p.exists() else None


def cmd_process(a: argparse.Namespace) -> int:
    from .pipeline import format_summary, process

    cfg = load_config(_default_config(a.config))
    if a.device:
        cfg.detector.device = a.device
    if a.stride:
        cfg.video.stride = a.stride
    s = process(a.camera, a.data, cfg, a.start, a.end)
    print(format_summary(s))
    return 0


def cmd_render(a: argparse.Namespace) -> int:
    from .render import render

    cfg = load_config(_default_config(a.config))
    res = render(a.cam, a.data, a.out, cfg, label=a.label, details=a.details, skeleton=a.skeleton,
                 zones=a.zones, minimap=a.minimap, fps_mode=a.fps_mode, scale=a.scale, start_s=a.start,
                 end_s=a.end, tids=a.tid, clip=a.clip, run_id=a.run)
    print(f"готово: {res['out']} ({res['frames']} кадров, {res['fps']:.3f} кадр/с, кадры {res['range']})")
    return 0


def cmd_gallery(a: argparse.Namespace) -> int:
    from .gallery import gallery

    cfg = load_config(_default_config(a.config))
    paths = gallery(a.cam, a.data, a.out, cfg)
    for p in paths:
        print(p)
    return 0


def cmd_retrack(a: argparse.Namespace) -> int:
    from .pipeline import retrack
    from .storage import Store

    cfg = load_config(_default_config(a.config))
    cams = a.cam or [int(c) for c in Store(a.data).load_cameras()["cam_id"]]
    for c in cams:
        s = retrack(c, a.data, cfg)
        print(f"cam{c}: треклетов {s.get('tracklets', 0)} (слабых {s.get('tracklets_weak', 0)}); разрезов: "
              f"пауза {s.get('cuts_gap', 0)}, скачок {s.get('cuts_jump', 0)}, перекрытие {s.get('cuts_occlusion', 0)}, "
              f"внешность {s.get('cuts_appearance', 0)}, из них подмена одежды {s.get('cuts_clothes', 0)}; "
              f"{s['wall_s']} с")
    return 0


def cmd_link(a: argparse.Namespace) -> int:
    from .link import format_link_summary, link

    cfg = load_config(_default_config(a.config))
    s = link(a.data, cfg)
    print(format_link_summary(s, cfg.link.major_min_s))
    return 0


def cmd_mosaic(a: argparse.Namespace) -> int:
    from .mosaic import mosaic

    cfg = load_config(_default_config(a.config))
    res = mosaic(a.data, a.out, cfg, cams=a.cams, label=a.label, run_id=a.run, scale=a.scale,
                 start_s=a.start, end_s=a.end, plan=not a.no_plan)
    print(f"готово: {res['out']} ({res['frames']} кадров)")
    return 0


def cmd_persons(a: argparse.Namespace) -> int:
    from .person_video import person_videos

    cfg = load_config(_default_config(a.config))
    res = person_videos(a.data, a.out_dir, cfg, persons=a.person, min_present_s=a.min_present, run_id=a.run,
                        layout=a.layout, scale=a.scale)
    if not res:
        print("нет людей под условие (уменьшите --min-present или укажите --person)")
    for r in res:
        print(f"P{r['person_id']}: в кадре {r['present_s']} с -> {r['out']} ({r['frames']} кадров)")
    return 0


def cmd_tracks(a: argparse.Namespace) -> int:
    from .tracks import build_tracks, format_tracks

    cfg = load_config(_default_config(a.config))
    if a.end_gap is not None:
        cfg.tracks.end_gap_s = a.end_gap
    res = build_tracks(a.data, cfg, run_id=a.run)
    print(format_tracks(res))
    print(f"\nсписок: {Path(a.data) / 'tracks.csv'}, {Path(a.data) / 'tracks.json'}")
    return 0


def cmd_ui(a: argparse.Namespace) -> int:
    from .ui import build_ui

    cfg = load_config(_default_config(a.config))
    res = build_ui(a.data, a.out, cfg, run_id=a.run, site=a.site)
    print(f"интерфейс: {res['index']}  (треков {res['n_tracks']}; открыть: python -m mtmc serve --dir {a.out})")
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    from .ui import serve

    serve(a.dir, a.port, open_browser=not a.no_browser)
    return 0


def cmd_calibrate(a: argparse.Namespace) -> int:
    from .calibrate import calibrate

    res = calibrate(a.passport, a.plan, a.px_per_m, points=a.points, frame_s=a.frame, out=a.out)
    print(f"H записана в {res['out']}: точек {res['n']}, ошибка репроекции медиана {res['err_med_m']:.3f} м, "
          f"макс {res['err_max_m']:.3f} м")
    return 0


def cmd_silhouettes(a: argparse.Namespace) -> int:
    from .silhouette import silhouettes
    from .storage import Store

    cfg = load_config(_default_config(a.config))
    cams = a.cam or [int(c) for c in Store(a.data).load_cameras()["cam_id"]]
    for c in cams:
        r = silhouettes(c, a.data, cfg)
        print(f"cam{c}: силуэтов {r['matched']} из {r['observations']} наблюдений ({r.get('share', 0):.0%}), "
              f"{r.get('wall_s', 0)} с")
    return 0


def cmd_autocalib(a: argparse.Namespace) -> int:
    import glob

    from .autocalib import autocalib, format_autocalib

    cfg = load_config(_default_config(a.config))
    files: list[str] = []
    for p in a.passports or []:
        files += sorted(glob.glob(p)) or [p]
    res = autocalib(a.data, cfg, a.site, files, apply=not a.no_apply, manual=a.manual, check_dir=a.check_dir,
                    only_missing=a.only_missing)
    print(format_autocalib(res))
    if res["applied"]:
        print(f"H записана и треклеты пересчитаны: камеры {', '.join(map(str, res['applied']))}. "
              f"Дальше: python -m mtmc link --data {a.data}")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    """Всё сразу: process каждой камеры -> link -> tracks -> ui."""
    import glob

    from .link import format_link_summary, link
    from .pipeline import format_summary, process
    from .tracks import build_tracks, format_tracks
    from .ui import build_ui

    cfg = load_config(_default_config(a.config))
    if a.device:
        cfg.detector.device = a.device
    files: list[str] = []
    for p in a.passports:
        files += sorted(glob.glob(p)) or [p]
    if not a.skip_process:
        for f in files:
            print(f"== process {f}")
            print(format_summary(process(f, a.data, cfg)))
    if a.autocalib:
        from .autocalib import autocalib, format_autocalib

        print("== autocalib")
        print(format_autocalib(autocalib(a.data, cfg, a.site, files, apply=True, manual=a.manual,
                                         only_missing=True)))
    print("== link")
    print(format_link_summary(link(a.data, cfg), cfg.link.major_min_s))
    print("== tracks")
    print(format_tracks(build_tracks(a.data, cfg)))
    if a.ui:
        res = build_ui(a.data, a.ui, cfg, site=a.site)
        print(f"== интерфейс: {res['index']}  (python -m mtmc serve --dir {a.ui})")
    return 0


def cmd_stats(a: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    from .storage import Store

    store = Store(a.data)
    cams = store.load_cameras()
    if cams.empty:
        print("БД пуста")
        return 0
    tr = store.load_tracklets()
    for c in cams.itertuples():
        t = tr[tr["cam_id"] == c.cam_id]
        op = store.obs_path(c.cam_id)
        n_obs = pq.ParquetFile(op).metadata.num_rows if op.exists() else 0
        meta = store.load_meta(c.cam_id)
        print(f"cam{c.cam_id} «{c.name or ''}»: {c.width}x{c.height}, fps {c.fps:.3f}, кадров {c.n_frames}, "
              f"обработано {c.processed_at}, stride {meta.get('stride')}, кадры {meta.get('start_frame')}–"
              f"{meta.get('end_frame')}")
        if t.empty:
            print("   треклетов нет")
            continue
        dur = t["t1"] - t["t0"]
        print(f"   наблюдений {n_obs}, локальных ID {t['local_id'].nunique()}, треклетов {len(t)} "
              f"(слабых {int(t['is_weak'].sum())}, с эмбеддингами {int((t['n_emb'] > 0).sum())})")
        print(f"   длительность треклета: медиана {dur.median():.1f} с, макс {dur.max():.1f} с; "
              f"mean_q {t['mean_q'].mean():.2f}")
        sr = t["start_reason"].value_counts().to_dict()
        er = t["end_reason"].value_counts().to_dict()
        print(f"   причины начала: {sr}")
        print(f"   причины конца:  {er}")
    with store.connect() as con:
        n_p = con.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
        n_l = con.execute("SELECT COUNT(*) FROM tracklet_person").fetchone()[0]
    print(f"persons: {n_p}, tracklet_person: {n_l}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", default="INFO", help="DEBUG | INFO | WARNING | ERROR")
    common.add_argument("--config", default=None, help="config.yaml (по умолчанию configs/config.yaml, если есть)")
    p = argparse.ArgumentParser(prog="python -m mtmc", description="MTMC часть 1: одна камера")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("process", parents=[common], help="обработать одну камеру")
    pp.add_argument("--camera", required=True, help="паспорт камеры (JSON)")
    pp.add_argument("--data", required=True, help="папка результатов")
    pp.add_argument("--start", type=float, default=None, help="начало, с видео")
    pp.add_argument("--end", type=float, default=None, help="конец, с видео")
    pp.add_argument("--device", default=None, help="переопределить detector.device (cpu, cuda:0)")
    pp.add_argument("--stride", type=int, default=None, help="переопределить video.stride")
    pp.set_defaults(func=cmd_process)

    pr = sub.add_parser("render", parents=[common], help="видео с разметкой (без нейросетей)")
    pr.add_argument("--cam", type=int, required=True)
    pr.add_argument("--data", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--label", choices=["local", "tracklet", "person"], default="tracklet")
    pr.add_argument("--run", type=int, default=None, help="run_id для --label person")
    pr.add_argument("--details", action="store_true", help="ракурс и q в подписи")
    pr.add_argument("--skeleton", action="store_true")
    pr.add_argument("--zones", action="store_true", help="зоны паспорта")
    pr.add_argument("--minimap", action="store_true", help="мини-план (нужна H)")
    pr.add_argument("--fps-mode", choices=["original", "processed"], default="original")
    pr.add_argument("--scale", type=float, default=1.0)
    pr.add_argument("--start", type=float, default=None)
    pr.add_argument("--end", type=float, default=None)
    pr.add_argument("--tid", type=int, action="append", default=None, help="только эти треклеты (можно несколько)")
    pr.add_argument("--clip", action="store_true", help="обрезать по времени --tid ±1 с")
    pr.set_defaults(func=cmd_render)

    pg = sub.add_parser("gallery", parents=[common], help="сетка кропов по треклетам")
    pg.add_argument("--cam", type=int, required=True)
    pg.add_argument("--data", required=True)
    pg.add_argument("--out", required=True)
    pg.set_defaults(func=cmd_gallery)

    pt = sub.add_parser("retrack", parents=[common], help="заново нарезать треклеты из сохранённых наблюдений")
    pt.add_argument("--data", required=True)
    pt.add_argument("--cam", type=int, nargs="*", default=None, help="камеры (по умолчанию все)")
    pt.set_defaults(func=cmd_retrack)

    pl = sub.add_parser("link", parents=[common], help="часть 2: склеить треклеты всех камер в людей")
    pl.add_argument("--data", required=True)
    pl.set_defaults(func=cmd_link)

    pm = sub.add_parser("mosaic", parents=[common], help="все камеры одним видео + план сверху")
    pm.add_argument("--data", required=True)
    pm.add_argument("--out", required=True)
    pm.add_argument("--cams", type=int, nargs="*", default=None, help="камеры (по умолчанию все)")
    pm.add_argument("--label", choices=["local", "tracklet", "person"], default="person")
    pm.add_argument("--run", type=int, default=None)
    pm.add_argument("--scale", type=float, default=1.0)
    pm.add_argument("--start", type=float, default=None, help="начало, с общего времени")
    pm.add_argument("--end", type=float, default=None)
    pm.add_argument("--no-plan", action="store_true", help="без панели вида сверху")
    pm.set_defaults(func=cmd_mosaic)

    pp2 = sub.add_parser("persons", parents=[common], help="отдельное видео на каждого человека")
    pp2.add_argument("--data", required=True)
    pp2.add_argument("--out-dir", required=True)
    pp2.add_argument("--person", type=int, nargs="*", default=None, help="только эти person_id")
    pp2.add_argument("--min-present", type=float, default=20.0, help="по умолчанию: люди, видимые ≥ N с")
    pp2.add_argument("--layout", choices=["grid", "best"], default="grid",
                     help="grid — все камеры; best — одна камера, где человек виден крупнее")
    pp2.add_argument("--run", type=int, default=None)
    pp2.add_argument("--scale", type=float, default=1.0)
    pp2.set_defaults(func=cmd_persons)

    ptk = sub.add_parser("tracks", parents=[common], help="сквозные треки: число и список (tracks.csv/json)")
    ptk.add_argument("--data", required=True)
    ptk.add_argument("--run", type=int, default=None, help="run_id склейки (по умолчанию последний)")
    ptk.add_argument("--end-gap", type=float, default=None, help="переопределить tracks.end_gap_s, с")
    ptk.set_defaults(func=cmd_tracks)

    pu = sub.add_parser("ui", parents=[common], help="собрать веб-интерфейс (видео камер + треки + план)")
    pu.add_argument("--data", required=True)
    pu.add_argument("--out", required=True, help="папка интерфейса")
    pu.add_argument("--run", type=int, default=None)
    pu.add_argument("--site", default=None, help="site.json: план офиса и камеры на плане")
    pu.set_defaults(func=cmd_ui)

    pv = sub.add_parser("serve", parents=[common], help="открыть интерфейс в браузере (локальный сервер)")
    pv.add_argument("--dir", required=True)
    pv.add_argument("--port", type=int, default=8765)
    pv.add_argument("--no-browser", action="store_true")
    pv.set_defaults(func=cmd_serve)

    pc = sub.add_parser("calibrate", parents=[common], help="гомография камера -> план по точкам")
    pc.add_argument("--passport", required=True)
    pc.add_argument("--plan", default=None, help="картинка плана (для щелчков мышью)")
    pc.add_argument("--px-per-m", type=float, default=None, help="масштаб плана, пикселей на метр")
    pc.add_argument("--points", default=None, help="JSON [[u, v, X, Y], ...] вместо щелчков (X, Y — пиксели плана)")
    pc.add_argument("--frame", type=float, default=0.0, help="кадр видео для щелчков, с")
    pc.add_argument("--out", default=None, help="куда записать паспорт (по умолчанию — поверх)")
    pc.set_defaults(func=cmd_calibrate)

    prn = sub.add_parser("run", parents=[common], help="всё сразу: process всех камер -> link -> tracks -> ui")
    prn.add_argument("--passports", nargs="+", required=True, help="паспорта (можно маску: configs/x/*.json)")
    prn.add_argument("--data", required=True)
    prn.add_argument("--ui", default=None, help="папка интерфейса")
    prn.add_argument("--site", default=None)
    prn.add_argument("--device", default=None)
    prn.add_argument("--skip-process", action="store_true", help="камеры уже обработаны")
    prn.add_argument("--autocalib", action="store_true",
                     help="после process привязать к плану автоматически камеры без ручной H (нужен --site)")
    prn.add_argument("--manual", action="store_true", help="где автопривязка не удалась — открыть окно ручной")
    prn.set_defaults(func=cmd_run)

    psl = sub.add_parser("silhouettes", parents=[common],
                         help="силуэты (контуры фигур) для уже обработанных камер; ID и треки не меняются")
    psl.add_argument("--data", required=True)
    psl.add_argument("--cam", type=int, nargs="*", default=None, help="камеры (по умолчанию все)")
    psl.set_defaults(func=cmd_silhouettes)

    pa = sub.add_parser("autocalib", parents=[common],
                        help="автопривязка камер к плану по идущим людям; не получилось — ручная")
    pa.add_argument("--data", required=True, help="папка с обработанными камерами (process можно без H)")
    pa.add_argument("--site", required=True, help="site.json: план (стены), масштаб, положение камер")
    pa.add_argument("--passports", nargs="*", default=None, help="паспорта: H запишется в них (с копией)")
    pa.add_argument("--manual", action="store_true", help="для камер со статусом fail открыть окно ручной привязки")
    pa.add_argument("--no-apply", action="store_true", help="только посчитать и нарисовать проверку")
    pa.add_argument("--only-missing", action="store_true", help="не трогать камеры с ручной H")
    pa.add_argument("--check-dir", default=None, help="куда картинки проверки (по умолчанию <data>/calib_check)")
    pa.set_defaults(func=cmd_autocalib)

    ps = sub.add_parser("stats", parents=[common], help="сводка по БД")
    ps.add_argument("--data", required=True)
    ps.set_defaults(func=cmd_stats)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.log_level)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, KeyError) as e:
        logging.getLogger("mtmc").error("%s", e)
        if args.log_level.upper() == "DEBUG":
            raise
        return 2


if __name__ == "__main__":
    sys.exit(main())
