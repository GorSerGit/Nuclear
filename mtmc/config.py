"""Загрузка config.yaml в dataclass-ы с дефолтами (раздел 6 ТЗ)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


WALK_MPS = 1.3       # опора шага: при ней замер на офисе (1.4 м/с) даёт пороги лучшего прогона (F 0.90); при k 0.93 F 0.875
METRIC_KEYS = {
    "tracklets": ("v_max_mps",),
    "link": ("d_same_m", "near_m", "d_diff_m", "group_handoff_m", "group_soft_m", "sure_max_m", "rescue_max_m",
             "app_tol_m", "shift_max_m", "shift_min_m", "shift_spread_m", "seq_v_max_mps", "seq_margin_m", "dup_max_m",
             "seq_cross_margin_m", "coord_far_m"),
    "tracks": ("absorb_m", "absorb_app_m", "absorb_near_m", "fov_margin_m", "bridge_v_max_mps", "cont_m",
               "person_twice_m"),
}


def rescale_metres(d: dict, k: float) -> None:
    """Расстояния склейки заданы в единицах «при шаге WALK_MPS». k — во сколько раз быстрее люди идут на этом плане:
    масштаб и растяжение плана учитываются сами, число px/м здесь не нужно."""
    for part, keys in METRIC_KEYS.items():
        for key in keys:
            if key in d.get(part, {}):
                d[part][key] = float(d[part][key]) * k


@dataclass
class VideoCfg:
    stride: int = 2
    exact_seek: bool = True          # перемотка к --start последовательным grab(), а не CAP_PROP_POS_FRAMES


@dataclass
class GateCfg:
    """Пропуск пустых кадров, см. mtmc/gate.py."""
    enabled: bool = False
    check_s: float = 1.0             # пустую сцену детектор всё равно проверяет так часто
    hold_s: float = 3.0               # после последнего человека кадры идут все, не меньше буфера трекера + 1 с
    width: int = 480                  # ширина уменьшенного кадра для сравнения
    diff: int = 14                    # изменение яркости пикселя, 0–255
    min_share: float = 0.0002         # изменилась такая доля кадра — запускаем детектор (дальний человек ~0.0004)
    noise_hits: int = 3               # столько раз изменилось без людей — шум: часы, экран


@dataclass
class DetectorCfg:
    weights: str = "weights/yolo11m-pose.pt"
    imgsz: int = 1280
    conf: float = 0.10
    half: bool = True
    device: str = "cuda:0"


@dataclass
class TrackerCfg:
    tracker_type: str = "botsort"
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer_s: float = 2.0
    match_thresh: float = 0.8
    fuse_score: bool = True
    gmc_method: str = "none"
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    with_reid: bool = True
    model: str = "auto"


@dataclass
class QualityCfg:
    h_min: float = 60
    h_good: float = 160
    edge_px: float = 4
    trunc_factor: float = 0.3
    occ_max: float = 0.4
    blur_ref: float = 150


@dataclass
class ObservationCfg:
    kp_conf: float = 0.5
    foot_body_ratio: float = 1.6
    sigma_px: float = 4.0
    sigma_base_m: float = 0.15
    sigma_mult: dict = field(default_factory=lambda: {"ankles": 1.0, "body": 1.6, "box": 2.0})
    view_side_thr: float = 0.15
    zone_share: float = 0.6           # такая доля рамки в ignore_zones и низ рамки там же — не человек (экран, зеркало)
    quality: QualityCfg = field(default_factory=QualityCfg)


@dataclass
class CropsCfg:
    min_q: float = 0.35
    every: int = 3
    pad: float = 0.05
    jpeg_quality: int = 90


@dataclass
class TrackletCfg:
    gap_cut_s: float = 1.5
    v_max_mps: float = 3.0
    jump_sigma_k: float = 3.0
    v_max_heights_ps: float = 2.5
    jump_px_margin_h: float = 0.5
    min_len: int = 5
    smooth_window: int = 7
    smooth_poly: int = 2
    edge_speed_window_s: float = 0.5
    # правило «перекрытие»: трекер подменяет людей, пока один заслонён другим
    occlusion_cut: bool = True
    occ_cut: float = 0.6             # вход в эпизод сильного перекрытия (доля бокса)
    occ_release: float = 0.3         # выход из эпизода (гистерезис)
    occ_gap_s: float = 0.3           # потеря дольше этого рядом с перекрытием — тоже разрез


@dataclass
class ReIDCfg:
    backend: str = "torchreid"       # torchreid | none
    model_name: str = "osnet_x1_0"
    weights: str = "weights/osnet_x1_0_msmt17.pth"
    device: str | None = None        # None -> как у детектора
    batch: int = 64
    flip_tta: bool = True
    max_crops: int = 32
    time_bins: int = 8
    app_cut_thr: float = 0.55         # верхний предел порога разреза
    app_cut_min: float = 0.55         # нижний предел (значение ТЗ)
    app_cut_margin: float = 0.10      # порог = медиана сходства людей, которых камера видела вместе, плюс это
    app_cut_min_side: int = 4
    app_cut_min_total: int = 8
    proto_min: int = 2
    clothes_cut: float = 75.0         # подмена в треклете: яркость верха или низа сменилась на столько (0–255)
    clothes_cut_side: int = 4         # кропов с каждой стороны от подмены
    clothes_spread: float = 2.5       # скачок больше разброса внутри стороны во столько раз


@dataclass
class RenderCfg:
    trail_s: float = 2.0
    box_thickness: int = 2
    weak_thickness: int = 1
    foot_radius: int = 4
    interp_max_gap_steps: int = 1    # интерполировать, если между наблюдениями не больше N шагов stride
    crf: int = 23
    preset: str = "veryfast"
    ffmpeg: str = "ffmpeg"
    font: str | None = None          # путь к TTF; None -> поиск системного шрифта
    font_size: int = 16
    minimap_px: int = 320
    minimap_margin_m: float = 1.0
    zones_alpha: float = 0.25
    clip_pad_s: float = 1.0


@dataclass
class GalleryCfg:
    crops_per_row: int = 8
    crop_h: int = 128
    rows_per_file: int = 50
    label_w: int = 330
    jpeg_quality: int = 90


@dataclass
class LinkCfg:
    """Часть 2: склейка треклетов в людей (положение на плане + внешность)."""
    grid_s: float = 0.08              # шаг общей сетки времени (stride / fps)
    min_obs: int = 3                  # треклеты короче не связываются (остаются отдельными людьми)
    # одновременные треклеты разных камер
    min_overlap_samples: int = 5
    d_same_m: float = 0.6             # медианное расстояние на плане для «один человек»
    sigma_k: float = 1.0              # + sigma_k * медиана hypot(sigma_a, sigma_b)
    near_m: float = 1.0
    near_frac_min: float = 0.6        # доля общих моментов, где расстояние < near_m
    d_diff_m: float = 1.2             # медиана больше — точно разные люди (запрет)
    cannot_far_min_samples: int = 12  # …но только при достаточно долгом пересечении (≈1 с)
    app_reject: float = 0.45          # если оба прототипа есть и сходство ниже — не связываем
    app_strong: float = 0.90          # такое сходство внешности важнее расхождения на плане (спасение, поправка камер)
    app_sure_min_obs: int = 24        # …если у обоих кусков столько наблюдений: вектор короткого куска шумный
    clothes_veto: float = 80.0        # яркость верха или низа разошлась сильнее — разные люди; 0 — выключено
    cam_debias: float = 0.0           # какую долю среднего вектора камеры вычитать; 0 — не вычитать
    scene_lam: float = 0.0            # подавление разброса внутри треклетов: меньше — сильнее; 0 — выключено
    purge_min: float = 0.0            # кусок группы похож на её части слабее — отделяется; 0 — выключено
    ahc_thr: float = 0.0              # люди по внешности: порог кластеризации; 0 — склейка по траекториям
    ahc_min_obs: int = 0              # куски короче не строят группы: их вектор слишком шумный
    ahc_short_thr: float = 0.0        # короткий кусок идёт к самой похожей группе при этом сходстве; 0 — нет
    cam_apart: list = field(default_factory=list)  # пары камер без общего обзора, [[1, 4], ...]
    cam_apart_s: float = 2.0          # быстрее этого между такими камерами не дойти
    occ_limit: float = 0.0            # вырезки с большим перекрытием не входят в вектор куска; 0 — входят все
    ahc_small_obs: int = 0            # группа меньше этого — обрывок: идёт к другой при пороге ahc_small_thr; 0 — нет
    ahc_small_thr: float = 0.0
    seq_cross_margin_m: float = 0.0   # запас передачи между камерами; 0 — как seq_margin_m
    seq_overlap_samples: int = 0      # перекрытие короче стольких отсчётов считается передачей; 0 — выключено
    part_min: float = 0.0             # сходство нижней половины тела ниже этого — разные люди; 0 — не сравнивается
    seq_cross_need_app: bool = False  # передача между камерами только при известной внешности
    seq_no_skip: bool = False         # пауза не перепрыгивает через прямое продолжение на той же камере
    overlap_app_w: float = 0.2       # вес внешности при выборе пары на другой камере: двое рядом различимы только ею
    group_seam_app_min: float = 0.0 # сходство кусков на стыке двух групп; 0 — стык не сверяется
    seq_gap_app_min: float = 0.0     # после паузы дольше seq_strict_gap_s внешность должна быть хотя бы такой
    seq_strict_gap_s: float = 0.0    # при паузе не дольше этой на одной камере сверяются сами концы кусков; 0 — выключено
    coord_far_m: float = 0.0          # одновременно дальше — разные люди; 0 — координаты в группировке не участвуют
    coord_near: float = 0.0           # добавка к сходству: одновременно рядом на двух камерах
    coord_seq: float = 0.0            # добавка к сходству: один кусок продолжает другой по времени и месту
    coord_penalty: float = 0.0        # штраф к сходству при расхождении на всю дистанцию coord_far_m
    sure_max_m: float = 5.0           # уверенной внешности прощается такое расхождение на плане
    xy_weight: float = 1.0            # вес расстояния при выборе пары: меньше — координаты решают меньше
    dup_cover: float = 0.9            # двойник: такая доля меньшей рамки лежит в другой рамке того же кадра…
    dup_share: float = 0.9            # …на такой доле общих кадров…
    dup_min_frames: int = 25          # …и общих кадров не меньше; мягче склеивает соседей (EPFL 4p: 40 ложных пар)
    app_margin: float = 0.13          # порог сцены: настолько похожее, чем люди, которых одна камера видела вместе
    app_handoff_min: float = 0.60     # порог сцены не опускается ниже
    app_ambiguity: float = 0.03       # соперник ближе этого — пара неоднозначна, не соединяем
    handoff_max_s: float = 15.0       # переход между камерами по внешности: макс. пауза; 0 — выключено
    handoff_overlap_s: float = 1.0    # допустимое наложение времени при переходе
    rescue_max_m: float = 2.5         # спасение: куски дальше не соединяем; 0 — выключено
    app_tol_m: float = 0.6            # спасение передачи между камерами: столько метров сверх обычного запаса
    shift_max_m: float = 2.0          # поправка положения камер по людям: 0 — выключена; пара дальше не голосует
    shift_min_m: float = 0.4          # поправка меньше этого не применяется: при точной привязке это шум
    shift_spread_m: float = 0.5       # разброс смещения в паре: шире — разные люди, а не ошибка привязки
    shift_min_samples: int = 25       # общих моментов в паре
    shift_min_pairs: int = 2          # пар на две камеры: один похоже одетый сосед камеру не сдвинет
    # последовательные треклеты одной камеры
    seq_max_gap_s: float = 4.0
    seq_v_max_mps: float = 2.0
    seq_margin_m: float = 0.5
    seq_app_min: float = 0.6          # нужное сходство внешности, если оба прототипа есть
    seq_noapp_gap_s: float = 1.0      # без прототипов — только короткие паузы
    seq_edge_window_s: float = 0.5    # положение начала/конца — среднее за это окно
    same_cam_overlap_s: float = 0.2   # пересечение по времени на одной камере больше — запрет
    dup_max_m: float = 0.4            # …если только треклеты не стоят в одной точке (дубль одного человека)
    group_passes: int = 3             # проходы склейки уже собранных групп по их общей траектории
    # передача между группами: одна кончилась — другая началась рядом, внешность групп очень похожа
    group_handoff_m: float = 1.2
    group_handoff_app: float = 0.9
    group_handoff_max_overlap_s: float = 1.0
    group_handoff_min_obs: int = 50
    # группы, долго идущие вместе с одинаковой внешностью, — один человек, даже если между их треклетами
    # есть редкие запреты (шум: дубль или ошибочный треклет)
    group_soft_m: float = 0.8
    group_soft_near: float = 0.7
    group_soft_app: float = 0.9
    group_soft_min_samples: int = 10
    group_soft_conflict_frac: float = 0.1
    major_min_s: float = 10.0         # «основной» человек в отчёте: суммарно не меньше


@dataclass
class TracksCfg:
    """Сквозные треки: человек (person_id) режется там, где его не видит ни одна камера дольше end_gap_s."""
    end_gap_s: float = 2.0            # пропуск во всех камерах дольше — трек завершён
    min_track_s: float = 2.0          # короче — шум (ложная детекция), в счёт не идёт
    min_track_obs: int = 10           # …или меньше наблюдений по всем камерам
    cam_merge_gap_s: float = 1.0      # интервалы одной камеры с паузой меньше — один интервал в списке
    map_step_s: float = 0.2           # шаг точек пути на плане
    # дубль: короткий трек всё время рядом с другим (тот же человек, недосклеенный кусок) — поглощается им
    absorb_m: float = 0.8             # медиана расстояния на плане
    absorb_overlap: float = 0.8       # доля моментов трека, когда рядом идёт другой трек
    # …или дальше (расхождение калибровок на краю кадра), но с той же внешностью
    absorb_app_m: float = 2.0
    absorb_app_sim: float = 0.93      # EPFL: части одного человека 0.94–0.99, разные люди ≤ 0.80
    absorb_app_overlap: float = 0.5   # …и трек целиком внутри другого по времени, общих моментов не меньше
    absorb_near_m: float = 1.0        # ближе этого хватает сходства absorb_near_sim: чем ближе, тем меньше нужна внешность
    absorb_near_sim: float = 0.90
    # фантом: трек видела одна камера, хотя он стоял в зоне обзора других (≥ phantom_min_other_cams) — ложная детекция
    phantom_max_s: float = 15.0
    phantom_inside: float = 0.8       # доля точек внутри наблюдаемого участка других камер
    phantom_min_other_cams: int = 2
    # пропуск детекции: человек пропал дольше end_gap_s, но не выходил — исчез и появился внутри зоны обзора
    # камеры (не у края кадра, не в зоне выхода), рядом с тем же местом -> трек и ID сохраняются
    # продолжение: куски на одной камере встык по времени и месту — один трек, даже если группы людей разные
    cont_gap_s: float = 0.0           # пауза не больше; 0 — выключено
    cont_m: float = 1.0               # расстояние конец -> начало не больше (+ 2 м/с за паузу)
    cont_app: float = 0.30            # внешность не противоречит: сходство не ниже
    phantom_by_fov: bool = True       # False: «место видят другие» — только где они реально видели людей (стены в офисе)
    detect_gap_max_s: float = 8.0     # дольше — всё равно новый трек
    fov_margin_m: float = 0.5         # точка исчезновения/появления не ближе к краю зоны обзора
    bridge_v_max_mps: float = 1.5     # за время пропуска человек мог уйти не дальше (+ 1 м)
    # призрак: короткий трек плохого качества, почти всё время перекрытый другим человеком, — не человек,
    # а ложная детекция рядом с ним. EPFL: у людей q ≥ 0.23 и перекрытие ≤ 0.19; у призрака q 0.07 и 0.45
    ghost_max_s: float = 15.0
    ghost_q_max: float = 0.15         # средняя оценка качества наблюдений не выше
    ghost_occ_min: float = 0.35       # среднее перекрытие не ниже
    ghost_conf_max: float = 0.65      # средняя уверенность детектора не выше
    # люди: треки одного человека (ушёл из всех камер и вернулся) — по внешности, треки не пересекаются по времени
    person_app_min: float = 0.88      # сходство прототипов пары треков (EPFL: один человек ≥ 0.95, разные ≤ 0.80)
    person_app_floor: float = 0.80    # при слиянии групп ни одна пара треков не должна быть ниже
    person_app_margin: float = 0.20   # порог сцены: настолько похожее, чем люди, которых одна камера видела вместе
    clothes_veto: float = 80.0        # одежда явно разная — разные люди; 0 — выключено
    cam_debias: float = 0.0           # какую долю среднего вектора камеры вычитать; 0 — не вычитать
    scene_lam: float = 0.0            # подавление разброса внутри треклетов: меньше — сильнее; 0 — выключено
    clothes_bright: float = 90.0      # заметная одежда: хотя бы одна зона светлее этого
    clothes_near: float = 30.0        # совпала: обе зоны ближе этого
    clothes_bonus: float = 0.06       # заметная совпавшая одежда снижает порог внешности настолько
    part_min: float = 0.0             # номер человека: сходство нижней половины тела не ниже этого; 0 — не сравнивается
    person_far_sim: float = 0.0       # короткий трек далеко по времени присоединяется только при таком сходстве; 0 — нет
    person_far_gap_s: float = 60.0    # «далеко»: столько секунд между треками
    person_far_short_s: float = 8.0   # «короткий»: не дольше стольких секунд
    person_small_s: float = 0.0      # человек, виденный не дольше этого, считается обрывком другого; 0 — нет
    person_small_sim: float = 0.5     # сходство обрывка с группой
    person_small_margin: float = 0.08 # и на столько лучше следующей группы
    person_link_min: float = 0.0    # треки одной группы склейки — один человек только при таком сходстве; 0 — всегда
    absorb_same_cam_s: float = 0.0   # одна камера видела обоих дольше этого — не дубль; 0 — проверки нет
    person_overlap_tol_s: float = 0.5 # треки одного человека не могут идти одновременно (допуск)
    person_twice_sim: float = 0.0     # с таким сходством одновременные треки — один человек на дальних концах двух камер; 0 — нет
    person_twice_s: float = 10.0      # и вместе идут не дольше этого
    person_twice_m: float = 8.0       # и на плане расходятся не дальше этого: вдали план ошибается на метры


@dataclass
class AutoCalibCfg:
    """Автопривязка камеры к плану по идущим людям (python -m mtmc autocalib)."""
    person_height_m: float = 1.70     # средний рост: даёт масштаб в метрах (если высота камеры не задана)
    q_min: float = 0.5                # отбор наблюдений для самокалибровки
    occ_max: float = 0.1
    aspect_min: float = 1.8           # высота/ширина рамки: стоящий человек (сидящие портят оценку)
    min_h_px: float = 40
    min_samples: int = 150
    max_samples: int = 3000
    resid_ok: float = 0.15            # медиана ошибки «голова» / рост человека в кадре — калибровка хорошая
    wall_thr: int = 110               # пиксели плана темнее — стены
    near_cam_m: float = 0.6           # возле камеры стены не учитываются (камера висит на стене)
    los_ok: float = 0.9               # доля точек, которые камера «видит» сквозь план без стен
    los_check: float = 0.75
    ambiguity_max: float = 0.9        # второй по качеству поворот не должен быть почти так же хорош
    overlap_ok_m: float = 0.5         # медиана расхождения с соседними камерами в общей зоне
    pos_prior_m: float = 0.5          # насколько можно сдвинуть камеру от положения в site.json
    refine_rounds: int = 3
    # фокус по людям определяется плохо (на EPFL завышался в 2.3 раза): берётся типичный угол обзора,
    # а при общих зонах с соседями выбирается из сетки тот, где камеры совпадают лучше
    hfov_prior_deg: float = 65.0
    hfov_grid: list = field(default_factory=lambda: [45, 55, 65, 75, 90, 105])
    f_prior_weight: float = 0.15


@dataclass
class SilhouetteCfg:
    """Силуэты (контуры фигур) поверх найденных людей: YOLO11-seg, сопоставление по рамкам."""
    enabled: bool = True              # строить после process (и в run)
    weights: str = "weights/yolo11m-seg.pt"
    imgsz: int | None = 960           # None -> как у детектора; маске хватает меньшего разрешения
    conf: float = 0.25
    device: str | None = None         # None -> как у детектора
    batch: int = 4                    # больше — на 8 ГБ видеопамяти Windows уходит в общую память, в 10–20 раз медленнее
    iou_match: float = 0.5            # маска относится к человеку, если рамки совпадают не хуже
    simplify_frac: float = 0.01       # допуск упрощения контура — доля высоты человека в кадре
    max_pts: int = 48
    every: int = 1                    # маска на каждом N-м кадре; в промежутках контур соседнего кадра сдвигается
                                      # за рамкой (на процессоре маски — почти половина времени обработки)


@dataclass
class UICfg:
    max_height: int = 720             # видео выше — уменьшаются для браузера
    crf: int = 24
    gop: int = 12                     # опорный кадр каждые N кадров (быстрая перемотка)
    skeleton: bool = True             # передавать позу (силуэт) в интерфейс


@dataclass
class IOCfg:
    flush_rows: int = 50_000


@dataclass
class Config:
    video: VideoCfg = field(default_factory=VideoCfg)
    gate: GateCfg = field(default_factory=GateCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    tracker: TrackerCfg = field(default_factory=TrackerCfg)
    observation: ObservationCfg = field(default_factory=ObservationCfg)
    crops: CropsCfg = field(default_factory=CropsCfg)
    tracklets: TrackletCfg = field(default_factory=TrackletCfg)
    reid: ReIDCfg = field(default_factory=ReIDCfg)
    render: RenderCfg = field(default_factory=RenderCfg)
    gallery: GalleryCfg = field(default_factory=GalleryCfg)
    link: LinkCfg = field(default_factory=LinkCfg)
    tracks: TracksCfg = field(default_factory=TracksCfg)
    ui: UICfg = field(default_factory=UICfg)
    autocalib: AutoCalibCfg = field(default_factory=AutoCalibCfg)
    silhouette: SilhouetteCfg = field(default_factory=SilhouetteCfg)
    io: IOCfg = field(default_factory=IOCfg)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _merge(obj: Any, data: dict, path: str) -> None:
    names = {f.name for f in dataclasses.fields(obj)}
    for key, value in data.items():
        if key not in names:
            raise ValueError(f"config: неизвестный параметр '{path}{key}'")
        cur = getattr(obj, key)
        if dataclasses.is_dataclass(cur):
            if not isinstance(value, dict):
                raise ValueError(f"config: '{path}{key}' должен быть словарём")
            _merge(cur, value, f"{path}{key}.")
        elif isinstance(cur, dict) and isinstance(value, dict):
            merged = dict(cur)
            merged.update(value)
            setattr(obj, key, merged)
        else:
            setattr(obj, key, value)


def config_from_dict(data: dict | None) -> Config:
    """Собирает Config из словаря поверх дефолтов; неизвестные ключи — ошибка."""
    cfg = Config()
    if data:
        _merge(cfg, data, "")
    return cfg


def load_config(path: str | Path | None) -> Config:
    """Читает config.yaml; если путь не задан — дефолты из ТЗ."""
    if path is None:
        return Config()
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return config_from_dict(data)
