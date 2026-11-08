# Справка по командной строке `python -m mtmc`

> Это справка разработчика по отдельным командам конвейера. Как запустить решение — в [README](../README.md).
> Примеры EPFL (`configs/examples/`) и тесты (`tests/`) в сдаваемую версию не входят; команды для офиса — в [SOLUTION.md](../SOLUTION.md#запуск).

## MTMC, часть 1: обработка одной камеры

Программа принимает видео одной камеры и её паспорт (JSON). Она находит людей (YOLO11-pose) и ведёт их трекером BoT-SORT, а затем режет треки на надёжные треклеты по трём правилам: пауза, скачок и смена внешности (OSNet). Результат пишется в SQLite, parquet и npz, кропы сохраняются отдельно. Видео с разметкой собирает отдельная команда, которая не запускает нейросети.

> **Новые видео:** запустите `!start.bat` без параметров. На одном сайте можно разметить план, выбрать камеры и отрезок, запустить обработку и открыть любой сохранённый результат. Подробнее в [НОВЫЕ_ВИДЕО.md](../НОВЫЕ_ВИДЕО.md).
>
> **Хакатон (сквозные треки по нескольким камерам, интерфейс, план):** см. [SOLUTION.md](../SOLUTION.md).
> Всё одной командой:
> ```bash
> python -m mtmc run --passports "configs/examples/epfl_lab_6p_c*.json" --data results_6p --ui ui_6p
> ```
> Затем: `python -m mtmc serve --dir ui_6p`.

## Установка (≈10 минут)

```bash
python -m venv .venv && .venv\Scripts\activate        # Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu118   # под свою CUDA, или CPU-версию
pip install -r requirements.txt
python scripts/download_weights.py                     # Linux: bash scripts/download_weights.sh
python -m pytest -q tests                              # 80 тестов, без GPU и без моделей
```

Нужен системный `ffmpeg` с `libx264` в PATH. Другой путь можно задать в `render.ffmpeg` в конфиге.

## Быстрый пример (EPFL Laboratory, уже лежит в `data/`)

```bash
python -m mtmc process --camera configs/examples/epfl_lab_4p_c0.json --data results --end 60
python -m mtmc render  --cam 0 --data results --out renders/cam0.mp4 --label tracklet --details
python -m mtmc gallery --cam 0 --data results --out renders/cam0_gallery.jpg
python -m mtmc stats   --data results
```

Если `--config` не указан, берётся `configs/config.yaml` (если он есть), иначе встроенные значения по умолчанию.

## Команды

| Команда | Что делает |
| --- | --- |
| `process --camera P.json --data D [--start s --end s] [--device cpu] [--stride N]` | проходы А и Б, запись в `D`; повторный запуск заменяет данные этой камеры целиком |
| `render --cam N --data D --out X.mp4` | видео с разметкой из БД и файлов; флаги `--label local/tracklet/person`, `--details`, `--skeleton`, `--zones`, `--minimap`, `--fps-mode original/processed`, `--scale`, `--start/--end`, `--tid T [--tid T2] --clip`, `--run` |
| `gallery --cam N --data D --out G.jpg` | сетка кропов по треклетам; если строк больше 50, файлы `G_001.jpg`, … |
| `stats --data D` | сводка по камерам, треклетам и причинам разрезов |

У всех команд есть `--log-level DEBUG|INFO|WARNING`.

## Паспорт камеры

Обязательны только `cam_id` и `video`; остальные поля описаны в разделе 2.2 ТЗ. Без `H` координаты плана пишутся как NULL, а правило скачка считается в пикселях. Без `zones` и `overlaps` соответствующие поля тоже NULL. Ошибки паспорта называют поле, например `поле 'H': должна быть матрица 3×3`.

Примеры лежат в `configs/examples/`:
- `epfl_lab_4p_c0.json` — видео из `data/4 person/`, только обязательные поля.
- `epfl_terrace_cam0.json` — положите `terrace1-c0.avi` в `datasets/epfl_terrace/`. Гомографию с сайта EPFL CVLab можно добавить в поле `H`.
- `wildtrack_cam1.json` — для полного паспорта с `H`, `K` и `dist` сгенерируйте его из калибровки:
  ```bash
  python scripts/wildtrack_passport.py --root datasets/Wildtrack --cam 1 --out configs/examples/wildtrack_cam1.json
  ```

## Что лежит в `--data`

```
mtmc.sqlite                      cameras, tracklets; runs/persons/tracklet_person — пустые заготовки для части 2
observations/cam0.parquet        все наблюдения (раздел 4.2 ТЗ), сортировка по frame_idx
observations/cam0.meta.json      stride и отрезок кадров последней обработки (для render)
embeddings/cam0/<tid>.npz        embs (float16), q, view, frame_idx, proto_*, model
crops/cam0/<local_id>/<frame>.jpg
runs/cam0_<время>.json           конфиг, версии, хэши весов, время по этапам, скорость, счётчики
```

Камеры можно обрабатывать параллельно в разных процессах в одну папку `--data`. SQLite работает в режиме WAL, транзакции короткие. Если запуск упал на середине, прежние данные камеры остаются как были.

## Часть 2: склейка людей между камерами

Нужно, чтобы все камеры одной сцены были обработаны в одну папку `--data` и имели гомографию.

```bash
python scripts/epfl_passports.py --calib datasets/epfl_lab/calibration-6p.txt --video-dir "data/6 person" --prefix 6p
python -m mtmc process --camera configs/examples/epfl_lab_6p_c0.json --data results_6p     # и так для c1..c3
python -m mtmc link    --data results_6p                       # треклеты -> люди (новый run_id)
python -m mtmc mosaic  --data results_6p --out renders/mosaic_6p.mp4 --scale 1.5
python -m mtmc render  --cam 1 --data results_6p --out renders/6p_c0_person.mp4 --label person
python scripts/eval_persons.py --data results_6p --gt datasets/epfl_lab/gt_lab_6p.txt
```

`link` заполняет таблицы `runs`, `persons` и `tracklet_person`. `mosaic` показывает все камеры одним видео, справа — план сверху: у каждого человека один номер и один цвет на всех камерах. Пороги задаются в разделе `link:` конфига, решения записаны в `DECISIONS.md`.

Результат на EPFL Lab 6p по разметке EPFL: покрытие 99.6%, чистота людей 98.5%, полнота 91.8%. Пятеро из шести человек почти целиком (90–99% времени) идут под одним номером.

Чистота треклетов части 1 по той же разметке — 96% (`scripts/eval_tracklet_purity.py`). Пороги нарезки можно пробовать без повторной детекции: `python -m mtmc retrack --data results_6p` занимает секунды на камеру.

## Проверка качества

```bash
python scripts/eval_purity.py --data results --cam 1 --gt-wildtrack datasets/Wildtrack/annotations_positions --view 1
python scripts/eval_purity.py --data results --cam 0 --gt-mot gt.txt          # MOT-формат frame,id,x,y,w,h
```

Скрипт выводит долю треклетов с чистотой ≥ 95% (цель ТЗ — не меньше 95%) и сколько в среднем треклетов приходится на одного человека.

## Скорость (замер)

RTX 4060, EPFL Laboratory 360×288 @ 25 fps, `stride: 2`, `imgsz: 1280`, FP16:

| этап | время на 375 обработанных кадров (30 с видео) |
| --- | --- |
| декодирование | 0.2 с |
| детекция + трекинг | 16.8 с (≈22 кадр/с) — **узкое место** |
| наблюдения + кропы | 0.6–1.0 с |
| Re-ID (OSNet, flip-TTA) | 0.5–0.7 с |
| нарезка, сводка, запись | < 0.2 с |
| итого | ≈ 24 с, **×1.2 быстрее реального времени** (без 5.5 с загрузки моделей) |

Почти всё время уходит на детекцию. Для видео низкого разрешения `detector.imgsz: 640` даёт ускорение примерно в 3 раза. Точные цифры каждого запуска лежат в `runs/*.json` (`timings_s`, `speed`).

## Структура кода

`mtmc/`: `config`, `passport`, `video_io`, `detect_track`, `observation`, `geometry`, `crops`, `tracklets`, `reid`, `storage`, `pipeline`, `render`, `gallery`, `draw`, `__main__`. Модули `observation`, `geometry` и `tracklets` состоят из чистых функций, их покрывают тесты в `tests/`. Решения по неоднозначным местам ТЗ записаны в `DECISIONS.md`.
