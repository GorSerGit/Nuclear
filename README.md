# Nuclear MTMC — сквозной трекинг людей по нескольким камерам

Решение кейса NtechLab на хакатоне Nuclear IT Hack. Команда «Новое название(2)», НИЯУ МИФИ:
Горбунов Сергей Михайлович, Черепков Павел Александрович, Маврицкий Иван Алексеевич.

Программа обрабатывает записи пяти камер офиса, находит людей и их силуэты, связывает наблюдения одного человека
на разных камерах в **сквозные треки с одним ID** и завершает трек, только когда человека дольше 2 с не видит ни одна камера.
Результат — число и список уникальных треков и веб-интерфейс с синхронным видео, силуэтами, ID и планом этажа.

**Отчёт:** [Отчёт.pdf](Отчёт.pdf) · **Скринкаст:** [Google Drive](https://drive.google.com/drive/folders/1wetiH9qStdNXzojK7xymWDFdKbYeugFR) ·
**Видео, веса и готовые результаты:** [Release v1.0](https://github.com/GorSerGit/Nuclear/releases/tag/v1.0)

## Отчётные материалы

| Пункт ТЗ | Где |
| --- | --- |
| 1. Запускаемый прототип и исходный код | код — [`mtmc/`](mtmc), [`scripts/`](scripts); запуск — `!start.bat`; описание кода — [src/README.md](src/README.md) |
| 2. Инструкция запуска и зависимости | раздел «Запуск» ниже, [`requirements.txt`](requirements.txt) |
| 3. Работа алгоритма в интерфейсе на предоставленном видео | [artifacts/demo](artifacts/demo/README.md): скринкаст, скриншоты в отчёте, готовый просмотр в Release |
| 4. Число уникальных треков; ID, начало, конец, камеры | **108 треков** — [artifacts/test-results/tracks.csv](artifacts/test-results/tracks.csv), [report.md](artifacts/test-results/report.md) |
| 5. Описание решения: алгоритм, готовые компоненты, ограничения, улучшения | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/LIMITATIONS.md](docs/LIMITATIONS.md), [docs/TEST_RESULTS.md](docs/TEST_RESULTS.md), подробно — [SOLUTION.md](SOLUTION.md), [Отчёт.pdf](Отчёт.pdf) |

## Результат на предоставленном видео

5 камер, 20 минут, 2592×1944. **108 уникальных сквозных треков**, из них 69 прошли через 2 и более камер, 9 — через все 5.
Медианная длина трека 13.6 с. Треки одного человека собраны в 34 человека (P1…P34) — это подсказка, в счёт треков не влияет.
Полный список: [artifacts/test-results/tracks.csv](artifacts/test-results/tracks.csv) (разделитель `;`, открывается в Excel).

## Запуск

### Быстрый старт (Windows)

1. Склонируйте или скачайте репозиторий.
2. Скачайте из [Release v1.0](https://github.com/GorSerGit/Nuclear/releases/tag/v1.0) три архива и распакуйте их **в папку репозитория**
   (пути внутри архивов уже от корня):
   - `Nuclear_data_weights.zip` — входные видео `data/office/` и веса моделей `weights/`;
   - `Nuclear_results.zip` — данные полного прогона `results_office/` и история обработок;
   - `Nuclear_ui_office.zip` — готовый просмотр всей записи `ui_office/`.
3. Запустите `!start.bat`. При первом запуске он сам ставит Python 3.13 (через winget), создаёт `.venv`,
   ставит зависимости и открывает интерфейс http://127.0.0.1:8780/.
4. В разделе «Результат» выберите «Вся запись» — готовый результат открывается без повторной обработки.
   Вкладка «Аналитика»: 3D-вид сверху, поиск по описанию одежды, видеоотчёт по человеку, проходы через линию и время в зонах.

Новую обработку можно запустить там же, в разделе «Обработка». Как подготовить свои видео и план — [НОВЫЕ_ВИДЕО.md](НОВЫЕ_ВИДЕО.md).

### Вручную (Python 3.13)

```bash
python -m venv .venv && .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu118   # под свою CUDA или CPU-версия
pip install -r requirements.txt
python scripts/download_weights.py          # если не скачан архив с весами
python -m mtmc run --passports "configs/office/cam*.json" --data results_office --ui ui_office --site configs/office/site.json
python -m mtmc serve --dir ui_office
```

### Зависимости

- Python 3.13 x64; GPU NVIDIA с CUDA желателен (на CPU обработка в разы медленнее).
- Пакеты — [`requirements.txt`](requirements.txt): torch 2.7, ultralytics 8.4 (YOLO11, BoT-SORT), torchreid (OSNet), timm и safetensors (PersonViT),
  opencv-python, numpy, scipy, pandas, pyarrow, pydantic, PyYAML, imageio-ffmpeg.
- Веса: YOLO11m-pose, YOLO11m-seg, PersonViT (MSMT17), OSNet x1.0 (MSMT17) — в архиве `Nuclear_data_weights.zip` или через `scripts/download_weights.py`.
- Интерфейс работает в любом современном браузере без интернета.

## Структура репозитория

| Путь | Что внутри |
| --- | --- |
| `mtmc/` | конвейер: детекция, треклеты, склейка, сквозные треки, интерфейс просмотра (`mtmc/ui_static/`) |
| `scripts/` | веб-приложение `office_web.py` (обработка, разметка, просмотр), инструменты разметки камер и плана |
| `configs/` | общий конфиг `config.yaml`; `configs/office/` — паспорта камер `cam1–5.json`, план, `site.json`, пороги `overrides.yaml` |
| `config/` | пояснение к настройкам (по шаблону хакатона) |
| `docs/` | архитектура, ограничения, результаты, справка по командам ([CLI.md](docs/CLI.md)) |
| `artifacts/` | материалы демонстрации и результаты прогона |
| `src/` | описание исходного кода и запуска (по шаблону хакатона) |
| `DECISIONS.md` | решения по спорным местам ТЗ и журнал доработок |
