# Исходный код

Платформа: **windows** (основная), linux — скрипт `start.sh`. Интерфейс — web (локальный сервер, браузер).

Код лежит в корне репозитория, а не в `src/`: так устроены пути запуска (`!start.bat` и `.bat`-скрипты обращаются к
`mtmc/`, `scripts/` и `configs/` относительно корня), и переносить их перед сдачей мы не стали.

| Путь | Что внутри |
| --- | --- |
| [`../mtmc/`](../mtmc) | конвейер: `detect_track`, `observation`, `lens`, `geometry`, `silhouette`, `tracklets`, `reid`, `link`, `tracks`, `ui`, `storage`, CLI `__main__` |
| [`../mtmc/ui_static/`](../mtmc/ui_static) | интерфейс просмотра: `app.js` (видео, силуэты, план, шкала), `features.js` (вкладка «Аналитика») |
| [`../scripts/office_web.py`](../scripts/office_web.py) | веб-приложение: загрузка видео, разметка камер и плана, запуск обработки, история результатов |
| [`../scripts/`](../scripts) | разметка камер (`camera_app`, `camera_auto`), плана (`site_tool`), диагностика, скачивание весов |
| [`../configs/`](../configs) | конфиг и паспорта камер офиса |

## Инструкция по запуску и сборке проекта

Сборка не нужна (Python). Запуск — в [README](../README.md#запуск):

1. Скачать три архива из [Release v1.0](https://github.com/GorSerGit/Nuclear/releases/tag/v1.0) и распаковать в папку репозитория.
2. Запустить `!start.bat` — он поставит Python 3.13 и зависимости и откроет http://127.0.0.1:8780/.

Командная строка: `python -m mtmc --help`, справка по командам — [docs/CLI.md](../docs/CLI.md).
