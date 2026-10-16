# Эксперименты

Все методы на **одних** хостах, окнах, эталонах и кандидатах — одной командой.

```bash
pip install -r requirements.txt

# синтетика в формате LANL, без скачивания
python -m experiments.end_to_end --fixture --workdir runs/demo

# настоящие данные (без --data-kind lanl прогон помечается 'unverified', а не LANL)
python -m experiments.end_to_end --netflow data/raw/netflow_day-0*.bz2 --config my_config.json \
    --workdir runs/lanl --data-kind lanl --source-note "форма LANL, <дата>, <кто>" \
    --methods all [--submissions-dir submissions]
```

Этапы можно запускать по отдельности: `pipeline.prepare` → `pipeline.reference` → `experiments.run_experiment`.

## Что получится в `<workdir>/run`

| файл | что внутри |
|---|---|
| `report.md`, `figures/` | отчёт: таблицы с интервалами, графики, аудит |
| `scoreboard_<окно>.md` / `.csv` | таблица результатов (`evaluation.scoreboard`): главные числа по группам и случайный уровень |
| `results/<окно>/<метод>.json` | полный результат оценки: метрики, интервалы, детали, хеши эталонов |
| `identity/<окно>/<метод>.json` | узнаётся ли хост через время: без выравнивания и с Procrustes |
| `audit.json` | чувствительность эталона к весам признаков, стабильность окон, случайный уровень, пересечение с эталонами |
| `manifest.json` | тип данных, хеши когорты и эталонов, seed, версии библиотек, git SHA |

## Какие методы участвуют

Бейзлайны (`random`, `behavioral`, `behavioral_pca`, `behavioral_svd`, `graph_spectral`), графовые методы
из `--methods` (`all` — все пять), потолок `reference_ceiling` (копия эталона, «утекает» намеренно) и любые
`--submissions-dir/<метод>/embeddings.csv`. Как подготовить свой файл — [`evaluation/README.md`](../evaluation/README.md).
Методы, которые ждали, но не получили, перечисляются в отчёте как «не оценены» — ничего не симулируется.

Перед тем как делать выводы по числам, прочитайте [`docs/METHODOLOGY.md`](../docs/METHODOLOGY.md).

## Другие команды

| команда | что делает |
|---|---|
| `python -m experiments.tune_methods …` | подбор гиперпараметров метода только на validation |
| `python -m experiments.feature_ablation …` | абляция признаков: профиль с портами и без (#43) |
| `python -m experiments.robustness --processed-dir … --out-dir … --methods deepwalk` | устойчивость к удалению связей (идея Тихона, #40) |
| `python -m experiments.make_demo_bundle` | пересобрать синтетический пример в `demo/synthetic_run` для дашборда и сайта |

### Устойчивость к удалению связей

Удаляем долю случайных пар хостов (все потоки между ними) из train-окна, обучаем метод заново и сравниваем
с эмбеддингом на полном графе: доля общих k ближайших соседей и косинус после поворота. «Шум» — тот же метод
с другим seed без удаления: это потолок, выше которого ждать нечего.
