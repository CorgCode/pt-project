# Прогон на мощной машине

Сейчас методы учатся всего на **1 часе** трафика — это главное ограничение балла. Здесь — как обучить на 6 часах
и как проверить на **другом дне**, которого методы не видели.

## Какие прогоны

| конфиг | train | validation | test | зачем |
|---|---|---|---|---|
| `config_day02.json` (сейчас) | день 02, 11:00–12:00 | 12:00–12:30 | 12:30–13:00 | текущая таблица |
| `config_train6h.json` | день 02, 06:00–12:00 | 12:00–12:30 | 12:30–13:00 | тот же test, но в 6 раз больше данных для обучения |
| `config_holdout_day03.json` | день 02, 06:00–12:00 | 12:00–12:30 | **день 03**, 12:30–13:00 | проверка на дне, которого не видели |

- В когорту попадают хосты с ≥ 1800 событиями в train (300 в час, как сейчас), чтобы хостов было примерно столько же.
- Окна заданы до прогона. Подбирать что-то по test нельзя — только по validation.
- Числа нового прогона нельзя ставить в одну таблицу с текущими: другая когорта и другой train. Таблица сама
  откажется их смешивать.

## Машина

| что | 6 часов train | целый день train |
|---|---|---|
| память | от 32 ГБ, лучше 64 | от 128 ГБ |
| диск | 30 ГБ | 60 ГБ |
| процессор | 8+ ядер | 8+ ядер |

Оценка грубая: 1 час train занимает ≈ 3 ГБ в памяти, графовые методы растут с числом хостов и связей.
Видеокарта не нужна.

## Шаги

```bash
# 1. код
git clone https://github.com/CorgCode/pt-project && cd pt-project
git checkout -b <логин>/research-60-train6h
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

# 2. данные: день 02 (и 03 — для проверки на другом дне), токен — из формы LANL
W=~/lanl6h
LANL_TOKEN=<токен> experiments/lanl/fetch.sh 02 $W/raw
LANL_TOKEN=<токен> experiments/lanl/fetch.sh 03 $W/raw                  # только для holdout_day03

# 3. подготовка и эталоны (для holdout — оба файла и config_holdout_day03.json)
python -m pipeline.prepare run --netflow $W/raw/netflow_day-02.bz2 --config experiments/lanl/config_train6h.json \
    --output-dir $W/processed --data-kind lanl --source-note "форма LANL, <дата>, <кто>"
python -m pipeline.reference --processed-dir $W/processed --output-dir $W/reference --pairs-per-anchor 10

# 4. новые данные — в таблицу ЭТОЙ ветки
cp $W/processed/cohort.csv $W/processed/manifest.json scoreboard/data/processed/
rm -rf scoreboard/data/reference && cp -r $W/reference scoreboard/data/reference
find scoreboard/submissions -name '*.csv' -delete && rm -f scoreboard/results/*.json   # старые векторы — от другого train
export PT_DATA=$W/processed

# 5. обучить и оценить: бейзлайны и графовые методы, затем лидер и гибриды
python -m evaluation.submit --builtin all --force
python scoreboard/hybrid_tfidf.py
python scoreboard/combine.py behavioral node2vec graph_spectral --name hybrid_weighted --tune-weights
python -m evaluation.submit --all
```

`*_services` и `hybrid_services` этими командами не пересчитываются (нужен `run_experiment` с
`methods_v2.json`, см. [README](README.md#улучшенный-прогон-v2)) — в новой таблице их не будет.

Таблица — `scoreboard/README.md` в этой ветке. Commit и push в ветку, **в `main` не сливать**: это другой
протокол. Числа сравнить с текущей таблицей и написать в #60.

## Чего ждать

- Больше всего должна вырасти узнаваемость (Identity): портрет хоста за 6 часов стабильнее.
- Neighbors и Pairs проверяются на тех же 30 минутах — вырастут меньше.
- New partners почти не вырастет: даже эталон из будущего даёт ≈ 0.23.
- holdout_day03 — честная проверка: если на другом дне порядок методов тот же, результат не случайность одного дня.

Это прогноз, на LANL такой прогон ещё не делали.

## Если не хватает памяти

- Взять 3 часа: `"train": [118800, 129600]`, `"min_events_per_window": 900`.
- Убрать самые тяжёлые методы: `python -m evaluation.submit --builtin random behavioral behavioral_pca behavioral_svd graph_spectral deepwalk node2vec`.
- Следить за памятью: `top` или `htop`; при нехватке процесс упадёт с `Killed`.
