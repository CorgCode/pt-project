# PT-05 — эмбеддинги хостов по сетевому трафику

Командный проект Positive Technologies × МФТИ.
Сайт проекта: https://corgcode.github.io/pt-project/ (пока только синтетические результаты).

## Цель

В корпоративной сети есть журнал соединений: кто с кем связывался, по какому порту, сколько трафика.
Какой хост чем занимается (сервер, рабочая станция, принтер) — неизвестно. Мы хотим по этим логам
**научиться строить для каждого хоста вектор — эмбеддинг**, чтобы по нему:

- находить хосты с похожим поведением;
- видеть группы и роли хостов без ручной разметки;
- замечать подозрительные изменения поведения;
- объяснять, почему хосты считаются похожими.

Данные — [LANL Unified Host and Network Dataset](https://csr.lanl.gov/data/2017/) (NetFlow).

## Как это устроено

```text
сырые логи LANL ─▶ pipeline ─▶ окна train / validation / test, когорта хостов, графы
                        │
                        └─▶ эталоны для оценки (строятся один раз, без участия моделей)

train-окно ─▶ методы и бейзлайны ─▶ эмбеддинги host_id, emb_0, emb_1, …
                                          │
эталоны ──────────────────────────────────┴─▶ evaluation ─▶ таблица результатов, отчёт, сайт
```

Главный принцип: **все методы учатся на одних данных и проверяются на одних эталонах**,
иначе их нельзя сравнивать. Модели видят только train-окно; оценка — по более поздним окнам.

## Быстрый старт

```bash
pip install -r requirements.txt

# всё сразу на синтетике в формате LANL (без скачивания данных), ~1 минута
python -m experiments.end_to_end --fixture --workdir runs/demo
# результат: runs/demo/run/report.md, runs/demo/run/scoreboard_test.md

# на настоящих файлах LANL
python -m experiments.end_to_end --netflow data/raw/netflow_day-0*.bz2 --config cfg.json \
    --workdir runs/lanl --data-kind lanl --source-note "откуда, когда, кто скачал" --methods all

# тесты
python -m pytest -q evaluation benchmark app pipeline experiments
```

## Где что лежит

| папка | что внутри | подробнее |
|---|---|---|
| `pipeline/` | LANL → окна → когорта → графы → эталоны | [`pipeline/README.md`](pipeline/README.md) |
| `benchmark/` | графовые методы (DeepWalk, node2vec, RolX, struc2vec, GraphSAGE) и бейзлайны | [`benchmark/README.md`](benchmark/README.md) |
| `evaluation/` | **оценка**: все метрики одной командой, таблица результатов, как подключить метод | [`evaluation/README.md`](evaluation/README.md) |
| `experiments/` | запуск всего вместе, подбор параметров, абляции, устойчивость | [`experiments/README.md`](experiments/README.md) |
| `app/` | дашборд на Streamlit | [`app/README.md`](app/README.md) |
| `docs/` | сайт на GitHub Pages и [методология](docs/METHODOLOGY.md) | |
| `research/` | ресёрч первого этапа, по файлу на участника | [`research/README.md`](research/README.md) |
| `demo/` | маленький синтетический результат для дашборда и сайта | |

## Где мы сейчас

- Готово и проверено **на синтетике**: пайплайн данных, 5 графовых методов, бейзлайны, единая оценка и таблица результатов.
- Не сделано: прогон на настоящих данных LANL (#60), кластеризация и аномалии (#21, PR #67), внешние метки для зонда (#25).
- Научных выводов о качестве методов пока **нет** — синтетика проверяет код, а не методы.

Задачи — в [Issues](https://github.com/CorgCode/pt-project/issues), кто что делал — в [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

## Команда

| | GitHub |
|---|---|
| Даниил | @physcorgi |
| Дима | @DrMogger |
| Тихон | @Tikhon2783 |
| Даня | @kalivixZ |
| Даша | @ElliOrNora |

## Как работать

Issue → своя ветка → pull request → ревью → `main`. Напрямую в `main` не пишем.
Ветки: `<логин>/<тип>-<номер issue>-<коротко>`, например `physcorgi/feature-68-eval`.
Подробно — [`CONTRIBUTING.md`](CONTRIBUTING.md).

Никогда не коммитим пароли, токены, ключи, `.env`, VPN-конфиги и сами данные (`data/` и `runs/` в `.gitignore`).
