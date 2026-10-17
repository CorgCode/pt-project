# PT-05: эмбеддинги хостов по сетевому трафику

Проект Positive Technologies × МФТИ. Сайт: https://corgcode.github.io/pt-project/

## Что делаем

По логам сети (NetFlow) строим для каждого хоста вектор — эмбеддинг. По нему хотим:

- находить похожие хосты;
- видеть группы и роли хостов;
- замечать странное поведение.

Данные — [LANL 2017](https://csr.lanl.gov/data/2017/). Разметки «кто есть кто» нет.

## Как это работает

```text
логи LANL → pipeline → окна train / validation / test
                     → эталоны для оценки
train → методы → эмбеддинги → evaluation → таблица результатов
```

Все методы учатся на одном train-окне и проверяются на одних и тех же эталонах.

## Быстрый старт

```bash
pip install -r requirements.txt
python -m experiments.end_to_end --fixture --workdir runs/demo   # самопроверка на тестовых данных, ~2 минуты
python -m pytest -q evaluation benchmark app pipeline experiments
```

## Проверить свой метод

1. `cp scoreboard/template_method.py scoreboard/<метод>.py` и поменяй в нём две функции: `fit` и `embed`.
2. `python scoreboard/<метод>.py` — обучит, сохранит и покажет место.
3. Commit и push. CI добавит метод в [таблицу](scoreboard/README.md).

## Где что

| папка | что |
|---|---|
| [`scoreboard/`](scoreboard/README.md) | общая таблица результатов |
| [`evaluation/`](evaluation/README.md) | метрики |
| [`benchmark/`](benchmark/README.md) | методы и бейзлайны |
| [`pipeline/`](pipeline/README.md) | подготовка данных |
| [`experiments/`](experiments/README.md) | запуски, в том числе [на LANL](experiments/lanl/README.md) |
| [`app/`](app/README.md) | дашборд |
| `docs/` | сайт и [методология](docs/METHODOLOGY.md) |
| [`research/`](research/README.md) | ресёрч участников |

## Результаты на LANL

Первый прогон — один день, 2 242 активных хоста ([таблица](scoreboard/README.md), [разбор](docs/METHODOLOGY.md)):

- общий балл (0 — случайно, 100 — идеально) у лучших — **48**: это гибриды поведенческого профиля с node2vec / DeepWalk
  (`scoreboard/combine.py`); чистый поведенческий профиль — 47, графовые методы — 33–43;
- **связи**: AUC 0.99 — выше правила «пара уже общалась» (0.94) и случайного уровня (0.64–0.76);
- **новые собеседники через похожих хостов**: в ~3.4 раза лучше прогноза без эмбеддинга;
- **узнать тот же хост через полчаса**: до 18% с первой попытки (случайно — 0.1%);
- **соседи по поведению** лучше всего у чистого поведенческого профиля (0.27).

Дальше: подбор параметров и больше дней (#60), кластеры и аномалии (#21).

## Команда

Даниил @physcorgi · Дима @DrMogger · Тихон @Tikhon2783 · Даня @kalivixZ · Даша @ElliOrNora

Как работаем — [`CONTRIBUTING.md`](CONTRIBUTING.md). Задачи — в [Issues](https://github.com/CorgCode/pt-project/issues).
