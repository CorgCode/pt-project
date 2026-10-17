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

Первый прогон — один день, 2 242 активных хоста ([все числа](scoreboard/README.md), [разбор](docs/METHODOLOGY.md)):

- все методы, кроме struc2vec, заметно лучше случайных векторов;
- **связи**: обученный классификатор на эмбеддингах behavioral и спектрального графа даёт ROC-AUC 0.98 —
  выше простого правила «пара уже общалась» (0.94);
- **новые связи**: лучше всех DeepWalk (0.65), остальные около 0.5;
- **новые собеседники через похожих хостов**: node2vec и DeepWalk в ~3 раза лучше прогноза без эмбеддинга;
- узнать тот же хост через полчаса пока трудно всем (до 10%).

Дальше: больше дней и подбор параметров (#60), кластеры и аномалии (#21).

## Команда

Даниил @physcorgi · Дима @DrMogger · Тихон @Tikhon2783 · Даня @kalivixZ · Даша @ElliOrNora

Как работаем — [`CONTRIBUTING.md`](CONTRIBUTING.md). Задачи — в [Issues](https://github.com/CorgCode/pt-project/issues).
