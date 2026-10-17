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
python -m experiments.end_to_end --fixture --workdir runs/demo   # всё сразу на синтетике, ~2 минуты
python -m pytest -q evaluation benchmark app pipeline experiments
```

## Проверить свой метод

1. Обучись на `scoreboard/data/processed` (только train).
2. Положи `scoreboard/submissions/<метод>/embeddings.csv`.
3. Commit и push. CI сам посчитает и добавит метод в [таблицу](scoreboard/README.md).

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

## Статус

- Готово: данные, 5 графовых методов, бейзлайны, оценка, таблица.
- Идёт: первый прогон на LANL (#60).
- В работе: кластеры и аномалии (#21), внешние метки (#25).

## Команда

Даниил @physcorgi · Дима @DrMogger · Тихон @Tikhon2783 · Даня @kalivixZ · Даша @ElliOrNora

Как работаем — [`CONTRIBUTING.md`](CONTRIBUTING.md). Задачи — в [Issues](https://github.com/CorgCode/pt-project/issues).
