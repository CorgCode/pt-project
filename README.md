# PT-05: эмбеддинги хостов по сетевому трафику

Проект Positive Technologies × МФТИ. Сайт: https://corgcode.github.io/pt-project/

## Что делаем

По логам сети (NetFlow) строим для каждого хоста вектор — эмбеддинг. По нему хотим:

- находить похожие хосты;
- видеть группы и роли хостов;
- замечать странное поведение.

Данные — [LANL 2017](https://csr.lanl.gov/data/2017/). Разметки «кто есть кто» нет.

## С чего начать

| хочу | куда |
|---|---|
| посмотреть результаты | [таблица](scoreboard/README.md), [сайт](https://corgcode.github.io/pt-project/), `streamlit run app/streamlit_app.py` |
| проверить свой метод | [3 шага](CONTRIBUTING.md#свой-метод-в-таблицу) |
| понять, что значат метрики | [`evaluation/README.md`](evaluation/README.md) |
| понять, можно ли верить числам | [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) |
| повторить прогон на LANL | [`experiments/lanl/README.md`](experiments/lanl/README.md) |
| сделать PR | [`CONTRIBUTING.md`](CONTRIBUTING.md) |

## Словарь

| слово | что значит |
|---|---|
| хост | компьютер или сервер в сети (`Comp123456`, `ActiveDirectory`) |
| поток | одна запись NetFlow: кто, кому, порт, сколько байт |
| эмбеддинг | вектор чисел, который метод строит для хоста |
| окно | отрезок времени: train — учимся, validation — подбираем, test — проверяем |
| когорта | хосты, которых оцениваем (активные в train) |
| эталон | заранее посчитанное по логам «кто на кого похож»; с ним сравниваем эмбеддинг |
| балл | одно число 0–100: 0 — как у случайных векторов, 100 — идеально |

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
python -m pytest -q evaluation methods app pipeline experiments
```

## Проверить свой метод

1. `cp scoreboard/template_method.py scoreboard/<метод>.py` и поменяй в нём две функции: `fit` и `embed`.
2. `python scoreboard/<метод>.py` — обучит, сохранит и покажет место.
3. Commit и push. CI добавит метод в [таблицу](scoreboard/README.md).

## Где что

| папка | что |
|---|---|
| [`scoreboard/`](scoreboard/README.md) | общая таблица результатов и шаблон своего метода |
| [`results/`](results/README.md) | итоги прогонов на LANL |
| [`evaluation/`](evaluation/README.md) | метрики |
| [`methods/`](methods/README.md) | методы и бейзлайны |
| [`pipeline/`](pipeline/README.md) | подготовка данных |
| [`experiments/`](experiments/README.md) | запуски, в том числе [на LANL](experiments/lanl/README.md) |
| [`app/`](app/README.md) | дашборд |
| [`docs/`](docs/README.md) | [методология](docs/METHODOLOGY.md) |
| [`research/`](research/README.md) | ресёрч участников |

## Результаты на LANL

Проверка на трёх днях: вектор строится по часу дня 02, проверяется по часу в днях 02, 03 и 04; 5 445 хостов
([таблица](scoreboard/README.md), [разбор](docs/METHODOLOGY.md)):

- общий балл (0 — случайно, 100 — идеально; [как считается](scoreboard/README.md#как-считается-score)) у лучшего — **39.1**:
  `temporal_encoder` — поведение + собеседники + сервисы + нейросеть, которая узнаёт хост во времени
  ([`scoreboard/temporal_encoder.py`](scoreboard/temporal_encoder.py)); лёгкая версия без нейросети — 38.2;
  чистый поведенческий профиль — 32, графовые методы — 4–20;
- места лучших методов одинаковы на всех трёх днях;
- **связи**: AUC 0.99 (тот же классификатор на случайных векторах — 0.82);
- **новые собеседники через похожих хостов**: MRR 0.31 против 0.085 без эмбеддинга;
- **узнать тот же хост на другой день**: до 41% с первой попытки (случайно — 0.1%);
- **соседи по поведению** лучше всего у чистого поведенческого профиля (0.15).

Дальше: подбор параметров и больше дней (#60), кластеры и аномалии (#21).

## Команда

Даниил @physcorgi · Дима @DrMogger · Тихон @Tikhon2783 · Даня @kalivixZ · Даша @ElliOrNora

Как работаем — [`CONTRIBUTING.md`](CONTRIBUTING.md). Задачи — в [Issues](https://github.com/CorgCode/pt-project/issues).
