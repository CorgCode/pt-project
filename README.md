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
| проверить свой метод | [3 шага в таблице](scoreboard/README.md#как-добавить-свой-метод) |
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
| [`docs/`](docs/README.md) | сайт и [методология](docs/METHODOLOGY.md) |
| [`research/`](research/README.md) | ресёрч участников |

## Результаты на LANL

Первый прогон — один день, 2 242 активных хоста ([таблица](scoreboard/README.md), [разбор](docs/METHODOLOGY.md)):

- общий балл (0 — случайно, 100 — идеально; [как считается](scoreboard/README.md#как-считается-балл)) у лучшего — **45.3**:
  поведенческий профиль + кто собеседники + какие сервисы (TF-IDF → SVD, [`scoreboard/hybrid_tfidf.py`](scoreboard/hybrid_tfidf.py));
  чистый поведенческий профиль — 40, графовые методы — 12–24: они почти не узнают хосты;
- **связи**: AUC 0.99 — выше правила «пара уже общалась» (0.94) и случайного уровня (0.64–0.76);
- **новые собеседники через похожих хостов**: в ~3.4 раза лучше прогноза без эмбеддинга;
- **узнать тот же хост через полчаса**: до 52% с первой попытки (случайно — 0.1%);
- **соседи по поведению** лучше всего у чистого поведенческого профиля (0.27).

Дальше: подбор параметров и больше дней (#60), кластеры и аномалии (#21).

## Команда

Даниил @physcorgi · Дима @DrMogger · Тихон @Tikhon2783 · Даня @kalivixZ · Даша @ElliOrNora

Как работаем — [`CONTRIBUTING.md`](CONTRIBUTING.md). Задачи — в [Issues](https://github.com/CorgCode/pt-project/issues).
