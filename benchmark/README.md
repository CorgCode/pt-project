# Методы и бейзлайны

Здесь живут те, кто **строит** эмбеддинги хостов. Оценивает их [`evaluation/`](../evaluation/README.md).

| папка | что внутри |
|---|---|
| `methods/` | графовые методы: DeepWalk, node2vec, RolX, struc2vec, GraphSAGE |
| `baselines/` | простые бейзлайны, которые методы обязаны обогнать |
| `config.json` | общие соглашения: имя колонки хоста, префикс `emb_`, значения k |
| `example_embeddings.csv` | пример файла эмбеддинга (проверяется в CI) |

## Общее правило

Каждый метод: `fit(data)` — учится **только на train-окне** (`data.training_flows()`, тест это проверяет);
`embed(data, окно)` — возвращает `DataFrame` с `host_id` в индексе и колонками `emb_*` для всех хостов когорты.

## Графовые методы (`methods/`)

Написаны на numpy / scipy / scikit-learn, без gensim и torch: CI лёгкий, результаты воспроизводятся точно.

| метод | идея | статья | заметки |
|---|---|---|---|
| `deepwalk` | случайные блуждания по графу + skip-gram | Perozzi et al., KDD 2014 | — |
| `node2vec` | блуждания с перекосом (p, q) между «вглубь» и «вширь» + skip-gram | Grover & Leskovec, KDD 2016 | `node2vec.md` |
| `rolx` | рекурсивные структурные признаки + NMF → роли | Henderson et al., KDD 2012 | `rolx.md` |
| `struc2vec` | граф структурной похожести (DTW по степеням) + блуждания | Ribeiro et al., KDD 2017 | `struc2vec.md` |
| `graphsage` | GraphSAGE без учителя, усреднение соседей; переносится на новые окна | Hamilton et al., NeurIPS 2017 | `graphsage.md` |

**Вход:** взвешенный неориентированный граф одного окна (`graph_input.py`): все хосты (когорта и их собеседники),
вес ребра — `log1p(число потоков)`, петли выброшены.

**Другие окна** (для узнаваемости во времени) — тот же алгоритм с теми же параметрами на графе того окна.
Координаты между окнами **не общие**, поэтому честное сравнение — после выравнивания (Procrustes).
GraphSAGE переносит обученные веса и объявляет общую систему координат.

**Пересечение с эталонами.** Графовые методы используют структуру собеседников — как и эталоны «будущие связи»
и «общие собеседники». Часть совпадения с ними — по построению; каждый метод описывает это в `info()["overlap"]`.

Общие кирпичики: `walks.py` (блуждания, в том числе точные node2vec-переходы), `sgns.py` (skip-gram
с negative sampling или hierarchical softmax), `base.py` (класс `GraphEmbeddingMethod`).

**Скорость** (одно ядро): skip-gram ≈ 150–200 тыс. пар (центр, контекст) в секунду. С параметрами из статей
граф на 20 000 хостов — около 15 минут на окно и эпоху, на 100 000 — около часа. Для первого прогона на реальных
данных берите `num_walks=5`.

### Запуск

```bash
python -m experiments.end_to_end --fixture --workdir runs/demo --methods deepwalk node2vec   # синтетика
python -m experiments.end_to_end … --methods all \
    --method-config '{"deepwalk": {"num_walks": 10, "walk_length": 80, "window": 10}}'
```

`--fast-methods` — крошечные параметры для CI (числа бессмысленны). Неизвестный параметр — ошибка.

### Подбор гиперпараметров — только на validation

```bash
python -m experiments.tune_methods --processed-dir data/processed --reference-dir data/reference \
    --method deepwalk --grid '{"walk_length": [40, 80], "window": [5, 10]}' --metric pair_roc_auc --out runs/tune.json
```

Метрику выбора фиксируйте **до** просмотра результатов, сетку держите маленькой (до 64 точек). `--target test` запрещён.
Лучший результат на validation завышен числом попыток — итог считается один раз на test.

### Добавить метод

Новый модуль в `methods/`: наследник `GraphEmbeddingMethod` с `name`, `default_params`, `reference` и методом
`embed_graph(graph) -> массив (n_nodes, d)`, плюс `METHODS = {"имя": Класс}` — метод найдётся сам. Тесты — рядом.

## Бейзлайны (`baselines/`)

Входят в каждый запуск экспериментов (`temporal.py`):

| имя | что это | общие координаты между окнами |
|---|---|---|
| `random` | случайный шум — нижняя граница | нет |
| `behavioral` | нормированный поведенческий профиль: объём, степень, направление, порты, протоколы, час суток | да |
| `behavioral_pca`, `behavioral_svd` | PCA / SVD этого профиля, обученные на train | да |
| `graph_spectral` | спектральное вложение графа окна | нет |
| `behavioral_portfree`, `_pca` | профиль без признаков по портам (#43), флаг `--port-free-baselines` | да |

Поведенческие бейзлайны похожи по признакам на эталон соседей, поэтому на «Поведении» они сильны отчасти
по построению. Это точки калибровки, а не победители.

Отдельно — калибровка по готовому `features.csv` (`methods.py`, `run_all.py`): `random`, `pca`, `svd` и
`reference_identity` (копия эталона — потолок, «утекает» намеренно). Нужна для CI:

```bash
python -m benchmark.baselines.make_synthetic_reference --output-dir /tmp/ref
python -m benchmark.baselines.run_all --reference-dir /tmp/ref --out-dir /tmp/sub --results-dir /tmp/res --evaluate
```
