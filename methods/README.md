# Методы и бейзлайны

Здесь то, что строит эмбеддинги. Оценивает их [`evaluation/`](../evaluation/README.md).

## Графовые методы (`methods/`)

| метод | идея | статья |
|---|---|---|
| `deepwalk` | случайные блуждания + skip-gram | Perozzi et al., 2014 |
| `node2vec` | блуждания с параметрами p, q | Grover & Leskovec, 2016 |
| `rolx` | структурные роли из признаков графа | Henderson et al., 2012 |
| `struc2vec` | похожесть по структуре окрестности | Ribeiro et al., 2017 |
| `graphsage` | нейросеть по соседям, переносится на новые окна | Hamilton et al., 2017 |

Без gensim и torch, только numpy / scipy / scikit-learn. Заметки — `methods/<метод>.md`.

```bash
python -m experiments.end_to_end --fixture --workdir runs/demo --methods all
```

### Какой граф видят методы

DeepWalk и node2vec умеют три вида графа (`--method-config '{"node2vec": {"graph": "both"}}'`):

| `graph` | вершины и рёбра | когда полезен |
|---|---|---|
| `hosts` (по умолчанию) | хост — хост, если общались | кто с кем общается |
| `services` | хост — сервис (протокол/порт), отдельно «пользуется» и «обслуживает» | роль хоста: какие сервисы он использует |
| `both` | оба вида рёбер | и то и другое |

`weight: hub_norm` ослабляет хабы: в LANL 5 серверов общаются с 70–80% сети, и без этого все блуждания уходят в них.
Готовый набор для LANL — `experiments/lanl/methods_v2.json`.

Сравнивать методы честнее при одинаковой размерности: `--dim 32` для всех (в таблице есть колонка «разм.»).

## Бейзлайны (`baselines/`)

| имя | что это |
|---|---|
| `random` | случайные векторы — ниже нельзя |
| `behavioral` | поведенческий профиль: объём, порты, протоколы, время суток |
| `behavioral_pca`, `behavioral_svd` | то же, сжатое до 16 чисел |
| `graph_spectral` | спектральное разложение графа связей |
| `behavioral_portfree` | профиль без портов (#43), флаг `--port-free-baselines` |

Бейзлайны должны обгонять все методы, иначе метод не нужен.

## Как добавить метод

Файл в `methods/`: класс-наследник `GraphEmbeddingMethod` с методом `embed_graph(graph)` и строкой
`METHODS = {"имя": Класс}`. Метод подхватится сам. Тесты — рядом.

Или проще: посчитать эмбеддинг где угодно и положить в [`scoreboard/`](../scoreboard/README.md).

## Подбор параметров

Только на validation:

```bash
python -m experiments.tune_methods --processed-dir data/processed --reference-dir data/reference \
    --method node2vec --grid '{"p": [0.25, 1, 4], "q": [0.25, 1, 4]}' --metric pair_roc_auc --out runs/tune.json
```
