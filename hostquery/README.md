# hostquery — запросы к хостам по сохранённым артефактам

Пакет отвечает на вопросы о уже посчитанном прогоне: кто на кого похож, на какие
группы распался набор, что изменилось между двумя окнами и кто выглядит нетипично.
Это слой чтения и объяснения, а не новая модель: он не обучает эмбеддинги, не
пересчитывает метрики и не пишет в артефакты.

Задача — [#157](https://github.com/CorgCode/pt-project/issues/157). Интерфейс лаборатории (#156)
получает те же числа через локальный HTTP API. Оценка качества кластеризации и
аномалий остаётся в #21: здесь нет ни accuracy, ни порогов «это атака».

```
python -m hostquery hosts     --bundle app/testdata/synthetic_run
python -m hostquery similar   --bundle app/testdata/synthetic_run --host Comp100014 -k 10
python -m hostquery groups    --bundle app/testdata/synthetic_run -n 4
python -m hostquery pair      --bundle app/testdata/synthetic_run --a Comp100014 --b Comp100021
python -m hostquery changes   --before окно1.csv --after окно2.csv -k 3
python -m hostquery outliers  --bundle app/testdata/synthetic_run -k 10
python -m hostquery serve     --bundle app/testdata/synthetic_run --port 8847
```

`--json` включает машинный ответ, `--pretty` — отступы. Ошибки в JSON-режиме тоже
JSON, код возврата `2`.

## Источники данных

| Источник | Что это | Пример |
| --- | --- | --- |
| `--bundle` | папка прогона с `run/manifest.json` | `results/lanl_day02`, `app/testdata/synthetic_run` |
| `--embeddings` | CSV с готовыми векторами: `host_id`, `emb_0…` | выгрузка `run/embeddings/<метод>/train.csv` |
| `--embeddings` | CSV с записями обмена: `src`, `dst`, `bytes`, `packets`, `proto`, `dst_port` | выгрузка netflow |

Вид CSV определяется по заголовку, а не по имени файла. Для трафика строятся те же
20 признаков, что в `noomo-ui/lab/core.js`: 8 простых показателей активности и 12
хешированных блоков сервисов (`proto:port mod 6`, исходящие/входящие). Блок — это
не протокол и не роль; в ответах он помечен `hashed_service_bucket`.

Ограничения браузерного примера сохранены и для CSV: до 500 хостов, 1024 координат,
10 000 строк, 3 МБ. Готовый прогон (`--bundle`) сервер читает без этих ограничений.

Готовые векторы не получают придуманных имён координат: `name` равен `null`,
`kind` равен `embedding_coordinate`. Смысл оси эмбеддинга по номеру не восстановить.

Координаты читаются в числовом порядке `emb_0, emb_1, …`, а не в порядке столбцов
файла: `host_id,emb_1,emb_0` даёт те же векторы, что и `host_id,emb_0,emb_1`, и при
сравнении файла с самим собой сходство равно 1. Номера должны идти подряд от `emb_0`.

## Контракт ответа

Версия контракта — `schema: "hostquery/1"`. Конверт одинаков у CLI и HTTP:
сначала запрос, затем происхождение, затем данные, в конце оговорки.

```json
{
  "schema": "hostquery/1",
  "query": {"kind": "similar", "host": "Comp100014", "k": 2, "method": "behavioral"},
  "provenance": {
    "schema": "hostquery/1",
    "source": "bundle",
    "bundle": "app/testdata/synthetic_run",
    "method": "behavioral",
    "synthetic": true,
    "data_kind": "synthetic_fixture",
    "embeddings": "app/testdata/synthetic_run/run/embeddings/behavioral/train.csv",
    "embeddings_sha256": "a289…655c",
    "reference": "app/testdata/synthetic_run/reference/test/neighbors.csv",
    "hosts": 300,
    "dimensions": 56
  },
  "host": "Comp100014",
  "k": 2,
  "neighbors": [
    {"id": "Comp102030", "rank": 1, "similarity": 0.9702, "reference_rank": 6, "in_reference_top_k": false}
  ],
  "warnings": []
}
```

Правила, на которые можно опираться:

- `provenance.embeddings_sha256` — sha256 файла, из которого прочитаны числа
  (для `changes` — первого окна). У `changes` дополнительно есть
  `provenance.compare.before`, `.after` и `.after_sha256`: путь и хеш второго окна.
- `query.kind` — тип запроса: `hosts`, `similar`, `consensus`, `groups`, `group`,
  `pair`, `changes`, `outliers`, `dataset`.
- Группы нумеруются **с нуля**, как метки `core.js`. Интерфейс показывает `group + 1`.
- `similarity` — косинус в `[-1, 1]`, не вероятность и не оценка риска.
- Ничьи решаются детерминированно: у соседей — порядком хостов в наборе (stable
  sort), у изменений и нетипичности — по `id`. Повторный запрос на тех же данных
  даёт тот же ответ.
- `warnings` — список строк; пустой список означает, что оговорок нет, а не что
  данные проверены на безопасность.

### Похожие хосты и консенсус

`similar` исключает сам хост, сортирует по убыванию косинуса и добавляет
`reference_rank`/`in_reference_top_k`, если рядом лежит сохранённый эталон
`reference/test/neighbors.csv`. `--vector "0.1,0.2,…"` ищет по произвольному вектору
той же размерности; переданный вектор хоста сам окажется первым.

`consensus` усредняет ранги одного хоста по нескольким методам:

```
python -m hostquery consensus --bundle runs/lanl --host Comp1 -k 10 \
  --methods behavioral,deepwalk,temporal_encoder --json
```

Ответ содержит `n_methods`, `votes`, `mean_rank`, `rank_spread`, `mean_similarity` и
`warnings`, если метод пропущен из-за отсутствия хоста.

### Группы

```
python -m hostquery groups --bundle app/testdata/synthetic_run -n 4 --json
python -m hostquery group  --bundle app/testdata/synthetic_run -n 4 --group 1 --json
```

`groups` отдаёт `labels` (длина равна числу хостов), `sizes`, `centers`, `projection`
(`x`, `y`, `hostId`) и `profiles` с тремя крайними координатами центроида.
`group` — профиль одной группы: `members`, `size`, `centroid`, `mean_similarity`,
`top_features` и напоминание `note`, что группа — номер кластера, а не роль.

Совпадающие профили могут оставить кластер без хостов: это допустимый вход.
Пустая группа не роняет ответ, а описывается явно — `empty: true`, `size: 0`,
`members: []`, `centroid: null`, `mean_similarity: null`, `top_features: []`.
Метки и `sizes` при этом сохраняют исходную нумерацию, поэтому `group + 1`
в интерфейсе не сдвигается.

Алгоритм портирован один в один из `noomo-ui/lab/core.js`: farthest-point init,
20 итераций, стабильный выбор при равенстве, 18 итераций степенного метода для
2D-проекции. `sklearn.KMeans`/`PCA` дали бы другие метки и другой поворот, поэтому
здесь они не используются.

### Объяснение пары

```
python -m hostquery pair --bundle app/testdata/synthetic_run --a Comp100014 --b Comp100021
```

Ответ: `cosine`, `coordinates` (до 8 координат у эмбеддинга, 8 простых признаков у
трафика), `top_contributions`, а для трафика ещё `shared_contacts` — общие
собеседники. Для готовых векторов `shared_contacts` равно `null`: журнала соединений
в них нет. Объяснение перечисляет совпадения значений, но не утверждает причину
похожести.

### Изменения между окнами

```
python -m hostquery changes --before окно1/ --after окно2/ -k 3 --json
python -m hostquery changes --before netflow_1.csv --after netflow_2.csv -k 3
```

- Группы обучаются на **первом** окне, второе раскладывается по тем же центрам.
- Для трафика оба окна приводятся к одной шкале: `log1p`, затем максимум по
  объединению окон. Иначе рост объёмов выглядел бы как смена профиля.
- `changes` отсортированы по возрастанию `similarity`: сначала самые изменившиеся,
  при равенстве — по `id`.
- `newHosts` и `missingHosts` показывают хосты, которых нет во втором/первом окне.
- `--alignment procrustes` поворачивает второе окно к первому по общим хостам. По
  умолчанию выравнивания нет.
- `--before`/`--after` принимают папку прогона или CSV; формат и размерность окон
  должны совпадать.

Каждый элемент `changes` совместим с `changes.js`: `id`, `similarity`,
`beforeGroup`, `afterGroup`, `added`, `removed`, `beforeContacts`, `afterContacts`, а
также `before` и `after` с объектами хоста (`id`, `v`, `norm`; у трафика ещё
`logValues` и `activity`). Номера групп — с нуля.

### Нетипичность

```
python -m hostquery outliers --bundle app/testdata/synthetic_run -k 10 --json
```

`score = 1 - средний косинус с k ближайшими`; сортировка по убыванию `score`.
Никакого порога и никакого «подозрительного» статуса нет. В `warnings` прямо
сказано, что это не вероятность атаки, а качество разбиения оценивает #21.

## HTTP API

```
python -m hostquery serve --bundle app/testdata/synthetic_run --port 8847 \
  --allow-origin http://127.0.0.1:8845
python -m hostquery serve --bundle окно1/ --compare окно2/ --port 8847 \
  --allow-origin http://127.0.0.1:8845
```

Сервер слушает только `127.0.0.1`, проверяет `Host` и `Origin`, отвечает только на
`GET` (остальные методы — `405`) и ничего не пишет на диск. Ответы — тот же конверт,
что у CLI.

`--allow-origin` — явный список локальных origin лаборатории; флаг можно повторять.
Разрешены только `http://127.0.0.1:<порт>`, `http://localhost:<порт>` и их `https`
варианты. Для разрешённого origin сервер отдаёт `Access-Control-Allow-Origin` и
`Vary: Origin`, включая ответы с ошибками; на preflight `OPTIONS` отвечает `204` с
`Access-Control-Allow-Methods: GET, OPTIONS`. Без флага cross-origin `fetch` из
лаборатории отклоняется: так API не открывается даже другим локальным страницам
по умолчанию.

| Путь | Что отдаёт |
| --- | --- |
| `GET /` | список путей и пометка «только чтение» |
| `GET /api/lab/health` | `status`, `methods`, наличие второго окна |
| `GET /api/lab/meta` | хосты, размерность, методы, эталон |
| `GET /api/lab/dataset?method=NAME` | векторы хостов для отрисовки |
| `GET /api/lab/similar?host=ID&k=10&groups=0` | похожие хосты |
| `GET /api/lab/groups?n=6` | метки, центры, проекция, профили |
| `GET /api/lab/group?n=6&group=0` | профиль группы |
| `GET /api/lab/pair?a=ID&b=ID` | объяснение пары |
| `GET /api/lab/changes?k=3&alignment=none` | изменения (нужен `--compare`) |
| `GET /api/lab/outliers?k=10&groups=0` | exploratory-нетипичность |

Ошибки — `400` с `{"error", "hint"?}`, неизвестный путь — тоже `400` с подсказкой
`GET /`. Заголовки ответа: `Cache-Control: no-store`, `X-Content-Type-Options: nosniff`,
`Content-Security-Policy: default-src 'none'; frame-ancestors 'none'`.

### Подключение к лаборатории #156

Никаких новых зависимостей: только стандартный `fetch` браузера. Конверт можно
проверять по `schema`, а `group` показывать как `group + 1`.

```js
const BASE = "http://127.0.0.1:8847";

async function lab(path) {
  const response = await fetch(BASE + path, { headers: { Accept: "application/json" } });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "hostquery: ошибка запроса");
  if (body.schema !== "hostquery/1") throw new Error("hostquery: неизвестная версия");
  return body;
}

const similar = await lab("/api/lab/similar?host=Comp100014&k=10");
const groups = await lab("/api/lab/groups?n=4");
const changes = await lab("/api/lab/changes?k=3");

// Проекция и метки уже в форме core.js: hostId, x, y, labels.
renderScatter(groups.projection, groups.labels);
renderChanges(changes.changes);           // before/after/logValues/activity
```

Сервер запускается рядом с лабораторией и читает те же артефакты. Origin страницы
лаборатории нужно передать в `--allow-origin`; пример выше рассчитан на страницу
`http://127.0.0.1:8845`. Если прогон обновляется, сервер нужно перезапустить: кэш
намеренно не разделяется между процессами.

## Паритет и тесты

```
python -m pytest -q hostquery
```

Тесты покрывают чтение CSV, порядок координат `emb_*`, косинус и консенсус,
детерминированные группы и пустые кластеры, объяснение пары, общую шкалу и
Procrustes, нетипичность, CLI, CORS и HTTP-контракт.
`hostquery/test_parity.py` сверяет константы с `noomo-ui/lab/core.js`, а при
установленном `node` прогоняет настоящий `core.js` на тех же CSV и сравнивает
проекцию, метки и `compareWindows` с точностью `1e-12`. В CI пакет подключён к
`.github/workflows/benchmark.yml`: там ставится Node.js 20, а при отсутствии
`noomo-ui/lab/core.js` в `main` подтягивается ветка лаборатории #156. Локально без
Node.js этот тест пропускается, остальные проверки паритета остаются. Соседи
дополнительно сверяются с уже существующим `app.results.Bundle.neighborhood`.

## Границы

- Пакет не меняет `noomo-ui/`, `app/operator/`, `ui-prototype/` и не строит графику.
- `pair` и `outliers` объясняют числа, но не называют роли и не заявляют об атаке.
- `changes` — сравнение двух наблюдаемых окон, а не модель алертов.
- Артефакты открываются только для чтения; `provenance` фиксирует sha256 источника.
- Оценка качества групп и аномалий — задача #21, а не этого пакета.
- Пакет не зависит от слияния #156: контракт HTTP описан здесь и проверяется тестами.
