# Данные: LANL → окна → граф → эталоны

Сырые логи → нормализованные потоки → окна train / validation / test → фиксированная когорта хостов →
граф на каждое окно → замороженные эталоны для оценки. Каждый шаг — отдельная команда с тестами.
Про файлы LANL ничего не додумываем: используются только задокументированные поля.

## Источник: `netflow_day-NN.bz2`

LANL Unified Host and Network Dataset (2017), <https://csr.lanl.gov/data/2017/> (скачивание через форму).
bzip2, CSV **без заголовка**, запись на строку, поля по порядку:

| # | поле | смысл (из описания источника) |
|---|---|---|
| 1 | `Time` | время начала события, «epoch time format» |
| 2 | `Duration` | длительность, с |
| 3 | `SrcDevice` | устройство, которое, вероятно, начало соединение |
| 4 | `DstDevice` | принимающее устройство |
| 5 | `Protocol` | номер протокола |
| 6 | `SrcPort` | порт на стороне SrcDevice |
| 7 | `DstPort` | порт на стороне DstDevice |
| 8 | `SrcPackets` | пакетов от SrcDevice |
| 9 | `DstPackets` | пакетов от DstDevice |
| 10 | `SrcBytes` | байт от SrcDevice |
| 11 | `DstBytes` | байт от DstDevice |

Устройства обезличены (`Comp132598`, иногда имена вроде `Mail`), редкие порты тоже (`Port12597`).
**Единицы и начало отсчёта `Time` источник не указывает**, поэтому в даты мы его не переводим: окна задаются
в тех же единицах. Перед выбором окон посмотрите реальный диапазон командой `describe`.
Логи событий Windows (`wls_day-NN.bz2`) не поддерживаются — работаем только с NetFlow.

**Нормализованный поток:** `src, dst, dst_port, proto, bytes, packets, ts` (+ `src_port, duration, resp_bytes, resp_packets`):
`src=SrcDevice, dst=DstDevice, dst_port=DstPort, proto=Protocol, bytes=SrcBytes, packets=SrcPackets, ts=Time`.
Направление — от инициатора к получателю; поведенческие признаки считаются только по стороне инициатора.
`host_id` — строка устройства как есть (регистр важен, `007` и `7` — разные хосты).

Битые строки (пустое устройство, нечисловые или отрицательные значения, пустой протокол или порт) выбрасываются
и **считаются по причинам** в `manifest.json`. Неверное число полей — ошибка, с `--lenient` — просто считается.

## Запуск

```bash
# 1. посмотреть реальный диапазон Time (файлов не пишет)
python -m pipeline.prepare describe --netflow data/raw/netflow_day-0*.bz2

# 2. объявить окна ДО просмотра любых результатов (пример — pipeline/example_config.json)
python -m pipeline.prepare run --netflow data/raw/netflow_day-0*.bz2 \
    --config my_config.json --output-dir data/processed \
    --data-kind lanl --source-note "форма LANL, <дата>, <кто>"   # без этих флагов данные помечаются 'unverified'

# 3. заморозить эталоны для оценки
python -m pipeline.reference --processed-dir data/processed --output-dir data/reference
```

Шаг 2 пишет в `data/processed`: `flows/{train,validation,test}.csv`, `cohort.csv` (активность хостов по окнам),
`graph/edges_<окно>.csv`, `manifest.json` (схема, конфиг, причины отбраковки, SHA-256 входных файлов, проверки утечек).

Шаг 3 пишет в `data/reference/{validation,test}`: `features.csv`, `neighbors.csv`, `edges.csv`, `partners.csv`,
`pairs.csv`, `triplets.csv`, `eval_hosts.csv`, `manifest.json` (+ `labels.csv`, только если переданы внешние метки
через `--labels F --labels-source "…"`). Что в каждом файле — [`evaluation/README.md`](../evaluation/README.md).

Модели читают только train-окно:

```python
from pipeline.data import ProcessedData
d = ProcessedData("data/processed")
flows, graph, hosts = d.training_flows(), d.training_graph(), d.cohort
```

## Происхождение данных

LANL не публикует контрольных сумм, поэтому происхождение файлов **никогда не проверяется**. Каждый результат
несёт одну из трёх пометок (`pipeline/provenance.py`):

| `data_kind` | смысл |
|---|---|
| `synthetic_fixture` | синтетика из `pipeline.fixtures` (есть файл-маркер или узнаваемые ID генератора) |
| `unverified` | **по умолчанию** для любых других файлов: про их происхождение ничего не утверждаем |
| `lanl_declared` | оператор заявил `--data-kind lanl --source-note "…"`; имена файлов должны быть `netflow_day-NN.bz2` |

`provenance_verified` всегда `false`. Противоречивые заявления (синтетика с `--data-kind lanl`, LANL без
`--source-note`) отклоняются. Отчёты и дашборд всегда показывают пометку; синтетика никогда не выдаётся за LANL.

## Когорта: каких хостов оцениваем

Решается в `prepare`, до появления моделей:

| `cohort_policy` | когорта | кого оцениваем в позднем окне |
|---|---|---|
| `train_only` (**по умолчанию**) | хосты, активные в train | те из них, кто активен и в этом окне; покрытие пишется в отчёт |
| `all_windows` | хосты, активные во **всех** окнах | всех |

`all_windows` выбирает хостов по их будущей активности: результаты завышают стабильность «выживших» хостов.
Оставлен только для сравнения.

## Время и защита от утечки

| окно | для чего |
|---|---|
| `train` | единственное окно, на котором учатся модели |
| `validation` | подбор параметров; эталоны в `reference/validation` |
| `test` | только итоговые числа; эталоны в `reference/test` |

- Окна — полуинтервалы `[start, end)`, строго по порядку, без пересечений (проверяется).
- Поток относится к окну по времени начала и **выбрасывается**, если ещё идёт на границе (`Time + Duration >= end`).
- Одна когорта на все методы, зафиксированная до моделей.
- Эталоны строятся по более позднему окну и только из логов.
- Разные эталоны — разные вопросы, их не смешиваем. Пары и тройки размечаются только в явных случаях;
  всё между — не размечено и **не** считается «непохожим».
- Метки ролей не генерируются. Линейный зонд недоступен без настоящих внешних меток.

## Синтетика (тесты, CI, демо)

`python -m pipeline.fixtures --output-dir data/fixture` пишет файлы **ровно в формате LANL** и маркер
`SYNTHETIC_FIXTURE`; всё, что из них построено, помечено `synthetic: true`. Результаты на синтетике ничего не говорят о LANL.

## Что нужно для настоящего эксперимента

1. Скачать файлы `netflow_day-NN.bz2` через форму LANL (один файл — один день; для начала хватит нескольких дней).
   Ссылки из формы одноразовые.
2. `describe`, выбрать окна train / validation / test в единицах `Time`, выполнить шаги 2–3.
3. Прогнать методы: `python -m experiments.end_to_end … --methods all` (см. [`experiments/README.md`](../experiments/README.md)).

Тесты: `python -m pytest -q pipeline`. Первый набросок «события → граф → эталон» был в PR #37 (Даня);
этот пайплайн заменил его однооконный прототип.
