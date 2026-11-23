# Методы DrMogger с 17 ноября

Код четырёх семейств, математическое описание и локальная таблица:

- [Таблица с актуальным пересчётом](../../research/DrMogger/nov17/README.md).
- [Устройство методов и используемых основ](../../research/DrMogger/reports/methods_since_2026_11_17.md).
- [Directed Bytes GNN](../directed_bytes_gnn/README.md).
- [Temporal GATv2](../temporal_gatv2/README.md).
- [Codebook Directed GNN](../codebook_directed_gnn/README.md).
- [Temporal Graph Mixer](../temporal_graph_mixer/README.md).

Из Similarity Graph включён только необходимый inference замороженной основы Directed Bytes GNN. Оригинальный
multiday_codebook загружается из локального checkpoint с проверкой SHA; код его
загрузки и получения архивных модулей описан в Codebook Directed GNN.

Данные, веса и embedding CSV находятся в локальных runs/PT_PROJECT_ML. В Git
включены код, описания, малый JSON пересчёта и таблицы. Development13 отделён
от benchmark14–16; никакого официального места этим таблицам не присваивается.
Сохранён только актуальный пересчёт. GATv2 на13 шагах нужен как matched-контроль
абляций, а не как отдельная историческая версия.

## Проверка

К стандартным зависимостям нужен PyTorch для нейросетевых методов. Для загрузки
оригинального codebook дополнительно нужны psutil, pyarrow и cloudpickle.
Без torch нейросетевые unit-тесты пропускаются; оценка сохранённых CSV работает.

~~~sh
python -m pip install -r requirements.txt torch joblib
python -m pytest -q evaluation methods app pipeline experiments
python -m experiments.methods_nov17.scoreboard --output runs/nov17_table
~~~

Последняя команда не читает трафик и воспроизводит таблицу из опубликованного
results.json. Links/Pairs и остальные пять метрик остаются доступны в CSV/JSON.

## Полный пересчёт локальных векторов

Из корня checkout этой ветки; укажите существующие локальные каталоги:

~~~sh
python -m experiments.methods_nov17.rescore \
  --runs /path/to/runs --data /path/to/PT_PROJECT_ML \
  --output /path/to/runs/nov17_fresh
python -m experiments.methods_nov17.scoreboard \
  --results /path/to/runs/nov17_fresh/results.json \
  --output /path/to/runs/nov17_table
~~~

Выходная папка пересчёта должна быть новой. После прерывания разрешён --resume:
результат принимается лишь при совпадении SHA source/late, эталонов и fit-edges.
Расчёт использует штатные функции проекта; Links C=1 ограничивает RAM через
memmap. Эквивалентность штатному оценщику проверяется отдельными тестами.

## Защита кэшей

По замечанию physcorgi в PR175 добавлены проверки:

- SHA содержимого flows, даже при неизменных размере и mtime;
- порядок хостов и fitted-словарь в ключах;
- новый vectorizer/profiles при повторном fit matcher;
- повторное построение индексов Episode при смене порядка партнёров;
- запрет повторного использования12-part графа как6-part;
- порядок host_id и полные source-значения в кэше метрик.

Legacy-основа загружается с очищенными оконными кэшами. Это не исправление
чужого PR175; здесь проверены наши пути обучения, inference и оценки.
