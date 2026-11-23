# Методы DrMogger с 17 ноября

Объединённый код методов, математическое описание и локальная таблица:

- [Таблица с актуальным пересчётом](../../research/DrMogger/nov17/README.md).
- [Устройство методов и используемых основ](../../research/DrMogger/reports/methods_since_2026_11_17.md).
- [Directed Bytes GNN](../directed_bytes_gnn/README.md).
- [Temporal GATv2](../temporal_gatv2/README.md).
- [Codebook Directed GNN](../codebook_directed_gnn/README.md).
- [Temporal Graph Mixer](../temporal_graph_mixer/README.md).

Similarity Graph включает обучение и inference; Directed Bytes GNN использует его замороженную основу. Оригинальный
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

Последняя команда не читает трафик и воспроизводит таблицу из сохранённого
results.json. Links/Pairs и остальные пять метрик остаются доступны в CSV/JSON.

## Пересчёт векторов четырёх новых семейств

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

Direct Identity: код и команды в ../temporal_donor_identity/README.md. Его сохранённая штатная оценка перенесена в общую локальную таблицу.
