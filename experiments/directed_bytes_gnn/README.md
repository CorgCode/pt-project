# Directed Bytes GNN

Замороженная основа Similarity Graph Recent на512 координат дополняется64
координатами обучаемой направленной GNN: [base;0,5×gnn], всего576. Два слоя
раздельно обрабатывают out/in counts и out/in bytes, есть собственный skip.
Обучение: InfoNCE между train-окнами и прогноз узлового профиля на1/2 дня.

Train — по часу02,10,11,12; development13; benchmark14–16. Реальные TX/RX
байты учитываются для обоих концов потока. Собственные ID служат для группировки
и положительных пар, а не входом нейросети.

Актуальные результаты: [локальная таблица](../../research/DrMogger/nov17/README.md).
Математика вместе с устройством основы:
[общий отчёт](../../research/DrMogger/reports/methods_since_2026_11_17.md).

## Повторение

Нужны torch и обученная основа в runs/similarity_graph_recent. Полный код её обучения
и inference включён в experiments/similarity_graph. Выбор основы фиксируется до benchmark.

~~~sh
python -m experiments.directed_bytes_gnn.prepare_windows --part development --output runs/my_gnn/windows
python -m experiments.directed_bytes_gnn.train --windows runs/my_gnn/windows --output runs/my_gnn/training --steps 80 --seed 17
python -m experiments.similarity_graph.infer \
  --checkpoint runs/similarity_graph_recent/refine01/model.joblib \
  --source runs/similarity_graph_recent/refine01/source.csv \
  --flows PT_PROJECT_ML/fingerprint_recent/day13/flows/test.csv \
  --start 1081800 --end 1085400 --day 13 \
  --cache runs/my_gnn/cache --output runs/my_gnn/baseline_day13.csv
python -m experiments.directed_bytes_gnn.evaluate_dev --run runs/my_gnn
# После фиксации development:
python -m experiments.directed_bytes_gnn.prepare_windows --part benchmark --output runs/my_gnn/windows
python -m experiments.directed_bytes_gnn.benchmark --run runs/my_gnn
~~~

В исходном recipe некоторые пути в runner заданы относительно корня проекта:
PT_PROJECT_ML и runs/similarity_graph_recent. Их нужно сохранить либо изменить
в конфигурации своей подготовки. Benchmark runner проверяет SHA зафиксированных
base-векторов; новый fit основы требует собственного manifest, а не отключения
проверки. Локальные имена test-файлов в данных не означают, что они допустимы
для обучения: допустимость определяется day/cutoff.
