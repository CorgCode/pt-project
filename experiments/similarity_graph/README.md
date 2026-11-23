# Similarity Graph Recent как зависимость

Здесь полный код обучения матрицы сходства, сжатия в векторы и сопоставления позднего графа.

Обучение: `python -m experiments.similarity_graph.run --config config.json --output runs/sg_new`.
Пути и окна задаются по example.json; выбор на development фиксируется до benchmark.

Для нового метода нужны model.joblib и source.csv из
runs/similarity_graph_recent/refine01. Это зафиксированная история02,10,11,12
с выбором на development13; математическое устройство объяснено в
[описании методов](../../research/DrMogger/reports/methods_since_2026_11_17.md).

~~~sh
python -m experiments.similarity_graph.infer \
  --checkpoint BASE/model.joblib --source BASE/source.csv \
  --flows late.csv --start START --end END --day DAY \
  --cache runs/sg_cache --output runs/sg_late.csv
~~~

Inference проверяет source-координаты и временной cutoff, не вызывает fit.
Поздний собственный ID не подставляется вместо поиска по профилям.
