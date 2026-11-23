# Similarity Graph Recent как зависимость

Здесь только загрузка и inference замороженной512-мерной основы Directed Bytes GNN:
model, matcher, извлечение признаков и кэш с SHA содержимого flows. Обучающие
скрипты прежнего исследования не дублируются: они есть в
[PR162](https://github.com/CorgCode/pt-project/pull/162) и
[коммите d5430f5](https://github.com/CorgCode/pt-project/tree/d5430f5/experiments/similarity_graph).

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
