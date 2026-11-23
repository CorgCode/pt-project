# Локальная таблица объединённых методов

Все пять метрик обязательны. Direct Identity перенесён из сохранённой штатной оценки его ветки; остальные строки пересчитаны из CSV с новым кэшем. Это исследовательская таблица, не публикация в общем рейтинге. Окна ранее изучались; закрытого holdout нет.

## Benchmark дней 14–16

| Метод | D | День 14 | День 15 | День 16 | Средний score |
|---|---:|---:|---:|---:|---:|
| Similarity Graph Recent · контроль | 512 | 62.8725 | 62.7577 | 65.3520 | 63.6608 |
| Directed Bytes GNN | 576 | 64.1753 | 63.4198 | 66.8273 | 64.8075 |
| Temporal Graph Mixer | 512 | 59.9336 | 60.5175 | 64.7680 | 61.7397 |
| Direct Identity Codebook | 896 | 63.3796 | 63.3088 | 68.3425 | 65.0103 |

Среднее вычислено из трёх оконных score. У Mixer выбран no_graph_global_w2 по development. История: Similarity Graph и Directed Bytes — по часу 02,10,11,12; Mixer — по часу 02–12.

## Development дня 13

Пилоты GATv2 и Codebook Directed GNN на benchmark не оценивались. Строки GATv2 с 13 шагами нужны для сравнения абляций при одинаковом бюджете. У надстроек no_graph/no_bytes изменяют только новую ветку; основа фиксирована.

| Метод | D | Score | Neighbors@10 | New MRR | Identity Top-1 |
|---|---:|---:|---:|---:|---:|
| GATv2 · 13 шагов, контроль абляций | 512 | 35.3219 | 0.14001 | 0.10690 | 0.16515 |
| GATv2 · 30 шагов | 512 | 35.2263 | 0.13973 | 0.10552 | 0.16743 |
| GATv2 · без графа, 13 шагов | 512 | 34.5287 | 0.14570 | 0.10857 | 0.14579 |
| GATv2 · без байтов, 13 шагов | 512 | 38.5852 | 0.15054 | 0.11179 | 0.20615 |
| GATv2 · без donor-loss, 13 шагов | 512 | 35.1931 | 0.13933 | 0.10686 | 0.16401 |
| Codebook · нулевая поправка | 896 | 61.5230 | 0.27473 | 0.16036 | 0.59567 |
| Codebook Directed GNN · Identity | 896 | 61.4935 | 0.27473 | 0.16036 | 0.59453 |
| Codebook · Identity без графа | 896 | 61.5525 | 0.27473 | 0.16036 | 0.59681 |
| Codebook · Identity без TX/RX-различия | 896 | 61.4048 | 0.27473 | 0.16036 | 0.59112 |
| Codebook Directed GNN · donor-поправка | 896 | 60.2208 | 0.27752 | 0.14760 | 0.58884 |
| Similarity Graph Recent · контроль | 512 | 62.0684 | 0.24963 | 0.15654 | 0.58428 |
| Directed Bytes GNN · вес 0,5 | 576 | 63.3867 | 0.25868 | 0.16475 | 0.58542 |
| Directed Bytes · без графа | 576 | 63.1672 | 0.25777 | 0.15931 | 0.58428 |
| Directed Bytes · без байтов | 576 | 63.1448 | 0.25680 | 0.15955 | 0.58428 |
| Temporal Graph Mixer | 512 | 58.1533 | 0.25378 | 0.13109 | 0.51708 |

Все пять сырых метрик и оконные результаты: [CSV](scoreboard.csv), [JSON с SHA входов](results.json).

## Код и описания

- [Directed Bytes GNN](../../../experiments/directed_bytes_gnn/README.md)
- [Temporal GATv2](../../../experiments/temporal_gatv2/README.md)
- [Codebook Directed GNN](../../../experiments/codebook_directed_gnn/README.md)
- [Temporal Graph Mixer](../../../experiments/temporal_graph_mixer/README.md)
- [Математическое устройство, включая основы](../reports/methods_since_2026_11_17.md)

В таблице используется штатное чтение экспортированных CSV. Веса, потоки и embedding CSV в Git не включены.

Повтор генерации без данных: python -m experiments.methods_nov17.scoreboard --output runs/nov17_table.
Полный пересчёт из локальных векторов: команды в [experiments/methods_nov17/README.md](../../../experiments/methods_nov17/README.md).
