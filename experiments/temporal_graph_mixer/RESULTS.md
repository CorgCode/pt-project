# Temporal Graph Mixer

Актуальный полный пересчёт: **61,7397** в среднем по benchmark14–16.
Development13: **58,1533**. Выбран no_graph_global_w2, веса Own/Stable/Dynamic1/1/1, historical source.

| День | Score | Neighbors@10 | Links AUC | New partners MRR | Pairs AUC | Identity Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| 14 | 59,9336 | 0,264896 | 0,960737 | 0,138138 | 0,997654 | 0,616456 |
| 15 | 60,5175 | 0,234960 | 0,963565 | 0,151646 | 0,998598 | 0,534620 |
| 16 | 64,7680 | 0,103088 | 0,982933 | 0,353810 | 0,998229 | 0,429727 |

Train:11часов, по одному на каждый день02–12; source02 фиксирован для всех
окон, development13 только для выбора. Окна14–16 ранее изучались: независимый
закрытый holdout не заявляется. Размерность512, source-когорта6453, seed17.
Цели70/90 не достигнуты; преимущество графового входа не установлено.

[Архитектура и команды](README.md).
[Математика](../../research/DrMogger/reports/methods_since_2026_11_17.md).
[Общая локальная таблица и все пять метрик](../../research/DrMogger/nov17/README.md).

Переносимая модель: runs/temporal_graph_mixer_01/v2/bundle и selected_model.tar.gz.
Checkpoint не содержит обучающие class-proxy. Повторный inference из bundle
точно воспроизвёл оценённые координаты6453×512; source одинаков для всех трёх
benchmark-окон. В Git нет весов, flows, feature-cache или embedding CSV.
