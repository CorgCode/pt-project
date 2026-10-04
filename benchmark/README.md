# Общий benchmark

Этот каталог задаёт единые правила оценки host embeddings для всей команды.

## Формат submission

Каждый метод должен экспортировать CSV вида:

```csv
host_id,emb_0,emb_1,emb_2
C1,0.12,-0.81,0.44
C2,0.09,-0.77,0.39
```

Правила:
- первый обязательный столбец: `host_id`;
- все embedding-координаты начинаются с `emb_`;
- `host_id` уникален;
- пропуски и NaN запрещены;
- порядок строк не важен: сопоставление идёт по `host_id`.

## Единые параметры

Параметры лежат в `benchmark/config.json`.

Сейчас зафиксированы:
- `k = 5, 10, 20`;
- cosine distance для embedding-space;
- одинаковый список основных метрик для всех методов.

## Основные метрики

Все методы в итоговом сравнении должны оцениваться одним набором:

- Triplet Accuracy;
- Pairwise ROC-AUC;
- Pairwise PR-AUC;
- Precision@k / Recall@k;
- k-NN Consistency;
- Linear / Logistic Probe.

Silhouette и Davies–Bouldin считаются вспомогательными: они оценивают геометрию кластеров, но не заменяют проверку правильности разделения хостов.

## Workflow

1. Метод строит embeddings.
2. Экспортирует их в общий CSV-формат.
3. `benchmark/validate_submission.py` проверяет файл.
4. После объединения реализаций метрик общий evaluator считает все метрики на одинаковых reference data / splits.
5. Результаты сводятся в одну таблицу.

## Почему не Kaggle сейчас

GitHub остаётся источником истины для кода и результатов. Это бесплатно, прозрачно и удобно для PR workflow.

Kaggle/Codabench или GitHub Pages можно добавить позже как внешний leaderboard, не меняя формат submission и evaluator.
