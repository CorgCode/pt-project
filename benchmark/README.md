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

## Единые reference data

Все методы должны оцениваться на одинаковых reference artifacts и splits. Форматы зафиксированы в `benchmark/REFERENCE_DATA.md`:

- `reference/features.csv` — независимые reference features для neighborhood metrics;
- `reference/neighbors.csv` — фиксированные reference-соседи по логам (предпочтительнее пересчёта для разных методов);
- `reference/pairs.csv` — positive/negative пары для ROC-AUC / PR-AUC и retrieval;
- `reference/triplets.csv` — `(anchor, positive, negative)` для Triplet Accuracy;
- `reference/labels.csv` — targets/splits для Linear / Logistic Probe.

Нельзя генерировать отдельную разметку или новый random split внутри каждого метода: иначе результаты разных embeddings не будут сопоставимы.

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

## Запуск

Сначала проверить submission:

```bash
python benchmark/validate_submission.py submissions/METHOD/embeddings.csv
```

Затем запустить общий evaluator:

```bash
python -m evaluation.evaluate \
  --embeddings submissions/METHOD/embeddings.csv \
  --reference-dir reference \
  --method METHOD \
  --output benchmark/results/METHOD.json
```

Сейчас общий CLI уже считает интегрированную k-NN consistency / Jaccard / nDCG оценку. Реализации из issues #22–#25 подключаются к этому же CLI по мере merge; отсутствующие метрики явно отмечаются как unavailable, а не подменяются фиктивными значениями.

Итоговый leaderboard строится одной командой:

```bash
python benchmark/build_leaderboard.py
```

Она собирает `benchmark/results/*.json` в `benchmark/leaderboard.csv`.

## Workflow

1. Метод строит embeddings.
2. Экспортирует их в общий CSV-формат.
3. `benchmark/validate_submission.py` проверяет файл.
4. Общий evaluator использует один и тот же набор reference data / splits из `benchmark/REFERENCE_DATA.md`.
5. Все доступные согласованные метрики считаются без специальных изменений под конкретный embedding-метод.
6. Результаты записываются в JSON и сводятся в один leaderboard.
7. GitHub Actions проверяет submission format и запускает evaluation tests.

## Почему не Kaggle сейчас

GitHub остаётся источником истины для кода и результатов. Это бесплатно, прозрачно и удобно для PR workflow.

Kaggle/Codabench или GitHub Pages можно добавить позже как внешний leaderboard, не меняя формат submission и evaluator.


## Temporal holdout и честная интерпретация

Для метрики k-NN consistency теперь можно построить reference из более позднего
временного окна, не используя тестовые сетевые события для построения embedding.
Протокол и команды: [Temporal reference](../evaluation/TEMPORAL_REFERENCE.md).

**Важно:** это оценка сохранения поведенческой похожести во времени, а не
истинная разметка функциональных ролей. Скрипт проверяет непересечение
предоставленных временных окон, но не может доказать, какие данные использовала
модель при обучении. Источники входных файлов должны быть задокументированы.

Evaluator добавляет к результату SHA-256 reference/features.csv и информацию
о протоколе (если есть manifest.json). Leaderboard не должен сравнивать метрики
по разным reference-файлам или наборам хостов: теперь такие смешанные результаты
приводят к ошибке вместо вводящей в заблуждение общей таблицы.

## Контракт задач #40–#45

См. [BENCHMARK_CONTRACT.md](../BENCHMARK_CONTRACT.md): единый `host_id,emb_*`,
manifest временных окон, `validate_benchmark.py`. Это предотвращает смешение
несовместимых старого и нового форматов, встречающихся в описаниях Issues.
