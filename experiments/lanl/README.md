# Прогон на LANL

Как повторить прогон на настоящих данных (#60). Данные в git не кладём, только итоги.

## 1. Скачать

1. Заполнить форму на https://csr.lanl.gov/data/2017/ — выдадут токен в ссылке `data-fence/<токен>/`.
2. Скачать день (≈ 1–2 ГБ, 10–20 минут):

   ```bash
   LANL_TOKEN=<токен> experiments/lanl/fetch.sh 02 /путь/raw
   ```

## 2. Подготовить

```bash
W=/путь
python -m pipeline.prepare run --netflow $W/raw/netflow_day-02.bz2 --config experiments/lanl/config_day02.json \
    --output-dir $W/lanl/processed --data-kind lanl --source-note "форма LANL, дата, кто"
python -m pipeline.reference --processed-dir $W/lanl/processed --output-dir $W/lanl/reference --pairs-per-anchor 10
```

Окна: train — 1 час, validation и test — по 30 минут. Когорта — хосты с ≥ 300 событиями в train (≈ 6 450).

## 3. Обучить и оценить

```bash
PYTHONUNBUFFERED=1 python -m experiments.run_experiment --processed-dir $W/lanl/processed \
    --reference-dir $W/lanl/reference --out-dir $W/lanl/run --methods all \
    --method-config '{"deepwalk": {"num_walks": 5}, "node2vec": {"num_walks": 5}}' > $W/logs/experiment.log 2>&1
```

Смотреть прогресс: `LANL_WORKDIR=$W experiments/lanl/status.sh --watch`.

### Улучшенный прогон (v2)

Что меняется и почему:

- **граф «хост — сервис»** (`graph: both`): роль хоста определяют сервисы (порты), а не только собеседники;
  в примере, где роли различаются лишь портами, обычный граф ролей не видит (0.00), новый — видит (+0.55);
- **ослабление хабов** (`weight: hub_norm`): 5 серверов общаются с 70–80% сети, и блуждания уходили в них;
- **10 блужданий на вершину** вместо 5 и **одинаковая размерность 32** у всех методов — честное сравнение;
- **node2vec: p = 0.5, q = 2** — подобраны на validation по «новым собеседникам» (`results/lanl_day02/tune_node2vec.json`).

```bash
PYTHONUNBUFFERED=1 python -m experiments.run_experiment --processed-dir $W/lanl/processed \
    --reference-dir $W/lanl/reference --out-dir $W/lanl/run_v2 --methods deepwalk node2vec --dim 32 \
    --method-config experiments/lanl/methods_v2.json --reuse-embeddings > $W/logs/experiment_v2.log 2>&1
```

Около 2 часов (DeepWalk 66 мин, node2vec 41 мин). Потом — `combine.py` для гибридов.

Итог (в таблице — `*_services`):

| метод | общий балл: прежний граф → «хост — сервис» |
|---|---|
| node2vec | 32.6 → **33.4**: соседи 0.073 → 0.088, связь 0.78 → 0.82, новые связи 0.56 → 0.69 |
| DeepWalk | 33.2 → 29.1: соседи и новые связи лучше, но пары 0.98 → 0.91 и новые собеседники хуже |
| гибрид поведение×4 + node2vec (сервисы) + спектр | 48.9 → **49.6** — лучший в таблице |

Сервисы помогают узнавать роль и новые связи, но размывают «кто с кем общается» — поэтому пары и
новые собеседники падают. В гибриде это закрывает спектральный граф.

## 4. Перенести итоги в репозиторий

```bash
python experiments/lanl/publish.py $W/lanl results/lanl_day02
```

## Ресурсы (ноутбук, 8 ГБ)

| шаг | время |
|---|---|
| скачивание дня | 10–20 мин |
| подготовка | 7 мин |
| бейзлайны | 1.5 мин |
| DeepWalk / node2vec / GraphSAGE / RolX | 14 / 9 / 6.5 / 1 мин |
| struc2vec | дольше всех |

- Узкое место — память: окно в 1 час ≈ 3 ГБ. Не запускайте рядом ничего тяжёлого.
- Больше ~10 тыс. хостов в когорте не влезает в 8 ГБ.
- Для всех дней нужна машина с 32+ ГБ.
