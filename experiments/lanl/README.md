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

## 4. Перенести итоги в репозиторий

```bash
python experiments/lanl/publish.py $W/lanl/run results/lanl_day02
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
