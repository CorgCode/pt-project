# Подготовка данных

Сырые логи LANL → окна train / validation / test → когорта хостов → графы → эталоны для оценки.

## Запуск

```bash
python -m pipeline.prepare describe --netflow data/raw/netflow_day-02.bz2          # диапазон времени
python -m pipeline.prepare run --netflow data/raw/netflow_day-02.bz2 \
    --config cfg.json --output-dir data/processed --data-kind lanl --source-note "откуда, когда"
python -m pipeline.reference --processed-dir data/processed --output-dir data/reference
```

Пример конфига — `example_config.json`, конфиг для LANL — `experiments/lanl/config_day02.json`.

## Входные данные

`netflow_day-NN.bz2` из [LANL 2017](https://csr.lanl.gov/data/2017/): CSV без заголовка, 11 полей
`Time, Duration, SrcDevice, DstDevice, Protocol, SrcPort, DstPort, SrcPackets, DstPackets, SrcBytes, DstBytes`.
Битые строки выбрасываются и считаются в `manifest.json`.

## Что получается

- `data/processed/` — потоки по окнам, `cohort.csv`, графы, `manifest.json`.
- `data/reference/{validation,test}/` — эталоны (что в них — [`evaluation/README.md`](../evaluation/README.md)).

Методы читают только train:

```python
from pipeline.data import ProcessedData
d = ProcessedData("data/processed")
flows, hosts = d.training_flows(), d.cohort
```

## Правила

- Окна не пересекаются. Поток на границе окна выбрасывается.
- Когорта — хосты, активные в train (`cohort_policy: train_only`). Её выбирают до моделей.
- Эталоны строятся по более позднему окну и только из логов.
- Каждый результат помечен: `synthetic_fixture`, `unverified` или `lanl_declared`. Синтетика никогда не выдаётся за LANL.

## Синтетика

`python -m pipeline.fixtures --output-dir data/fixture` — файлы ровно в формате LANL для тестов и демо.

Тесты: `python -m pytest -q pipeline`.
