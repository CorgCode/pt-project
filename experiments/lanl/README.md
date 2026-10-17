# Прогон на настоящих данных LANL

Как повторить первый прогон на LANL (#60): от скачивания до таблицы результатов. Данные и эмбеддинги в git
не попадают — только итоги (`results/lanl_day02/`).

## 1. Доступ и скачивание

1. На https://csr.lanl.gov/data/2017/ заполните форму (email и цель использования) — выдадут ссылку вида
   `https://csr.lanl.gov/data-fence/<токен>/...`. Данные под лицензией CC0; в публикациях просят ссылаться на
   *Turcotte, Kent, Hash. Unified Host and Network Data Set, 2018*.
2. Скачайте нужный день. Сервер отдаёт ~0.3 МБ/с на соединение, поэтому скрипт качает параллельными кусками
   (12 соединений ≈ 1.7 МБ/с, день ≈ 10–20 минут) и проверяет архив:

   ```bash
   LANL_TOKEN=<токен> experiments/lanl/fetch.sh 02 /путь/к/работе/raw
   ```

   Один день NetFlow — 1–2 ГБ в bz2 (день 02 — 1.08 ГБ, 116 млн записей, Time 118 781 → 172 799, 31 142 хоста).

## 2. Подготовка и эталоны

```bash
W=/путь/к/работе
python -m pipeline.prepare run --netflow $W/raw/netflow_day-02.bz2 --config experiments/lanl/config_day02.json \
    --output-dir $W/lanl/processed --data-kind lanl --source-note "csr.lanl.gov/data/2017, <дата>, <кто>"
python -m pipeline.reference --processed-dir $W/lanl/processed --output-dir $W/lanl/reference --pairs-per-anchor 10
```

`config_day02.json`: train — 1 час (Time 126 000–129 600), validation и test — по 30 минут сразу после;
в когорту попадают хосты с ≥ 300 событиями в train (≈ 6 450 активных хостов). Порог применяется и к поздним
окнам, а они вдвое короче — поэтому оценивается 35–43% когорты; это видно в отчёте.

## 3. Обучение и оценка

```bash
PYTHONUNBUFFERED=1 python -m experiments.run_experiment --processed-dir $W/lanl/processed \
    --reference-dir $W/lanl/reference --out-dir $W/lanl/run --methods all \
    --method-config '{"deepwalk": {"num_walks": 5}, "node2vec": {"num_walks": 5}}' > $W/logs/experiment.log 2>&1
```

Прогресс: `LANL_WORKDIR=$W experiments/lanl/status.sh --watch` — какой метод учится, сколько памяти,
последние строки лога.

Время на ноутбуке (8 ГБ ОЗУ, одно ядро на метод): бейзлайны — 1.5 мин, DeepWalk — 14 мин, GraphSAGE — 6.5 мин,
node2vec — 9 мин, RolX — 1 мин; struc2vec и оценка с аудитом — дольше всего.

## 4. Итоги в репозиторий

```bash
python experiments/lanl/publish.py $W/lanl/run results/lanl_day02
```

## Что мы узнали про ресурсы

- **Память, а не процессор — узкое место.** Окно в 1 час ≈ 7.7 млн потоков ≈ 3 ГБ в памяти; при 8 ГБ ОЗУ
  параллельные прогоны и тяжёлые тесты рядом загоняют процесс в swap, и он замедляется в разы.
  Запускайте один прогон и ничего тяжёлого рядом.
- **Когорта больше ~10 тыс. хостов не помещается**: метрики по общим собеседникам строят матрицу хосты × хосты.
  Отсюда порог ≥ 300 событий.
- **Хабы** (`ActiveDirectory`, почтовые серверы) делают struc2vec тяжёлым: кольца соседей на расстоянии 2–3
  почти совпадают со всей сетью.
- Для больших окон и всех дней нужна машина с 32+ ГБ ОЗУ.
