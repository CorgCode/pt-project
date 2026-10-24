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
| node2vec | 17.6 → **23.8**: соседи 0.073 → 0.088, связь 0.78 → 0.82, новые связи 0.56 → 0.69, узнаваемость выше |
| DeepWalk | 16.3 → **18.3**: соседи и новые связи лучше, но пары 0.98 → 0.91 и новые собеседники хуже |
| гибрид поведение×4 + node2vec (сервисы) + спектр | 33.6 → **35.4** — лучший в таблице |

Сервисы помогают узнавать роль и новые связи, но размывают «кто с кем общается» — поэтому пары и
новые собеседники падают. В гибриде это закрывает спектральный граф.

## 4. Перенести итоги в репозиторий

```bash
python experiments/lanl/publish.py $W/lanl results/lanl_day02
```

## Несколько дней за ночь (#60)

Одна команда: скачать → подготовить → обучить (все методы, без портов, граф host–service) → перенести итоги.

```bash
git clone https://github.com/CorgCode/pt-project && cd pt-project
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
LANL_TOKEN=<токен> PY=.venv/bin/python experiments/lanl/night.sh ~/lanl 03 04 05
```

- Окна — те же часы суток, что в дне 02, сдвинутые на сутки. День ≈ 3 часа (16 ГБ), три дня — на ночь.
- Упавший день не останавливает остальные; повторный запуск продолжит с места остановки.
- Архив дня удаляется после подготовки (`KEEP_RAW=1` — оставить). Свободного места — от 15 ГБ.
- struc2vec не входит (дольше и хуже всех): `METHODS="deepwalk node2vec graphsage rolx struc2vec"`.
- Прогресс: `LANL_WORKDIR=~/lanl/day03 experiments/lanl/status.sh --watch`.
- Итоги — `results/lanl_dayNN` и `results/lanl_dayNN_services`; смотреть — `streamlit run app/streamlit_app.py`.

## Проверка на нескольких днях

Один день и полчаса test — мало: метод может просто угадать этот отрезок. Таблица с проверкой на трёх днях:
train — день 02, 11:00–12:00; test — по часу (12:30–13:30) в днях 02, 03 и 04. Итог — среднее по дням,
в разделе «По дням» — score на каждом. День 05 не трогаем: он для последней проверки, когда методы выбраны.

```bash
LANL_TOKEN=<токен> NOTE="форма LANL, <дата>, <кто>" experiments/lanl/multiday.sh ~/lanl_multiday
```

Скрипт докачает дни 03 и 04, подготовит данные (`config_multiday.json`), построит эталоны, обучит все методы
и гибриды и пересоберёт `scoreboard/`. С установленным torch добавится и `temporal_encoder`. Train 6 часов: `CONFIG=experiments/lanl/config_multiday_train6h.json`
(нужно больше памяти, см. [BIG_MACHINE.md](BIG_MACHINE.md)). Окна и день 05 выбраны до просмотра результатов.

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

Обучение на 6 часах и проверка на другом дне — [BIG_MACHINE.md](BIG_MACHINE.md).

Быстрый временной графовый энкодер и PCA-сжатие — [код, запуск и ограничения](fast_temporal/README.md), [результаты двух окон](../../results/fast_temporal_day02/README.md).
