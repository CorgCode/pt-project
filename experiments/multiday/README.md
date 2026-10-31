# Обучение на нескольких днях

Отдельный протокол: модель учится на истории дней 02–12, а не на одном часе дня 02.
**С исходной таблицей не сравнивается** — там правило «1 час train». Числа ниже — на своих
закрытых днях, у каждого этапа свой контроль на тех же данных.

## Итог

Закрытый день 14, 12:30–13:30, 1581 оцениваемый хост, проверка один раз после фиксации модели:

| Модель | Score | New partners, MRR | Identity, Top-1 |
|---|---:|---:|---:|
| `temporal_encoder`, 1 час дня 02 | 54.3 | 0.128 | 0.438 |
| кодировщик на 32 часах (этап 3) | 58.3 | 0.126 | 0.628 |
| **codebook с историей до дня 12** (этап 5) | **63.3** | **0.172** | **0.648** |

+5.1 к прежнему многодневному кодировщику, +9.1 к `temporal_encoder`. Рост не во всех задачах:
Neighbors 0.304 → 0.294, Links AUC 0.943 → 0.936. Интервалов на общий score нет.

## Протокол

- Каждый этап: выбор по validation-дню → фиксация модели и хешей кода → один раз закрытый test-день.
  Test-день этапа дальше используется только как validation/development.
- Когорта исходная (6453 хоста), эталоны строятся так же, как в таблице, потолок Neighbors — persistence на том же дне.
- Test открывается, только если прирост на validation ≥ 1 балла (порог фиксируется заранее в `protocol.json`).

## Этапы

Каждый этап читает итоги предыдущего из `$PT_ML_ROOT` (по умолчанию `/Volumes/KINGSTON/PT_PROJECT_ML`).

| # | Модуль | Что делает | Закрытый test | Результат |
|---|---|---|---|---|
| 1 | `extended` | train — 11:00–12:00 дней 02–07; семейства `gain`, `roles`, `posterior`, `dc` | день 09 | 66.5 против 59.9 у `temporal_encoder` |
| 2 | `search` | отпечатки трафика (`memory`, `bag`) поверх победителя этапа 1 | день 11 | 69.1 против 65.4 |
| 3 | `expanded_search` | 32 часовых окна (часы 10–13 дней 02–09), `expanded_data`, `expanded_train` | день 12 | 69.1 против 69.0 — прироста нет |
| 4 | `quick`, `donor_search`, `transfer`, `aligned`, `bigraph`, `listwise`, `recent`, `codebook` | доноры контактов, ранжирование, обучаемые профили источника | — | порог validation не пройден, test не открывался |
| 5 | `codebook_recent` | профили источника с целями дней 03–12 и адаптер трафика | день 14 | 63.3 против 58.3 |

Служебные: `prepare` (подготовка дня из архива), `reference` (эталоны окна), `finish` (фиксация и итоговая оценка).

## Запуск

```bash
pip install -r requirements.txt -r experiments/multiday/requirements.txt
export PT_ML_ROOT=/путь/PT_PROJECT_ML PT_COMPACT=1
python -m experiments.multiday.extended --work $PT_ML_ROOT/multiday_gain
python -m experiments.multiday.search --work $PT_ML_ROOT/fingerprint_memory
python -m experiments.multiday.expanded_search --work $PT_ML_ROOT/fingerprint_expanded
python -m experiments.multiday.donor_search --work $PT_ML_ROOT/fingerprint_donors
python -m experiments.multiday.transfer --work $PT_ML_ROOT/fingerprint_transfer
python -m experiments.multiday.aligned --work $PT_ML_ROOT/fingerprint_aligned
python -m experiments.multiday.recent --work $PT_ML_ROOT/fingerprint_recent
python -m experiments.multiday.codebook_recent --work $PT_ML_ROOT/fingerprint_codebook_recent
```

Нужны архивы `netflow_day-02…14.bz2` в `$PT_ML_ROOT/multiday/raw` (`experiments/lanl/fetch.sh`, ~1.4 ГБ на день)
и подготовленный день 02 в `$PT_ML_ROOT/multiday/processed`. Повторный запуск продолжает с места остановки.
Итог каждого этапа — `<work>/final/REPORT.md`, `summary.json`, `frozen_selection.json`.

## Ограничения

- Другой объём данных: заметная часть прироста к `temporal_encoder` — просто больше истории.
- Дни 10, 12, 13 многократно использовались для выбора; честные числа — только закрытые test-дни этапов.
- Модели и отчёты лежат на SSD, в git — только код. Хеши кода в `frozen_selection.json` старых прогонов
  указывают на прежние имена файлов (`experiments/fingerprint_*.py`), поэтому продолжать старые прогоны — из исходной ветки.
